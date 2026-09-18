
"""
Splitting of approximately spherical 3D objects at convexity defects.

Adapted from the original `split_along_convexity_defects_3D` with the
following changes:

* anisotropic voxel sizes are handled everywhere (EDT, marching cubes,
  plane geometry, volume/solidity), instead of being silently ignored;
* the cut plane normal is fitted to the defect ring itself (weighted PCA)
  instead of being taken from the vector joining the two strongest EDT
  maxima, with the peak-to-peak direction kept only as a fallback;
* the defect ring is recovered from *all* components above a depth
  threshold that is relative to the deepest defect, and ring fragments
  are merged, so a ring broken by a fuzzy surface is not truncated;
* seeds come from the h-maxima of a smoothed EDT rather than from raw
  `local_maxima`, which is what makes this survive fuzzy boundaries;
* the plane is used to *group seeds*, not to cut directly. A watershed
  guided by that grouping is generated as one candidate, alongside a
  hard planar cut and a plane-independent EDT-seeded watershed, so the
  interface is free to curve the way two spheres actually meet;
* candidates are no longer tried in a fixed order until one "works".
  All of them are generated, filtered for structural validity (and,
  where possible, for improving solidity over the parent), and the
  survivor with the smallest interface area is chosen -- the cut
  through the narrowest neck, in the spirit of bottleneck-detection
  clump-splitting methods;
* an explicit gate decides *whether* to split at all (solidity + seed
  count), separate from *which* split wins among the candidates.

Public API:
    should_split_3D(mask, voxel_size=None, ...)      -> bool
    split_along_convexity_defects_3D(ID, lab, max_ID, ...) -> (lab, was_split, IDs)
"""

import cv2
import numpy as np
import scipy.ndimage
import scipy.sparse
import scipy.sparse.csgraph
import scipy.spatial

import skimage.measure
import skimage.morphology
import skimage.segmentation

from . import core

CONNECTIVITY_3D = np.ones((3, 3, 3), dtype=bool)

# depth below the convex hull, as a fraction of the deepest defect, above
# which a surface vertex is considered part of the defect ring
DEFECT_DEPTH_FRAC = 0.5
# max ratio (smallest / middle eigenvalue) for a defect cloud to count as
# a planar ring. A ring gives ~0, a compact blob gives ~1.
MAX_RINGNESS = 0.35


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------

def _as_spacing(voxel_size, ndim=3):
    """Physical voxel size as a length-`ndim` array, in (Z, Y, X) order."""
    if voxel_size is None:
        return np.ones(ndim, dtype=float)
    spacing = np.asarray(voxel_size, dtype=float).ravel()
    if spacing.size == 1:
        spacing = np.repeat(spacing, ndim)
    if spacing.size != ndim:
        raise ValueError(
            f'`voxel_size` must have 1 or {ndim} elements, got {spacing.size}.'
        )
    if np.any(spacing <= 0):
        raise ValueError(f'`voxel_size` must be strictly positive, got {spacing}.')
    return spacing


def _surface_mesh(mask, spacing):
    """Marching-cubes surface of `mask`, in physical units. None if degenerate."""
    padded_mask = np.pad(mask, 1)
    try:
        vertices, faces, _, _ = skimage.measure.marching_cubes(
            padded_mask.astype(np.uint8), level=0.5, spacing=tuple(spacing)
        )
    except (RuntimeError, ValueError):
        return None
    if len(vertices) < 4 or len(faces) == 0:
        return None
    # undo the padding, in physical units
    vertices = vertices - spacing
    return vertices, faces


def _hull_depths(vertices, block=20_000):
    """
    Depth of every vertex below the convex hull.

    Hull facet equations satisfy ``v @ n + d <= 0`` inside the hull, so
    ``-(v @ n + d)`` is the distance to that facet plane and the minimum
    over facets is the depth below the hull surface. Computed in blocks
    because the dense (n_vertices x n_facets) product is the memory
    bottleneck on large meshes.
    """
    try:
        hull = scipy.spatial.ConvexHull(vertices)
    except scipy.spatial.QhullError:
        return None, None
    equations = hull.equations
    depths = np.empty(len(vertices), dtype=float)
    for start in range(0, len(vertices), block):
        stop = start + block
        chunk = vertices[start:stop]
        # note the parentheses: the negation must happen before the min
        # over facets, otherwise this returns the distance to the
        # *farthest* hull plane instead of the depth below the hull
        depths[start:stop] = (
            -(chunk @ equations[:, :-1].T + equations[:, -1])
        ).min(axis=1)
    return depths, hull


def _solidity(mask, spacing, hull=None, vertices=None):
    """
    Volume / convex-hull-volume, in physical units.

    Cheap compared to `regionprops.solidity`, which rasterises the hull.
    Values close to 1 mean the object has no convexity defect worth
    splitting on.
    """
    if hull is None:
        if vertices is None:
            mesh = _surface_mesh(mask, spacing)
            if mesh is None:
                return 1.0
            vertices = mesh[0]
        try:
            hull = scipy.spatial.ConvexHull(vertices)
        except scipy.spatial.QhullError:
            return 1.0
    volume = float(mask.sum()) * float(np.prod(spacing))
    if hull.volume <= 0:
        return 1.0
    return float(np.clip(volume / hull.volume, 0.0, 1.0))


def _smoothed_edt(mask, spacing, sigma):
    """
    EDT in physical units, Gaussian-smoothed with `sigma` in physical units.

    Smoothing the *distance transform* (not the mask) is what suppresses
    the spurious maxima that a fuzzy, ragged boundary produces, without
    eroding the object.
    """
    distance = scipy.ndimage.distance_transform_edt(mask, sampling=spacing)
    if sigma and sigma > 0:
        distance = scipy.ndimage.gaussian_filter(distance, sigma / spacing)
        distance[~mask] = 0.0
    return distance


def _h_maxima_seeds(distance, h_frac=0.25, h=None):
    """
    Seeds as h-maxima of the EDT, sorted by peak distance (descending).

    h-maxima merge maxima whose prominence over the connecting saddle is
    below `h`, which is a contrast criterion. This is far more robust on
    fuzzy masks than `peak_local_max(min_distance=...)` or plain
    `local_maxima`, which return every plateau on a noisy boundary.

    Returns (seed_labels, n_seeds, peak_values_sorted, label_order).
    """
    dmax = float(distance.max())
    if dmax <= 0:
        return None, 0, None, None
    if h is None:
        h = h_frac * dmax
    h = max(float(h), 1e-6)

    maxima = skimage.morphology.h_maxima(distance, h)
    seed_labels, n_seeds = scipy.ndimage.label(maxima, structure=CONNECTIVITY_3D)
    if n_seeds == 0:
        return None, 0, None, None

    index = np.arange(1, n_seeds + 1)
    peaks = np.asarray(
        scipy.ndimage.maximum(distance, seed_labels, index), dtype=float
    )
    order = index[np.argsort(peaks)[::-1]]
    return seed_labels, n_seeds, np.sort(peaks)[::-1], order


def _seed_peak_coords(distance, seed_labels, labels):
    """Voxel coordinate of the EDT peak inside each given seed component."""
    coords = []
    for label in labels:
        component = seed_labels == label
        masked = np.where(component, distance, -np.inf)
        coords.append(np.unravel_index(np.argmax(masked), distance.shape))
    return [np.asarray(c, dtype=float) for c in coords]


# ----------------------------------------------------------------------
# convexity defect -> cut plane
# ----------------------------------------------------------------------

def _defect_ring(vertices, faces, depths, depth_threshold, merge_radius):
    """
    Largest connected group of deep vertices, with nearby fragments merged.

    A fuzzy surface breaks the concave ring into arcs; connecting
    components whose vertices come within `merge_radius` of each other
    stitches those arcs back into one ring, instead of keeping a single
    arc and biasing the plane origin towards it.
    """
    deep = depths >= depth_threshold
    if deep.sum() < 6:
        return None

    edges = np.concatenate((faces[:, :2], faces[:, 1:], faces[:, ::2]), axis=0)
    edges = edges[np.all(deep[edges], axis=1)]
    if len(edges) == 0:
        return None

    n_vertices = len(vertices)
    adjacency = scipy.sparse.coo_matrix(
        (
            np.ones(2 * len(edges), dtype=bool),
            (
                np.concatenate((edges[:, 0], edges[:, 1])),
                np.concatenate((edges[:, 1], edges[:, 0])),
            ),
        ),
        shape=(n_vertices, n_vertices),
    )
    _, components = scipy.sparse.csgraph.connected_components(adjacency)
    components = components.copy()
    components[~deep] = -1

    labels = np.unique(components[components >= 0])
    if labels.size == 0:
        return None

    # merge ring fragments that are spatially close
    if labels.size > 1 and merge_radius > 0:
        centres = {}
        for label in labels:
            centres[label] = vertices[components == label]
        tree_labels = list(labels)
        parent = {label: label for label in tree_labels}

        def _find(label):
            while parent[label] != label:
                parent[label] = parent[parent[label]]
                label = parent[label]
            return label

        for i, label_i in enumerate(tree_labels):
            tree_i = scipy.spatial.cKDTree(centres[label_i])
            for label_j in tree_labels[i + 1:]:
                if tree_i.query(centres[label_j], distance_upper_bound=merge_radius)[0].min() < merge_radius:
                    parent[_find(label_j)] = _find(label_i)
        merged = np.full(components.shape, -1, dtype=np.int64)
        for label in tree_labels:
            merged[components == label] = _find(label)
        components = merged
        labels = np.unique(components[components >= 0])

    # keep the group carrying the most total defect depth, not the most
    # vertices: a large shallow dimple should not outvote a deep neck
    scores = [depths[components == label].sum() for label in labels]
    best = labels[int(np.argmax(scores))]
    selection = components == best
    if selection.sum() < 6:
        return None
    return vertices[selection], depths[selection]


def _fit_plane(points, weights):
    """
    Weighted PCA plane fit. Returns (origin, normal, ringness).

    For a concave ring the covariance has one small eigenvalue (out of
    plane) and two comparable large ones (in plane), so the eigenvector
    of the smallest eigenvalue is the cut normal. `ringness` is the ratio
    of the two smallest eigenvalues: ~0 for a clean ring, ~1 for a blob,
    and it is the signal that the fit should not be trusted.
    """
    origin = np.average(points, axis=0, weights=weights)
    centred = points - origin
    covariance = (centred * weights[:, None]).T @ centred / weights.sum()
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, 0]
    normal = normal / np.linalg.norm(normal)
    ringness = float(eigenvalues[0] / max(eigenvalues[1], 1e-12))
    return origin, normal, ringness


def _convexity_defect_plane_3D(
        mask,
        spacing,
        depth_frac=DEFECT_DEPTH_FRAC,
        min_defect_depth=None,
        max_ringness=MAX_RINGNESS,
        peak_coords=None,
    ):
    """
    Cut plane through the dominant convexity defect.

    Returns ``(origin, normal)`` in physical units, or None if the object
    has no defect deep enough, or if the defect is not ring-like and no
    fallback direction is available.
    """
    mesh = _surface_mesh(mask, spacing)
    if mesh is None:
        return None
    vertices, faces = mesh

    depths, _ = _hull_depths(vertices)
    if depths is None:
        return None

    max_depth = float(depths.max())
    if min_defect_depth is None:
        min_defect_depth = float(spacing.max())
    if max_depth < min_defect_depth:
        # object is convex to within one voxel: nothing to split on
        return None

    threshold = max(min_defect_depth, depth_frac * max_depth)
    ring = _defect_ring(
        vertices, faces, depths, threshold, merge_radius=4.0 * float(spacing.max())
    )
    if ring is None:
        return None
    ring_points, ring_depths = ring

    origin, normal, ringness = _fit_plane(ring_points, ring_depths)

    if ringness > max_ringness:
        # defect cloud is not planar; fall back to the EDT peak-to-peak
        # direction through the same origin, if two peaks are known
        if peak_coords is None or len(peak_coords) < 2:
            return None
        fallback = (peak_coords[1] - peak_coords[0]) * spacing
        norm = np.linalg.norm(fallback)
        if norm == 0:
            return None
        normal = fallback / norm

    # sanity check: the two strongest EDT peaks should end up on opposite
    # sides of the plane, otherwise the plane does not separate the lobes
    if peak_coords is not None and len(peak_coords) >= 2:
        signed = [
            float((coord * spacing - origin) @ normal) for coord in peak_coords[:2]
        ]
        if signed[0] * signed[1] > 0:
            return None

    return origin, normal


def _signed_distance_to_plane(shape, spacing, origin, normal):
    """Signed plane distance for every voxel of a crop, in physical units."""
    grids = np.meshgrid(
        *[np.arange(size, dtype=float) * step for size, step in zip(shape, spacing)],
        indexing='ij',
    )
    signed = np.zeros(shape, dtype=float)
    for grid, component, shift in zip(grids, normal, origin):
        signed += (grid - shift) * component
    return signed


# ----------------------------------------------------------------------
# splitting
# ----------------------------------------------------------------------

def _markers_from_plane(mask, distance, signed, seed_labels, n_seeds, spacing):
    """
    Two-label marker image: seeds grouped by which side of the plane they
    fall on.

    The plane decides the *grouping*; the watershed decides the actual
    interface. That is the point of doing it this way rather than cutting
    on the plane: the cut is free to curve into the EDT valley, which is
    where the true neck is, while still being anchored at the concavity.
    """
    markers = np.zeros(mask.shape, dtype=np.int32)

    if seed_labels is not None and n_seeds > 0:
        index = np.arange(1, n_seeds + 1)
        centroids = np.asarray(
            scipy.ndimage.center_of_mass(seed_labels > 0, seed_labels, index),
            dtype=float,
        ).reshape(-1, 3)
        sides = np.array([
            signed[tuple(np.round(centroid).astype(int))] >= 0
            for centroid in centroids
        ])
        negative = index[~sides]
        positive = index[sides]
        if negative.size:
            markers[np.isin(seed_labels, negative)] = 1
        if positive.size:
            markers[np.isin(seed_labels, positive)] = 2

    # if a side received no seed, plant one at its deepest point
    for marker_ID, side_mask in ((1, signed < 0), (2, signed >= 0)):
        if np.any(markers == marker_ID):
            continue
        candidate = mask & side_mask
        if not candidate.any():
            return None
        masked = np.where(candidate, distance, -np.inf)
        markers[np.unravel_index(np.argmax(masked), mask.shape)] = marker_ID

    return markers


def _markers_from_seeds(distance, seed_labels, order):
    """Two-label marker image from the two strongest EDT seeds."""
    markers = np.zeros(seed_labels.shape, dtype=np.int32)
    markers[seed_labels == order[0]] = 1
    markers[seed_labels == order[1]] = 2
    return markers


def _watershed_two(mask, distance, markers, compactness):
    """Compact watershed on the inverted EDT, restricted to the object."""
    labels = skimage.segmentation.watershed(
        -distance, markers=markers, mask=mask, compactness=compactness
    )
    first = labels == 1
    second = labels == 2
    if not first.any() or not second.any():
        return None
    return first, second


def _interface_area(first, second, spacing):
    """
    Physical area of the boundary between two disjoint voxel sets.

    Sum of voxel-face areas wherever a `first` voxel is 6-connected to a
    `second` voxel. This is the discrete cut-surface area used in
    graph-cut / min-cut formulations of clump splitting (cf. bottleneck
    detection, Wang/Zhang/Ray): the candidate with the smallest such area
    is the one passing through the narrowest neck, which for
    near-spherical, mutually overlapping objects is exactly where the
    boundary between them should sit. Two voxel sets that share no face
    (e.g. separated by a plane that leaves a diagonal-only contact) score
    correctly as having zero shared-face area at that contact.
    """
    area = 0.0
    ndim = first.ndim
    for axis in range(ndim):
        face_area = float(np.prod([spacing[a] for a in range(ndim) if a != axis]))
        sl_lo = [slice(None)] * ndim
        sl_hi = [slice(None)] * ndim
        sl_lo[axis] = slice(0, -1)
        sl_hi[axis] = slice(1, None)
        lo, hi = tuple(sl_lo), tuple(sl_hi)
        n_faces = (
            np.count_nonzero(first[lo] & second[hi])
            + np.count_nonzero(second[lo] & first[hi])
        )
        area += n_faces * face_area
    return area


def _valid_partition(first, second, min_volume_frac):
    """
    Structural validity only: both halves non-empty and not wildly
    unbalanced. Kept separate from the solidity check below so that
    validity and "is this an improvement" can be tested independently.
    """
    if not first.any() or not second.any():
        return False
    sizes = np.array([first.sum(), second.sum()], dtype=float)
    return bool(sizes.min() >= min_volume_frac * sizes.sum())


def _solidity_improves(parent_solidity, first, second, spacing):
    """Whether a split made both children rounder than the parent."""
    child_solidities = [_solidity(child, spacing) for child in (first, second)]
    return min(child_solidities) > parent_solidity


def should_split_3D(
        mask,
        voxel_size=None,
        min_solidity=0.93,
        smooth_sigma=1.0,
        h_maxima_frac=0.25,
    ):
    """
    Whether an object is worth splitting at all.

    Two independent votes: a convexity vote (solidity below threshold)
    and a multiplicity vote (more than one h-maximum in the smoothed
    EDT). Either one is enough.
    """
    mask = np.asarray(mask, dtype=bool)
    spacing = _as_spacing(voxel_size, mask.ndim)
    if not mask.any():
        return False
    if _solidity(mask, spacing) < min_solidity:
        return True
    distance = _smoothed_edt(mask, spacing, smooth_sigma)
    _, n_seeds, _, _ = _h_maxima_seeds(distance, h_frac=h_maxima_frac)
    return n_seeds > 1


def split_along_convexity_defects_3D(
        ID, lab, max_ID, max_i=1, eps_percent=0.01, rp=None,
        voxel_size=None,
        smooth_sigma=1.0,
        h_maxima_frac=0.25,
        compactness=0.01,
        min_volume_frac=0.1,
        min_solidity=0.93,
        require_improvement=True,
        gate_on_solidity=True,
    ):
    """
    Split the object `ID` in the 3D label image `lab` at its dominant
    convexity defect.

    Parameters
    ----------
    ID, lab, max_ID, max_i, eps_percent, rp
        As in the original function. `eps_percent` is kept for API
        compatibility and is unused.
    voxel_size : float or (dz, dy, dx), optional
        Physical voxel size. Anisotropic stacks *must* pass this or the
        EDT saddle sits in the wrong place and the cut plane tilts.
    smooth_sigma : float
        Gaussian sigma applied to the EDT, in physical units. This is the
        main knob for fuzzy masks.
    h_maxima_frac : float
        Seed prominence, as a fraction of the object's maximum EDT value.
        Larger = fewer seeds = fewer splits.
    compactness : float
        Compact-watershed weight. Small positive values bias basins
        towards round shapes, which suits near-spherical objects and
        suppresses leaky cuts along fuzzy borders. 0 disables it.
    min_volume_frac : float
        Reject a split whose smaller child is below this fraction of the
        parent volume.
    min_solidity, gate_on_solidity : float, bool
        Objects at or above `min_solidity` with a single EDT seed are
        left alone.
    require_improvement : bool
        Prefer candidates where both children are more solid than the
        parent. If none of the generated candidates qualify, this is not
        a hard rejection: splitting falls back to ranking all
        structurally valid candidates by interface area instead.

    Returns
    -------
    (lab, was_split, IDs)

    Notes
    -----
    Internally this generates up to three candidate splits (a
    defect-plane-guided watershed, a hard planar cut through the same
    plane, and a plane-independent EDT-seeded watershed), filters them,
    and picks the one whose interface -- the boundary surface between
    the two children -- has the smallest physical area. This favours the
    cut that passes through the narrowest neck between two overlapping
    near-spherical objects, rather than whichever candidate happened to
    be tried first.
    """
    if lab.ndim != 3:
        raise ValueError(f'Expected a 3D label image, got {lab.ndim}D.')

    spacing = _as_spacing(voxel_size, 3)

    if rp is not None:
        obj = rp.get_obj_from_ID(ID)
        lab_ID_bool = np.zeros_like(lab[obj.slice], dtype=bool)
        lab_ID_bool[obj.image] = True
        obj_slice = obj.slice
    else:
        lab_ID_bool = lab == ID
        obj_slice = np.s_[:, :, :]

    if not lab_ID_bool.any():
        return lab, False, []

    # ------------------------------------------------------------------
    # 1. already-disconnected components: relabel, no geometry needed
    # ------------------------------------------------------------------
    components = skimage.measure.label(lab_ID_bool, connectivity=3)
    components_rp = skimage.measure.regionprops(components)
    if len(components_rp) > 1:
        components_out = np.zeros_like(components, dtype=lab.dtype)
        components_rp.sort(key=lambda component: component.area, reverse=True)
        separateIDs = [ID]
        for component_i, component in enumerate(components_rp):
            component_ID = ID if component_i == 0 else max_ID + max_i
            components_out[component.slice][component.image] = component_ID
            if component_i > 0:
                separateIDs.append(component_ID)
                max_i += 1
        lab[obj_slice][lab_ID_bool] = components_out[lab_ID_bool]
        return lab, True, separateIDs

    # ------------------------------------------------------------------
    # 2. EDT, seeds, and the decision of whether to split at all
    # ------------------------------------------------------------------
    distance = _smoothed_edt(lab_ID_bool, spacing, smooth_sigma)
    seed_labels, n_seeds, _, order = _h_maxima_seeds(
        distance, h_frac=h_maxima_frac
    )

    parent_solidity = _solidity(lab_ID_bool, spacing)
    if gate_on_solidity and parent_solidity >= min_solidity and n_seeds < 2:
        return lab, False, []

    peak_coords = None
    if n_seeds >= 2:
        peak_coords = _seed_peak_coords(distance, seed_labels, order[:2])

    # ------------------------------------------------------------------
    # 3. generate every candidate split this function knows how to make.
    # Nothing is accepted or rejected yet -- that happens in step 4.
    # ------------------------------------------------------------------
    candidates = []  # list of (first, second, method_name)

    defect_plane = _convexity_defect_plane_3D(
        lab_ID_bool, spacing, peak_coords=peak_coords
    )
    if defect_plane is not None:
        plane_origin, plane_normal = defect_plane
        signed = _signed_distance_to_plane(
            lab_ID_bool.shape, spacing, plane_origin, plane_normal
        )

        # 3a. defect-plane-guided watershed: the plane groups the seeds,
        # the watershed finds where the surface actually narrows
        markers = _markers_from_plane(
            lab_ID_bool, distance, signed, seed_labels, n_seeds, spacing
        )
        if markers is not None:
            plane_watershed = _watershed_two(lab_ID_bool, distance, markers, compactness)
            if plane_watershed is not None:
                candidates.append((*plane_watershed, 'plane_watershed'))

        # 3b. hard planar cut through the same plane. Not just a last
        # resort: for two cleanly overlapping, similarly sized spheres
        # this can legitimately be the minimal-area cut and win step 4.
        first_planar = lab_ID_bool & (signed < 0)
        second_planar = lab_ID_bool & (signed >= 0)
        candidates.append((first_planar, second_planar, 'planar_cut'))

    # 3c. EDT-seeded watershed on the two strongest h-maxima, independent
    # of whether a usable convexity defect was found at all
    if n_seeds >= 2:
        markers = _markers_from_seeds(distance, seed_labels, order)
        edt_watershed = _watershed_two(lab_ID_bool, distance, markers, compactness)
        if edt_watershed is not None:
            candidates.append((*edt_watershed, 'edt_watershed'))

    # ------------------------------------------------------------------
    # 4. keep structurally valid candidates; prefer ones that improve
    # solidity, but don't refuse to split just because none happen to
    # (the area ranking below already favours the most natural cut);
    # then take the candidate with the smallest interface area, i.e.
    # the narrowest neck between the two children
    # ------------------------------------------------------------------
    candidates = [
        candidate for candidate in candidates
        if _valid_partition(candidate[0], candidate[1], min_volume_frac)
    ]
    if require_improvement:
        improving = [
            candidate for candidate in candidates
            if _solidity_improves(parent_solidity, candidate[0], candidate[1], spacing)
        ]
        if improving:
            candidates = improving

    if not candidates:
        return lab, False, []

    first, second, method = min(
        candidates,
        key=lambda candidate: _interface_area(candidate[0], candidate[1], spacing),
    )

    # ------------------------------------------------------------------
    # 5. write the two children back into the label image
    # ------------------------------------------------------------------
    if second.sum() > first.sum():
        first, second = second, first

    ID1 = ID
    ID2 = max_ID + max_i
    split_lab = np.zeros(lab_ID_bool.shape, dtype=lab.dtype)
    split_lab[first] = ID1
    split_lab[second] = ID2
    # voxels the watershed left unassigned stay with the larger child
    unassigned = lab_ID_bool & (split_lab == 0)
    if unassigned.any():
        split_lab[unassigned] = ID1

    lab[obj_slice][lab_ID_bool] = split_lab[lab_ID_bool]
    return lab, True, [ID1, ID2]


def split_all_along_convexity_defects_3D(lab, voxel_size=None, max_iter=3, **kwargs):
    """
    Apply the splitter to every object, repeatedly, until nothing splits.

    Repetition matters for clumps of three or more: each pass separates
    one lobe, and the children are re-examined on the next pass.
    """
    lab = lab.copy()
    for _ in range(max_iter):
        was_split_any = False
        for ID in np.unique(lab[lab > 0]):
            max_ID = int(lab.max())
            lab, was_split, _ = split_along_convexity_defects_3D(
                ID, lab, max_ID, max_i=1, voxel_size=voxel_size, **kwargs
            )
            was_split_any |= was_split
        if not was_split_any:
            break
    return lab

def convexity_defects(img, eps_percent):
    img = img.astype(np.uint8)
    contours, _ = cv2.findContours(img,2,1)
    cnt = max(contours, key=cv2.contourArea)
    cnt = cv2.approxPolyDP(cnt,eps_percent*cv2.arcLength(cnt,True),True) # see https://www.programcreek.com/python/example/89457/cv22.convexityDefects
    hull = cv2.convexHull(cnt,returnPoints = False) # see https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
    defects = cv2.convexityDefects(cnt,hull) # see https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
    return cnt, defects

def split_connected_components(lab, rp=None, max_ID=None):  
    if rp is None:
        lab = skimage.measure.regionprops(lab)
    
    if max_ID is None:
        max_ID = max([obj.label for obj in rp], default=1)
        
    split_occured = False
    for obj in rp:
        lab_obj = skimage.measure.label(obj.image)
        rp_lab_obj = skimage.measure.regionprops(lab_obj)
        if len(rp_lab_obj)<=1:
            continue
        lab_obj += max_ID
        _slice = obj.slice # self.getObjSlice(obj.slice)
        _objMask = obj.image # self.getObjImage(obj.image)
        lab[_slice][_objMask] = lab_obj[_objMask]
        split_occured = True
        max_ID += 1
    return split_occured

def split_along_convexity_defects(
        ID, lab, max_ID, max_i=1, eps_percent=0.01, rp=None
    ):
    if rp is not None:
        obj = rp.get_obj_from_ID(ID)
        lab_ID_bool = np.zeros_like(lab[obj.slice], dtype=bool)
        lab_ID_bool[obj.image] = True
    else:
        lab_ID_bool = lab == ID
    # First try separating by labelling
    lab_ID = lab_ID_bool.astype(int)
    rp_ID = skimage.measure.regionprops(lab_ID)
    split_occured = split_connected_components(lab_ID, rp=rp_ID, max_ID=max_ID)
    if split_occured:
        success = True
        if rp is not None:
            lab[obj.slice][obj.image] = lab_ID[obj.image]
        else:
            lab[lab_ID_bool] = lab_ID[lab_ID_bool]
            
        rp_ID = skimage.measure.regionprops(lab_ID)
        separateIDs = [obj.label for obj in rp_ID]
        return lab, success, separateIDs

    cnt, defects = convexity_defects(lab_ID_bool, eps_percent)
    success = False
    if defects is None:
        return lab, success, []

    if len(defects) != 2:
        return lab, success, []

    # This line is needed since opencv-python-headless > 5.0
    defects = np.asarray(defects).reshape(-1, 4)
    defects_points = [0]*len(defects)
    for i, defect in enumerate(defects):
        s,e,f,d = defect
        x,y = tuple(cnt[f][0])
        defects_points[i] = (y,x)
    (r0, c0), (r1, c1) = defects_points
    rr, cc, _ = skimage.draw.line_aa(r0, c0, r1, c1)
    sep_bud_img = np.copy(lab_ID_bool)
    sep_bud_img[rr, cc] = False
    
    sep_bud_label = skimage.measure.label(
        sep_bud_img, connectivity=2
    )
    
    rp_sep = skimage.measure.regionprops(sep_bud_label)
    if len(rp_sep) != 2:
        return lab, False, []

    IDs_sep = [obj.label for obj in rp_sep]
    areas = [obj.area for obj in rp_sep]
    bud_idx, moth_idx = np.argsort(areas)
    curr_ID_bud = IDs_sep[bud_idx]
    curr_ID_moth = IDs_sep[moth_idx]
    orig_sblab = np.copy(sep_bud_label)
    # sep_bud_label = np.zeros_like(sep_bud_label)
    ID1 = ID
    ID2 = max_ID+max_i
    sep_bud_label[orig_sblab==curr_ID_moth] = ID1
    sep_bud_label[orig_sblab==curr_ID_bud] = ID2
    splittedIDs = [ID1, ID2]
    # sep_bud_label *= (max_ID+max_i)
    temp_sep_bud_lab = sep_bud_label.copy()
    for r, c in zip(rr, cc):
        if lab_ID_bool[r, c]:
            nearest_ID = core.nearest_nonzero_2D(sep_bud_label, r, c)
            temp_sep_bud_lab[r,c] = nearest_ID
    sep_bud_label = temp_sep_bud_lab
    sep_bud_label_mask = sep_bud_label != 0
    # plt.imshow_tk(sep_bud_label, dots_coords=np.asarray(defects_points))
    if rp is not None:
        lab[obj.slice][sep_bud_label_mask] = sep_bud_label[sep_bud_label_mask]
    else:
        lab[sep_bud_label_mask] = sep_bud_label[sep_bud_label_mask]
    max_i += 1
    success = True
    return lab, success, splittedIDs

