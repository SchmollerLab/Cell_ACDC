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

import scipy.ndimage as ndimage

CONNECTIVITY_3D = np.ones((3, 3, 3), dtype=bool)

# depth below the convex hull, as a fraction of the deepest defect, above
# which a surface vertex is considered part of the defect ring
DEFECT_DEPTH_FRAC = 0.2
# max ratio (smallest / middle eigenvalue) for a defect cloud to count as
# a planar ring. A ring gives ~0, a compact blob gives ~1.
MAX_RINGNESS = 0.35


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------

def _cut_side_mask(shape, curve_y, curve_x):
    """Rasterize the drawn curve and endpoint-line extensions inside an ROI."""
    height, width = shape
    if height < 2 or width < 2 or len(curve_y) < 2:
        return np.zeros(shape, dtype=bool)

    curve = np.column_stack((curve_y, curve_x)).astype(float)
    direction = curve[-1] - curve[0]
    if not np.any(direction):
        return np.zeros(shape, dtype=bool)

    intersections = []
    y0, x0 = curve[0]
    dy, dx = direction
    if dx:
        for x in (0, width - 1):
            t = (x - x0) / dx
            y = y0 + t * dy
            if 0 <= y <= height - 1:
                intersections.append((t, (y, x)))
    if dy:
        for y in (0, height - 1):
            t = (y - y0) / dy
            x = x0 + t * dx
            if 0 <= x <= width - 1:
                intersections.append((t, (y, x)))
    if len(intersections) < 2:
        return np.zeros(shape, dtype=bool)
    intersections.sort(key=lambda intersection: intersection[0])
    if np.allclose(intersections[0][1], intersections[-1][1]):
        return np.zeros(shape, dtype=bool)
    start_edge = np.rint(intersections[0][1]).astype(int)
    end_edge = np.rint(intersections[-1][1]).astype(int)

    # Close the open cut along the shorter image-border path to form a
    # polygon; filling it identifies one side without assuming curve shape.
    perimeter = np.array(
        [(0, x) for x in range(width)]
        + [(y, width - 1) for y in range(1, height)]
        + [(height - 1, x) for x in range(width - 2, -1, -1)]
        + [(y, 0) for y in range(height - 2, 0, -1)],
        dtype=int
    )
    start_idx = np.sum((perimeter - start_edge)**2, axis=1).argmin()
    end_idx = np.sum((perimeter - end_edge)**2, axis=1).argmin()
    clockwise = [end_idx]
    while clockwise[-1] != start_idx:
        clockwise.append((clockwise[-1] + 1) % len(perimeter))
    counterclockwise = [end_idx]
    while counterclockwise[-1] != start_idx:
        counterclockwise.append((counterclockwise[-1] - 1) % len(perimeter))
    edge_path = min((clockwise, counterclockwise), key=len)

    polygon = np.vstack((
        start_edge,
        curve,
        end_edge,
        perimeter[edge_path],
    ))
    side_mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(side_mask, [polygon[:, ::-1].astype(np.int32)], 1)
    return side_mask.astype(bool)

def split_cut_components(
        previous_lab, cut_lab, cut_coords, curve_y, curve_x, max_ID,
        split_IDs=None, bbox=None
    ):
    """Split every crossed label between the two sides of the drawn cut.

    IDs touched by ``cut_coords`` are split by the side mask formed from the
    drawn curve and its endpoint extensions (see :func:`_cut_side_mask`). 
    The larger side keeps its old ID. Smaller sides reuse IDs from ``split_IDs`` 
    when available, otherwise they receive new IDs above ``max_ID``. IDs not 
    touched by the cut are unchanged.

    Parameters
    ----------
    previous_lab : ndarray
        Labels before the cut. Used to determine which IDs the cut intersects.
    cut_lab : ndarray
        Labels after the cut line has been removed. This array is modified
        in-place within ``bbox`` as split IDs are assigned.
    cut_coords : tuple of ndarray
        Row and column coordinates of cut pixels, used to find intersected IDs.
    curve_y, curve_x : array-like
        Row and column coordinates of the complete drawn curve. These define
        which side of the cut each pixel belongs to.
    max_ID : int
        Highest ID already in use; newly allocated IDs start above this value.
    split_IDs : sequence of int, optional
        Previously allocated split IDs available for reuse, typically from
        splitting the same object in an adjacent z-slice.
    bbox : tuple of int, optional
        Region of interest as ``(min_row, min_col, max_row, max_col)``.
        Processing is restricted to this half-open bounding box. If omitted,
        the bounding box of all nonzero pixels in ``previous_lab`` is used.

    Returns
    -------
    cut_lab : ndarray
        The modified label image, with untouched IDs preserved.
    max_ID : int
        The updated highest ID after allocating any new split IDs.
    new_IDs : list of int
        IDs allocated during this call (reused ``split_IDs`` are excluded).
    """
    if cut_lab.shape != previous_lab.shape:
        raise ValueError(
            'Cut and previous label images must have the same shape.'
        )
    yy, xx = cut_coords
    crossed_IDs = np.unique(previous_lab[yy, xx]) 
    # should be fast, as it only looks at the cut coordinates
    crossed_IDs = crossed_IDs[crossed_IDs != 0]
    if not crossed_IDs.size:
        return cut_lab, max_ID, []
    
    # preferred IDs are those available for reuse from previous splits
    preferred_IDs = list(split_IDs or ())
    used_preferred_IDs = set()
    new_IDs = []

    #  bbox format (min_row, min_col, max_row, max_col)
    if bbox is None:
        object_y, object_x = np.nonzero(previous_lab)
        if not object_y.size:
            return cut_lab, max_ID, new_IDs
        bbox = (
            object_y.min(), object_x.min(), object_y.max() + 1, object_x.max() + 1
        )
    min_row, min_col, max_row, max_col = bbox
    roi = np.s_[min_row:max_row, min_col:max_col]
    cut_roi = cut_lab[roi]
    
    # generate the side mask for the cut region
    side_mask = _cut_side_mask(
        cut_roi.shape,
        np.asarray(curve_y) - min_row,
        np.asarray(curve_x) - min_col,
    )

    for ID in crossed_IDs:
        side_masks = (
            (cut_roi == ID) & side_mask,
            (cut_roi == ID) & ~side_mask,
        )
        side_areas = [int(mask.sum()) for mask in side_masks]
        if not all(side_areas):
            continue

        # region with the largest area keeps its old ID
        largest_side = max(range(2), key=lambda i: side_areas[i])
        assignments = {largest_side: int(ID)}

        for side_i, mask in enumerate(side_masks):
            if side_i in assignments:
                continue
            preferred_ID = next(
                (
                    candidate for candidate in preferred_IDs
                    if candidate not in used_preferred_IDs
                    and candidate not in assignments.values()
                ),
                None
            )
            if preferred_ID is not None:
                assignments[side_i] = preferred_ID
                used_preferred_IDs.add(preferred_ID)
                continue
            max_ID += 1
            assignments[side_i] = max_ID
            new_IDs.append(max_ID)
        for side_i, mask in enumerate(side_masks):
            cut_roi[mask] = assignments[side_i]

    return cut_lab, max_ID, new_IDs


def track_split_slice(lab, neighboring_labs, unique_ID, split_IDs=None):
    """Track split labels against each adjacent slice and keep the best match.
    This function can later also be used for segmenting 2D slices and merging 
    between them."""
    from .trackers.CellACDC.CellACDC_tracker import track_frame

    if not np.any(lab):
        return lab, 0

    from . import regionprops
    current_rp = regionprops.acdcRegionprops(lab, precache_centroids=False)
    best_lab = lab
    best_track_count = -1
    split_IDs_new = split_IDs[:] if split_IDs is not None else []

    for neighbor_lab in neighboring_labs:
        if neighbor_lab is None or not np.any(neighbor_lab):
            continue
        if neighbor_lab.shape != lab.shape:
            raise ValueError(
                'Neighboring and current label images must have the same shape.'
            )

        prev_rp = regionprops.acdcRegionprops(neighbor_lab, precache_centroids=False)
        tracked_lab, add_info = track_frame(
            neighbor_lab,
            prev_rp,
            lab,
            current_rp,
            unique_ID=unique_ID,
            return_all=True,
            assign_unique_new_IDs=False,
            return_assignments=True,
        )
        assignments = add_info['assignments']
        track_count = len(assignments)
        if track_count > best_track_count:
            best_lab = tracked_lab
            best_track_count = track_count
            split_IDs_new = [assignments.get(split_ID, split_ID) 
                             for split_ID in (split_IDs or [])]

    # return the best tracked label image and the number of successful tracks
    if split_IDs is None:
        return best_lab, max(best_track_count, 0)
    
    
    return best_lab, max(best_track_count, 0), split_IDs_new


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
    Extract the dominant connected group of deep surface vertices.

    Vertices at or deeper than ``depth_threshold`` are treated as candidate
    points on the concavity ring. Their connectivity is calculated from the
    triangle-mesh edges in ``faces``, so shallow vertices cannot bridge two
    separate deep regions. On fuzzy or coarse surfaces the ring may be
    represented by disconnected arcs; if their closest vertices are within
    ``merge_radius``, those components are joined using spatial proximity.
    The resulting component with the greatest *sum* of vertex depths is
    returned. Summed depth favors a substantial deep neck over a larger but
    shallow surface indentation.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
        Surface-vertex coordinates, in physical units.
    faces : ndarray, shape (M, 3)
        Triangle mesh; each row contains indices into ``vertices``.
    depths : ndarray, shape (N,)
        Convex-hull depth for each surface vertex, in physical units.
    depth_threshold : float
        Minimum depth for a vertex to be included in the candidate ring.
    merge_radius : float
        Maximum physical distance between fragments for them to be merged.
        A merge can join components transitively through intermediate
        fragments.

    Returns
    -------
    (ring_vertices, ring_depths) : tuple of ndarray
        Coordinates and corresponding depths for the selected component.
        Returns ``None`` when fewer than six deep vertices are available,
        no candidate mesh edges connect them, or the selected component
        contains fewer than six vertices.
    """
    deep = depths >= depth_threshold
    if deep.sum() < 6:
        return None

    # Keep only triangle edges whose two endpoints pass the depth threshold;
    # this forms connected ring segments without shallow bridges.
    edges = np.concatenate((faces[:, :2], faces[:, 1:], faces[:, ::2]), axis=0)
    edges = edges[np.all(deep[edges], axis=1)]
    if len(edges) == 0:
        return None

    # Connected-components operates on a sparse graph of mesh vertices.
    # Masking shallow vertices afterward excludes isolated graph nodes.
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
        # Union-find allows close-fragment links to merge transitively.
        parent = {label: label for label in tree_labels}

        def _find(label):
            """Return a fragment's merged-set root, compressing the path."""
            while parent[label] != label:
                parent[label] = parent[parent[label]]
                label = parent[label]
            return label

        for i, label_i in enumerate(tree_labels):
            tree_i = scipy.spatial.cKDTree(centres[label_i])
            for label_j in tree_labels[i + 1:]:
                # A KD-tree nearest-neighbor query avoids a dense all-pairs
                # distance matrix for large surface fragments.
                if (
                    tree_i.query(
                        centres[label_j], 
                        distance_upper_bound=merge_radius
                        )[0].min() 
                    < merge_radius):
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
    # Require enough ring vertices for a stable plane fit; tiny fragments
    # may be spurious defects.
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


def _markers_from_seeds(seed_labels, order):
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
        defect_depth_frac=0.2,
        max_ringness=0.35,
        split_disconnected=True,
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
    split_disconnected : bool
        Assign separate IDs to disconnected 3D components before trying
        a convexity-based split.
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
    if split_disconnected:
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
        lab_ID_bool, spacing, peak_coords=peak_coords,
        depth_frac=defect_depth_frac,
        max_ringness=max_ringness
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
        markers = _markers_from_seeds(seed_labels, order)
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

def split_along_convexity_defects_slice_by_slice(
        ID, lab, max_ID, eps_percent=0.01, split_disconnected=False, rp=None
    ):
    """Split a labeled 3D object by components or by 2D convexity per slice.

    First separate disconnected 3D components, preserving the original ID
    for the largest one. If the object is a single component, apply the
    legacy 2D convexity-defect splitter to each z-slice containing it. When
    ``split_disconnected`` is false, only the largest connected 2D piece in a
    slice is considered for a convexity cut; other same-ID pieces are kept.

    Parameters
    ----------
    ID : int
        Label value of the object to split.
    lab : ndarray
        3D integer label image. Updated in-place.
    max_ID : int
        Highest label currently in use; new labels are allocated above it
        (and above the maximum label in ``lab``).
    eps_percent : float
        Relative contour-approximation tolerance passed to the 2D splitter.
    split_disconnected : bool
        Whether to separate disconnected 2D pieces before trying a
        convexity-defect split.
    rp : object, optional
        Regionprops-like lookup exposing ``get_obj_from_ID``. When supplied,
        its object mask is used instead of finding ``ID`` directly in ``lab``.

    Returns
    -------
    lab : ndarray
        Updated label image.
    was_split : bool
        Whether disconnected components or any slice was split.
    IDs : list of int
        IDs assigned when a split occurred; empty when the object remains
        unchanged.
    """
    if lab.ndim != 3:
        raise ValueError(f'Expected a 3D label image, got {lab.ndim}D.')
    max_ID = max(int(max_ID), int(lab.max()))

    if rp is None:
        object_mask = lab == ID
        if not object_mask.any():
            return lab, False, []
        bbox = ndimage.find_objects(object_mask.astype(np.uint8))[0]
    else:
        obj = rp.get_obj_from_ID(ID)
        if obj is None:
            raise ValueError(f'Object with ID {ID} was not found in regionprops.')
        object_mask = np.zeros_like(lab, dtype=bool)
        object_mask[obj.slice][obj.image] = True
        bbox = obj.slice

    # Label in 3D first so pieces separated in one slice but connected
    # elsewhere in the volume are not mistaken for separate objects.
    if split_disconnected:
        components = skimage.measure.label(object_mask, connectivity=3)
        component_props = skimage.measure.regionprops(components)
        component_props.sort(key=lambda component: component.area, reverse=True)

        if len(component_props) > 1:
            component_IDs = []
            for component_i, component in enumerate(component_props):
                component_ID = ID if component_i == 0 else max_ID + 1
                if component_i > 0:
                    max_ID += 1
                lab[component.slice][component.image] = component_ID
                component_IDs.append(component_ID)

            if len(component_IDs) > 1:
                return lab, True, component_IDs
            if not component_IDs:
                return lab, False, []
        
    was_split = False
    all_split_IDs = []
    # Iterate over each 2D slice within the bounding box of the object.
    for z, lab_2D in enumerate(lab[bbox]):
        slice_lab = lab_2D.copy()
        split_lab, success, split_IDs = split_along_convexity_defects(
            ID, slice_lab, max_ID,
            eps_percent=eps_percent,
            split_disconnected=False,
        )
        if not success:
            continue
        
        neighboring_slices = []
        if z > 0:
            neighboring_slices.append(lab[bbox][z-1])
        if z < lab[bbox].shape[0] - 1:
            neighboring_slices.append(lab[bbox][z+1])
        
        split_lab, _, split_IDs = track_split_slice(
            split_lab,
            neighboring_slices,
            max_ID + 1,
            split_IDs=split_IDs
        )

        # If tracking could not link the new piece to a neighbor (e.g. the
        # neighbor slice was not split), reuse the ID already allocated in
        # other slices instead of introducing yet another ID.
        previous_new_IDs = set(all_split_IDs) - {ID}
        new_IDs = [split_ID for split_ID in split_IDs if split_ID != ID]
        if len(previous_new_IDs) == 1 and len(new_IDs) == 1:
            previous_new_ID = next(iter(previous_new_IDs))
            if new_IDs[0] != previous_new_ID:
                split_lab[split_lab == new_IDs[0]] = previous_new_ID
                split_IDs = [
                    previous_new_ID if split_ID == new_IDs[0] else split_ID
                    for split_ID in split_IDs
                ]

        split_lab = np.where(lab_2D == ID, split_lab, lab_2D)
        lab[bbox][z] = split_lab
        all_split_IDs.extend(split_IDs)
        was_split = True
        max_ID = max(*all_split_IDs, max_ID) if all_split_IDs else max_ID

    
    if not was_split:
        return lab, False, []
    all_split_IDs = list(set(all_split_IDs))
    return lab, True, all_split_IDs

def convexity_defects(img, eps_percent):
    """Return a simplified outer contour and its OpenCV convexity defects.

    The contour is approximated to reduce boundary noise before computing
    hull defects. ``None`` is returned for defects when the hull indices do
    not have the ordering required by OpenCV's convexity-defect routine.

    Parameters
    ----------
    img : ndarray
        Binary 2D object mask.
    eps_percent : float
        Fraction of the contour perimeter used as the approximation
        tolerance.

    Returns
    -------
    contour : ndarray
        Approximated contour of the largest foreground component.
    defects : ndarray or None
        OpenCV convexity-defect records, or ``None`` if they cannot be
        computed from the contour/hull ordering.
    """
    img = img.astype(np.uint8)
    contours, _ = cv2.findContours(img,2,1)
    cnt = max(contours, key=cv2.contourArea)
    cnt = cv2.approxPolyDP(cnt,eps_percent*cv2.arcLength(cnt,True),True) # see https://www.programcreek.com/python/example/89457/cv22.convexityDefects
    hull = cv2.convexHull(cnt,returnPoints = False) # see https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
    hull_indices = hull.ravel()
    hull_diffs = np.diff(hull_indices)
    if not (
        np.all(hull_diffs > 0) or np.all(hull_diffs < 0)
    ):
        return cnt, None
    # OpenCV expects hull indices in contour order, rather than an arbitrary
    # ordering of the hull vertices.
    defects = cv2.convexityDefects(cnt,hull) # see https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
    return cnt, defects

def split_connected_components(lab, rp=None, max_ID=None):  
    """Give each disconnected component of each regionprops object its own ID.

    Within each object, the largest component retains its existing label and
    every smaller component receives a fresh label above ``max_ID``.

    Parameters
    ----------
    lab : ndarray
        Label image, modified in-place.
    rp : sequence, optional
        Regionprops records to process. Computed from ``lab`` if omitted.
    max_ID : int, optional
        Highest ID already allocated. Defaults to the largest label in
        ``rp`` (or 1 when ``rp`` is empty).

    Returns
    -------
    bool
        Whether at least one object contained multiple connected components.
    """
    if rp is None:
        rp = skimage.measure.regionprops(lab)
    
    if max_ID is None:
        max_ID = max([obj.label for obj in rp], default=1)
        
    split_occured = False
    for obj in rp:
        lab_obj = skimage.measure.label(obj.image)
        rp_lab_obj = skimage.measure.regionprops(lab_obj)
        if len(rp_lab_obj)<=1:
            continue
        rp_lab_obj.sort(key=lambda component: component.area, reverse=True)
        components_out = np.zeros_like(lab_obj, dtype=lab.dtype)
        # Keeping the largest piece under the original ID avoids needless
        # identity changes; the remaining pieces receive monotonically new IDs.
        for component_i, component in enumerate(rp_lab_obj):
            component_ID = (
                obj.label if component_i == 0 else max_ID + component_i
            )
            components_out[component.slice][component.image] = component_ID
        _slice = obj.slice
        _objMask = obj.image
        lab[_slice][_objMask] = components_out[_objMask]
        split_occured = True
        max_ID += len(rp_lab_obj) - 1
    return split_occured

def split_along_convexity_defects(
    ID, lab, max_ID, max_i=1, eps_percent=0.01, rp=None,
    split_disconnected=True
    ):
    """Split one 2D labeled object along its dominant convexity defects.

    Disconnected pieces are optionally separated first. Otherwise the
    largest contour's two deepest defect points define a straight rasterized
    cut. Selecting the deepest pair allows the split to proceed when a
    ragged contour has more than two reported defects. The two resulting
    regions retain the original label on the larger side and receive
    ``max_ID + max_i`` on the smaller side. Cut pixels inside the object are
    assigned to the nearest resulting region.

    Parameters
    ----------
    ID : int
        Label value of the target object.
    lab : ndarray
        2D integer label image, modified in-place.
    max_ID : int
        Highest label currently in use.
    max_i : int
        Offset used when allocating the new region's label.
    eps_percent : float
        Relative contour-approximation tolerance used to find defects.
    rp : object, optional
        Regionprops-like lookup exposing ``get_obj_from_ID``.
    split_disconnected : bool
        Whether to separate disconnected pieces before analyzing contour
        convexity.

    Returns
    -------
    lab : ndarray
        Updated label image.
    success : bool
        Whether the object was split.
    split_IDs : list of int
        IDs of the resulting regions, or an empty list when no split occurred.
    """
    if rp is not None:
        obj = rp.get_obj_from_ID(ID)
        lab_ID_bool = np.zeros_like(lab[obj.slice], dtype=bool)
        lab_ID_bool[obj.image] = True
    else:
        lab_ID_bool = lab == ID
    if split_disconnected:
        # First try separating by labelling
        lab_ID = np.zeros_like(lab_ID_bool, dtype=lab.dtype)
        lab_ID[lab_ID_bool] = ID
        rp_ID = skimage.measure.regionprops(lab_ID)
        split_occured = split_connected_components(
            lab_ID, rp=rp_ID, max_ID=max_ID
        )
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
        print("No convexity defects found.")
        return lab, success, []

    defects = np.asarray(defects).reshape(-1, 4)
    if len(defects) < 2:
        return lab, success, []

    # Ragged contours can produce extra shallow defects. Use the two deepest
    # valleys as the candidate neck endpoints rather than rejecting the
    # whole object whenever OpenCV reports more than two.
    deepest_indices = np.argsort(defects[:, 3])[-2:]
    defects_points = []
    for defect_index in deepest_indices:
        farthest_index = defects[defect_index, 2]
        x, y = tuple(cnt[farthest_index][0])
        defects_points.append((y, x))
    (r0, c0), (r1, c1) = defects_points
    # The line between the two deepest contour defects approximates the
    # neck cut; connected-component labeling checks whether it separates
    # exactly two pieces before any labels are reassigned.
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
