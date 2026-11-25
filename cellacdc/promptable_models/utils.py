from typing import List, Literal

import numpy as np

import skimage.measure
import skimage.morphology

from cellacdc import printl

BRUSH_STROKES_ZPROJ_FUNCS = {
    'max z-projection': lambda img: img.max(axis=0),
    'mean z-projection': lambda img: img.mean(axis=0),
    'median z-proj.': lambda img: np.median(img, axis=0),
}

def _get_segm_info_value(segmInfo_df, idx, col):
    """Return `segmInfo_df.at[idx, col]` handling duplicated index (first
    value is returned) and missing entries (None is returned)."""
    if segmInfo_df is None:
        return

    try:
        value = segmInfo_df.at[idx, col]
    except ValueError:
        try:
            value = segmInfo_df.loc[idx, col].iloc[0]
        except Exception as err:
            return
    except Exception as err:
        return

    if hasattr(value, 'iloc'):
        if len(value) == 0:
            return
        value = value.iloc[0]

    return value

def get_brush_strokes_image_source(posData, frame_i, annot_ndim):
    """Get the source of the image corresponding to 2D brush strokes of
    z-stack data at frame `frame_i`, i.e., the z-projection or the z-slice
    used in the GUI (columns 'which_z_proj_gui' and 'z_slice_used_gui' of
    `posData.segmInfo_df`).

    Parameters
    ----------
    posData : cellacdc.load.loadData
        Position data (see `get_brush_strokes_image`).
    frame_i : int
        Frame index.
    annot_ndim : int
        Number of dimensions of the annotation array (2 for (Y, X) and 3 for
        (Z, Y, X)).

    Returns
    -------
    str, int, or None
        Name of the z-projection (key of `BRUSH_STROKES_ZPROJ_FUNCS`) or
        index of the z-slice. None if the image does not depend on the
        z-slice or z-projection used in the GUI (3D annotations or 2D data).
    """
    SizeZ = getattr(posData, 'SizeZ', 1)
    if annot_ndim == 3 or SizeZ == 1:
        return

    img = posData.img_data[frame_i]
    if img.ndim != 3:
        return

    segmInfo_df = getattr(posData, 'segmInfo_df', None)
    idx = (getattr(posData, 'filename', None), frame_i)
    zProjHow = _get_segm_info_value(segmInfo_df, idx, 'which_z_proj_gui')
    if zProjHow in BRUSH_STROKES_ZPROJ_FUNCS:
        return zProjHow

    z = _get_segm_info_value(segmInfo_df, idx, 'z_slice_used_gui')
    try:
        z = int(z)
    except Exception as err:
        z = len(img)//2
    return min(max(z, 0), len(img)-1)

def get_brush_strokes_image(posData, frame_i, annot_ndim, source=None):
    """Get the image corresponding to the brush strokes annotation of
    frame `frame_i`, i.e., the image the promptable model sees.

    Parameters
    ----------
    posData : cellacdc.load.loadData
        Position data. Required attributes are `img_data`, `SizeZ`, and,
        for 2D annotations of z-stack data, `segmInfo_df` and `filename`.
    frame_i : int
        Frame index.
    annot_ndim : int
        Number of dimensions of the annotation array (2 for (Y, X) and 3 for
        (Z, Y, X)).
    source : str or int, optional
        Source of the image of 2D annotations of z-stack data (name of the
        z-projection or index of the z-slice, see
        `get_brush_strokes_image_source`), e.g., the one displayed when the
        brush strokes were painted. If None, the source used in the GUI at
        frame `frame_i` is used. Ignored with 3D annotations or 2D data.
        Default is None

    Returns
    -------
    np.ndarray
        Image with `annot_ndim` dimensions. With 2D annotations of z-stack
        data, this is the z-slice or the z-projection `source` (default is
        the one used in the GUI at frame `frame_i`, i.e., columns
        'z_slice_used_gui' and 'which_z_proj_gui' of `posData.segmInfo_df`).

    Raises
    ------
    ValueError
        If the number of dimensions of the image is different from
        `annot_ndim` (e.g., RGB images).
    """
    img = posData.img_data[frame_i]
    SizeZ = getattr(posData, 'SizeZ', 1)
    if annot_ndim == 3 or SizeZ == 1 or img.ndim != 3:
        image = img
    else:
        # 2D annotations of z-stack data --> use the z-slice or projection
        # `source` (default is the one used in the GUI)
        if source is None:
            source = get_brush_strokes_image_source(
                posData, frame_i, annot_ndim
            )
        zproj_func = BRUSH_STROKES_ZPROJ_FUNCS.get(source)
        if zproj_func is not None:
            image = zproj_func(img)
        else:
            z = min(max(int(source), 0), len(img)-1)
            image = img[z]

    if image.ndim != annot_ndim:
        raise ValueError(
            f'The image of frame n. {frame_i+1} has shape {image.shape} '
            f'which is not compatible with {annot_ndim}D brush strokes. '
            'Brush strokes require single-channel (grayscale) 2D or 3D '
            'images (e.g., RGB images are not supported).'
        )

    return image

def train_model_on_brush_strokes(model, prompts_data, logger_func=print):
    """Add brush strokes prompts to a promptable model and train it.

    Parameters
    ----------
    model : promptable model
        Either an in-process `acdcPromptSegment.Model` or a service
        `ModelClient`. Must have `add_prompt` and `train` methods.
    prompts_data : list of tuples
        Each item is (pos_i, frame_i, annot, image) where `annot` is the
        brush strokes annotation (0 = unlabelled, 1 = background,
        2 = foreground) with the same shape as `image`.
    logger_func : callable, optional
        Function used to log progress. Default is `print`

    Returns
    -------
    dict or None
        Training summary returned by `model.train()`.
    """
    for pos_i, frame_i, annot, image in prompts_data:
        logger_func(
            f'Adding brush strokes of frame n. {frame_i+1} '
            f'(position n. {pos_i+1})...'
        )
        model.add_prompt(
            prompt=annot,
            prompt_id=0,
            image=image,
            prompt_type='brush_stroke',
            frame_i=int(frame_i),
            pos_i=int(pos_i)
        )

    logger_func('Training model on brush strokes...')
    return model.train()

def get_brush_disk_mask(radius):
    """Boolean disk mask with shape (2*radius+1, 2*radius+1)"""
    radius = max(int(radius), 0)
    return skimage.morphology.disk(radius, dtype=bool)

def stamp_brush_disk(annot2D, yc, xc, disk_mask, value):
    """Set `annot2D` to `value` inside `disk_mask` centered at (yc, xc),
    clipped at all image borders.

    Parameters
    ----------
    annot2D : np.ndarray
        2D (Y, X) annotation array modified in-place.
    yc, xc : float
        Center of the disk in pixel coordinates (floored to integers).
    disk_mask : np.ndarray
        Boolean disk mask with odd shape (e.g., from `get_brush_disk_mask`).
    value : int
        Value to paint.

    Returns
    -------
    bool
        True if at least part of the disk is inside the image.
    """
    Y, X = annot2D.shape[-2:]
    dh, dw = disk_mask.shape
    ry, rx = dh//2, dw//2
    yc, xc = int(np.floor(yc)), int(np.floor(xc))
    y0, y1 = yc - ry, yc + ry + 1
    x0, x1 = xc - rx, xc + rx + 1
    y0c, y1c = max(y0, 0), min(y1, Y)
    x0c, x1c = max(x0, 0), min(x1, X)
    if y0c >= y1c or x0c >= x1c:
        return False

    local_mask = disk_mask[y0c-y0:dh-(y1-y1c), x0c-x0:dw-(x1-x1c)]
    annot2D[y0c:y1c, x0c:x1c][local_mask] = value
    return True

def paint_brush_stroke(
        annot2D, y, x, value, radius, last_yx=None, disk_mask=None
    ):
    """Paint a brush stroke with a disk of radius `radius` at (y, x) and,
    if `last_yx` is not None, along the segment from `last_yx` to (y, x)
    (stamping disks every `max(radius/2, 1)` pixels to avoid gaps between
    consecutive mouse positions).

    Parameters
    ----------
    annot2D : np.ndarray
        2D (Y, X) annotation array modified in-place.
    y, x : float
        Current position in pixel coordinates.
    value : int
        Value to paint (e.g., 0 = erase, 1 = background, 2 = foreground).
    radius : int
        Radius of the brush in pixels.
    last_yx : tuple of floats, optional
        Previous position of the stroke. Default is None
    disk_mask : np.ndarray, optional
        Precomputed disk mask (see `get_brush_disk_mask`). Default is None
    """
    if disk_mask is None:
        disk_mask = get_brush_disk_mask(radius)

    if last_yx is None:
        stamp_brush_disk(annot2D, y, x, disk_mask, value)
        return

    y0, x0 = last_yx
    dist = float(np.hypot(y - y0, x - x0))
    step = max(radius/2, 1)
    num_steps = int(np.ceil(dist/step))
    if num_steps == 0:
        stamp_brush_disk(annot2D, y, x, disk_mask, value)
        return

    for t in np.linspace(0, 1, num_steps+1)[1:]:
        yc = y0 + t*(y - y0)
        xc = x0 + t*(x - x0)
        stamp_brush_disk(annot2D, yc, xc, disk_mask, value)

def add_objects_to_brush_strokes(annot, lab, foreground_ids):
    """Add segmentation masks to the brush strokes annotation (in-place):
    the objects of `lab` with IDs in `foreground_ids` become foreground
    (value 2) and the background of `lab` (value 0) becomes background
    (value 1). The other objects are left not annotated.

    Only the pixels that are not annotated yet (value 0 in `annot`) are
    modified, i.e., the existing brush strokes are kept.

    Parameters
    ----------
    annot : np.ndarray
        Brush strokes annotation (0 = not annotated, 1 = background,
        2 = foreground) modified in-place. It can be a view (e.g., a
        z-slice or a cutout) of a larger annotation array.
    lab : np.ndarray
        Segmentation masks (labels) with the same shape as `annot`.
    foreground_ids : iterable of int
        IDs of the objects of `lab` used as foreground. IDs that are not
        in `lab` are ignored.

    Returns
    -------
    tuple of ints
        Number of foreground and background pixels added.

    Raises
    ------
    ValueError
        If `annot` and `lab` do not have the same shape.
    """
    if annot.shape != lab.shape:
        raise ValueError(
            f'The brush strokes shape {annot.shape} is different from the '
            f'segmentation masks shape {lab.shape}.'
        )

    foreground_ids = np.asarray(list(foreground_ids), dtype=np.int64)
    not_annotated_mask = annot == 0
    foreground_mask = (
        np.isin(lab, foreground_ids) & (lab > 0) & not_annotated_mask
    )
    background_mask = (lab == 0) & not_annotated_mask
    annot[foreground_mask] = 2
    annot[background_mask] = 1
    num_foreground = int(np.count_nonzero(foreground_mask))
    num_background = int(np.count_nonzero(background_mask))
    return num_foreground, num_background


def log_mask_selection(prompt_id, masks, scores, mask_idx, z_slice=None):
    """Log mask sizes and scores for debugging prompt segmentation."""
    z_info = f" z={z_slice}" if z_slice is not None else ""
    printl(f"[DEBUG] Prompt ID {prompt_id}{z_info}:")
    for i, (mask, score) in enumerate(zip(masks, scores)):
        selected = " <-- selected" if i == mask_idx else ""
        printl(f"  mask[{i}]: {int(mask.sum()):7} px, score={score:.6f}{selected}")


def build_combined_mask(model_out):
    """Build a single labeled mask from model output, sorted by area (largest first)."""
    rp_model_out = skimage.measure.regionprops(model_out)
    # Sort by area descending so smaller objects paint on top
    rp_model_out = sorted(rp_model_out, key=lambda x: x.area, reverse=True)

    combined = np.zeros(model_out.shape, dtype=model_out.dtype)
    for obj in rp_model_out:
        combined[obj.slice][obj.image] = obj.label
    return combined


def _apply_overlap_rule(
        lab_old,
        lab_new,
        mode: Literal['union', 'intersection']
    ):
    """
    Apply overlap rules between old and new label masks.

    For each overlapping pair (old ID p, new ID q):
    - union: p OR q region → all become p (old absorbs new)
    - intersection: only p AND q → p; p XOR q → 0 (deleted)

    Non-overlapping regions:
    - Old-only IDs: preserved in all modes
    - New-only IDs: added in 'union', deleted in 'intersection'
    """
    result = np.zeros_like(lab_old)

    old_ids = set(np.unique(lab_old)) - {0}
    new_ids = set(np.unique(lab_new)) - {0}

    # Track which new IDs overlap with old
    overlapping_new_ids = set()

    # Process each old object
    for p in old_ids:
        p_mask = lab_old == p

        # Find new IDs that overlap with this old ID
        overlapping_q_ids = set(np.unique(lab_new[p_mask])) - {0}
        overlapping_new_ids.update(overlapping_q_ids)

        if not overlapping_q_ids:
            # No overlap - keep old object as is
            result[p_mask] = p
        else:
            # Has overlap with one or more new IDs
            for q in overlapping_q_ids:
                q_mask = lab_new == q

                p_and_q = np.logical_and(p_mask, q_mask)  # Overlap region
                p_only = np.logical_and(p_mask, ~q_mask)  # Old only
                q_only = np.logical_and(q_mask, ~p_mask)  # New only

                if mode == 'union':
                    # p OR q → all become p
                    result[p_and_q] = p
                    result[p_only] = p
                    result[q_only] = p

                elif mode == 'intersection':
                    # Only p AND q → p; p XOR q → 0
                    result[p_and_q] = p
                    # p_only and q_only become 0 (already 0 in result)

    # Handle new IDs that don't overlap with any old ID
    non_overlapping_new_ids = new_ids - overlapping_new_ids
    for q in non_overlapping_new_ids:
        q_mask = lab_new == q
        if mode == 'union':
            result[q_mask] = q
        # In 'intersection' mode, non-overlapping new IDs are not added

    return result


def insert_model_output_into_labels(
        lab,
        model_out,
        edited_IDs: int | List[int] = 0,
    ):
    """
    Combine model output with existing labels using three strategies.

    Parameters
    ----------
    lab : np.ndarray
        Existing label mask
    model_out : np.ndarray
        New label mask from model
    edited_IDs : int or List[int]
        Deprecated, kept for API compatibility. No longer used.

    Returns
    -------
    lab_new, lab_union, lab_intersection : tuple of np.ndarray
        - lab_new: just the model output (new masks only)
        - lab_union: old absorbs new in overlap regions
        - lab_intersection: only overlap regions kept
    """
    # Build combined new mask from model output (sorted by area, smallest on top)
    lab_new = build_combined_mask(model_out)

    # Apply overlap rules for union and intersection
    lab_union = _apply_overlap_rule(lab, lab_new, mode='union')
    lab_intersection = _apply_overlap_rule(lab, lab_new, mode='intersection')

    return lab_new, lab_union, lab_intersection
