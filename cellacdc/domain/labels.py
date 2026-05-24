"""Pure label-array operations (no Qt)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage
import skimage.measure
import skimage.morphology
import skimage.segmentation


@dataclass(frozen=True)
class LabelResizeResult:
    """Result of resizing one object in a 2D label plane."""

    labels_2d: np.ndarray
    active_labels_2d: np.ndarray
    seed_labels: np.ndarray
    previous_coords: tuple[np.ndarray, np.ndarray]
    resized_coords: tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class LabelMoveResult:
    """Result of moving one object in a label image."""

    labels: np.ndarray
    previous_coords: np.ndarray
    moved_coords: np.ndarray


@dataclass(frozen=True)
class LabelIdMappingResult:
    """Result of applying one or more label-ID remappings."""

    labels: np.ndarray
    max_id: int
    swapped_pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class LabelBorderClearResult:
    """Result of clearing labels that touch the image border."""

    labels: np.ndarray
    removed_ids: list[int]


@dataclass(frozen=True)
class LabelIdsRemovalResult:
    """Result of removing labels by identity."""

    labels: np.ndarray
    removed_ids: list[int]


@dataclass(frozen=True)
class LabelHoleFillResult:
    """Result of filling holes for one label ID."""

    labels: np.ndarray
    filled_pixels: int


@dataclass(frozen=True)
class LabelRegionSelectionResult:
    """Result of selecting labels through a drawn region."""

    labels: np.ndarray
    selected_ids: list[int]


@dataclass(frozen=True)
class LabelRoiIndexResult:
    """Result of indexing a label ROI into a label image."""

    labels: np.ndarray
    roi_labels: np.ndarray
    inserted_ids: list[int]
    replaced_ids: list[int]


@dataclass(frozen=True)
class DeletedRoiRestoreResult:
    """Result of restoring labels from a deleted-ROI mask."""

    labels_2d: np.ndarray
    display_labels_2d: np.ndarray
    deleted_mask: np.ndarray
    remaining_deleted_ids: set[int]
    restored_ids: set[int]
    restored_masks: list[tuple[int, np.ndarray]]


@dataclass(frozen=True)
class DeletedRoiApplyResult:
    """Result of applying deletion ROI masks to a label image."""

    labels_2d: np.ndarray
    deleted_masks: list[np.ndarray]
    deleted_ids_by_roi: list[set[int]]
    deleted_ids: set[int]


def frame_slice(labels: np.ndarray, frame_i: int) -> np.ndarray:
    if labels.ndim == 3:
        return labels[frame_i]
    return labels


def clicked_label_at(
    labels: np.ndarray,
    x: int,
    y: int,
    frame_i: int = 0,
) -> int:
    sl = frame_slice(labels, frame_i)
    if y < 0 or x < 0 or y >= sl.shape[0] or x >= sl.shape[1]:
        return 0
    return int(sl[y, x])


def label_ids_from_labels(labels: np.ndarray) -> list[int]:
    """Return non-background label IDs in ascending order."""
    ids = np.unique(labels)
    return [int(label_id) for label_id in ids if int(label_id) != 0]


def clear_border_labels(
    labels: np.ndarray,
    *,
    buffer_size: int = 0,
    bgval: int = 0,
    mask: np.ndarray | None = None,
) -> LabelBorderClearResult:
    """Return labels with border-touching objects removed."""
    original_ids = set(label_ids_from_labels(labels))
    cleared_labels = skimage.segmentation.clear_border(
        labels,
        buffer_size=buffer_size,
        bgval=bgval,
        mask=mask,
    )
    remaining_ids = set(label_ids_from_labels(cleared_labels))
    return LabelBorderClearResult(
        labels=cleared_labels,
        removed_ids=sorted(original_ids - remaining_ids),
    )


def remove_new_label_ids(
    labels: np.ndarray,
    previous_ids,
    current_ids,
) -> LabelIdsRemovalResult:
    """Remove labels present in ``current_ids`` but absent from ``previous_ids``."""
    previous_ids = {int(label_id) for label_id in previous_ids}
    removed_ids = sorted(
        int(label_id)
        for label_id in set(current_ids) - previous_ids
        if int(label_id) > 0
    )

    updated_labels = labels.copy()
    if removed_ids:
        updated_labels[np.isin(updated_labels, removed_ids)] = 0

    return LabelIdsRemovalResult(
        labels=updated_labels,
        removed_ids=removed_ids,
    )


def fill_label_holes(labels_2d: np.ndarray, label_id: int) -> LabelHoleFillResult:
    """Fill holes inside one 2D label object."""
    label_id = int(label_id)
    updated_labels = labels_2d.copy()
    mask = labels_2d == label_id
    filled_mask = scipy.ndimage.binary_fill_holes(mask)
    filled_pixels = int(np.count_nonzero(filled_mask & ~mask))
    updated_labels[filled_mask] = label_id
    return LabelHoleFillResult(
        labels=updated_labels,
        filled_pixels=filled_pixels,
    )


def _clear_labels_not_fully_in_mask(
    labels_2d: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    selected_labels = labels_2d.copy()
    for obj in skimage.measure.regionprops(labels_2d):
        if np.all(mask[obj.slice][obj.image]):
            continue
        selected_labels[obj.slice][obj.image] = 0
    return selected_labels


def select_labels_in_region(
    labels: np.ndarray,
    mask: np.ndarray,
    *,
    enclosed_only: bool = False,
) -> LabelRegionSelectionResult:
    """Return labels selected by a 2D region mask.

    If ``enclosed_only`` is true, only objects fully enclosed by the region are
    selected. Otherwise, every object touching the region is selected.
    """
    selected_labels = labels.copy()
    if enclosed_only:
        if selected_labels.ndim == 2:
            selected_labels = _clear_labels_not_fully_in_mask(
                selected_labels,
                mask,
            )
        else:
            for z, labels_2d in enumerate(selected_labels):
                selected_labels[z] = _clear_labels_not_fully_in_mask(
                    labels_2d,
                    mask,
                )
    else:
        selected_labels[..., ~mask] = 0

    return LabelRegionSelectionResult(
        labels=selected_labels,
        selected_ids=label_ids_from_labels(selected_labels),
    )


def _xy_border_mask(shape: tuple[int, ...]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[..., 1:-1, 1:-1] = True
    return mask


def index_label_roi(
    labels: np.ndarray,
    roi_labels: np.ndarray,
    roi_slice,
    brush_id: int,
    *,
    clear_border: bool = False,
    replace_existing: bool = False,
) -> LabelRoiIndexResult:
    """Insert ROI labels into ``labels`` using Cell-ACDC label ROI semantics."""
    indexed_roi_labels = roi_labels.copy()
    if clear_border:
        indexed_roi_labels = skimage.segmentation.clear_border(
            indexed_roi_labels,
            mask=_xy_border_mask(indexed_roi_labels.shape),
        )

    roi_mask = indexed_roi_labels > 0
    inserted_ids = sorted(
        int(label_id) + int(brush_id) - 1
        for label_id in np.unique(indexed_roi_labels[roi_mask])
    )
    indexed_roi_labels[roi_mask] += int(brush_id) - 1

    updated_labels = labels.copy()
    target = updated_labels[roi_slice]
    replaced_ids = []
    if replace_existing and np.any(roi_mask):
        replaced_ids = [
            int(label_id) for label_id in np.unique(target[roi_mask])
            if int(label_id) != 0
        ]
        for label_id in replaced_ids:
            updated_labels[updated_labels == label_id] = 0
        target = updated_labels[roi_slice]

    target[roi_mask] = indexed_roi_labels[roi_mask]
    return LabelRoiIndexResult(
        labels=updated_labels,
        roi_labels=indexed_roi_labels,
        inserted_ids=inserted_ids,
        replaced_ids=replaced_ids,
    )


def merge_label_ids(
    labels: np.ndarray,
    source_id: int,
    target_id: int,
    frame_i: int | None = None,
) -> np.ndarray:
    """Replace ``source_id`` with ``target_id`` in ``labels``."""
    if source_id == target_id or source_id == 0:
        return labels
    if frame_i is not None and labels.ndim == 3:
        sl = labels[frame_i]
        sl[sl == source_id] = target_id
    else:
        labels[labels == source_id] = target_id
    return labels


def merge_multiple_ids(
    labels: np.ndarray,
    ids_to_merge: np.ndarray | list[int],
    target_id: int,
    frame_i: int | None = None,
) -> np.ndarray:
    """Merge each ID in ``ids_to_merge`` into ``target_id``."""
    for label_id in np.asarray(ids_to_merge).ravel():
        label_id = int(label_id)
        if label_id == 0 or label_id == target_id:
            continue
        merge_label_ids(labels, label_id, target_id, frame_i=frame_i)
    return labels


def apply_label_id_mapping(
        labels: np.ndarray,
        old_new_pairs,
        *,
        existing_ids=None,
        merge_existing: bool = False,
        start_max_id: int | None = None,
) -> LabelIdMappingResult:
    """Apply Cell-ACDC edit-ID semantics to a label image in place.

    If the target ID already exists and ``merge_existing`` is false, IDs are
    swapped through a temporary label. Otherwise the old ID is replaced by the
    new ID, which allows explicit merge workflows.
    """
    max_id = (
        int(np.max(labels)) if start_max_id is None and labels.size else
        int(start_max_id or 0)
    )
    existing_ids_set = (
        None if existing_ids is None else {int(label_id) for label_id in existing_ids}
    )
    swapped_pairs = []

    for old_id, new_id in old_new_pairs:
        old_id = int(old_id)
        new_id = int(new_id)
        has_target = (
            bool(np.any(labels == new_id))
            if existing_ids_set is None else new_id in existing_ids_set
        )

        if has_target and not merge_existing:
            temp_id = max_id + 1
            labels[labels == old_id] = temp_id
            labels[labels == new_id] = old_id
            labels[labels == temp_id] = new_id
            max_id = temp_id
            swapped_pairs.append((old_id, new_id))
        else:
            labels[labels == old_id] = new_id
            max_id = max(max_id, new_id)

    return LabelIdMappingResult(
        labels=labels,
        max_id=max_id,
        swapped_pairs=tuple(swapped_pairs),
    )


def next_available_label_id(
        id_groups=(),
        *,
        manual_edit_info=(),
        base_id: int = 0,
) -> int:
    """Return the next label ID after all known and manually edited IDs."""
    max_id = int(base_id)
    for ids in id_groups:
        for label_id in ids:
            max_id = max(max_id, int(label_id))

    for info in manual_edit_info:
        try:
            label_id = info[2]
        except (TypeError, IndexError):
            label_id = info
        max_id = max(max_id, int(label_id))

    return max_id + 1


def remap_id_set(ids, old_ids, new_ids) -> set[int]:
    """Return an ID set remapped through parallel old/new ID sequences."""
    id_mapper = dict(zip(old_ids, new_ids))
    return {int(id_mapper[label_id]) for label_id in ids}


def restore_deleted_roi_labels(
        labels_2d: np.ndarray,
        display_labels_2d: np.ndarray,
        deleted_mask: np.ndarray,
        roi_mask: np.ndarray,
        deleted_ids,
        *,
        enforce: bool = True,
) -> DeletedRoiRestoreResult:
    """Restore labels that were previously removed by a deletion ROI.

    ``deleted_mask`` stores the deleted object IDs. If ``enforce`` is false,
    IDs still overlapping the current ROI mask are kept deleted.
    """
    deleted_ids = {int(label_id) for label_id in deleted_ids}
    overlap_roi_deleted_ids = {
        int(label_id) for label_id in np.unique(deleted_mask[roi_mask])
    }
    restored_ids = set()
    restored_masks = []

    for label_id in deleted_ids:
        if label_id in overlap_roi_deleted_ids and not enforce:
            continue

        restore_mask = deleted_mask == label_id
        restored_ids.add(label_id)
        restored_masks.append((label_id, restore_mask.copy()))
        display_labels_2d[restore_mask] = label_id
        labels_2d[restore_mask] = label_id
        deleted_mask[restore_mask] = 0

    return DeletedRoiRestoreResult(
        labels_2d=labels_2d,
        display_labels_2d=display_labels_2d,
        deleted_mask=deleted_mask,
        remaining_deleted_ids=deleted_ids - restored_ids,
        restored_ids=restored_ids,
        restored_masks=restored_masks,
    )


def label_ids_in_masks(
        labels: np.ndarray,
        masks,
        *,
        additional_labels: np.ndarray | None = None,
) -> set[int]:
    """Return all label IDs under one or more boolean masks."""
    label_ids = set()
    for mask in masks:
        label_ids.update(int(label_id) for label_id in labels[mask])
        if additional_labels is not None:
            label_ids.update(int(label_id) for label_id in additional_labels[mask])
    return label_ids


def collect_deleted_roi_ids(deleted_ids_by_roi) -> set[int]:
    """Flatten stored deleted-ID collections for multiple deletion ROIs."""
    label_ids = set()
    for deleted_ids in deleted_ids_by_roi:
        label_ids.update(int(label_id) for label_id in deleted_ids)
    return label_ids


def apply_deleted_roi_masks(
        labels_2d: np.ndarray,
        roi_masks,
        deleted_masks,
        deleted_ids_by_roi,
) -> DeletedRoiApplyResult:
    """Delete labelled objects intersecting ROI masks and record them."""
    deleted_masks = list(deleted_masks)
    deleted_ids_by_roi = [
        {int(label_id) for label_id in deleted_ids}
        for deleted_ids in deleted_ids_by_roi
    ]
    all_deleted_ids = set()

    for idx, roi_mask in enumerate(roi_masks):
        deleted_mask = deleted_masks[idx]
        deleted_ids = deleted_ids_by_roi[idx]
        for obj in skimage.measure.regionprops(labels_2d):
            object_mask = obj.image
            object_slice = obj.slice
            is_deleted_object = np.any(roi_mask[object_slice][object_mask])
            if not is_deleted_object:
                continue

            label_id = int(obj.label)
            deleted_mask[object_slice][object_mask] = label_id
            labels_2d[object_slice][object_mask] = 0
            deleted_ids.add(label_id)
            all_deleted_ids.add(label_id)

        deleted_masks[idx] = deleted_mask
        deleted_ids_by_roi[idx] = deleted_ids

    return DeletedRoiApplyResult(
        labels_2d=labels_2d,
        deleted_masks=deleted_masks,
        deleted_ids_by_roi=deleted_ids_by_roi,
        deleted_ids=all_deleted_ids,
    )


def _empty_roi_mask(shape: tuple[int, ...]) -> np.ndarray:
    return np.zeros(shape, dtype=bool)


def _paint_roi_coords(
        roi_mask: np.ndarray,
        rr: np.ndarray,
        cc: np.ndarray,
        *,
        z_slice=None,
) -> np.ndarray:
    if roi_mask.ndim == 3:
        roi_mask[z_slice, rr, cc] = True
    else:
        roi_mask[rr, cc] = True
    return roi_mask


def polygon_roi_mask(
        shape: tuple[int, ...],
        points,
        *,
        z_slice=None,
) -> np.ndarray:
    """Rasterize a polyline or polygon ROI from ``(x, y)`` points."""
    roi_mask = _empty_roi_mask(shape)
    if not points:
        return roi_mask

    rr_points = [int(y) for x, y in points]
    cc_points = [int(x) for x, y in points]
    if not rr_points or not cc_points:
        return roi_mask

    plane_shape = shape[-2:]
    if len(rr_points) == 2:
        rr, cc, _ = skimage.draw.line_aa(
            rr_points[0], cc_points[0], rr_points[1], cc_points[1],
        )
    else:
        rr, cc = skimage.draw.polygon(rr_points, cc_points, shape=plane_shape)

    height, width = plane_shape
    keep = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
    return _paint_roi_coords(roi_mask, rr[keep], cc[keep], z_slice=z_slice)


def line_roi_mask(
        shape: tuple[int, ...],
        point1,
        point2,
        *,
        z_slice=None,
) -> np.ndarray:
    """Rasterize a line ROI from two ``(x, y)`` points."""
    roi_mask = _empty_roi_mask(shape)
    x1, y1 = [int(coord) for coord in point1]
    x2, y2 = [int(coord) for coord in point2]
    rr, cc, _ = skimage.draw.line_aa(y1, x1, y2, x2)
    return _paint_roi_coords(roi_mask, rr, cc, z_slice=z_slice)


def rectangle_roi_mask(
        shape: tuple[int, ...],
        origin,
        size,
        *,
        z_slice=None,
) -> np.ndarray:
    """Rasterize an axis-aligned rectangular ROI."""
    roi_mask = _empty_roi_mask(shape)
    x0, y0 = [int(coord) for coord in origin]
    width, height = [int(coord) for coord in size]
    if roi_mask.ndim == 3:
        roi_mask[z_slice, y0:y0+height, x0:x0+width] = True
    else:
        roi_mask[y0:y0+height, x0:x0+width] = True
    return roi_mask


def build_disk_mask(
    shape: tuple[int, int],
    x: int,
    y: int,
    radius: int,
) -> np.ndarray:
    """Build a circular boolean mask centered at ``(x, y)``."""
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = (yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2
    mask[y0:y1, x0:x1] = disk
    return mask


def _label_target(
    labels: np.ndarray,
    frame_i: int,
) -> np.ndarray:
    if labels.ndim == 3:
        return labels[frame_i]
    return labels


def apply_label_mask(
    labels: np.ndarray,
    mask: np.ndarray,
    label_id: int,
    frame_i: int = 0,
) -> np.ndarray:
    """Paint ``label_id`` wherever ``mask`` is True."""
    target = _label_target(labels, frame_i)
    target[mask] = label_id
    return labels


def apply_eraser_mask(
    labels: np.ndarray,
    mask: np.ndarray,
    frame_i: int = 0,
    only_id: int | None = None,
) -> np.ndarray:
    """Zero labels under ``mask``; optionally restrict to ``only_id``."""
    target = _label_target(labels, frame_i)
    if only_id is not None:
        erase_mask = np.logical_and(mask, target == only_id)
    else:
        erase_mask = mask
    target[erase_mask] = 0
    return labels


def resize_label_object(
    labels_2d: np.ndarray,
    active_labels_2d: np.ndarray,
    object_coords: np.ndarray,
    label_id: int,
    footprint_size: int,
    *,
    dilation: bool = True,
    seed_labels: np.ndarray | None = None,
) -> LabelResizeResult:
    """Dilate or erode one label object without overwriting neighbouring IDs.

    ``labels_2d`` is the persisted label plane to edit, while
    ``active_labels_2d`` is the collision mask used by interactive tools. Both
    arrays are updated in place and returned for scriptable callers.
    """
    coords = np.asarray(object_coords)
    yy = coords[:, -2].astype(int, copy=True)
    xx = coords[:, -1].astype(int, copy=True)
    previous_coords = (yy.copy(), xx.copy())

    if seed_labels is None:
        seed_labels = np.zeros_like(active_labels_2d)
        seed_labels[yy, xx] = label_id
    else:
        seed_labels = np.asarray(seed_labels)

    active_labels_2d[yy, xx] = 0
    labels_2d[yy, xx] = 0

    footprint = skimage.morphology.disk(int(footprint_size))
    if dilation:
        resized_labels = skimage.morphology.dilation(seed_labels, footprint)
    else:
        resized_labels = skimage.morphology.erosion(seed_labels, footprint)

    # Keep the edited object from growing into still-occupied pixels.
    resized_labels = np.asarray(resized_labels)
    resized_labels[active_labels_2d > 0] = 0

    resized_regions = skimage.measure.regionprops(resized_labels.astype(np.int32))
    if not resized_regions:
        raise ValueError(f'Label {label_id} vanished during resize')

    resized_obj_coords = resized_regions[0].coords
    resized_yy = resized_obj_coords[:, -2].astype(int, copy=False)
    resized_xx = resized_obj_coords[:, -1].astype(int, copy=False)
    resized_coords = (resized_yy.copy(), resized_xx.copy())

    active_labels_2d[resized_yy, resized_xx] = label_id
    labels_2d[resized_yy, resized_xx] = label_id

    return LabelResizeResult(
        labels_2d=labels_2d,
        active_labels_2d=active_labels_2d,
        seed_labels=seed_labels,
        previous_coords=previous_coords,
        resized_coords=resized_coords,
    )


def move_label_object(
    labels: np.ndarray,
    object_coords: np.ndarray,
    label_id: int,
    *,
    delta_y: int,
    delta_x: int,
    shape: tuple[int, int] | None = None,
) -> LabelMoveResult:
    """Move one 2D or z-stacked label object, clipping at image boundaries."""
    moved_coords = np.asarray(object_coords).copy()
    previous_coords = moved_coords.copy()

    if shape is None:
        shape = labels.shape[-2:]
    height, width = shape

    yy = previous_coords[:, -2].astype(int, copy=False)
    xx = previous_coords[:, -1].astype(int, copy=False)

    if labels.ndim >= 3 and previous_coords.shape[1] >= 3:
        zz = previous_coords[:, 0].astype(int, copy=False)
        labels[zz, yy, xx] = 0
    else:
        labels[yy, xx] = 0

    moved_coords[:, -2] = np.clip(
        moved_coords[:, -2] + int(delta_y), 0, height - 1,
    )
    moved_coords[:, -1] = np.clip(
        moved_coords[:, -1] + int(delta_x), 0, width - 1,
    )

    moved_yy = moved_coords[:, -2].astype(int, copy=False)
    moved_xx = moved_coords[:, -1].astype(int, copy=False)
    if labels.ndim >= 3 and moved_coords.shape[1] >= 3:
        moved_zz = moved_coords[:, 0].astype(int, copy=False)
        labels[moved_zz, moved_yy, moved_xx] = label_id
    else:
        labels[moved_yy, moved_xx] = label_id

    return LabelMoveResult(
        labels=labels,
        previous_coords=previous_coords,
        moved_coords=moved_coords,
    )
