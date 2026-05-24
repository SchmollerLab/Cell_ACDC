"""View-model commands for label image edits."""

from __future__ import annotations

import numpy as np

from cellacdc.core import (
    binary_fill_holes as core_binary_fill_holes,
    convex_hull_mask as core_convex_hull_mask,
    count_objects as core_count_objects,
    nearest_nonzero_2D,
    nearest_nonzero_z_idx_from_z_centroid,
    split_connected_components as core_split_connected_components,
    split_along_convexity_defects,
)
from cellacdc.domain.labels import (
    DeletedRoiApplyResult,
    DeletedRoiRestoreResult,
    LabelBorderClearResult,
    LabelHoleFillResult,
    LabelIdMappingResult,
    LabelIdsRemovalResult,
    LabelMoveResult,
    LabelRegionSelectionResult,
    LabelResizeResult,
    LabelRoiIndexResult,
    apply_deleted_roi_masks,
    apply_label_id_mapping,
    clear_border_labels,
    collect_deleted_roi_ids,
    fill_label_holes,
    index_label_roi,
    label_ids_from_labels,
    label_ids_in_masks,
    line_roi_mask,
    move_label_object,
    next_available_label_id,
    polygon_roi_mask,
    rectangle_roi_mask,
    remap_id_set,
    remove_new_label_ids,
    resize_label_object,
    restore_deleted_roi_labels,
    select_labels_in_region,
)
from cellacdc.measure import separate_with_label
from cellacdc.myutils import get_trimmed_list


class LabelEditViewModel:
    """Application-facing commands for editing label arrays."""

    def clear_border_labels(
        self,
        labels: np.ndarray,
        *,
        buffer_size: int = 0,
    ) -> LabelBorderClearResult:
        return clear_border_labels(labels, buffer_size=buffer_size)

    def remove_new_labels(
        self,
        labels: np.ndarray,
        previous_ids,
        current_ids,
    ) -> LabelIdsRemovalResult:
        return remove_new_label_ids(labels, previous_ids, current_ids)

    def fill_label_holes(
        self,
        labels_2d: np.ndarray,
        label_id: int,
    ) -> LabelHoleFillResult:
        return fill_label_holes(labels_2d, label_id)

    def select_labels_in_region(
        self,
        labels: np.ndarray,
        mask: np.ndarray,
        *,
        enclosed_only: bool = False,
    ) -> LabelRegionSelectionResult:
        return select_labels_in_region(
            labels,
            mask,
            enclosed_only=enclosed_only,
        )

    def index_label_roi(
        self,
        labels: np.ndarray,
        roi_labels: np.ndarray,
        roi_slice,
        brush_id: int,
        *,
        clear_border: bool = False,
        replace_existing: bool = False,
    ) -> LabelRoiIndexResult:
        return index_label_roi(
            labels,
            roi_labels,
            roi_slice,
            brush_id,
            clear_border=clear_border,
            replace_existing=replace_existing,
        )

    def resize_label_object(
        self,
        labels_2d: np.ndarray,
        active_labels_2d: np.ndarray,
        object_coords: np.ndarray,
        label_id: int,
        footprint_size: int,
        *,
        dilation: bool = True,
        seed_labels: np.ndarray | None = None,
    ) -> LabelResizeResult:
        return resize_label_object(
            labels_2d,
            active_labels_2d,
            object_coords,
            label_id,
            footprint_size,
            dilation=dilation,
            seed_labels=seed_labels,
        )

    def move_label_object(
        self,
        labels: np.ndarray,
        object_coords: np.ndarray,
        label_id: int,
        *,
        delta_y: int,
        delta_x: int,
        shape: tuple[int, int] | None = None,
    ) -> LabelMoveResult:
        return move_label_object(
            labels,
            object_coords,
            label_id,
            delta_y=delta_y,
            delta_x=delta_x,
            shape=shape,
        )

    def apply_id_mapping(
        self,
        labels: np.ndarray,
        old_new_pairs,
        *,
        existing_ids=None,
        merge_existing: bool = False,
        start_max_id: int | None = None,
    ) -> LabelIdMappingResult:
        return apply_label_id_mapping(
            labels,
            old_new_pairs,
            existing_ids=existing_ids,
            merge_existing=merge_existing,
            start_max_id=start_max_id,
        )

    def restore_deleted_roi_labels(
        self,
        labels_2d: np.ndarray,
        display_labels_2d: np.ndarray,
        deleted_mask: np.ndarray,
        roi_mask: np.ndarray,
        deleted_ids,
        *,
        enforce: bool = True,
    ) -> DeletedRoiRestoreResult:
        return restore_deleted_roi_labels(
            labels_2d,
            display_labels_2d,
            deleted_mask,
            roi_mask,
            deleted_ids,
            enforce=enforce,
        )

    def label_ids_in_masks(
        self,
        labels: np.ndarray,
        masks,
        *,
        additional_labels: np.ndarray | None = None,
    ) -> set[int]:
        return label_ids_in_masks(
            labels,
            masks,
            additional_labels=additional_labels,
        )

    def collect_deleted_roi_ids(self, deleted_ids_by_roi) -> set[int]:
        return collect_deleted_roi_ids(deleted_ids_by_roi)

    def apply_deleted_roi_masks(
        self,
        labels_2d: np.ndarray,
        roi_masks,
        deleted_masks,
        deleted_ids_by_roi,
    ) -> DeletedRoiApplyResult:
        return apply_deleted_roi_masks(
            labels_2d,
            roi_masks,
            deleted_masks,
            deleted_ids_by_roi,
        )

    def polygon_roi_mask(
        self,
        shape: tuple[int, ...],
        points,
        *,
        z_slice=None,
    ) -> np.ndarray:
        return polygon_roi_mask(shape, points, z_slice=z_slice)

    def line_roi_mask(
        self,
        shape: tuple[int, ...],
        point1,
        point2,
        *,
        z_slice=None,
    ) -> np.ndarray:
        return line_roi_mask(shape, point1, point2, z_slice=z_slice)

    def rectangle_roi_mask(
        self,
        shape: tuple[int, ...],
        origin,
        size,
        *,
        z_slice=None,
    ) -> np.ndarray:
        return rectangle_roi_mask(shape, origin, size, z_slice=z_slice)

    def next_available_label_id(
        self,
        id_groups=(),
        *,
        manual_edit_info=(),
        base_id: int = 0,
    ) -> int:
        return next_available_label_id(
            id_groups,
            manual_edit_info=manual_edit_info,
            base_id=base_id,
        )

    def label_ids_from_labels(self, labels: np.ndarray) -> list[int]:
        return label_ids_from_labels(labels)

    def remap_id_set(self, ids, old_ids, new_ids) -> set[int]:
        return remap_id_set(ids, old_ids, new_ids)

    def separate_with_label(
        self,
        labels,
        regionprops,
        ids_to_separate,
        max_id: int,
        *,
        click_coords_list=None,
    ):
        return separate_with_label(
            labels,
            regionprops,
            ids_to_separate,
            max_id,
            click_coords_list=click_coords_list,
        )

    def nearest_nonzero_2d(
        self,
        labels_2d: np.ndarray,
        y,
        x,
        *,
        max_dist=None,
        return_coords: bool = False,
    ):
        return nearest_nonzero_2D(
            labels_2d,
            y,
            x,
            max_dist=max_dist,
            return_coords=return_coords,
        )

    def nearest_nonzero_z_from_centroid(self, obj, *, current_z: int = -1):
        return nearest_nonzero_z_idx_from_z_centroid(obj, current_z=current_z)

    def split_along_convexity_defects(
        self,
        label_id: int,
        labels_2d: np.ndarray,
        max_id: int,
        *,
        max_i: int = 1,
        eps_percent: float = 0.01,
    ):
        return split_along_convexity_defects(
            label_id,
            labels_2d,
            max_id,
            max_i=max_i,
            eps_percent=eps_percent,
        )

    def split_connected_components(
        self,
        labels: np.ndarray,
        *,
        regionprops=None,
        max_id=None,
    ):
        return core_split_connected_components(
            labels,
            rp=regionprops,
            max_ID=max_id,
        )

    def binary_fill_holes(self, mask: np.ndarray, *, slice_by_slice: bool = True):
        return core_binary_fill_holes(mask, slice_by_slice=slice_by_slice)

    def convex_hull_mask(self, mask: np.ndarray, *, slice_by_slice: bool = True):
        return core_convex_hull_mask(mask, slice_by_slice=slice_by_slice)

    def count_objects(self, position_data, logger_func):
        return core_count_objects(position_data, logger_func)

    def format_trimmed_ids(self, ids, *, max_num_digits=10):
        return get_trimmed_list(list(ids), max_num_digits=max_num_digits)
