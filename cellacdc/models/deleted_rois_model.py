"""Scriptable model rules for deleted ROI workflows."""

from __future__ import annotations

from collections.abc import Iterable


class DeletedRoisModel:
    """Headless decisions for deleted-ROI display and propagation."""

    def roi_axis(
        self,
        *,
        is_polyline: bool,
        labels_image_visible: bool,
    ) -> str:
        if is_polyline or not labels_image_visible:
            return 'left'
        return 'right'

    def should_render_deleted_roi(self, annotation_mode: str) -> bool:
        return 'nothing' not in annotation_mode

    def should_render_deleted_roi_contours(self, annotation_mode: str) -> bool:
        return 'contours' in annotation_mode

    def should_render_deleted_roi_overlay(self, annotation_mode: str) -> bool:
        return 'overlay segm. masks' in annotation_mode

    def should_initialize_overlay_masks(
        self,
        init: bool,
        annotation_mode: str,
    ) -> bool:
        return init and not self.should_render_deleted_roi_contours(
            annotation_mode
        )

    def labels_to_skip(self, deleted_ids: Iterable[int]) -> dict[int, bool]:
        return {deleted_id: True for deleted_id in deleted_ids}
