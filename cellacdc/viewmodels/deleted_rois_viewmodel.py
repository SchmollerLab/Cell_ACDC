"""View-model contracts for deleted ROI workflows."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from cellacdc.models.deleted_rois_model import DeletedRoisModel


@dataclass(frozen=True)
class DeletedRoisViewModel:
    """Application-facing deleted-ROI decisions."""

    model: DeletedRoisModel = field(default_factory=DeletedRoisModel)

    def roi_axis(
        self,
        *,
        is_polyline: bool,
        labels_image_visible: bool,
    ) -> str:
        return self.model.roi_axis(
            is_polyline=is_polyline,
            labels_image_visible=labels_image_visible,
        )

    def should_render_deleted_roi(self, annotation_mode: str) -> bool:
        return self.model.should_render_deleted_roi(annotation_mode)

    def should_render_deleted_roi_contours(self, annotation_mode: str) -> bool:
        return self.model.should_render_deleted_roi_contours(annotation_mode)

    def should_render_deleted_roi_overlay(self, annotation_mode: str) -> bool:
        return self.model.should_render_deleted_roi_overlay(annotation_mode)

    def should_initialize_overlay_masks(
        self,
        init: bool,
        annotation_mode: str,
    ) -> bool:
        return self.model.should_initialize_overlay_masks(
            init,
            annotation_mode,
        )

    def labels_to_skip(self, deleted_ids: Iterable[int]) -> dict[int, bool]:
        return self.model.labels_to_skip(deleted_ids)
