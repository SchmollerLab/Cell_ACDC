"""View-model contracts for frame and position navigation."""

from __future__ import annotations

from dataclasses import dataclass, field

from cellacdc.models.frame_navigation_model import FrameNavigationModel

from .frame_metadata_viewmodel import FrameMetadataViewModel
from .label_edits_viewmodel import LabelEditViewModel


@dataclass(frozen=True)
class FrameNavigationViewModel:
    """Application-facing frame/position navigation decisions."""

    model: FrameNavigationModel = field(default_factory=FrameNavigationModel)
    frame_metadata: FrameMetadataViewModel = field(
        default_factory=FrameMetadataViewModel
    )
    label_edits: LabelEditViewModel = field(default_factory=LabelEditViewModel)

    def should_show_next_frame_image(self, **kwargs) -> bool:
        return self.model.should_show_next_frame_image(**kwargs)

    def next_frame_index(self, **kwargs) -> int:
        return self.model.next_frame_index(**kwargs)

    def navigation_position(self, **kwargs) -> int:
        return self.model.navigation_position(**kwargs)

    def navigation_limit(self, **kwargs):
        return self.model.navigation_limit(**kwargs)

    def should_store_when_slider_moves(self, *, mode: str) -> bool:
        return self.model.should_store_when_slider_moves(mode=mode)

    def should_warn_lost_objects(self, **kwargs) -> bool:
        return self.model.should_warn_lost_objects(**kwargs)

    def blocks_future_manual_annotation(self, **kwargs) -> bool:
        return self.model.blocks_future_manual_annotation(**kwargs)

    def should_apply_new_frame_tools(self, **kwargs) -> bool:
        return self.model.should_apply_new_frame_tools(**kwargs)

    def is_single_z_slice_projection(self, how: str) -> bool:
        return self.model.is_single_z_slice_projection(how)

    def should_disable_overlay_z_slice(self, how: str) -> bool:
        return self.model.should_disable_overlay_z_slice(how)

    def projection_frame_indices(self, **kwargs):
        return self.model.projection_frame_indices(**kwargs)

    def z_slice_frame_indices(self, **kwargs):
        return self.model.z_slice_frame_indices(**kwargs)

    def nearest_nonzero_z_from_centroid(self, *args, **kwargs):
        return self.label_edits.nearest_nonzero_z_from_centroid(*args, **kwargs)

    def empty_frame_record(self):
        return self.frame_metadata.empty_frame_record()

    def empty_frame_records(self, count: int):
        return self.frame_metadata.empty_frame_records(count)
