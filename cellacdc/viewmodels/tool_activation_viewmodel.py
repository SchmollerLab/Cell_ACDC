"""View-model contracts for active-tool workflows."""

from __future__ import annotations

from dataclasses import dataclass, field

from cellacdc.models.tool_activation_model import ToolActivationModel

from .label_edits_viewmodel import LabelEditViewModel
from .tracking_viewmodel import TrackingViewModel


@dataclass(frozen=True)
class ToolActivationViewModel:
    """Application-facing decisions for active tools."""

    model: ToolActivationModel = field(default_factory=ToolActivationModel)
    label_edits: LabelEditViewModel = field(default_factory=LabelEditViewModel)
    tracking: TrackingViewModel = field(default_factory=TrackingViewModel)

    def manual_annotation_highlight_color(
        self,
        *,
        current_frame_i: int,
        frame_to_restore: int | None,
    ) -> str:
        return self.model.manual_annotation_highlight_color(
            current_frame_i=current_frame_i,
            frame_to_restore=frame_to_restore,
        )

    def should_highlight_hover_lost_object(
        self,
        *,
        has_no_modifier: bool,
        copy_lost_object_checked: bool,
        is_exit_event: bool,
    ) -> bool:
        return self.model.should_highlight_hover_lost_object(
            has_no_modifier=has_no_modifier,
            copy_lost_object_checked=copy_lost_object_checked,
            is_exit_event=is_exit_event,
        )

    def point_in_shape(self, x: int, y: int, shape: tuple[int, int]) -> bool:
        return self.model.point_in_shape(x, y, shape)

    def should_hide_hover_objects(
        self,
        *,
        brush_auto_hide_checked: bool,
        force: bool,
    ) -> bool:
        return self.model.should_hide_hover_objects(
            brush_auto_hide_checked=brush_auto_hide_checked,
            force=force,
        )

    def should_disable_non_functional_buttons(self, is_segm_3d: bool) -> bool:
        return self.model.should_disable_non_functional_buttons(is_segm_3d)
