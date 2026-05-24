"""Scriptable model rules for active-tool workflows."""

from __future__ import annotations


class ToolActivationModel:
    """Headless decisions for active-tool and hover workflows."""

    def manual_annotation_highlight_color(
        self,
        *,
        current_frame_i: int,
        frame_to_restore: int | None,
    ) -> str:
        if current_frame_i == frame_to_restore:
            return 'green'
        if frame_to_restore is not None and current_frame_i < frame_to_restore:
            return 'gold'
        return 'red'

    def should_highlight_hover_lost_object(
        self,
        *,
        has_no_modifier: bool,
        copy_lost_object_checked: bool,
        is_exit_event: bool,
    ) -> bool:
        return (
            has_no_modifier
            and copy_lost_object_checked
            and not is_exit_event
        )

    def point_in_shape(self, x: int, y: int, shape: tuple[int, int]) -> bool:
        height, width = shape
        return x >= 0 and x < width and y >= 0 and y < height

    def should_hide_hover_objects(
        self,
        *,
        brush_auto_hide_checked: bool,
        force: bool,
    ) -> bool:
        return brush_auto_hide_checked or force

    def should_disable_non_functional_buttons(self, is_segm_3d: bool) -> bool:
        return is_segm_3d
