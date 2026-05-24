"""Qt-free model rules for canvas selection interactions."""

from __future__ import annotations


class CanvasSelectionModel:
    """Headless decisions for canvas selection workflows."""

    viewer_mode = 'Viewer'
    segmentation_mode = 'Segmentation and Tracking'

    def should_drag_image(
        self,
        *,
        left_click: bool,
        eraser_on: bool,
        brush_on: bool,
        middle_click: bool,
        pan_click: bool,
    ) -> bool:
        return pan_click or (
            left_click and not eraser_on and not brush_on and not middle_click
        )

    def should_blink_viewer_mode(
        self,
        *,
        mode: str,
        middle_click: bool,
        right_action_on: bool = False,
        custom_action_on: bool = False,
        right_click: bool = False,
    ) -> bool:
        if mode != self.viewer_mode:
            return False
        if middle_click:
            return True
        return (right_action_on or custom_action_on) and (
            right_click or middle_click
        )

    def should_show_labels_menu(
        self,
        *,
        right_click: bool,
        right_action_on: bool,
        middle_click: bool,
        event_from_img1: bool,
    ) -> bool:
        return (
            right_click
            and not right_action_on
            and not middle_click
            and not event_from_img1
        )

    def can_delete(self, *, mode: str, is_snapshot: bool) -> bool:
        return mode == self.segmentation_mode or is_snapshot

    def is_viewer_mode(self, mode: str) -> bool:
        return mode == self.viewer_mode

    def should_process_release(
        self,
        *,
        mode: str,
        in_bounds: bool,
    ) -> bool:
        return mode != self.viewer_mode and in_bounds
