"""Scriptable model rules for canvas tool interaction decisions."""

from __future__ import annotations


class CanvasToolModel:
    """Headless canvas tool decision rules."""

    manual_separate_draw_mode_key = 'manual_separate_draw_mode'

    def viewer_mode_allows_press(
        self,
        mode: str,
        *,
        can_add_point: bool = False,
        can_ruler: bool = False,
    ) -> bool:
        return mode != 'Viewer' or can_add_point or can_ruler

    def should_forward_img1_press_to_img2(
        self,
        *,
        right_click: bool,
        middle_click: bool,
        can_add_point: bool,
        mode: str,
        is_snapshot: bool,
        is_annotate_division: bool,
        manual_background_on: bool,
    ) -> bool:
        return (
            (right_click or (middle_click and not can_add_point))
            and (mode == 'Segmentation and Tracking' or is_snapshot)
            and not is_annotate_division
            and not manual_background_on
        )

    def should_forward_img1_release_to_img2(
        self,
        *,
        right_click: bool,
        mode: str,
        is_snapshot: bool,
    ) -> bool:
        return (
            (mode == 'Segmentation and Tracking' or is_snapshot)
            and right_click
        )

    def manual_separate_draw_mode_update(self, mode) -> tuple[str, object]:
        return self.manual_separate_draw_mode_key, mode
