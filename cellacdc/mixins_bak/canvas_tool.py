"""View adapter for canvas tool interaction decisions."""

from __future__ import annotations


class CanvasToolMixin:
    """Qt-facing adapter around the scriptable canvas tool decision rules."""

    manual_separate_draw_mode_key = "manual_separate_draw_mode"

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
            and (mode == "Segmentation and Tracking" or is_snapshot)
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
        return (mode == "Segmentation and Tracking" or is_snapshot) and right_click

    def store_manual_separate_draw_mode(self, settings, settings_csv_path, mode):
        settings.at[self.manual_separate_draw_mode_key, "value"] = mode
        settings.to_csv(settings_csv_path)

    def viewer_mode_allows_press(
        self,
        mode: str,
        *,
        can_add_point: bool = False,
        can_ruler: bool = False,
    ) -> bool:
        return mode != "Viewer" or can_add_point or can_ruler
