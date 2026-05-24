"""View adapter for canvas tool interaction decisions."""

from __future__ import annotations

from cellacdc.viewmodels.canvas_tool_viewmodel import CanvasToolViewModel


class CanvasToolView:
    """Qt-facing adapter around the scriptable canvas tool view-model."""

    def __init__(self, view_model: CanvasToolViewModel):
        self.view_model = view_model

    def viewer_mode_allows_press(
        self,
        mode: str,
        *,
        can_add_point: bool = False,
        can_ruler: bool = False,
    ) -> bool:
        return self.view_model.viewer_mode_allows_press(
            mode,
            can_add_point=can_add_point,
            can_ruler=can_ruler,
        )

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
        return self.view_model.should_forward_img1_press_to_img2(
            right_click=right_click,
            middle_click=middle_click,
            can_add_point=can_add_point,
            mode=mode,
            is_snapshot=is_snapshot,
            is_annotate_division=is_annotate_division,
            manual_background_on=manual_background_on,
        )

    def should_forward_img1_release_to_img2(
        self,
        *,
        right_click: bool,
        mode: str,
        is_snapshot: bool,
    ) -> bool:
        return self.view_model.should_forward_img1_release_to_img2(
            right_click=right_click,
            mode=mode,
            is_snapshot=is_snapshot,
        )

    def store_manual_separate_draw_mode(self, settings, settings_csv_path, mode):
        self.view_model.apply_manual_separate_draw_mode(settings, mode)
        settings.to_csv(settings_csv_path)
