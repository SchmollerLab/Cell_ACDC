"""View-model contracts for canvas tool interaction decisions."""

from __future__ import annotations

from dataclasses import dataclass, field

from cellacdc.models.canvas_tool_model import CanvasToolModel


@dataclass(frozen=True)
class CanvasToolViewModel:
    """Application-facing canvas tool commands."""

    model: CanvasToolModel = field(default_factory=CanvasToolModel)

    def viewer_mode_allows_press(
        self,
        mode: str,
        *,
        can_add_point: bool = False,
        can_ruler: bool = False,
    ) -> bool:
        return self.model.viewer_mode_allows_press(
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
        return self.model.should_forward_img1_press_to_img2(
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
        return self.model.should_forward_img1_release_to_img2(
            right_click=right_click,
            mode=mode,
            is_snapshot=is_snapshot,
        )

    def apply_manual_separate_draw_mode(self, settings, mode):
        key, value = self.model.manual_separate_draw_mode_update(mode)
        settings.at[key, 'value'] = value
        return settings
