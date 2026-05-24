"""Scriptable model rules for canvas context menus."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeletedRoiClickDecision:
    """Decision for clicks on deleted-ROI overlays."""

    handled: bool
    show_context_menu: bool = False
    drag_roi: bool = False


class CanvasContextMenuModel:
    """Headless canvas context-menu decision rules."""

    scale_bar_target = 'scale_bar'
    timestamp_target = 'timestamp'
    gradient_target = 'gradient'

    def image_gradient_menu_target(
        self,
        *,
        scale_bar_highlighted: bool,
        timestamp_highlighted: bool,
    ) -> str:
        if scale_bar_highlighted:
            return self.scale_bar_target
        if timestamp_highlighted:
            return self.timestamp_target
        return self.gradient_target

    def deleted_roi_click_decision(
        self,
        *,
        clicked_on_roi: bool,
        left_click: bool,
        right_click: bool,
    ) -> DeletedRoiClickDecision:
        if not clicked_on_roi:
            return DeletedRoiClickDecision(handled=False)
        if right_click:
            return DeletedRoiClickDecision(
                handled=True,
                show_context_menu=True,
            )
        if left_click:
            return DeletedRoiClickDecision(handled=True, drag_roi=True)
        return DeletedRoiClickDecision(handled=False)
