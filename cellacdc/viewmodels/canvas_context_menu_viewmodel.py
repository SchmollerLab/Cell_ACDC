"""View-model contracts for canvas context menus."""

from __future__ import annotations

from dataclasses import dataclass, field

from cellacdc.models.canvas_context_menu_model import (
    CanvasContextMenuModel,
    DeletedRoiClickDecision,
)


@dataclass(frozen=True)
class CanvasContextMenuViewModel:
    """Application-facing canvas context-menu commands."""

    model: CanvasContextMenuModel = field(
        default_factory=CanvasContextMenuModel
    )

    @property
    def scale_bar_target(self) -> str:
        return self.model.scale_bar_target

    @property
    def timestamp_target(self) -> str:
        return self.model.timestamp_target

    def image_gradient_menu_target(
        self,
        *,
        scale_bar_highlighted: bool,
        timestamp_highlighted: bool,
    ) -> str:
        return self.model.image_gradient_menu_target(
            scale_bar_highlighted=scale_bar_highlighted,
            timestamp_highlighted=timestamp_highlighted,
        )

    def deleted_roi_click_decision(
        self,
        *,
        clicked_on_roi: bool,
        left_click: bool,
        right_click: bool,
    ) -> DeletedRoiClickDecision:
        return self.model.deleted_roi_click_decision(
            clicked_on_roi=clicked_on_roi,
            left_click=left_click,
            right_click=right_click,
        )
