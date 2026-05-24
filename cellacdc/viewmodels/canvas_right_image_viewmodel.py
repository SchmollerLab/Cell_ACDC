"""View-model contracts for duplicated right-image interactions."""

from __future__ import annotations

from dataclasses import dataclass, field

from cellacdc.models.canvas_right_image_model import CanvasRightImageModel


@dataclass(frozen=True)
class CanvasRightImageViewModel:
    """Application-facing duplicated right-image commands."""

    model: CanvasRightImageModel = field(default_factory=CanvasRightImageModel)

    def should_show_context_menu(
        self,
        *,
        right_click: bool,
        is_right_click_action_on: bool,
    ) -> bool:
        return self.model.should_show_context_menu(
            right_click=right_click,
            is_right_click_action_on=is_right_click_action_on,
        )
