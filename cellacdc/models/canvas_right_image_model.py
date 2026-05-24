"""Scriptable model rules for duplicated right-image interactions."""

from __future__ import annotations


class CanvasRightImageModel:
    """Headless duplicated right-image event rules."""

    def should_show_context_menu(
        self,
        *,
        right_click: bool,
        is_right_click_action_on: bool,
    ) -> bool:
        return right_click and not is_right_click_action_on
