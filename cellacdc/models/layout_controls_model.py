"""Scriptable model rules for layout-control workflows."""

from __future__ import annotations

import re


class LayoutControlsModel:
    """Headless decisions for GUI layout controls."""

    yes_value = 'Yes'
    no_value = 'No'

    def zoom_percentage_from_text(self, text: str) -> int:
        return int(re.findall(r'(\d+)%', text)[0])

    def zoom_factors(self, percentage: int) -> tuple[float, float] | None:
        if percentage == 100:
            return None
        factor = percentage / 100
        return factor, factor

    def checked_setting_value(self, checked: bool) -> str:
        return self.yes_value if checked else self.no_value

    def checked_from_setting_value(self, value) -> bool:
        return value == self.yes_value

    def should_retain_z_slider_space(
        self,
        *,
        checked: bool,
        z_slice_enabled: bool,
    ) -> bool:
        return checked and z_slice_enabled

    def tool_name_from_tooltip(self, tooltip: str) -> str:
        return re.findall(r'Name: (.*)', tooltip)[0]
