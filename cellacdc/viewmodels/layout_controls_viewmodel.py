"""View-model contracts for layout-control workflows."""

from __future__ import annotations

from dataclasses import dataclass, field

from cellacdc.models.layout_controls_model import LayoutControlsModel


@dataclass(frozen=True)
class LayoutControlsViewModel:
    """Application-facing decisions for GUI layout controls."""

    model: LayoutControlsModel = field(default_factory=LayoutControlsModel)

    def zoom_percentage_from_text(self, text: str) -> int:
        return self.model.zoom_percentage_from_text(text)

    def zoom_factors(self, percentage: int) -> tuple[float, float] | None:
        return self.model.zoom_factors(percentage)

    def checked_setting_value(self, checked: bool) -> str:
        return self.model.checked_setting_value(checked)

    def checked_from_setting_value(self, value) -> bool:
        return self.model.checked_from_setting_value(value)

    def should_retain_z_slider_space(
        self,
        *,
        checked: bool,
        z_slice_enabled: bool,
    ) -> bool:
        return self.model.should_retain_z_slider_space(
            checked=checked,
            z_slice_enabled=z_slice_enabled,
        )

    def tool_name_from_tooltip(self, tooltip: str) -> str:
        return self.model.tool_name_from_tooltip(tooltip)
