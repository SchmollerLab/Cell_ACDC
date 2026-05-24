"""View-model contracts for quick settings."""

from __future__ import annotations

from dataclasses import dataclass, field

from cellacdc.models.quick_settings_model import (
    FontSizeSetting,
    QuickSettingsModel,
)


@dataclass(frozen=True)
class QuickSettingsViewModel:
    """Application-facing quick-settings commands."""

    model: QuickSettingsModel = field(default_factory=QuickSettingsModel)

    def font_size_setting(
        self,
        saved_font_size,
        *,
        has_px_mode: bool,
    ) -> FontSizeSetting:
        return self.model.font_size_setting(
            saved_font_size,
            has_px_mode=has_px_mode,
        )

    def should_update_all_contours(self, *, is_data_loaded: bool) -> bool:
        return self.model.should_update_all_contours(
            is_data_loaded=is_data_loaded
        )
