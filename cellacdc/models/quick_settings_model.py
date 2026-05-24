"""Scriptable model rules for quick settings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FontSizeSetting:
    """Parsed font-size setting and migration requirement."""

    value: int
    add_px_mode_setting: bool = False


class QuickSettingsModel:
    """Headless quick-settings decision rules."""

    def font_size_setting(
        self,
        saved_font_size,
        *,
        has_px_mode: bool,
    ) -> FontSizeSetting:
        saved_font_size = str(saved_font_size)
        if saved_font_size.find('pt') != -1:
            saved_font_size = saved_font_size[:-2]
        font_size = int(saved_font_size)
        if has_px_mode:
            return FontSizeSetting(value=font_size)
        return FontSizeSetting(
            value=2*font_size,
            add_px_mode_setting=True,
        )

    def should_update_all_contours(self, *, is_data_loaded: bool) -> bool:
        return is_data_loaded
