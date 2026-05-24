"""View adapter for canvas tool interaction decisions."""

from __future__ import annotations


class CanvasTool:
    """Extracted from guiWin."""

    def storeManualSeparateDrawMode(self, mode):
        self.df_settings.at["manual_separate_draw_mode", "value"] = mode
        self.df_settings.to_csv(self.settings_csv_path)
