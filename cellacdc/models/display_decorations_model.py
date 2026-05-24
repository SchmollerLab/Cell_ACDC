"""Scriptable model rules for display decorations."""

from __future__ import annotations


class DisplayDecorationsModel:
    """Headless display-decoration decision rules."""

    def clamped_view_range(self, image_shape, view_range):
        y_size, x_size = image_shape[:2]
        x_range, y_range = view_range
        x_min = 0 if x_range[0] < 0 else x_range[0]
        y_min = 0 if y_range[0] < 0 else y_range[0]
        x_max = x_size if x_range[1] >= x_size else x_range[1]
        y_max = y_size if y_range[1] >= y_size else y_range[1]
        return int(y_min), int(y_max), int(x_min), int(x_max)

    def integer_view_range(self, view_range):
        x_range, y_range = view_range
        return (
            [round(x_range[0]), round(x_range[1])],
            [round(y_range[0]), round(y_range[1])],
        )

    def should_move_decoration(
        self,
        *,
        dialog_open: bool,
        move_with_zoom: bool,
    ) -> bool:
        return dialog_open or move_with_zoom

    def should_store_view_range(
        self,
        *,
        has_range_reset_state: bool,
        is_range_reset: bool = False,
    ) -> bool:
        return has_range_reset_state and is_range_reset

    def should_update_timestamp_frame(
        self,
        *,
        has_timestamp: bool,
        timestamp_enabled: bool,
    ) -> bool:
        return has_timestamp and timestamp_enabled
