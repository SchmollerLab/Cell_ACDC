"""Scriptable model rules for brush and eraser tools."""

from __future__ import annotations

from typing import Any

import skimage.morphology


class BrushToolsModel:
    """Headless decisions and geometry for brush/eraser tools."""

    yes_value = 'Yes'
    no_value = 'No'

    def checked_setting_value(self, checked: bool) -> str:
        return self.yes_value if checked else self.no_value

    def default_delete_object_info_value(self) -> str:
        return self.yes_value

    def should_show_delete_object_info(self, setting_value: Any) -> bool:
        return setting_value == self.yes_value

    def delete_object_info_value(
        self,
        do_not_show_again_checked: bool,
    ) -> str:
        return (
            self.no_value
            if do_not_show_again_checked
            else self.yes_value
        )

    def should_fill_holes(
        self,
        sender: str,
        *,
        auto_fill_checked: bool,
    ) -> bool:
        return sender == 'brush' and auto_fill_checked

    def brush_toolbar_visible(
        self,
        edit_id_visible: bool,
        *,
        brush_size_visible: bool,
        auto_fill_visible: bool,
        auto_hide_visible: bool,
    ) -> bool:
        return any(
            (
                edit_id_visible,
                brush_size_visible,
                auto_fill_visible,
                auto_hide_visible,
            )
        )

    def disk_mask(self, brush_size: int):
        return skimage.morphology.disk(brush_size, dtype=bool)

    def disk_mask_bounds(
        self,
        image_shape: tuple[int, int],
        brush_size: int,
        xdata: int,
        ydata: int,
        disk_mask,
    ):
        y_size, x_size = image_shape
        y_bottom, x_left = ydata - brush_size, xdata - brush_size
        y_top, x_right = ydata + brush_size + 1, xdata + brush_size + 1

        if x_left < 0:
            if y_bottom < 0:
                disk_mask = disk_mask.copy()
                disk_mask = disk_mask[-y_bottom:, -x_left:]
                y_bottom = 0
            elif y_top > y_size:
                disk_mask = disk_mask.copy()
                disk_mask = disk_mask[0:y_size - y_bottom, -x_left:]
                y_top = y_size
            else:
                disk_mask = disk_mask.copy()
                disk_mask = disk_mask[:, -x_left:]
            x_left = 0

        elif x_right > x_size:
            if y_bottom < 0:
                disk_mask = disk_mask.copy()
                disk_mask = disk_mask[-y_bottom:, 0:x_size - x_left]
                y_bottom = 0
            elif y_top > y_size:
                disk_mask = disk_mask.copy()
                disk_mask = disk_mask[0:y_size - y_bottom, 0:x_size - x_left]
                y_top = y_size
            else:
                disk_mask = disk_mask.copy()
                disk_mask = disk_mask[:, 0:x_size - x_left]
            x_right = x_size

        elif y_bottom < 0:
            disk_mask = disk_mask.copy()
            disk_mask = disk_mask[-y_bottom:]
            y_bottom = 0

        elif y_top > y_size:
            disk_mask = disk_mask.copy()
            disk_mask = disk_mask[0:y_size - y_bottom]
            y_top = y_size

        return y_bottom, x_left, y_top, x_right, disk_mask

    def magic_wand_flood_tolerance(
        self,
        tolerance_percent: float,
        image_min: float,
        image_max: float,
    ):
        if tolerance_percent == 0:
            return None
        return (image_max - image_min) * (tolerance_percent / 100)
