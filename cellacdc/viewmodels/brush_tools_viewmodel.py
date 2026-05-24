"""View-model contracts for brush and eraser tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cellacdc.models.brush_tools_model import BrushToolsModel


@dataclass(frozen=True)
class BrushToolsViewModel:
    """Application-facing brush/eraser decisions."""

    model: BrushToolsModel = field(default_factory=BrushToolsModel)

    def checked_setting_value(self, checked: bool) -> str:
        return self.model.checked_setting_value(checked)

    def default_delete_object_info_value(self) -> str:
        return self.model.default_delete_object_info_value()

    def should_show_delete_object_info(self, setting_value: Any) -> bool:
        return self.model.should_show_delete_object_info(setting_value)

    def delete_object_info_value(
        self,
        do_not_show_again_checked: bool,
    ) -> str:
        return self.model.delete_object_info_value(do_not_show_again_checked)

    def should_fill_holes(
        self,
        sender: str,
        *,
        auto_fill_checked: bool,
    ) -> bool:
        return self.model.should_fill_holes(
            sender,
            auto_fill_checked=auto_fill_checked,
        )

    def brush_toolbar_visible(
        self,
        edit_id_visible: bool,
        *,
        brush_size_visible: bool,
        auto_fill_visible: bool,
        auto_hide_visible: bool,
    ) -> bool:
        return self.model.brush_toolbar_visible(
            edit_id_visible,
            brush_size_visible=brush_size_visible,
            auto_fill_visible=auto_fill_visible,
            auto_hide_visible=auto_hide_visible,
        )

    def disk_mask(self, brush_size: int):
        return self.model.disk_mask(brush_size)

    def disk_mask_bounds(
        self,
        image_shape: tuple[int, int],
        brush_size: int,
        xdata: int,
        ydata: int,
        disk_mask,
    ):
        return self.model.disk_mask_bounds(
            image_shape,
            brush_size,
            xdata,
            ydata,
            disk_mask,
        )

    def magic_wand_flood_tolerance(
        self,
        tolerance_percent: float,
        image_min: float,
        image_max: float,
    ):
        return self.model.magic_wand_flood_tolerance(
            tolerance_percent,
            image_min,
            image_max,
        )
