"""View-model contracts for canvas hover interactions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cellacdc.models.canvas_hover_model import CanvasHoverModel


@dataclass(frozen=True)
class CanvasHoverViewModel:
    """Application-facing canvas hover decisions."""

    model: CanvasHoverModel = field(default_factory=CanvasHoverModel)

    def point_in_bounds(
        self,
        image_shape: tuple[int, int],
        xdata: int,
        ydata: int,
    ) -> bool:
        return self.model.point_in_bounds(image_shape, xdata, ydata)

    def hover_position(self, is_exit: bool, position) -> tuple[Any, Any]:
        return self.model.hover_position(is_exit, position)

    def should_set_mirrored_cursor(
        self,
        *,
        override_cursor_is_none: bool,
        is_exit: bool,
        mirrored_cursor_enabled: bool,
        is_hover_img1: bool = True,
    ) -> bool:
        return self.model.should_set_mirrored_cursor(
            override_cursor_is_none=override_cursor_is_none,
            is_exit=is_exit,
            mirrored_cursor_enabled=mirrored_cursor_enabled,
            is_hover_img1=is_hover_img1,
        )

    def should_draw_ruler_line(
        self,
        *,
        ruler_checked: bool,
        add_deleted_polyline_checked: bool,
        temp_segment_on: bool,
        is_exit: bool,
    ) -> bool:
        return self.model.should_draw_ruler_line(
            ruler_checked=ruler_checked,
            add_deleted_polyline_checked=add_deleted_polyline_checked,
            temp_segment_on=temp_segment_on,
            is_exit=is_exit,
        )

    def cursor_flags(self, **kwargs) -> dict[str, bool]:
        return self.model.cursor_flags(**kwargs)
