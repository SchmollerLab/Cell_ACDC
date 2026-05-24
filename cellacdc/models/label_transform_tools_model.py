"""Scriptable model rules for label transform tools."""

from __future__ import annotations


class LabelTransformToolsModel:
    """Headless decision rules for label transform tools."""

    def reset_expand_label_id(self) -> int:
        return -1

    def should_reinitialize_expansion(
        self,
        *,
        expanding_id: int,
        hover_label_id: int,
        dilation: bool,
        is_dilation: bool,
    ) -> bool:
        return expanding_id != hover_label_id or dilation != is_dilation

    def should_start_moving_label(self, label_id: int) -> bool:
        return label_id != 0

    def point_in_shape(self, *, x: int, y: int, shape) -> bool:
        y_size, x_size = shape
        return x >= 0 and y >= 0 and x < x_size and y < y_size

    def move_delta(self, *, previous_pos, current_pos) -> tuple[int, int]:
        x_start, y_start = previous_pos
        x_current, y_current = current_pos
        return x_current - x_start, y_current - y_start

    def should_clear_move_state(self, *, checked: bool) -> bool:
        return not checked
