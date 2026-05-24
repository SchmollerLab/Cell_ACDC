"""View-model contracts for label transform tools."""

from __future__ import annotations

from dataclasses import dataclass, field

from cellacdc.models.label_transform_tools_model import (
    LabelTransformToolsModel,
)


@dataclass(frozen=True)
class LabelTransformToolsViewModel:
    """Application-facing label transform commands."""

    model: LabelTransformToolsModel = field(
        default_factory=LabelTransformToolsModel
    )

    def reset_expand_label_id(self) -> int:
        return self.model.reset_expand_label_id()

    def should_reinitialize_expansion(
        self,
        *,
        expanding_id: int,
        hover_label_id: int,
        dilation: bool,
        is_dilation: bool,
    ) -> bool:
        return self.model.should_reinitialize_expansion(
            expanding_id=expanding_id,
            hover_label_id=hover_label_id,
            dilation=dilation,
            is_dilation=is_dilation,
        )

    def should_start_moving_label(self, label_id: int) -> bool:
        return self.model.should_start_moving_label(label_id)

    def point_in_shape(self, *, x: int, y: int, shape) -> bool:
        return self.model.point_in_shape(x=x, y=y, shape=shape)

    def move_delta(self, *, previous_pos, current_pos) -> tuple[int, int]:
        return self.model.move_delta(
            previous_pos=previous_pos,
            current_pos=current_pos,
        )

    def should_clear_move_state(self, *, checked: bool) -> bool:
        return self.model.should_clear_move_state(checked=checked)
