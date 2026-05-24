"""View-model contracts for display decorations."""

from __future__ import annotations

from dataclasses import dataclass, field

from cellacdc.models.display_decorations_model import (
    DisplayDecorationsModel,
)


@dataclass(frozen=True)
class DisplayDecorationsViewModel:
    """Application-facing display-decoration commands."""

    model: DisplayDecorationsModel = field(
        default_factory=DisplayDecorationsModel
    )

    def clamped_view_range(self, image_shape, view_range):
        return self.model.clamped_view_range(image_shape, view_range)

    def integer_view_range(self, view_range):
        return self.model.integer_view_range(view_range)

    def should_move_decoration(
        self,
        *,
        dialog_open: bool,
        move_with_zoom: bool,
    ) -> bool:
        return self.model.should_move_decoration(
            dialog_open=dialog_open,
            move_with_zoom=move_with_zoom,
        )

    def should_store_view_range(
        self,
        *,
        has_range_reset_state: bool,
        is_range_reset: bool = False,
    ) -> bool:
        return self.model.should_store_view_range(
            has_range_reset_state=has_range_reset_state,
            is_range_reset=is_range_reset,
        )

    def should_update_timestamp_frame(
        self,
        *,
        has_timestamp: bool,
        timestamp_enabled: bool,
    ) -> bool:
        return self.model.should_update_timestamp_frame(
            has_timestamp=has_timestamp,
            timestamp_enabled=timestamp_enabled,
        )
