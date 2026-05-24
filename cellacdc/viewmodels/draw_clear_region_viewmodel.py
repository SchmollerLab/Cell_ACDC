"""View-model contracts for draw-clear-region workflows."""

from __future__ import annotations

from dataclasses import dataclass, field

from cellacdc.models.draw_clear_region_model import (
    DrawClearRegionModel,
    DrawClearRegionToolbarState,
)


@dataclass(frozen=True)
class DrawClearRegionViewModel:
    """Application-facing draw-clear-region commands."""

    model: DrawClearRegionModel = field(
        default_factory=DrawClearRegionModel
    )

    def toolbar_state(
        self,
        *,
        checked: bool,
        is_segm_3d: bool,
        size_z: int,
    ) -> DrawClearRegionToolbarState:
        return self.model.toolbar_state(
            checked=checked,
            is_segm_3d=is_segm_3d,
            size_z=size_z,
        )

    def z_range_for_projection(
        self,
        *,
        is_segm_3d: bool,
        z_projection: str,
        size_z: int,
        single_z_range,
    ):
        return self.model.z_range_for_projection(
            is_segm_3d=is_segm_3d,
            z_projection=z_projection,
            size_z=size_z,
            single_z_range=single_z_range,
        )

    def is_single_z_projection(self, z_projection: str) -> bool:
        return self.model.is_single_z_projection(z_projection)

    def empty_selection_warning(self, *, enclosed_only: bool) -> str:
        return self.model.empty_selection_warning(
            enclosed_only=enclosed_only
        )
