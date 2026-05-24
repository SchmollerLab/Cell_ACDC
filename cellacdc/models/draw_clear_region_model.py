"""Scriptable model rules for draw-clear-region workflows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DrawClearRegionToolbarState:
    """Desired z-slice toolbar state for the draw-clear tool."""

    update_z_control: bool
    z_control_enabled: bool = False
    size_z: int | None = None


class DrawClearRegionModel:
    """Headless draw-clear-region decision rules."""

    single_z_slice_projection = 'single z-slice'

    def toolbar_state(
        self,
        *,
        checked: bool,
        is_segm_3d: bool,
        size_z: int,
    ) -> DrawClearRegionToolbarState:
        if not is_segm_3d:
            return DrawClearRegionToolbarState(update_z_control=True)
        if not checked:
            return DrawClearRegionToolbarState(update_z_control=False)
        return DrawClearRegionToolbarState(
            update_z_control=True,
            z_control_enabled=True,
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
        if not is_segm_3d:
            return None
        if z_projection == self.single_z_slice_projection:
            return single_z_range
        return (0, size_z)

    def is_single_z_projection(self, z_projection: str) -> bool:
        return z_projection == self.single_z_slice_projection

    def empty_selection_warning(self, *, enclosed_only: bool) -> str:
        if enclosed_only:
            return (
                'None of the objects in the freehand region are fully enclosed'
            )
        return 'None of the objects are touching the freehand region'
