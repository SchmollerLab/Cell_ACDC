"""Scriptable model rules for points-layer workflows."""

from __future__ import annotations

from collections.abc import Mapping


class PointsLayersModel:
    """Headless decisions for points-layer GUI workflows."""

    recovery_tolerance_seconds = 15

    def click_entry_table_filename(
        self,
        basename: str,
        table_endname: str,
    ) -> str:
        table_basename = basename if basename.endswith('_') else f'{basename}_'
        filename = f'{table_basename}{table_endname}'
        if not filename.endswith('.csv'):
            filename = f'{filename}.csv'
        return filename

    def should_load_recovery_table(
        self,
        *,
        recovery_exists: bool,
        main_exists: bool,
        recovery_mtime: float | None,
        main_mtime: float | None,
    ) -> bool:
        if not recovery_exists:
            return False
        if not main_exists:
            return True
        if recovery_mtime is None or main_mtime is None:
            return False
        return (
            recovery_mtime
            > main_mtime + self.recovery_tolerance_seconds
        )

    def should_compute_points_layer(
        self,
        *,
        layer_type_index: int,
        compute_points_layers: bool,
    ) -> bool:
        return layer_type_index < 2 and compute_points_layers

    def should_log_missing_frame_points(self, layer_type_index: int) -> bool:
        return layer_type_index != 4

    def should_use_z_slice(
        self,
        *,
        z_projection_mode: str,
        size_z: int,
        frame_points_data: Mapping,
    ) -> bool:
        return (
            z_projection_mode == 'single z-slice'
            and size_z > 1
            and 'x' not in frame_points_data
        )
