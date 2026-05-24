"""View-model contracts for points-layer workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from cellacdc.models.points_layers_model import PointsLayersModel

from .points import PointsViewModel


@dataclass(frozen=True)
class PointsLayersViewModel:
    """Application-facing points-layer decisions and data transforms."""

    model: PointsLayersModel = field(default_factory=PointsLayersModel)
    points: PointsViewModel = field(default_factory=PointsViewModel)

    def click_entry_table_filename(
        self,
        basename: str,
        table_endname: str,
    ) -> str:
        return self.model.click_entry_table_filename(basename, table_endname)

    def should_load_recovery_table(
        self,
        *,
        recovery_exists: bool,
        main_exists: bool,
        recovery_mtime: float | None,
        main_mtime: float | None,
    ) -> bool:
        return self.model.should_load_recovery_table(
            recovery_exists=recovery_exists,
            main_exists=main_exists,
            recovery_mtime=recovery_mtime,
            main_mtime=main_mtime,
        )

    def should_compute_points_layer(
        self,
        *,
        layer_type_index: int,
        compute_points_layers: bool,
    ) -> bool:
        return self.model.should_compute_points_layer(
            layer_type_index=layer_type_index,
            compute_points_layers=compute_points_layers,
        )

    def should_log_missing_frame_points(self, layer_type_index: int) -> bool:
        return self.model.should_log_missing_frame_points(layer_type_index)

    def should_use_z_slice(
        self,
        *,
        z_projection_mode: str,
        size_z: int,
        frame_points_data: Mapping,
    ) -> bool:
        return self.model.should_use_z_slice(
            z_projection_mode=z_projection_mode,
            size_z=size_z,
            frame_points_data=frame_points_data,
        )
