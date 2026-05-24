"""View-model commands for point-layer data."""

from __future__ import annotations

from cellacdc.io.readers.points import load_click_points_table, load_points_table
from cellacdc.io.writers.points import (
    click_points_table_filename,
    save_click_points_table,
)

from cellacdc.domain.points import (
    add_click_point,
    click_points_table_to_data,
    flatten_frame_points_data,
    next_click_point_id,
    point_id_already_new,
    points_table_to_data,
    remove_click_points,
)


class PointsMixin:
    """Application-facing commands for point-layer data transforms."""

    def add_click_point(
        self,
        points_data_pos,
        frame_i: int,
        x: float,
        y: float,
        point_id: int,
        *,
        size_z: int = 1,
        z_slice: int | None = None,
    ):
        return add_click_point(
            points_data_pos,
            frame_i,
            x,
            y,
            point_id,
            size_z=size_z,
            z_slice=z_slice,
        )

    def click_points_table_filename(
        self,
        basename: str,
        table_endname: str,
    ) -> str:
        return click_points_table_filename(basename, table_endname)

    def click_points_table_to_data(self, df, *, size_z: int = 1):
        return click_points_table_to_data(df, size_z=size_z)

    def flatten_frame_points_data(
        self,
        frame_points_data,
        *,
        z_slice: int | None = None,
        z_radius: int = 0,
    ):
        return flatten_frame_points_data(
            frame_points_data,
            z_slice=z_slice,
            z_radius=z_radius,
        )

    def load_click_points_table(self, filepath):
        return load_click_points_table(filepath)

    def load_points_table(self, filepath):
        return load_points_table(filepath)

    def loaded_table_to_points_data(
        self,
        df,
        t_col,
        z_col,
        y_col,
        x_col,
    ):
        return points_table_to_data(df, t_col, z_col, y_col, x_col)

    def next_click_point_id(
        self,
        points_data_pos,
        frame_i: int,
        current_id: int,
        *,
        size_z: int = 1,
    ) -> int:
        return next_click_point_id(
            points_data_pos,
            frame_i,
            current_id,
            size_z=size_z,
        )

    def point_id_already_new(
        self,
        points_data_pos,
        frame_i: int,
        point_id: int,
        known_ids,
    ) -> bool:
        return point_id_already_new(
            points_data_pos,
            frame_i,
            point_id,
            known_ids,
        )

    def remove_click_points(
        self,
        frame_points_data,
        points,
        *,
        z_slice: int | None = None,
        z_radius: int = 0,
    ) -> list[int]:
        return remove_click_points(
            frame_points_data,
            points,
            z_slice=z_slice,
            z_radius=z_radius,
        )

    def save_click_points_table(
        self,
        filepath,
        df,
        sort_by=("frame_i", "Cell_ID"),
    ):
        return save_click_points_table(filepath, df, sort_by=sort_by)
