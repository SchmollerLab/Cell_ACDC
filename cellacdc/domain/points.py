"""Pure point-layer table transformations."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


def infer_points_column_mapping(columns: Iterable[str]) -> dict[str, str]:
    """Infer standard point-layer columns from table columns."""
    column_set = set(columns)
    return {
        'x': 'x' if 'x' in column_set else 'None',
        'y': 'y' if 'y' in column_set else 'None',
        'z': 'z' if 'z' in column_set else 'None',
        't': 'frame_i' if 'frame_i' in column_set else 'None',
    }


def points_table_to_data(
    df: pd.DataFrame,
    t_col: str,
    z_col: str,
    y_col: str,
    x_col: str,
    *,
    include_row_data: bool = True,
) -> dict[Any, dict[str, list[Any]] | dict[int, dict[str, list[Any]]]]:
    """Convert a points table to the nested point-layer data structure."""
    points_data = {}
    points_df = df.copy()
    if 'id' not in points_df.columns:
        points_df['id'] = ''

    if t_col != 'None':
        grouped = points_df.groupby(t_col)
    else:
        grouped = [(0, points_df)]

    for frame_i, df_frame in grouped:
        if z_col != 'None':
            df_frame = df_frame.copy()
            df_frame[z_col] = df_frame[z_col].round().astype(int)
            points_data[frame_i] = {}
            for z in df_frame[z_col].unique():
                z_int = round(z)
                df_z = df_frame[df_frame[z_col] == z_int]
                z_data = {
                    'x': df_z[x_col].to_list(),
                    'y': df_z[y_col].to_list(),
                    'id': df_z['id'].to_list(),
                }
                if include_row_data:
                    z_data['data'] = [
                        row.to_string() for _, row in df_z.iterrows()
                    ]
                points_data[frame_i][z_int] = z_data
        else:
            frame_data = {
                'x': df_frame[x_col].to_list(),
                'y': df_frame[y_col].to_list(),
                'id': df_frame['id'].to_list(),
            }
            if include_row_data:
                frame_data['data'] = [
                    row.to_string() for _, row in df_frame.iterrows()
                ]
            points_data[frame_i] = frame_data
    return points_data


def click_points_table_to_data(
    df: pd.DataFrame,
    *,
    size_z: int = 1,
) -> dict[Any, dict[str, list[Any]] | dict[int, dict[str, list[Any]]]]:
    """Convert click-entry point tables to GUI point-layer data."""
    if size_z > 1 and df['z'].isna().any():
        raise ValueError('3D point tables require z values for every row')

    z_col = 'z' if size_z > 1 else 'None'
    return points_table_to_data(
        df,
        'frame_i',
        z_col,
        'y',
        'x',
        include_row_data=False,
    )


def point_id_already_new(
    points_data_pos: dict[Any, Any] | None,
    frame_i: int,
    point_id: int,
    known_ids: Iterable[int],
) -> bool:
    """Return whether ``point_id`` is new and not already present in frame data."""
    if point_id in known_ids:
        return False

    if points_data_pos is None:
        return True

    frame_points_data = points_data_pos.get(frame_i)
    if frame_points_data is None:
        return True

    if 'x' not in frame_points_data:
        for z_data in frame_points_data.values():
            if point_id in z_data['id']:
                return False
        return True

    return point_id not in frame_points_data['id']


def next_click_point_id(
    points_data_pos: dict[Any, Any] | None,
    frame_i: int,
    current_id: int,
    *,
    size_z: int = 1,
) -> int:
    """Return the next point id for a click-entry layer."""
    if points_data_pos is None:
        return 1

    frame_points_data = points_data_pos.get(frame_i)
    if frame_points_data is None:
        return 1

    if size_z > 1:
        new_id = 1
        for z_data in frame_points_data.values():
            max_id = max(z_data.get('id', []), default=0) + 1
            if max_id > new_id:
                new_id = max_id
    else:
        new_id = max(frame_points_data.get('id', []), default=0) + 1

    if current_id >= new_id:
        return current_id
    return new_id


def add_click_point(
    points_data_pos: dict[Any, Any],
    frame_i: int,
    x: float,
    y: float,
    point_id: int,
    *,
    size_z: int = 1,
    z_slice: int | None = None,
) -> dict[Any, Any]:
    """Add one click-entry point to nested point-layer data."""
    frame_points_data = points_data_pos.get(frame_i)
    if frame_points_data is None:
        if size_z > 1:
            if z_slice is None:
                raise ValueError('z_slice is required for 3D point data')
            points_data_pos[frame_i] = {
                z_slice: {'x': [x], 'y': [y], 'id': [point_id]},
            }
        else:
            points_data_pos[frame_i] = {
                'x': [x], 'y': [y], 'id': [point_id],
            }
        return points_data_pos

    if size_z > 1:
        if z_slice is None:
            raise ValueError('z_slice is required for 3D point data')
        z_data = frame_points_data.get(z_slice)
        if z_data is None:
            frame_points_data[z_slice] = {
                'x': [x], 'y': [y], 'id': [point_id],
            }
        else:
            z_data['x'].append(x)
            z_data['y'].append(y)
            z_data['id'].append(point_id)
    else:
        frame_points_data['x'].append(x)
        frame_points_data['y'].append(y)
        frame_points_data['id'].append(point_id)

    points_data_pos[frame_i] = frame_points_data
    return points_data_pos


def remove_click_points(
    frame_points_data: dict[Any, Any],
    points: Iterable[tuple[float, float, int]],
    *,
    z_slice: int | None = None,
    z_radius: int = 0,
) -> list[int]:
    """Remove clicked points from one frame's nested point-layer data."""
    removed_ids = []
    for x, y, point_id in points:
        if z_slice is not None:
            z_range = range(z_slice - z_radius, z_slice + z_radius + 1)
            data_slices = [
                frame_points_data[z]
                for z in z_range
                if z in frame_points_data
            ]
        else:
            data_slices = [frame_points_data]

        for points_slice in data_slices:
            if point_id not in points_slice['id']:
                continue
            points_slice['x'].remove(x)
            points_slice['y'].remove(y)
            points_slice['id'].remove(point_id)
            removed_ids.append(point_id)

    return removed_ids


def flatten_frame_points_data(
    frame_points_data: dict[Any, Any],
    *,
    z_slice: int | None = None,
    z_radius: int = 0,
) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    """Flatten one frame's point-layer data for display or scripting."""
    if 'x' in frame_points_data:
        return (
            list(frame_points_data['x']),
            list(frame_points_data['y']),
            list(frame_points_data['id']),
            list(frame_points_data.get('data', [])),
        )

    xx, yy, ids, data = [], [], [], []
    if z_slice is None:
        z_items = frame_points_data.items()
    else:
        z_range = range(z_slice - z_radius, z_slice + z_radius + 1)
        z_items = (
            (z, frame_points_data[z])
            for z in z_range
            if z in frame_points_data
        )

    for _z, z_data in z_items:
        xx.extend(z_data['x'])
        yy.extend(z_data['y'])
        ids.extend(z_data['id'])
        data.extend(z_data.get('data', []))
    return xx, yy, ids, data


def _label_at(labels, y: float, x: float, z: float | None, is_segm_3d: bool):
    if is_segm_3d and z is not None:
        return labels[int(z), int(y), int(x)]
    return labels[int(y), int(x)]


def _linear_fit_3d(xx, yy, zz):
    points = np.column_stack((xx, yy, zz))
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid)
    return centroid, vh[0]


def interpolate_points_zslices(
    df: pd.DataFrame,
    labels,
    *,
    is_segm_3d: bool,
) -> pd.DataFrame:
    """Interpolate missing z-slice points for each frame/id point track."""
    if not is_segm_3d or 'z' not in df.columns:
        return df

    df_new_rows = []
    for (_frame_i, _point_id), df_id in df.groupby(['frame_i', 'id']):
        xx = df_id['x'].values
        yy = df_id['y'].values
        zz = df_id['z'].values
        point, direction = _linear_fit_3d(xx, yy, zz)

        new_row_df = df_id.iloc[[0]].copy()
        z0, z1 = int(np.min(zz)), int(np.max(zz))
        for z in range(z0, z1 + 1):
            if z in zz:
                continue

            t_int = (z - point[2]) / direction[2]
            x_new, y_new, z_new = point + t_int * direction
            new_row_df['z'] = round(z_new)
            new_row_df['y'] = round(y_new)
            new_row_df['x'] = round(x_new)
            new_row_df['Cell_ID'] = labels[
                int(round(z_new)),
                int(round(y_new)),
                int(round(x_new)),
            ]
            df_new_rows.append(new_row_df.copy())

    if not df_new_rows:
        return df

    df_new = pd.concat(df_new_rows, ignore_index=True)
    df = pd.concat([df, df_new], ignore_index=True)
    return df.sort_values(by=['frame_i', 'id', 'z']).reset_index(drop=True)


def points_data_to_table(
    points_data: dict[Any, Any],
    labels,
    *,
    is_segm_3d: bool = False,
    size_z: int = 1,
    interpolate_z: bool = False,
) -> pd.DataFrame:
    """Convert nested point-layer data to a table."""
    df = pd.DataFrame(columns=['frame_i', 'Cell_ID', 'z', 'y', 'x', 'id'])
    frames_vals = []
    cell_ids = []
    zz = []
    yy = []
    xx = []
    ids = []

    for frame_i, frame_points_data in points_data.items():
        if size_z > 1:
            for z, z_slice_points_data in frame_points_data.items():
                for y, x, point_id in zip(
                    z_slice_points_data['y'],
                    z_slice_points_data['x'],
                    z_slice_points_data['id'],
                ):
                    frames_vals.append(frame_i)
                    cell_ids.append(_label_at(labels, y, x, z, is_segm_3d))
                    zz.append(z)
                    yy.append(y)
                    xx.append(x)
                    ids.append(point_id)
        else:
            for y, x, point_id in zip(
                frame_points_data['y'],
                frame_points_data['x'],
                frame_points_data['id'],
            ):
                frames_vals.append(frame_i)
                cell_ids.append(_label_at(labels, y, x, None, is_segm_3d))
                yy.append(y)
                xx.append(x)
                ids.append(point_id)

    df['frame_i'] = frames_vals
    df['Cell_ID'] = cell_ids
    df['y'] = yy
    df['x'] = xx
    df['id'] = ids
    if zz:
        df['z'] = zz

    if interpolate_z:
        df = interpolate_points_zslices(df, labels, is_segm_3d=is_segm_3d)
    return df
