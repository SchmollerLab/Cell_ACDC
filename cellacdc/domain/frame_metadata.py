"""Per-frame ACDC metadata table transforms."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .edit_id import project_centroid


@dataclass(frozen=True)
class AcdcFrameMetadataResult:
    """Result of building one frame's ACDC metadata dataframe."""

    dataframe: pd.DataFrame
    max_id: int


def concat_visited_acdc_frames(
    frame_records,
    *,
    labels_key: str = 'labels',
    acdc_key: str = 'acdc_df',
) -> pd.DataFrame | None:
    """Concatenate ACDC frame tables until labels or metadata are missing."""
    acdc_dfs = []
    keys = []
    for frame_i, frame_record in enumerate(frame_records):
        if frame_record[labels_key] is None:
            break

        acdc_df = frame_record[acdc_key]
        if acdc_df is None:
            break

        acdc_dfs.append(acdc_df)
        keys.append(frame_i)

    if not acdc_dfs:
        return None

    return pd.concat(acdc_dfs, keys=keys, names=['frame_i'])


def build_acdc_frame_metadata(
        regionprops,
        *,
        edit_id_info=(),
        existing_df: pd.DataFrame | None = None,
        is_3d: bool = False,
        depth_axis: str = 'z',
) -> AcdcFrameMetadataResult:
    """Build or update dynamic per-object metadata for one frame."""
    ids = []
    is_cell_dead = []
    is_cell_excluded = []
    x_centroid = []
    y_centroid = []
    z_centroid = []
    was_manually_edited = []
    edited_new_ids = {int(vals[2]) for vals in edit_id_info}

    for obj in regionprops:
        label_id = int(obj.label)
        ids.append(label_id)
        is_cell_dead.append(getattr(obj, 'dead', False))
        is_cell_excluded.append(getattr(obj, 'excluded', False))
        y, x = project_centroid(
            obj.centroid,
            is_3d=is_3d,
            depth_axis=depth_axis,
        )
        y_centroid.append(int(y))
        x_centroid.append(int(x))
        if is_3d:
            z_centroid.append(int(obj.centroid[0]))
        was_manually_edited.append(int(label_id in edited_new_ids))

    if existing_df is None:
        df = pd.DataFrame(
            {
                'Cell_ID': ids,
                'is_cell_dead': is_cell_dead,
                'is_cell_excluded': is_cell_excluded,
                'x_centroid': x_centroid,
                'y_centroid': y_centroid,
                'was_manually_edited': was_manually_edited,
            }
        ).set_index('Cell_ID')
    else:
        df = existing_df.drop(columns=['time_seconds'], errors='ignore')
        df = df.reindex(ids, fill_value=0)
        df['is_cell_dead'] = is_cell_dead
        df['is_cell_excluded'] = is_cell_excluded
        df['x_centroid'] = x_centroid
        df['y_centroid'] = y_centroid
        df['was_manually_edited'] = was_manually_edited

    if is_3d:
        df['z_centroid'] = z_centroid

    return AcdcFrameMetadataResult(dataframe=df, max_id=max(ids, default=0))
