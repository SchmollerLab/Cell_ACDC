"""Edit-ID metadata transforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .labels import apply_label_id_mapping


@dataclass(frozen=True)
class ManualEditTrackingResult:
    """Result of replaying manual edit-ID corrections after tracking."""

    labels: np.ndarray
    remaining_edit_info: list[tuple[int, int, int]]
    removed_edit_info: list[tuple[int, int, int]]


def project_centroid(
        centroid,
        *,
        is_3d: bool = False,
        depth_axis: str = 'z',
) -> tuple[float, float]:
    """Project a regionprops centroid to the visible y/x plane."""
    if not is_3d:
        y, x = centroid
        return y, x

    zc, yc, xc = centroid
    if depth_axis == 'z':
        return yc, xc
    if depth_axis == 'y':
        return zc, xc
    return zc, yc


def add_yx_centroids_to_df(
        df: pd.DataFrame,
        regionprops,
        *,
        is_3d: bool = False,
        depth_axis: str = 'z',
) -> pd.DataFrame:
    """Add visible-plane centroid columns indexed by object label."""
    for obj in regionprops:
        y_centroid, x_centroid = project_centroid(
            obj.centroid,
            is_3d=is_3d,
            depth_axis=depth_axis,
        )
        df.at[obj.label, 'y_centroid'] = int(y_centroid)
        df.at[obj.label, 'x_centroid'] = int(x_centroid)
    return df


def edit_id_info_from_df(
        df: pd.DataFrame,
        regionprops=None,
        *,
        is_3d: bool = False,
        depth_axis: str = 'z',
) -> list[tuple[int, int, int]]:
    """Build replay tuples for manually edited IDs from an ACDC dataframe."""
    if 'was_manually_edited' not in df.columns:
        return []

    has_centroids = {'y_centroid', 'x_centroid'}.issubset(df.columns)
    if not has_centroids:
        if regionprops is None:
            raise ValueError(
                'regionprops are required when centroid columns are missing'
            )
        df = add_yx_centroids_to_df(
            df,
            regionprops,
            is_3d=is_3d,
            depth_axis=depth_axis,
        )

    manually_edited_df = df[df['was_manually_edited'] > 0]
    return [
        (row.y_centroid, row.x_centroid, row.Index)
        for row in manually_edited_df.itertuples()
    ]


def manual_edit_conflicts(
        labels: np.ndarray,
        edit_id_info,
) -> dict[int, int]:
    """Return tracked IDs that differ from requested manual edit IDs."""
    return {
        int(labels[y, x]): int(new_id)
        for y, x, new_id in edit_id_info
        if int(labels[y, x]) != int(new_id)
    }


def apply_manual_edit_tracking(
        tracked_labels: np.ndarray,
        edit_id_info,
        all_ids,
) -> ManualEditTrackingResult:
    """Replay manual ID edits onto a newly tracked label image in place."""
    all_ids_set = {int(label_id) for label_id in all_ids}
    max_id = max(all_ids_set, default=1)
    remaining_info = []
    removed_info = []

    for info in edit_id_info:
        y, x, new_id = info
        new_id = int(new_id)
        old_id = int(tracked_labels[y, x])
        normalized_info = (int(y), int(x), new_id)
        if old_id == 0 or old_id == new_id:
            removed_info.append(normalized_info)
            continue

        result = apply_label_id_mapping(
            tracked_labels,
            [(old_id, new_id)],
            existing_ids=all_ids_set,
            start_max_id=max_id,
        )
        max_id = result.max_id
        remaining_info.append(normalized_info)

    return ManualEditTrackingResult(
        labels=tracked_labels,
        remaining_edit_info=remaining_info,
        removed_edit_info=removed_info,
    )
