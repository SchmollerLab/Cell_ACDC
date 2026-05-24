"""View-model commands for manual edit-ID operations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cellacdc.domain.edit_id import (
    ManualEditTrackingResult,
    add_yx_centroids_to_df,
    apply_manual_edit_tracking,
    edit_id_info_from_df,
    manual_edit_conflicts,
    project_centroid,
)


class EditIdViewModel:
    """Application-facing commands for manual ID edit metadata."""

    def project_centroid(
        self,
        centroid,
        *,
        is_3d: bool = False,
        depth_axis: str = 'z',
    ) -> tuple[float, float]:
        return project_centroid(
            centroid,
            is_3d=is_3d,
            depth_axis=depth_axis,
        )

    def add_yx_centroids_to_df(
        self,
        df: pd.DataFrame,
        regionprops,
        *,
        is_3d: bool = False,
        depth_axis: str = 'z',
    ) -> pd.DataFrame:
        return add_yx_centroids_to_df(
            df,
            regionprops,
            is_3d=is_3d,
            depth_axis=depth_axis,
        )

    def edit_id_info_from_df(
        self,
        df: pd.DataFrame,
        regionprops=None,
        *,
        is_3d: bool = False,
        depth_axis: str = 'z',
    ) -> list[tuple[int, int, int]]:
        return edit_id_info_from_df(
            df,
            regionprops,
            is_3d=is_3d,
            depth_axis=depth_axis,
        )

    def manual_edit_conflicts(
        self,
        labels: np.ndarray,
        edit_id_info,
    ) -> dict[int, int]:
        return manual_edit_conflicts(labels, edit_id_info)

    def apply_manual_edit_tracking(
        self,
        tracked_labels: np.ndarray,
        edit_id_info,
        all_ids,
    ) -> ManualEditTrackingResult:
        return apply_manual_edit_tracking(
            tracked_labels,
            edit_id_info,
            all_ids,
        )
