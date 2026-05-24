"""View-model commands for ACDC frame metadata."""

from __future__ import annotations

import pandas as pd

from cellacdc.domain.frame_metadata import (
    AcdcFrameMetadataResult,
    build_acdc_frame_metadata,
    concat_visited_acdc_frames,
)
from cellacdc.myutils import get_empty_stored_data_dict


class FrameMetadataMixin:
    """Application-facing commands for per-frame ACDC metadata tables."""

    def build_acdc_frame_metadata(
        self,
        regionprops,
        *,
        edit_id_info=(),
        existing_df: pd.DataFrame | None = None,
        is_3d: bool = False,
        depth_axis: str = "z",
    ) -> AcdcFrameMetadataResult:
        return build_acdc_frame_metadata(
            regionprops,
            edit_id_info=edit_id_info,
            existing_df=existing_df,
            is_3d=is_3d,
            depth_axis=depth_axis,
        )

    def concat_visited_acdc_frames(
        self,
        frame_records,
        *,
        labels_key: str = "labels",
        acdc_key: str = "acdc_df",
    ) -> pd.DataFrame | None:
        return concat_visited_acdc_frames(
            frame_records,
            labels_key=labels_key,
            acdc_key=acdc_key,
        )

    def empty_frame_record(self):
        return get_empty_stored_data_dict()

    def empty_frame_records(self, count: int):
        return [self.empty_frame_record() for _ in range(count)]
