"""Scriptable model rules for lineage-tree interaction workflows."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np
import pandas as pd


class LineageInteractionsModel:
    """Headless decisions for lineage-tree interaction workflows."""

    lineage_mode = 'Normal division: Lineage tree'
    viewer_mode = 'Viewer'

    def is_lineage_mode(self, mode: str) -> bool:
        return mode == self.lineage_mode

    def should_initialize(
        self,
        *,
        force: bool,
        mode: str,
        lineage_tree_exists: bool,
    ) -> bool:
        if not force and lineage_tree_exists:
            return False
        return force or self.is_lineage_mode(mode)

    def default_mode_after_failed_init(self) -> str:
        return self.viewer_mode

    def last_annotated_frame_index(
        self,
        frame_records: Iterable[dict],
        *,
        acdc_key: str = 'acdc_df',
        generation_column: str = 'generation_num_tree',
    ) -> int:
        last_frame_i = 0
        for frame_i, record in enumerate(frame_records):
            acdc_df = record[acdc_key]
            if (
                acdc_df is None
                or generation_column not in acdc_df.columns
                or acdc_df[generation_column].isin([np.nan, 0]).all()
            ):
                break
            last_frame_i = frame_i
        return last_frame_i

    def missing_frame_indices(
        self,
        current_frame_i: int,
        present_frames: Iterable[int] | None,
    ) -> list[int]:
        present = set(present_frames or [])
        missing = [
            frame_i for frame_i in range(current_frame_i + 1)
            if frame_i not in present
        ]
        missing.sort()
        return missing

    def should_process_auto_frame(
        self,
        *,
        mode: str,
        frame_i: int,
        processed_frames: Iterable[int],
    ) -> bool:
        if not self.is_lineage_mode(mode):
            return False
        return frame_i not in processed_frames

    def should_backup_original(
        self,
        original_frame_i: int | None,
        current_frame_i: int,
    ) -> bool:
        return original_frame_i is None or original_frame_i != current_frame_i

    def next_candidate_index(
        self,
        click_index: int,
        candidates_count: int,
    ) -> int:
        if candidates_count <= 0:
            return 0
        return abs(click_index % candidates_count)

    def should_skip_original_mother(
        self,
        current_parent_id,
        candidate_parent_id,
        *,
        original_mother_skipped: bool,
    ) -> bool:
        return (
            current_parent_id == candidate_parent_id
            and not original_mother_skipped
        )

    def parent_id_differences(
        self,
        original_df: pd.DataFrame,
        new_df: pd.DataFrame,
        reset_index_cell_id: Callable[[pd.DataFrame], pd.DataFrame],
        *,
        compare_columns: Sequence[str] = ('parent_ID_tree',),
    ) -> pd.DataFrame | None:
        if original_df.equals(new_df):
            return None

        new_df = new_df[original_df.columns]
        new_df = reset_index_cell_id(new_df)
        new_df = new_df[list(compare_columns)]
        new_df = new_df.sort_index()

        original_df = reset_index_cell_id(original_df)
        original_df = original_df[list(compare_columns)]
        original_df = original_df.sort_index()

        differences = original_df.compare(new_df)
        if differences.empty:
            return None

        differences = reset_index_cell_id(differences)
        differences = differences[compare_columns[0]]
        return differences.reset_index()
