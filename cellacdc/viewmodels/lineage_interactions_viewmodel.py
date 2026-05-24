"""View-model contracts for lineage-tree interaction workflows."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import pandas as pd

from cellacdc.models.lineage_interactions_model import (
    LineageInteractionsModel,
)


@dataclass(frozen=True)
class LineageInteractionsViewModel:
    """Application-facing lineage-tree interaction decisions."""

    model: LineageInteractionsModel = field(
        default_factory=LineageInteractionsModel
    )

    def is_lineage_mode(self, mode: str) -> bool:
        return self.model.is_lineage_mode(mode)

    def should_initialize(
        self,
        *,
        force: bool,
        mode: str,
        lineage_tree_exists: bool,
    ) -> bool:
        return self.model.should_initialize(
            force=force,
            mode=mode,
            lineage_tree_exists=lineage_tree_exists,
        )

    def default_mode_after_failed_init(self) -> str:
        return self.model.default_mode_after_failed_init()

    def last_annotated_frame_index(self, frame_records: Iterable[dict]) -> int:
        return self.model.last_annotated_frame_index(frame_records)

    def missing_frame_indices(
        self,
        current_frame_i: int,
        present_frames: Iterable[int] | None,
    ) -> list[int]:
        return self.model.missing_frame_indices(
            current_frame_i,
            present_frames,
        )

    def should_process_auto_frame(
        self,
        *,
        mode: str,
        frame_i: int,
        processed_frames: Iterable[int],
    ) -> bool:
        return self.model.should_process_auto_frame(
            mode=mode,
            frame_i=frame_i,
            processed_frames=processed_frames,
        )

    def should_backup_original(
        self,
        original_frame_i: int | None,
        current_frame_i: int,
    ) -> bool:
        return self.model.should_backup_original(
            original_frame_i,
            current_frame_i,
        )

    def next_candidate_index(
        self,
        click_index: int,
        candidates_count: int,
    ) -> int:
        return self.model.next_candidate_index(click_index, candidates_count)

    def should_skip_original_mother(
        self,
        current_parent_id,
        candidate_parent_id,
        *,
        original_mother_skipped: bool,
    ) -> bool:
        return self.model.should_skip_original_mother(
            current_parent_id,
            candidate_parent_id,
            original_mother_skipped=original_mother_skipped,
        )

    def parent_id_differences(
        self,
        original_df: pd.DataFrame,
        new_df: pd.DataFrame,
        reset_index_cell_id: Callable[[pd.DataFrame], pd.DataFrame],
    ) -> pd.DataFrame | None:
        return self.model.parent_id_differences(
            original_df,
            new_df,
            reset_index_cell_id,
        )
