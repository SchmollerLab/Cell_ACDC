"""View-model commands for CCA table edits."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cellacdc.domain.cell_cycle import (
    add_base_cell_cycle_annotation,
    build_base_cell_cycle_annotations,
    concat_cell_cycle_annotations,
    CcaSnapshotIdEditResult,
    apply_manual_cca_changes,
    apply_snapshot_cca_id_edits,
    has_cell_cycle_annotations,
    last_annotated_cell_cycle_frame_index,
    merge_missing_cca_ids,
    relabel_cca_ids,
    remove_cell_cycle_annotations,
    remove_future_cell_cycle_annotations,
    reset_cca_future_flags,
    split_concat_cell_cycle_annotations,
)
from cellacdc.domain.cell_cycle_deletions import (
    CcaDeletedIdsResult,
    delete_cca_ids,
)
from cellacdc.domain.cell_cycle_frames import (
    normalize_loaded_cell_cycle_frame_annotations,
    prepare_cell_cycle_checker_annotations,
    resolve_cell_cycle_annotations,
    store_cell_cycle_frame_annotations,
)


@dataclass(frozen=True)
class CcaFrameEditResult:
    """Result of a current-frame CCA edit command."""

    cca_df: pd.DataFrame


class CcaEditViewModel:
    """Application-facing commands for editing CCA tables.

    The Qt view owns undo, persistence, and dialogs. This view model owns the
    command shape that binds view events to scriptable domain operations.
    """

    def add_missing_ids(
        self,
        cca_df: pd.DataFrame | None,
        base_cca_df: pd.DataFrame,
    ) -> CcaFrameEditResult:
        return CcaFrameEditResult(
            cca_df=merge_missing_cca_ids(cca_df, base_cca_df),
        )

    def relabel_ids(
        self,
        cca_df: pd.DataFrame,
        old_ids,
        new_ids,
    ) -> CcaFrameEditResult:
        return CcaFrameEditResult(
            cca_df=relabel_cca_ids(cca_df, old_ids, new_ids),
        )

    def delete_ids(
        self,
        cca_df: pd.DataFrame,
        deleted_ids,
    ) -> CcaDeletedIdsResult:
        return delete_cca_ids(cca_df, deleted_ids)

    def apply_snapshot_id_edits(
        self,
        cca_df: pd.DataFrame,
        edit_text: str,
        current_ids,
        base_cca_df: pd.DataFrame,
        *,
        base_values: dict | None = None,
    ) -> CcaSnapshotIdEditResult:
        return apply_snapshot_cca_id_edits(
            cca_df,
            edit_text,
            current_ids,
            base_cca_df,
            base_values=base_values,
        )

    def apply_manual_changes(
        self,
        cca_df: pd.DataFrame,
        changes,
    ) -> pd.DataFrame:
        return apply_manual_cca_changes(cca_df, changes)

    def normalize_loaded_frame_annotations(
        self,
        acdc_df: pd.DataFrame | None,
        cca_colnames,
        int_colnames=(),
    ) -> pd.DataFrame | None:
        return normalize_loaded_cell_cycle_frame_annotations(
            acdc_df,
            cca_colnames,
            int_colnames,
        )

    def add_base_annotation(
        self,
        cca_df: pd.DataFrame,
        cell_ids,
        *,
        base_values: dict | None = None,
    ) -> pd.DataFrame:
        return add_base_cell_cycle_annotation(
            cca_df,
            cell_ids,
            base_values=base_values,
        )

    def build_base_annotations(
        self,
        cell_ids,
        *,
        with_tree_cols: bool = False,
        base_values: dict | None = None,
        tree_values: dict | None = None,
    ) -> pd.DataFrame:
        return build_base_cell_cycle_annotations(
            cell_ids,
            with_tree_cols=with_tree_cols,
            base_values=base_values,
            tree_values=tree_values,
        )

    def last_annotated_frame_index(self, acdc_dfs) -> int:
        return last_annotated_cell_cycle_frame_index(acdc_dfs)

    def concat_annotations(
        self,
        frame_records,
        cca_colnames,
        *,
        acdc_key: str = 'acdc_df',
        size_t: int | None = None,
    ) -> pd.DataFrame | None:
        return concat_cell_cycle_annotations(
            frame_records,
            cca_colnames,
            acdc_key=acdc_key,
            size_t=size_t,
        )

    def split_concat_annotations(
        self,
        global_cca_df: pd.DataFrame | None,
        *,
        size_t: int | None = None,
        frame_level: str = 'frame_i',
    ) -> list[tuple[int, pd.DataFrame]]:
        return split_concat_cell_cycle_annotations(
            global_cca_df,
            size_t=size_t,
            frame_level=frame_level,
        )

    def remove_future_annotations(
        self,
        frame_records,
        cca_colnames,
        from_frame_i: int,
        *,
        size_t: int | None = None,
        concatenated_acdc_df: pd.DataFrame | None = None,
        acdc_key: str = 'acdc_df',
    ):
        return remove_future_cell_cycle_annotations(
            frame_records,
            cca_colnames,
            from_frame_i,
            size_t=size_t,
            concatenated_acdc_df=concatenated_acdc_df,
            acdc_key=acdc_key,
        )

    def remove_annotations(self, acdc_df: pd.DataFrame | None, cca_colnames):
        return remove_cell_cycle_annotations(acdc_df, cca_colnames)

    def reset_future_flags(self, cca_df: pd.DataFrame) -> pd.DataFrame:
        return reset_cca_future_flags(cca_df)

    def resolve_annotations(
        self,
        acdc_df: pd.DataFrame | None,
        cca_colnames,
        *,
        is_snapshot: bool = False,
        snapshot_cell_ids=(),
        dropna: bool = True,
        base_values: dict | None = None,
        tree_values: dict | None = None,
        with_tree_cols: bool = False,
    ):
        return resolve_cell_cycle_annotations(
            acdc_df,
            cca_colnames,
            is_snapshot=is_snapshot,
            snapshot_cell_ids=snapshot_cell_ids,
            dropna=dropna,
            base_values=base_values,
            tree_values=tree_values,
            with_tree_cols=with_tree_cols,
        )

    def prepare_checker_annotations(
        self,
        cca_df: pd.DataFrame | None,
        *,
        checker_running: bool = True,
    ) -> pd.DataFrame | None:
        return prepare_cell_cycle_checker_annotations(
            cca_df,
            checker_running=checker_running,
        )

    def store_frame_annotations(
        self,
        acdc_df: pd.DataFrame | None,
        cca_df: pd.DataFrame | None,
        cca_colnames,
        *,
        store_checker_copy: bool = False,
        store_cca_df_copy: bool = False,
    ):
        return store_cell_cycle_frame_annotations(
            acdc_df,
            cca_df,
            cca_colnames,
            store_checker_copy=store_checker_copy,
            store_cca_df_copy=store_cca_df_copy,
        )

    def has_annotations(self, acdc_df: pd.DataFrame | None) -> bool:
        return has_cell_cycle_annotations(acdc_df)
