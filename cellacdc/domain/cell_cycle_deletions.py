"""Deleted-ID cell-cycle annotation table operations."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .cell_cycle import (
    base_cell_cycle_annotation_status,
    build_base_cell_cycle_annotations,
)


@dataclass(frozen=True)
class CcaDeletedIdsResult:
    """CCA table after deleting IDs plus their relative-ID references."""

    cca_df: pd.DataFrame
    relative_ids: pd.Series


@dataclass(frozen=True)
class CcaRelativeRestoreResult:
    """CCA table after restoring relative statuses."""

    cca_df: pd.DataFrame
    any_restored: bool


@dataclass(frozen=True)
class CcaDeletedRelativeStatusesResult:
    """Current CCA table plus statuses restored for deleted relatives."""

    cca_df: pd.DataFrame
    relative_statuses: dict


@dataclass(frozen=True)
class CcaDeletedIdsPropagationResult:
    """Deleted-ID updates across a current frame plus visited neighbors."""

    current_cca_df: pd.DataFrame
    updated_cca_dfs_by_frame: dict[int, pd.DataFrame]
    undo_frame_indices: list[int]
    relative_statuses: dict


def _frame_value(frame_values, frame_i: int):
    if frame_values is None:
        return None
    if hasattr(frame_values, 'get'):
        return frame_values.get(frame_i)
    try:
        return frame_values[frame_i]
    except (IndexError, KeyError, TypeError):
        return None


def _frame_count(frame_values, size_t: int | None = None) -> int:
    if size_t is not None:
        return int(size_t)
    if hasattr(frame_values, 'keys'):
        keys = list(frame_values.keys())
        return max(keys) + 1 if keys else 0
    return len(frame_values)


def delete_cca_ids(
    cca_df: pd.DataFrame,
    deleted_ids,
) -> CcaDeletedIdsResult:
    """Return ``cca_df`` without ``deleted_ids`` and their relative IDs."""
    relative_ids = cca_df.reindex(deleted_ids, fill_value=-1)['relative_ID']
    updated_cca_df = cca_df.drop(deleted_ids, errors='ignore')
    return CcaDeletedIdsResult(
        cca_df=updated_cca_df,
        relative_ids=relative_ids,
    )


def apply_cca_deleted_ids_to_frame(
    cca_df: pd.DataFrame,
    deleted_ids,
    *,
    drop_deleted: bool = True,
    existing_ids=None,
    base_cca_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Drop deleted IDs or restore their base rows when labels still exist."""
    if drop_deleted:
        return cca_df.drop(deleted_ids, errors='ignore')

    existing_ids = set(deleted_ids if existing_ids is None else existing_ids)
    restore_ids = [label_id for label_id in deleted_ids if label_id in existing_ids]
    if not restore_ids:
        return cca_df
    if base_cca_df is None:
        raise ValueError('base_cca_df is required when restoring deleted IDs')

    updated_cca_df = cca_df.copy()
    for label_id in restore_ids:
        if label_id not in base_cca_df.index:
            continue
        updated_cca_df.loc[label_id] = base_cca_df.loc[label_id]
    return updated_cca_df


def apply_deleted_cell_cycle_ids_to_frame(
    cca_df: pd.DataFrame,
    deleted_ids,
    relative_statuses: dict,
    *,
    relative_ids=None,
    drop_deleted: bool = True,
    existing_ids=None,
    base_values: dict | None = None,
) -> CcaRelativeRestoreResult:
    """Apply deleted-ID updates and restore relative statuses for one frame."""
    base_cca_df = None
    if not drop_deleted:
        restore_ids = [label_id for label_id in deleted_ids if (
            existing_ids is None or label_id in existing_ids
        )]
        if restore_ids:
            base_cca_df = build_base_cell_cycle_annotations(
                restore_ids,
                base_values=base_values,
            )

    updated_cca_df = apply_cca_deleted_ids_to_frame(
        cca_df,
        deleted_ids,
        drop_deleted=drop_deleted,
        existing_ids=existing_ids,
        base_cca_df=base_cca_df,
    )
    return restore_cca_relative_statuses(
        updated_cca_df,
        relative_statuses,
        relative_ids=relative_ids,
    )


def restore_cca_relative_statuses(
    cca_df: pd.DataFrame,
    relative_statuses: dict,
    relative_ids=None,
) -> CcaRelativeRestoreResult:
    """Restore stored CCA statuses for relative IDs present in ``cca_df``."""
    if relative_ids is None:
        relative_ids = relative_statuses.keys()

    updated_cca_df = cca_df.copy()
    any_restored = False
    required_cols = {'cell_cycle_stage', 'relationship'}
    if not required_cols.issubset(updated_cca_df.columns):
        return CcaRelativeRestoreResult(updated_cca_df, any_restored)

    for relative_id in relative_ids:
        if relative_id not in relative_statuses:
            continue
        if relative_id not in updated_cca_df.index:
            continue
        updated_cca_df.loc[relative_id] = relative_statuses[relative_id]
        any_restored = True

    return CcaRelativeRestoreResult(updated_cca_df, any_restored)


def deleted_relative_cca_status(
    cca_df: pd.DataFrame,
    relative_id: int,
    base_status: pd.Series,
    past_cca_dfs=(),
) -> pd.Series | None:
    """Return the CCA status to restore for a deleted ID's relative."""
    try:
        cell_cycle_stage = cca_df.at[relative_id, 'cell_cycle_stage']
        relationship = cca_df.at[relative_id, 'relationship']
    except Exception:
        return None

    cca_status = base_status.copy()
    if relationship == 'mother' and cell_cycle_stage == 'S':
        for past_cca_df in past_cca_dfs:
            if past_cca_df is None or relative_id not in past_cca_df.index:
                continue
            cell_cycle_stage_past = past_cca_df.at[
                relative_id, 'cell_cycle_stage'
            ]
            if cell_cycle_stage_past == 'G1':
                cca_status = past_cca_df.loc[relative_id].copy()
                break

    return cca_status


def restore_deleted_relative_cell_cycle_statuses(
    cca_df: pd.DataFrame,
    relative_ids,
    *,
    past_cca_dfs=(),
    base_values: dict | None = None,
) -> CcaDeletedRelativeStatusesResult:
    """Restore statuses for relatives of deleted IDs on the current frame."""
    past_cca_dfs = list(past_cca_dfs)
    updated_cca_df = cca_df.copy()
    relative_statuses = {}
    for relative_id in relative_ids:
        base_status = base_cell_cycle_annotation_status(base_values)
        base_status.name = relative_id
        cca_status = deleted_relative_cca_status(
            updated_cca_df,
            relative_id,
            base_status,
            past_cca_dfs=past_cca_dfs,
        )
        if cca_status is None:
            continue

        updated_cca_df.loc[relative_id] = cca_status
        relative_statuses[relative_id] = cca_status

    return CcaDeletedRelativeStatusesResult(
        cca_df=updated_cca_df,
        relative_statuses=relative_statuses,
    )


def propagate_deleted_cell_cycle_ids(
    cca_dfs_by_frame,
    current_frame_i: int,
    deleted_ids,
    relative_ids,
    *,
    current_cca_df: pd.DataFrame | None = None,
    future_cca_frames=None,
    past_cca_frames=None,
    drop_in_past: bool = True,
    drop_in_future: bool = True,
    existing_ids_by_frame=None,
    base_values: dict | None = None,
    size_t: int | None = None,
) -> CcaDeletedIdsPropagationResult:
    """Return CCA frame updates after IDs were deleted on one frame.

    ``cca_dfs_by_frame`` can be a list-like object or mapping keyed by frame
    index. ``None`` frame values represent unvisited frames and stop traversal.
    """
    current_frame_i = int(current_frame_i)
    if current_cca_df is None:
        current_cca_df = _frame_value(cca_dfs_by_frame, current_frame_i)
    if current_cca_df is None:
        raise ValueError('current frame has no CCA table')

    if past_cca_frames is None:
        past_cca_frames = [
            (past_frame_i, _frame_value(cca_dfs_by_frame, past_frame_i))
            for past_frame_i in range(current_frame_i - 1, -1, -1)
        ]
    else:
        past_cca_frames = list(past_cca_frames)

    current_restore_result = restore_deleted_relative_cell_cycle_statuses(
        current_cca_df,
        relative_ids,
        past_cca_dfs=(cca_df_i for _, cca_df_i in past_cca_frames),
        base_values=base_values,
    )
    current_cca_df = current_restore_result.cca_df
    relative_statuses = current_restore_result.relative_statuses

    updated_cca_dfs_by_frame = {}
    if relative_statuses:
        updated_cca_dfs_by_frame[current_frame_i] = current_cca_df

    undo_frame_indices = []
    if future_cca_frames is None:
        stop_frame_i = _frame_count(cca_dfs_by_frame, size_t=size_t)
        future_cca_frames = (
            (future_frame_i, _frame_value(cca_dfs_by_frame, future_frame_i))
            for future_frame_i in range(current_frame_i + 1, stop_frame_i)
        )

    for future_frame_i, cca_df_i in future_cca_frames:
        if cca_df_i is None:
            break

        undo_frame_indices.append(future_frame_i)
        existing_ids = None
        if not drop_in_future:
            existing_ids = _frame_value(existing_ids_by_frame, future_frame_i)

        restore_result = apply_deleted_cell_cycle_ids_to_frame(
            cca_df_i,
            deleted_ids,
            relative_statuses,
            relative_ids=relative_ids,
            drop_deleted=drop_in_future,
            existing_ids=existing_ids,
            base_values=base_values,
        )
        if not restore_result.any_restored:
            break

        updated_cca_dfs_by_frame[future_frame_i] = restore_result.cca_df

    for past_frame_i, cca_df_i in past_cca_frames:
        if cca_df_i is None:
            break

        undo_frame_indices.append(past_frame_i)
        existing_ids = None
        if not drop_in_past:
            existing_ids = _frame_value(existing_ids_by_frame, past_frame_i)

        restore_result = apply_deleted_cell_cycle_ids_to_frame(
            cca_df_i,
            deleted_ids,
            relative_statuses,
            relative_ids=relative_ids,
            drop_deleted=drop_in_past,
            existing_ids=existing_ids,
            base_values=base_values,
        )
        if not restore_result.any_restored:
            break

        updated_cca_dfs_by_frame[past_frame_i] = restore_result.cca_df

    return CcaDeletedIdsPropagationResult(
        current_cca_df=current_cca_df,
        updated_cca_dfs_by_frame=updated_cca_dfs_by_frame,
        undo_frame_indices=undo_frame_indices,
        relative_statuses=relative_statuses,
    )
