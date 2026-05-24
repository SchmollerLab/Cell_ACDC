"""Cell-cycle history-known annotation propagation operations."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .cell_cycle import bud_known_history_status, toggle_history_knowledge


@dataclass(frozen=True)
class CcaHistoryKnowledgePropagationResult:
    """CCA table updates after toggling one cell's history knowledge."""

    current_cca_df: pd.DataFrame
    updated_cca_dfs_by_frame: dict[int, pd.DataFrame]
    undo_frame_indices: list[int]
    relative_id: int
    relative_status: pd.Series | None = None


def apply_history_knowledge_to_frame(
    cca_df: pd.DataFrame,
    cell_id: int,
    *,
    status_when_emerged: pd.Series | None = None,
    relative_id: int | None = None,
    relative_status: pd.Series | None = None,
) -> pd.DataFrame:
    """Return one CCA table after toggling a cell's history knowledge."""
    updated_cca_df = cca_df.copy()
    toggle_history_knowledge(
        updated_cca_df,
        cell_id,
        status_when_emerged=status_when_emerged,
    )
    if (
            relative_id is not None
            and relative_status is not None
            and relative_id in updated_cca_df.index
    ):
        updated_cca_df.loc[relative_id] = relative_status

    return updated_cca_df


def known_history_status_for_bud(
    bud_id: int,
    past_cca_frames,
    base_status: pd.Series,
) -> pd.Series | None:
    """Return status to restore when marking ``bud_id`` history as known."""
    return bud_known_history_status(
        bud_id,
        past_cca_frames,
        base_status,
    )


def propagate_history_knowledge(
    current_cca_df: pd.DataFrame,
    current_frame_i: int,
    cell_id: int,
    *,
    future_cca_frames=(),
    past_cca_frames=(),
    status_when_emerged: pd.Series | None = None,
    relative_id: int | None = None,
    relative_status: pd.Series | None = None,
) -> CcaHistoryKnowledgePropagationResult:
    """Return CCA frame updates after toggling history knowledge on a cell."""
    current_frame_i = int(current_frame_i)
    if current_cca_df is None:
        raise ValueError('current frame has no CCA table')

    if relative_id is None:
        relative_id = current_cca_df.at[cell_id, 'relative_ID']

    updated_current_cca_df = apply_history_knowledge_to_frame(
        current_cca_df,
        cell_id,
        status_when_emerged=status_when_emerged,
        relative_id=relative_id,
        relative_status=relative_status,
    )
    updated_cca_dfs_by_frame = {current_frame_i: updated_current_cca_df}

    undo_frame_indices = []
    for frame_i, cca_df_i in future_cca_frames:
        if cca_df_i is None:
            break

        undo_frame_indices.append(frame_i)
        if cell_id not in cca_df_i.index:
            continue

        updated_cca_dfs_by_frame[frame_i] = apply_history_knowledge_to_frame(
            cca_df_i,
            cell_id,
            status_when_emerged=status_when_emerged,
            relative_id=relative_id,
            relative_status=relative_status,
        )

    for frame_i, cca_df_i in past_cca_frames:
        if cca_df_i is None:
            break

        undo_frame_indices.append(frame_i)
        if cell_id not in cca_df_i.index:
            break

        frame_relative_id = cca_df_i.at[cell_id, 'relative_ID']
        updated_cca_dfs_by_frame[frame_i] = apply_history_knowledge_to_frame(
            cca_df_i,
            cell_id,
            status_when_emerged=status_when_emerged,
            relative_id=frame_relative_id,
            relative_status=relative_status,
        )

    return CcaHistoryKnowledgePropagationResult(
        current_cca_df=updated_current_cca_df,
        updated_cca_dfs_by_frame=updated_cca_dfs_by_frame,
        undo_frame_indices=undo_frame_indices,
        relative_id=relative_id,
        relative_status=relative_status,
    )
