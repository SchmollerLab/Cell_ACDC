"""Pure lineage annotation table operations."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class LineageAnnotationsRemovalResult:
    """ACDC frame after removing lineage annotation columns."""

    acdc_df: pd.DataFrame | None
    removed: bool
    missing_frame: bool = False


@dataclass(frozen=True)
class LineageFutureRemovalResult:
    """Future lineage removals for frame records."""

    acdc_dfs_by_frame: dict[int, pd.DataFrame]
    scanned_frame_indices: list[int]
    removed_frame_indices: list[int]
    missing_frame_indices: list[int]


def has_lineage_tree_annotations(
    acdc_df: pd.DataFrame | None,
    lineage_tree=None,
    *,
    parent_column: str = 'parent_ID_tree',
) -> bool:
    """Return whether lineage tree annotations are active or stored."""
    if lineage_tree is not None:
        return True
    return acdc_df is not None and parent_column in acdc_df.columns


def remove_lineage_tree_annotations(
    acdc_df: pd.DataFrame | None,
    lineage_tree_colnames,
) -> LineageAnnotationsRemovalResult:
    """Return an ACDC frame table without lineage tree columns."""
    if acdc_df is None:
        return LineageAnnotationsRemovalResult(
            acdc_df=None,
            removed=False,
            missing_frame=True,
        )

    existing_columns = acdc_df.columns.intersection(lineage_tree_colnames)
    if existing_columns.empty:
        return LineageAnnotationsRemovalResult(acdc_df=acdc_df, removed=False)

    return LineageAnnotationsRemovalResult(
        acdc_df=acdc_df.drop(columns=lineage_tree_colnames, errors='ignore'),
        removed=True,
    )


def remove_future_lineage_tree_annotations(
    frame_records,
    lineage_tree_colnames,
    from_frame_i: int,
    *,
    size_t: int | None = None,
    acdc_key: str = 'acdc_df',
) -> LineageFutureRemovalResult:
    """Return future frame-table lineage removals from ``from_frame_i`` onward."""
    acdc_dfs_by_frame = {}
    scanned_frame_indices = []
    removed_frame_indices = []
    missing_frame_indices = []
    stop_at = len(frame_records) if size_t is None else int(size_t)

    for frame_i in range(int(from_frame_i), stop_at):
        scanned_frame_indices.append(frame_i)
        acdc_df = frame_records[frame_i][acdc_key]
        result = remove_lineage_tree_annotations(
            acdc_df,
            lineage_tree_colnames,
        )
        if result.missing_frame:
            missing_frame_indices.append(frame_i)
            continue
        if not result.removed:
            continue

        acdc_dfs_by_frame[frame_i] = result.acdc_df
        removed_frame_indices.append(frame_i)

    return LineageFutureRemovalResult(
        acdc_dfs_by_frame=acdc_dfs_by_frame,
        scanned_frame_indices=scanned_frame_indices,
        removed_frame_indices=removed_frame_indices,
        missing_frame_indices=missing_frame_indices,
    )
