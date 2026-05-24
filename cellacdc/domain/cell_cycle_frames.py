"""Frame-level cell-cycle annotation table operations."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .cell_cycle import (
    build_base_cell_cycle_annotations,
    ensure_cca_columns,
    extract_cell_cycle_annotations,
    has_cell_cycle_annotations,
    last_annotated_cca_by_cell,
    store_cell_cycle_annotations,
)


@dataclass(frozen=True)
class CcaFrameResolutionResult:
    """Resolved CCA table for one frame."""

    cca_df: pd.DataFrame | None
    used_snapshot_fallback: bool = False


@dataclass(frozen=True)
class CcaFrameStoreResult:
    """Stored ACDC frame plus optional CCA cache tables."""

    acdc_df: pd.DataFrame | None
    checker_cca_df: pd.DataFrame | None = None
    cached_cca_df: pd.DataFrame | None = None


@dataclass(frozen=True)
class CcaMissingFramesInitResult:
    """Prepared missing CCA frame tables and their last known statuses."""

    acdc_dfs_by_frame: dict[int, pd.DataFrame]
    last_annotated_cca_df: pd.DataFrame


def prepare_missing_cell_cycle_frame_annotations(
    frame_records,
    cca_colnames,
    last_cca_frame_i: int,
) -> CcaMissingFramesInitResult:
    """Prepare missing CCA columns before initializing skipped frames."""
    acdc_dfs_by_frame = {}
    annotated_cca_dfs = []
    for frame_i in range(last_cca_frame_i + 1):
        acdc_df = frame_records[frame_i]['acdc_df']
        if not has_cell_cycle_annotations(acdc_df):
            acdc_df = ensure_cca_columns(acdc_df, cca_colnames)
            acdc_dfs_by_frame[frame_i] = acdc_df
        annotated_cca_dfs.append(acdc_df[list(cca_colnames)])

    return CcaMissingFramesInitResult(
        acdc_dfs_by_frame=acdc_dfs_by_frame,
        last_annotated_cca_df=last_annotated_cca_by_cell(annotated_cca_dfs),
    )


def normalize_loaded_cell_cycle_frame_annotations(
    acdc_df: pd.DataFrame | None,
    cca_colnames,
    int_colnames=(),
) -> pd.DataFrame | None:
    """Normalize CCA columns loaded from concatenated frame metadata."""
    if acdc_df is None or not has_cell_cycle_annotations(acdc_df):
        return acdc_df

    cca_cols = acdc_df.columns.intersection(cca_colnames)
    cca_df = acdc_df[cca_cols].dropna()
    if cca_df.empty:
        return acdc_df.drop(columns=cca_colnames, errors='ignore')

    normalized_acdc_df = acdc_df.loc[cca_df.index].copy()
    existing_int_cols = [
        col for col in int_colnames if col in normalized_acdc_df.columns
    ]
    if existing_int_cols:
        normalized_acdc_df[existing_int_cols] = (
            normalized_acdc_df[existing_int_cols].astype('Int64')
        )
    return normalized_acdc_df


def resolve_cell_cycle_annotations(
    acdc_df: pd.DataFrame | None,
    cca_colnames,
    *,
    is_snapshot: bool = False,
    snapshot_cell_ids=(),
    dropna: bool = True,
    base_values: dict | None = None,
    tree_values: dict | None = None,
    with_tree_cols: bool = False,
) -> CcaFrameResolutionResult:
    """Resolve a frame CCA table, optionally falling back to snapshot defaults."""
    cca_df = extract_cell_cycle_annotations(
        acdc_df,
        cca_colnames,
        dropna=False,
    )
    used_snapshot_fallback = False
    if cca_df is None and is_snapshot:
        cca_df = build_base_cell_cycle_annotations(
            snapshot_cell_ids,
            with_tree_cols=with_tree_cols,
            base_values=base_values,
            tree_values=tree_values,
        )
        used_snapshot_fallback = True

    if cca_df is not None and dropna:
        cca_df = cca_df.dropna()

    return CcaFrameResolutionResult(
        cca_df=cca_df,
        used_snapshot_fallback=used_snapshot_fallback,
    )


def prepare_cell_cycle_checker_annotations(
    cca_df: pd.DataFrame | None,
    *,
    checker_running: bool = True,
) -> pd.DataFrame | None:
    """Return a checker-safe CCA copy when integrity checks are active."""
    if not checker_running or cca_df is None:
        return None
    return cca_df.copy()


def store_cell_cycle_frame_annotations(
    acdc_df: pd.DataFrame | None,
    cca_df: pd.DataFrame | None,
    cca_colnames,
    *,
    store_checker_copy: bool = False,
    store_cca_df_copy: bool = False,
) -> CcaFrameStoreResult:
    """Return stored ACDC frame and optional CCA cache copies."""
    stored_acdc_df = store_cell_cycle_annotations(
        acdc_df,
        cca_df,
        cca_colnames,
    )
    checker_cca_df = prepare_cell_cycle_checker_annotations(
        cca_df,
        checker_running=store_checker_copy,
    )

    cached_cca_df = None
    if store_cca_df_copy and cca_df is not None:
        cached_cca_df = cca_df.copy()

    return CcaFrameStoreResult(
        acdc_df=stored_acdc_df,
        checker_cca_df=checker_cca_df,
        cached_cca_df=cached_cca_df,
    )
