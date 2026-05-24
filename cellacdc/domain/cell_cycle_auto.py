"""Automatic cell-cycle annotation assignment operations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .cell_cycle import MotherBudPair, has_cell_cycle_annotations


@dataclass(frozen=True)
class AutoCcaFrameInitResult:
    """Current-frame CCA table prepared for automatic assignment."""

    cca_df: pd.DataFrame


@dataclass(frozen=True)
class AutoCcaRepeatFrameResult:
    """Repeat-auto-CCA state plus the new IDs still eligible."""

    is_last_visited_again: bool
    new_ids: list[int]


@dataclass(frozen=True)
class AutoCcaAssignmentResult:
    """Mother-bud assignments selected from an auto-CCA cost matrix."""

    pairs: list[MotherBudPair]
    assigned_mother_ids: set[int]


def auto_cca_repeat_frame_state(
    current_acdc_df: pd.DataFrame | None,
    next_acdc_df: pd.DataFrame | None,
    new_ids,
    *,
    enforce_all: bool = False,
) -> AutoCcaRepeatFrameResult:
    """Return repeat-auto-CCA state and IDs eligible for reassignment."""
    original_new_ids = list(new_ids)
    if not has_cell_cycle_annotations(current_acdc_df):
        return AutoCcaRepeatFrameResult(False, original_new_ids)
    if enforce_all:
        return AutoCcaRepeatFrameResult(False, original_new_ids)

    filtered_new_ids = [
        cell_id for cell_id in original_new_ids
        if current_acdc_df.at[cell_id, 'is_history_known']
        and current_acdc_df.at[cell_id, 'cell_cycle_stage'] == 'S'
    ]
    is_last_visited_again = (
        not has_cell_cycle_annotations(next_acdc_df)
    )
    return AutoCcaRepeatFrameResult(
        is_last_visited_again=is_last_visited_again,
        new_ids=filtered_new_ids,
    )


def merge_current_with_found_cca_rows(
    current_cca_df: pd.DataFrame,
    found_cca_dfs,
) -> pd.DataFrame:
    """Merge current and found CCA rows, keeping current rows first."""
    found_cca_dfs = list(found_cca_dfs)
    if not found_cca_dfs:
        return current_cca_df

    cca_df = pd.concat([current_cca_df, *found_cca_dfs])
    unique_idx = ~cca_df.index.duplicated(keep='first')
    return cca_df[unique_idx]


def prepare_auto_cca_current_frame(
    previous_cca_df: pd.DataFrame,
    current_acdc_df: pd.DataFrame,
    cca_colnames,
    *,
    current_cca_df: pd.DataFrame | None = None,
    found_cca_dfs=(),
) -> AutoCcaFrameInitResult:
    """Return the current CCA table to use before auto-assignment."""
    if current_cca_df is None:
        cca_df = previous_cca_df.copy()
    else:
        cca_df = current_acdc_df[list(cca_colnames)].copy()

    cca_df = merge_current_with_found_cca_rows(cca_df, found_cca_dfs)
    return AutoCcaFrameInitResult(cca_df=cca_df)


def uncorrected_new_ids_for_auto_cca(
    new_ids,
    current_cca_df: pd.DataFrame,
) -> list[int]:
    """Filter out new IDs that were manually corrected already."""
    try:
        corrected_ids = set(
            current_cca_df[current_cca_df['corrected_on_frame_i'] > 0].index
        )
    except Exception:
        corrected_ids = set()

    return [cell_id for cell_id in new_ids if cell_id not in corrected_ids]


def auto_cca_candidate_mother_ids(
    previous_cca_df: pd.DataFrame,
    previous_acdc_df: pd.DataFrame,
    current_ids,
    *,
    current_cca_df: pd.DataFrame | None = None,
    include_current_g1: bool = False,
    current_frame_i: int | None = None,
):
    """Return candidate G1 mother IDs for automatic CCA assignment."""
    try:
        previous_g1_df = previous_cca_df[
            previous_cca_df['cell_cycle_stage'] == 'G1'
        ]
        previous_g1_df = previous_g1_df[
            ~previous_acdc_df.loc[previous_g1_df.index]['is_cell_dead']
        ]
        candidate_ids = set(previous_g1_df.index)
    except Exception:
        candidate_ids = set()

    if include_current_g1 and current_cca_df is not None:
        current_g1_df = current_cca_df[
            current_cca_df['cell_cycle_stage'] == 'G1'
        ]
        new_cell_g1 = [
            cell_id for cell_id in current_g1_df.index
            if cell_id not in previous_cca_df.index
        ]
        candidate_ids.update(new_cell_g1)

        if (
                current_frame_i is not None
                and 'corrected_on_frame_i' in current_cca_df.columns
        ):
            cells_s_current = current_cca_df[
                (current_cca_df['cell_cycle_stage'] == 'S')
                & (current_cca_df['corrected_on_frame_i'] == current_frame_i)
            ].index
            candidate_ids = candidate_ids - set(cells_s_current)

    current_ids = set(current_ids)
    return [cell_id for cell_id in candidate_ids if cell_id in current_ids]


def auto_cca_assignments_from_cost(
    cost,
    mother_ids,
    bud_ids,
) -> AutoCcaAssignmentResult:
    """Return minimum-cost mother-bud assignments from a cost matrix."""
    mother_ids = list(mother_ids)
    bud_ids = list(bud_ids)
    row_idx, col_idx = linear_sum_assignment(cost)
    pairs = [
        MotherBudPair(
            bud_id=bud_ids[bud_idx],
            mother_id=mother_ids[mother_idx],
        )
        for mother_idx, bud_idx in zip(row_idx, col_idx)
    ]
    return AutoCcaAssignmentResult(
        pairs=pairs,
        assigned_mother_ids={pair.mother_id for pair in pairs},
    )


def apply_auto_cca_assignments(
    cca_df: pd.DataFrame,
    assignments: AutoCcaAssignmentResult,
    frame_i: int,
    base_bud_status: pd.Series,
    *,
    previous_cca_df: pd.DataFrame | None = None,
    current_ids=None,
) -> pd.DataFrame:
    """Apply selected auto-CCA mother-bud assignments to one frame."""
    updated_cca_df = cca_df
    for pair in assignments.pairs:
        updated_cca_df = apply_auto_bud_assignment(
            updated_cca_df,
            pair.bud_id,
            pair.mother_id,
            frame_i,
            base_bud_status,
            previous_cca_df=previous_cca_df,
            new_mother_ids=assignments.assigned_mother_ids,
        )

    if current_ids is not None:
        updated_cca_df = updated_cca_df.loc[list(current_ids)]

    return updated_cca_df


def nearest_point_2d_yx(points, all_others):
    """Return minimum distance and nearest point between two YX point sets."""
    points = np.asarray(points)
    all_others = np.asarray(all_others)
    diff = points[:, np.newaxis] - all_others
    dist = np.linalg.norm(diff, axis=2)
    point_idx, other_idx = np.unravel_index(dist.argmin(), dist.shape)
    return float(dist[point_idx, other_idx]), all_others[other_idx]


def auto_cca_cost_matrix_from_contours(
    mother_ids,
    bud_ids,
    mother_contours,
    bud_contours,
) -> np.ndarray:
    """Build an auto-CCA cost matrix from mother and bud contours."""
    mother_ids = list(mother_ids)
    bud_ids = list(bud_ids)
    cost = np.full((len(mother_ids), len(bud_ids)), np.inf)
    for mother_idx, mother_id in enumerate(mother_ids):
        mother_contour = mother_contours.get(mother_id)
        if mother_contour is None:
            continue
        for bud_idx, bud_id in enumerate(bud_ids):
            bud_contour = bud_contours.get(bud_id)
            if bud_contour is None:
                continue
            min_dist, _ = nearest_point_2d_yx(mother_contour, bud_contour)
            cost[mother_idx, bud_idx] = min_dist
    return cost


def auto_cca_cost_matrix_from_distances(
    distance_matrix_df: pd.DataFrame,
    mother_ids,
    bud_ids,
) -> np.ndarray:
    """Select an auto-CCA cost matrix from a precomputed distance table."""
    return distance_matrix_df.loc[list(mother_ids), list(bud_ids)].values


def apply_auto_bud_assignment(
    cca_df: pd.DataFrame,
    bud_id: int,
    mother_id: int,
    frame_i: int,
    base_bud_status: pd.Series,
    *,
    previous_cca_df: pd.DataFrame | None = None,
    new_mother_ids=(),
) -> pd.DataFrame:
    """Return ``cca_df`` after one automatic bud-to-mother assignment."""
    updated_cca_df = cca_df.copy()
    new_mother_ids = set(new_mother_ids)

    if bud_id in updated_cca_df.index and previous_cca_df is not None:
        relative_id = updated_cca_df.at[bud_id, 'relative_ID']
        if relative_id in previous_cca_df.index and relative_id not in new_mother_ids:
            updated_cca_df.loc[relative_id] = previous_cca_df.loc[relative_id]

    updated_cca_df.at[mother_id, 'relative_ID'] = bud_id
    updated_cca_df.at[mother_id, 'cell_cycle_stage'] = 'S'

    bud_status = base_bud_status.copy()
    bud_status['cell_cycle_stage'] = 'S'
    bud_status['generation_num'] = 0
    bud_status['relative_ID'] = mother_id
    bud_status['relationship'] = 'bud'
    bud_status['emerg_frame_i'] = frame_i
    bud_status['is_history_known'] = True
    bud_status['corrected_on_frame_i'] = -1
    for column in bud_status.index:
        if column not in updated_cca_df.columns:
            updated_cca_df[column] = pd.NA
    updated_cca_df.loc[bud_id] = bud_status
    return updated_cca_df
