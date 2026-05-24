"""Cell-cycle division propagation operations."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .cell_cycle import (
    annotate_division,
    apply_mother_bud_pairing,
    assign_bud_to_mother,
    evaluate_mother_future_eligibility_frame,
    evaluate_mother_past_eligibility_frame,
    FutureBudDivisionResult,
    future_bud_division,
    mark_will_divide_frame,
    MotherEligibilityIssue,
    mother_status_before_wrong_bud,
    mother_not_g1_before_bud_emergence_frame,
    relative_status_before_bud_emergence,
    restore_mother_status_for_wrong_bud_frame,
    restore_mother_status_until_g1,
    undo_division_annotation,
    wrong_bud_id_for_mother,
)


@dataclass(frozen=True)
class CcaWillDividePropagationResult:
    """CCA frame updates after marking past frames as will-divide."""

    updated_cca_dfs_by_frame: dict[int, pd.DataFrame]
    generation_num: int | None = None
    stopped_frame_i: int | None = None


@dataclass(frozen=True)
class CcaManualDivisionPropagationResult:
    """CCA frame updates after annotating or undoing one division."""

    current_cca_df: pd.DataFrame
    updated_cca_dfs_by_frame: dict[int, pd.DataFrame]
    undo_frame_indices: list[int]
    clicked_stage: str
    relative_id: int


@dataclass(frozen=True)
class CcaSwapMothersFutureDivisionResult:
    """CCA frame updates for future division during mother-bud swaps."""

    updated_cca_dfs_by_frame: dict[int, pd.DataFrame]
    wrong_bud_id: int | None


@dataclass(frozen=True)
class CcaSwapMothersPastRestoreResult:
    """CCA frame updates restoring a mother before a wrong-bud assignment."""

    updated_cca_dfs_by_frame: dict[int, pd.DataFrame]
    wrong_bud_id: int | None
    mother_status: pd.Series | None = None


@dataclass(frozen=True)
class CcaMotherBudPairingsResult:
    """CCA table after applying one or more mother-bud pairings."""

    cca_df: pd.DataFrame
    pairings: dict[int, int]


@dataclass(frozen=True)
class CcaBudMotherAssignmentPropagationResult:
    """CCA updates after assigning a bud to a different mother."""

    current_cca_df: pd.DataFrame
    updated_cca_dfs_by_frame: dict[int, pd.DataFrame]
    undo_frame_indices: list[int]
    previous_mother_id: int


@dataclass(frozen=True)
class CcaMotherAssignmentEligibilityResult:
    """Mother-assignment eligibility issues across future and past frames."""

    future_issue: MotherEligibilityIssue | None = None
    past_issue: MotherEligibilityIssue | None = None
    g1_duration_future: int = 0

    @property
    def can_assign_without_user_action(self) -> bool:
        return self.future_issue is None and self.past_issue is None


@dataclass(frozen=True)
class CcaBudMotherChangeEligibilityResult:
    """Result of checking if a bud can change mother assignment."""

    future_division: FutureBudDivisionResult | None = None

    @property
    def can_change(self) -> bool:
        return self.future_division is None


@dataclass(frozen=True)
class CcaSwapMothersPairingPlan:
    """Bud/mother pairing maps for swapping two mother assignments."""

    correct_pairings: dict[int, int]
    wrong_pairings: dict[int, int]


@dataclass(frozen=True)
class CcaSwapMothersEligibilityResult:
    """Result of validating whether two mother assignments can be swapped."""

    can_swap: bool
    plan: CcaSwapMothersPairingPlan
    future_division_bud_id: int | None = None
    future_division_mother_id: int | None = None
    future_division_frame_i: int | None = None
    mother_not_g1_bud_id: int | None = None
    mother_not_g1_mother_id: int | None = None
    mother_not_g1_frame_i: int | None = None


@dataclass(frozen=True)
class CcaSwapMothersPropagationResult:
    """CCA updates after swapping two mother-bud assignments."""

    current_cca_df: pd.DataFrame
    updated_cca_dfs_by_frame: dict[int, pd.DataFrame]
    plan: CcaSwapMothersPairingPlan


def _frame_value(frame_values, frame_i: int):
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


def swap_mothers_pairing_plan(
    bud_id: int,
    other_bud_id: int,
    other_mother_id: int,
    mother_id: int,
) -> CcaSwapMothersPairingPlan:
    """Return correct and wrong pairings for a mother swap."""
    return CcaSwapMothersPairingPlan(
        correct_pairings={
            other_bud_id: mother_id,
            bud_id: other_mother_id,
        },
        wrong_pairings={
            mother_id: bud_id,
            other_mother_id: other_bud_id,
        },
    )


def swap_mothers_eligibility(
    bud_id: int,
    other_bud_id: int,
    other_mother_id: int,
    mother_id: int,
    future_cca_frames,
    past_cca_frames,
) -> CcaSwapMothersEligibilityResult:
    """Validate whether two mother assignments can be swapped."""
    plan = swap_mothers_pairing_plan(
        bud_id,
        other_bud_id,
        other_mother_id,
        mother_id,
    )

    future_cca_frames = list(future_cca_frames)
    for candidate_bud_id in (bud_id, other_bud_id):
        future_division = future_bud_division(
            candidate_bud_id,
            future_cca_frames,
        )
        if future_division is not None:
            return CcaSwapMothersEligibilityResult(
                can_swap=False,
                plan=plan,
                future_division_bud_id=candidate_bud_id,
                future_division_mother_id=future_division.mother_id,
                future_division_frame_i=future_division.frame_i,
            )

    past_cca_frames = list(past_cca_frames)
    for correct_bud_id, correct_mother_id in plan.correct_pairings.items():
        wrong_bud_id = plan.wrong_pairings[correct_mother_id]
        frame_not_g1 = mother_not_g1_before_bud_emergence_frame(
            correct_mother_id,
            correct_bud_id,
            wrong_bud_id,
            past_cca_frames,
        )
        if frame_not_g1 is not None:
            return CcaSwapMothersEligibilityResult(
                can_swap=False,
                plan=plan,
                mother_not_g1_bud_id=correct_bud_id,
                mother_not_g1_mother_id=correct_mother_id,
                mother_not_g1_frame_i=frame_not_g1,
            )

    return CcaSwapMothersEligibilityResult(can_swap=True, plan=plan)


def apply_mother_bud_pairings(
    cca_df: pd.DataFrame,
    pairings: dict[int, int],
    corrected_frame_i: int,
    *,
    set_mother_s_if_g1: bool = True,
) -> CcaMotherBudPairingsResult:
    """Return ``cca_df`` after applying multiple bud-to-mother pairings."""
    updated_cca_df = cca_df.copy()
    for bud_id, mother_id in pairings.items():
        updated_cca_df = apply_mother_bud_pairing(
            updated_cca_df,
            bud_id,
            mother_id,
            corrected_frame_i,
            set_mother_s_if_g1=set_mother_s_if_g1,
        )
    return CcaMotherBudPairingsResult(
        cca_df=updated_cca_df,
        pairings=dict(pairings),
    )


def mother_assignment_eligibility(
    bud_id: int,
    mother_id: int,
    future_cca_frames,
    past_cca_frames,
    last_cca_frame_i: int,
) -> CcaMotherAssignmentEligibilityResult:
    """Return the first eligibility issues for a proposed mother assignment."""
    g1_duration_future = 0
    future_issue = None
    for future_i, cca_df_i in future_cca_frames:
        result = evaluate_mother_future_eligibility_frame(
            cca_df_i,
            bud_id,
            mother_id,
            future_i,
            g1_duration_future,
            last_cca_frame_i,
        )
        g1_duration_future = result.g1_duration
        if result.issue is not None:
            future_issue = result.issue
            break
        if result.stop:
            break

    past_issue = None
    for past_i, cca_df_i in past_cca_frames:
        result = evaluate_mother_past_eligibility_frame(
            cca_df_i,
            bud_id,
            mother_id,
            past_i,
        )
        if result.issue is not None:
            past_issue = result.issue
            break
        if result.stop:
            break

    return CcaMotherAssignmentEligibilityResult(
        future_issue=future_issue,
        past_issue=past_issue,
        g1_duration_future=g1_duration_future,
    )


def bud_mother_change_eligibility(
    bud_id: int,
    future_cca_frames,
) -> CcaBudMotherChangeEligibilityResult:
    """Validate that ``bud_id`` has no future division annotation."""
    return CcaBudMotherChangeEligibilityResult(
        future_division=future_bud_division(bud_id, future_cca_frames),
    )


def previous_relative_status_before_bud_emergence(
    bud_id: int,
    current_mother_id: int,
    past_cca_frames,
    base_status: pd.Series,
) -> pd.Series:
    """Return relative status from before ``bud_id`` emerged."""
    base_mother_status = base_status.copy()
    base_mother_status.name = current_mother_id
    return relative_status_before_bud_emergence(
        bud_id,
        current_mother_id,
        past_cca_frames,
        base_mother_status,
        base_status,
    )


def propagate_bud_mother_assignment(
    current_cca_df: pd.DataFrame,
    current_frame_i: int,
    bud_id: int,
    mother_id: int,
    *,
    future_cca_frames=(),
    past_cca_frames=(),
    previous_mother_status: pd.Series | None = None,
) -> CcaBudMotherAssignmentPropagationResult:
    """Return CCA updates after assigning ``bud_id`` to ``mother_id``."""
    current_frame_i = int(current_frame_i)
    if current_cca_df is None:
        raise ValueError('current frame has no CCA table')

    previous_mother_id = current_cca_df.at[bud_id, 'relative_ID']
    if current_frame_i == 0:
        current_update = assign_bud_to_mother(
            current_cca_df,
            bud_id,
            mother_id,
            previous_mother_id=previous_mother_id,
            reset_previous_mother=True,
            mother_generation_num=2,
            mother_relationship=None,
        )
        return CcaBudMotherAssignmentPropagationResult(
            current_cca_df=current_update,
            updated_cca_dfs_by_frame={current_frame_i: current_update},
            undo_frame_indices=[],
            previous_mother_id=previous_mother_id,
        )

    current_update = assign_bud_to_mother(
        current_cca_df,
        bud_id,
        mother_id,
        corrected_frame_i=current_frame_i,
        previous_mother_id=previous_mother_id,
        previous_mother_status=previous_mother_status,
    )

    updated_cca_dfs_by_frame = {current_frame_i: current_update}
    undo_frame_indices = []

    for future_i, cca_df_i in future_cca_frames:
        if cca_df_i is None:
            break

        if bud_id not in cca_df_i.index or mother_id not in cca_df_i.index:
            continue

        undo_frame_indices.append(future_i)
        bud_relationship = cca_df_i.at[bud_id, 'relationship']
        bud_stage = cca_df_i.at[bud_id, 'cell_cycle_stage']
        if bud_relationship == 'mother' and bud_stage == 'S':
            break

        updated_cca_dfs_by_frame[future_i] = assign_bud_to_mother(
            cca_df_i,
            bud_id,
            mother_id,
            update_mother_only_if_g1=True,
            previous_mother_id=previous_mother_id,
            previous_mother_status=previous_mother_status,
        )

    for past_i, cca_df_i in past_cca_frames:
        if cca_df_i is None:
            break

        if bud_id not in cca_df_i.index:
            break

        undo_frame_indices.append(past_i)
        updated_cca_dfs_by_frame[past_i] = assign_bud_to_mother(
            cca_df_i,
            bud_id,
            mother_id,
            previous_mother_id=previous_mother_id,
            previous_mother_status=previous_mother_status,
        )

    return CcaBudMotherAssignmentPropagationResult(
        current_cca_df=current_update,
        updated_cca_dfs_by_frame=updated_cca_dfs_by_frame,
        undo_frame_indices=undo_frame_indices,
        previous_mother_id=previous_mother_id,
    )


def propagate_will_divide(
    cca_dfs_by_frame,
    current_frame_i: int,
    cell_id: int,
    relative_id: int,
    *,
    past_cca_frames=None,
) -> CcaWillDividePropagationResult:
    """Return past-frame CCA updates for a pending cell division."""
    generation_num = None
    stopped_frame_i = None
    updated_cca_dfs_by_frame = {}
    if past_cca_frames is None:
        past_cca_frames = (
            (past_frame_i, _frame_value(cca_dfs_by_frame, past_frame_i))
            for past_frame_i in range(int(current_frame_i) - 1, -1, -1)
        )

    for past_frame_i, cca_df_i in past_cca_frames:
        if cca_df_i is None:
            stopped_frame_i = past_frame_i
            break

        result = mark_will_divide_frame(
            cca_df_i,
            cell_id,
            relative_id,
            generation_num=generation_num,
        )
        generation_num = result.generation_num
        if result.stop:
            stopped_frame_i = past_frame_i
            break
        if result.should_store:
            updated_cca_dfs_by_frame[past_frame_i] = result.cca_df

    return CcaWillDividePropagationResult(
        updated_cca_dfs_by_frame=updated_cca_dfs_by_frame,
        generation_num=generation_num,
        stopped_frame_i=stopped_frame_i,
    )


def propagate_manual_division_annotation(
    cca_dfs_by_frame,
    current_frame_i: int,
    cell_id: int,
    *,
    current_cca_df: pd.DataFrame | None = None,
    future_cca_frames=None,
    past_cca_frames=None,
    size_t: int | None = None,
) -> CcaManualDivisionPropagationResult:
    """Return CCA updates for manual division annotation or undo."""
    current_frame_i = int(current_frame_i)
    if current_cca_df is None:
        current_cca_df = _frame_value(cca_dfs_by_frame, current_frame_i)
    if current_cca_df is None:
        raise ValueError('current frame has no CCA table')

    if past_cca_frames is not None:
        past_cca_frames = list(past_cca_frames)

    clicked_stage = current_cca_df.at[cell_id, 'cell_cycle_stage']
    relative_id = current_cca_df.at[cell_id, 'relative_ID']
    current_update = current_cca_df.copy()
    updated_cca_dfs_by_frame = {}

    if clicked_stage == 'S':
        will_divide_result = propagate_will_divide(
            cca_dfs_by_frame if past_cca_frames is None else None,
            current_frame_i,
            cell_id,
            relative_id,
            past_cca_frames=past_cca_frames,
        )
        updated_cca_dfs_by_frame.update(
            will_divide_result.updated_cca_dfs_by_frame
        )
        annotate_division(current_update, cell_id, relative_id, current_frame_i)
    else:
        undo_division_annotation(current_update, cell_id, relative_id)

    updated_cca_dfs_by_frame[current_frame_i] = current_update
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
        if cell_id not in cca_df_i.index:
            continue

        future_update = cca_df_i.copy()
        frame_stage = future_update.at[cell_id, 'cell_cycle_stage']
        frame_relative_id = future_update.at[cell_id, 'relative_ID']
        if clicked_stage == 'S':
            if frame_stage == 'G1':
                break
            annotate_division(
                future_update,
                cell_id,
                frame_relative_id,
                current_frame_i,
            )
            updated_cca_dfs_by_frame[future_frame_i] = future_update
        elif frame_stage == 'S':
            annotate_division(
                future_update,
                cell_id,
                frame_relative_id,
                current_frame_i,
            )
            updated_cca_dfs_by_frame[future_frame_i] = future_update
            break
        else:
            undo_division_annotation(future_update, cell_id, frame_relative_id)
            updated_cca_dfs_by_frame[future_frame_i] = future_update

    if past_cca_frames is None:
        past_cca_frames = (
            (past_frame_i, _frame_value(cca_dfs_by_frame, past_frame_i))
            for past_frame_i in range(current_frame_i - 1, -1, -1)
        )

    for past_frame_i, cca_df_i in past_cca_frames:
        if cca_df_i is None:
            break
        if cell_id not in cca_df_i.index or relative_id not in cca_df_i.index:
            break

        undo_frame_indices.append(past_frame_i)
        frame_stage = cca_df_i.at[cell_id, 'cell_cycle_stage']
        frame_relative_id = cca_df_i.at[cell_id, 'relative_ID']
        if frame_stage == 'S':
            break

        past_update = cca_df_i.copy()
        undo_division_annotation(past_update, cell_id, frame_relative_id)
        updated_cca_dfs_by_frame[past_frame_i] = past_update

    return CcaManualDivisionPropagationResult(
        current_cca_df=current_update,
        updated_cca_dfs_by_frame=updated_cca_dfs_by_frame,
        undo_frame_indices=undo_frame_indices,
        clicked_stage=clicked_stage,
        relative_id=relative_id,
    )


def propagate_swap_mothers_future_division(
    cca_dfs_by_frame,
    frame_i: int,
    mother_id: int,
    *,
    size_t: int | None = None,
) -> CcaSwapMothersFutureDivisionResult:
    """Return future-frame CCA updates after a swap-mothers division."""
    frame_i = int(frame_i)
    cca_df_at_division = _frame_value(cca_dfs_by_frame, frame_i)
    if cca_df_at_division is None:
        raise ValueError('division frame has no CCA table')

    wrong_bud_id = wrong_bud_id_for_mother(cca_df_at_division, mother_id)
    if wrong_bud_id is None:
        return CcaSwapMothersFutureDivisionResult(
            updated_cca_dfs_by_frame={},
            wrong_bud_id=None,
        )

    updated_cca_dfs_by_frame = {}
    division_cca_df = cca_df_at_division.copy()
    annotate_division(division_cca_df, mother_id, wrong_bud_id, frame_i)
    division_cca_df.at[mother_id, 'corrected_on_frame_i'] = frame_i
    updated_cca_dfs_by_frame[frame_i] = division_cca_df

    mother_status_to_restore = division_cca_df.loc[mother_id]
    stop_frame_i = _frame_count(cca_dfs_by_frame, size_t=size_t)
    for future_i in range(frame_i + 1, stop_frame_i):
        cca_df_i = _frame_value(cca_dfs_by_frame, future_i)
        if cca_df_i is None:
            break

        restore_result = restore_mother_status_until_g1(
            cca_df_i,
            mother_id,
            mother_status_to_restore,
            frame_i,
        )
        if not restore_result.restored:
            break

        updated_cca_dfs_by_frame[future_i] = restore_result.cca_df

    return CcaSwapMothersFutureDivisionResult(
        updated_cca_dfs_by_frame=updated_cca_dfs_by_frame,
        wrong_bud_id=wrong_bud_id,
    )


def restore_swap_mothers_past_status(
    cca_dfs_by_frame,
    frame_i: int,
    mother_id: int,
    base_status: pd.Series,
) -> CcaSwapMothersPastRestoreResult:
    """Return past-frame CCA updates restoring a mother before a wrong bud."""
    frame_i = int(frame_i)
    cca_df_at_disappearance = _frame_value(cca_dfs_by_frame, frame_i)
    if cca_df_at_disappearance is None:
        raise ValueError('disappearance frame has no CCA table')

    wrong_bud_id = wrong_bud_id_for_mother(
        cca_df_at_disappearance,
        mother_id,
    )
    if wrong_bud_id is None:
        return CcaSwapMothersPastRestoreResult(
            updated_cca_dfs_by_frame={},
            wrong_bud_id=None,
        )

    past_cca_frames = (
        _frame_value(cca_dfs_by_frame, past_i)
        for past_i in range(frame_i, -1, -1)
    )
    mother_status = mother_status_before_wrong_bud(
        mother_id,
        wrong_bud_id,
        past_cca_frames,
        base_status,
    )

    updated_cca_dfs_by_frame = {}
    for past_i in range(frame_i, -1, -1):
        cca_df_i = _frame_value(cca_dfs_by_frame, past_i)
        if cca_df_i is None:
            break

        restore_result = restore_mother_status_for_wrong_bud_frame(
            cca_df_i,
            mother_id,
            wrong_bud_id,
            mother_status,
            frame_i,
        )
        if not restore_result.restored:
            break

        updated_cca_dfs_by_frame[past_i] = restore_result.cca_df

    return CcaSwapMothersPastRestoreResult(
        updated_cca_dfs_by_frame=updated_cca_dfs_by_frame,
        wrong_bud_id=wrong_bud_id,
        mother_status=mother_status,
    )


def propagate_swap_mothers_assignment(
    current_cca_df: pd.DataFrame,
    current_frame_i: int,
    bud_id: int,
    other_bud_id: int,
    other_mother_id: int,
    mother_id: int,
    *,
    base_status: pd.Series,
    past_cca_frames=(),
    future_cca_frames=(),
) -> CcaSwapMothersPropagationResult:
    """Return CCA updates after swapping two incorrect mother assignments."""
    current_frame_i = int(current_frame_i)
    plan = swap_mothers_pairing_plan(
        bud_id,
        other_bud_id,
        other_mother_id,
        mother_id,
    )

    current_pairings_result = apply_mother_bud_pairings(
        current_cca_df,
        plan.correct_pairings,
        current_frame_i,
        set_mother_s_if_g1=False,
    )
    current_update = current_pairings_result.cca_df
    updated_cca_dfs_by_frame = {current_frame_i: current_update}

    past_dfs_by_frame = {
        int(frame_i): cca_df for frame_i, cca_df in past_cca_frames
    }
    corrected_bud_ids_past = set()
    for past_i in sorted(past_dfs_by_frame, reverse=True):
        if len(corrected_bud_ids_past) == len(plan.correct_pairings):
            break

        for correct_bud_id, correct_mother_id in plan.correct_pairings.items():
            if correct_bud_id in corrected_bud_ids_past:
                continue

            cca_df_i = past_dfs_by_frame[past_i]
            if cca_df_i is None:
                continue

            if correct_bud_id not in cca_df_i.index:
                corrected_bud_ids_past.add(correct_bud_id)
                if len(corrected_bud_ids_past) < len(plan.correct_pairings):
                    restore_frames = {
                        frame_i: past_dfs_by_frame[frame_i]
                        for frame_i in past_dfs_by_frame
                        if frame_i <= past_i
                    }
                    restore_result = restore_swap_mothers_past_status(
                        restore_frames,
                        past_i,
                        correct_mother_id,
                        base_status,
                    )
                    for frame_i, restored_df in (
                            restore_result.updated_cca_dfs_by_frame.items()
                    ):
                        past_dfs_by_frame[frame_i] = restored_df
                        updated_cca_dfs_by_frame[frame_i] = restored_df
                continue

            pairings_result = apply_mother_bud_pairings(
                cca_df_i,
                {correct_bud_id: correct_mother_id},
                current_frame_i,
            )
            past_dfs_by_frame[past_i] = pairings_result.cca_df
            updated_cca_dfs_by_frame[past_i] = pairings_result.cca_df

    future_dfs_by_frame = {
        int(frame_i): cca_df for frame_i, cca_df in future_cca_frames
    }
    corrected_bud_ids_future = set()
    for future_i in sorted(future_dfs_by_frame):
        if len(corrected_bud_ids_future) == len(plan.correct_pairings):
            break

        cca_df_i = updated_cca_dfs_by_frame.get(
            future_i,
            future_dfs_by_frame[future_i],
        )
        if cca_df_i is None:
            break

        for correct_bud_id, correct_mother_id in plan.correct_pairings.items():
            if correct_bud_id in corrected_bud_ids_future:
                continue

            if correct_bud_id not in cca_df_i.index:
                corrected_bud_ids_future.add(correct_bud_id)
                continue

            bud_stage = cca_df_i.at[correct_bud_id, 'cell_cycle_stage']
            if bud_stage == 'G1':
                corrected_bud_ids_future.add(correct_bud_id)
                if len(corrected_bud_ids_future) < len(plan.correct_pairings):
                    future_frames_for_division = {
                        frame_i: updated_cca_dfs_by_frame.get(frame_i, df)
                        for frame_i, df in future_dfs_by_frame.items()
                        if frame_i >= future_i
                    }
                    future_frames_for_division[future_i] = cca_df_i
                    division_result = propagate_swap_mothers_future_division(
                        future_frames_for_division,
                        future_i,
                        correct_mother_id,
                    )
                    for frame_i, division_df in (
                            division_result.updated_cca_dfs_by_frame.items()
                    ):
                        updated_cca_dfs_by_frame[frame_i] = division_df
                    if division_result.wrong_bud_id is not None:
                        will_divide_result = propagate_will_divide(
                            past_dfs_by_frame,
                            current_frame_i,
                            correct_mother_id,
                            division_result.wrong_bud_id,
                        )
                        for frame_i, will_divide_df in (
                                will_divide_result
                                .updated_cca_dfs_by_frame.items()
                        ):
                            past_dfs_by_frame[frame_i] = will_divide_df
                            updated_cca_dfs_by_frame[frame_i] = will_divide_df
                    cca_df_i = updated_cca_dfs_by_frame.get(future_i, cca_df_i)
                continue

            pairings_result = apply_mother_bud_pairings(
                cca_df_i,
                {correct_bud_id: correct_mother_id},
                current_frame_i,
            )
            cca_df_i = pairings_result.cca_df
            updated_cca_dfs_by_frame[future_i] = cca_df_i

    return CcaSwapMothersPropagationResult(
        current_cca_df=current_update,
        updated_cca_dfs_by_frame=updated_cca_dfs_by_frame,
        plan=plan,
    )
