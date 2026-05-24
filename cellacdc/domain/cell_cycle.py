"""Pure cell-cycle annotation table operations."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


DEFAULT_CELL_CYCLE_ANNOTATION_VALUES = {
    'cell_cycle_stage': 'G1',
    'generation_num': 2,
    'relative_ID': -1,
    'relationship': 'mother',
    'emerg_frame_i': -1,
    'division_frame_i': -1,
    'is_history_known': False,
    'corrected_on_frame_i': -1,
    'will_divide': 0,
    'daughter_disappears_before_division': 0,
    'disappears_before_division': 0,
}

DEFAULT_LINEAGE_TREE_ANNOTATION_VALUES = {
    'Cell_ID_tree': -1,
    'generation_num_tree': 1,
    'parent_ID_tree': -1,
    'root_ID_tree': -1,
    'sister_ID_tree': -1,
}


@dataclass(frozen=True)
class CcaSnapshotIdChanges:
    """ID additions/deletions needed after a segmentation edit."""

    new_ids: list[int]
    deleted_ids: list[int]
    reset_base: bool = False


@dataclass(frozen=True)
class CcaSnapshotIdEditResult:
    """CCA table after applying snapshot label-ID edits."""

    cca_df: pd.DataFrame
    changes: CcaSnapshotIdChanges
    restored_relative_ids: list[int]


@dataclass(frozen=True)
class CcaMotherStatusRestoreResult:
    """CCA table after restoring a mother status for one frame."""

    cca_df: pd.DataFrame
    restored: bool


@dataclass(frozen=True)
class CcaWillDivideFrameResult:
    """CCA table after applying one will-divide propagation step."""

    cca_df: pd.DataFrame
    generation_num: int | None
    should_store: bool
    stop: bool


@dataclass(frozen=True)
class CcaFrameRemovalResult:
    """ACDC frame after removing CCA columns."""

    acdc_df: pd.DataFrame | None
    removed: bool
    missing_frame: bool = False


@dataclass(frozen=True)
class CcaFutureRemovalResult:
    """Future CCA removals for frame records and concatenated ACDC data."""

    acdc_dfs_by_frame: dict[int, pd.DataFrame]
    cache_frame_indices: list[int]
    removed_frame_indices: list[int]
    concatenated_acdc_df: pd.DataFrame | None
    stopped_at_frame_i: int | None = None


@dataclass(frozen=True)
class ExistingNewIdCcaRowsResult:
    """Past CCA rows found for new IDs plus IDs that remain new."""

    found_cca_dfs: list[pd.DataFrame]
    remaining_new_ids: list[int]


@dataclass(frozen=True)
class CcaDisappearedBeforeDivisionFrameResult:
    """CCA frame update for disappeared-before-division propagation."""

    cca_df: pd.DataFrame | None
    should_store: bool
    stop: bool


@dataclass(frozen=True)
class CcaSPhaseDisappearancePropagationResult:
    """CCA updates after S-phase cells disappear before division."""

    previous_cca_df: pd.DataFrame
    current_cca_df: pd.DataFrame | None
    updated_cca_dfs_by_frame: dict[int, pd.DataFrame]
    disappeared_ids: list[int]
    automatically_divided_ids: list[int]


@dataclass(frozen=True)
class FutureBudDivisionResult:
    """Future frame where a bud is already annotated as divided."""

    frame_i: int
    mother_id: int


@dataclass(frozen=True)
class MotherBudPair:
    """Mother-bud ID pair."""

    bud_id: int
    mother_id: int


@dataclass(frozen=True)
class MotherEligibilityIssue:
    """Reason a mother candidate is not eligible on one frame."""

    mother_id: int
    bud_id: int
    frame_i: int
    reason: str
    blocks_assignment: bool = False


@dataclass(frozen=True)
class MotherEligibilityFrameResult:
    """Result of checking one mother-eligibility frame."""

    issue: MotherEligibilityIssue | None
    stop: bool
    g1_duration: int = 0


@dataclass(frozen=True)
class MissingCcaAnnotationItem:
    """One frame with missing values in CCA annotation columns."""

    position_i: int
    frame_i: int | None
    cca_df: pd.DataFrame


_ADD_NEW_ID_EDITS = {
    'Add new ID with brush tool',
    'Add new ID with curvature tool',
}
_DELETE_ID_EDITS = {
    'Delete ID',
    'Deleted non-selected objects',
    'Delete ID with eraser',
    'Delete IDs using ROI',
    'Merge IDs',
}
_SYNC_ID_EDITS = {
    'Separate IDs',
    'Edit ID',
}


def cca_snapshot_id_changes(
        edit_text: str,
        cca_ids,
        current_ids,
) -> CcaSnapshotIdChanges:
    """Classify CCA row additions/deletions after snapshot label edits."""
    cca_ids_set = {int(label_id) for label_id in cca_ids}
    current_ids_set = {int(label_id) for label_id in current_ids}
    new_ids = [
        int(label_id) for label_id in current_ids
        if int(label_id) not in cca_ids_set
    ]
    deleted_ids = [
        int(label_id) for label_id in cca_ids
        if int(label_id) not in current_ids_set
    ]

    if edit_text in _ADD_NEW_ID_EDITS:
        return CcaSnapshotIdChanges(new_ids=new_ids, deleted_ids=[])
    if edit_text in _DELETE_ID_EDITS:
        return CcaSnapshotIdChanges(new_ids=[], deleted_ids=deleted_ids)
    if edit_text in _SYNC_ID_EDITS:
        return CcaSnapshotIdChanges(new_ids=new_ids, deleted_ids=deleted_ids)
    if edit_text == 'Repeat segmentation':
        return CcaSnapshotIdChanges(
            new_ids=[],
            deleted_ids=[],
            reset_base=True,
        )
    return CcaSnapshotIdChanges(new_ids=[], deleted_ids=[])


def relabel_cca_ids(
    cca_df: pd.DataFrame,
    old_ids,
    new_ids,
) -> pd.DataFrame:
    """Return ``cca_df`` with IDs relabelled in index and references."""
    id_mapper = dict(zip(old_ids, new_ids))
    relabelled_cca_df = cca_df.copy()
    relabelled_cca_df['relative_ID'] = (
        relabelled_cca_df['relative_ID'].replace(old_ids, new_ids)
    )
    return relabelled_cca_df.rename(index=id_mapper)


def merge_missing_cca_ids(
    cca_df: pd.DataFrame | None,
    base_cca_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return ``cca_df`` with rows filled from ``base_cca_df`` where missing."""
    if cca_df is None:
        return base_cca_df.copy()
    return cca_df.combine_first(base_cca_df)


def apply_snapshot_cca_id_edits(
    cca_df: pd.DataFrame,
    edit_text: str,
    current_ids,
    base_cca_df: pd.DataFrame,
    *,
    base_values: dict | None = None,
) -> CcaSnapshotIdEditResult:
    """Return CCA table updated after snapshot label-ID edits."""
    changes = cca_snapshot_id_changes(edit_text, cca_df.index, current_ids)
    if changes.reset_base:
        return CcaSnapshotIdEditResult(
            cca_df=base_cca_df.copy(),
            changes=changes,
            restored_relative_ids=[],
        )

    updated_cca_df = cca_df.copy()
    for new_id in changes.new_ids:
        if new_id <= 0:
            continue
        updated_cca_df = add_base_cell_cycle_annotation(
            updated_cca_df,
            new_id,
            base_values=base_values,
        )

    restored_relative_ids = []
    if changes.deleted_ids:
        relative_ids = updated_cca_df.reindex(
            changes.deleted_ids,
            fill_value=-1,
        )['relative_ID']
        updated_cca_df = updated_cca_df.drop(
            changes.deleted_ids,
            errors='ignore',
        )
        for relative_id in relative_ids:
            if relative_id <= 0:
                continue
            updated_cca_df = add_base_cell_cycle_annotation(
                updated_cca_df,
                relative_id,
                base_values=base_values,
            )
            restored_relative_ids.append(int(relative_id))

    return CcaSnapshotIdEditResult(
        cca_df=updated_cca_df,
        changes=changes,
        restored_relative_ids=restored_relative_ids,
    )


def collect_existing_new_id_cca_rows(
    new_ids,
    past_cca_dfs,
) -> ExistingNewIdCcaRowsResult:
    """Collect past CCA rows for IDs that were classified as new."""
    if not new_ids:
        return ExistingNewIdCcaRowsResult(
            found_cca_dfs=[],
            remaining_new_ids=[],
        )

    remaining_new_ids = list(new_ids)
    found_cca_dfs = []
    for cca_df in past_cca_dfs:
        if not remaining_new_ids:
            break

        intersect_idx = cca_df.index.intersection(remaining_new_ids)
        found_cca_df = cca_df.loc[intersect_idx]
        if found_cca_df.empty:
            continue

        found_cca_dfs.append(found_cca_df)
        found_ids = set(found_cca_df.index)
        remaining_new_ids = [
            cell_id for cell_id in remaining_new_ids
            if cell_id not in found_ids
        ]

    return ExistingNewIdCcaRowsResult(
        found_cca_dfs=found_cca_dfs,
        remaining_new_ids=remaining_new_ids,
    )


def collect_existing_new_id_cca_rows_from_frames(
    new_ids,
    past_acdc_frames,
    cca_colnames,
) -> ExistingNewIdCcaRowsResult:
    """Collect past CCA rows for new IDs from frame ACDC tables."""
    past_cca_dfs = (
        acdc_df[list(cca_colnames)]
        for _, acdc_df in past_acdc_frames
        if acdc_df is not None
    )
    return collect_existing_new_id_cca_rows(new_ids, past_cca_dfs)


def ensure_cca_columns(
    acdc_df: pd.DataFrame,
    cca_colnames,
    fill_value='',
) -> pd.DataFrame:
    """Return ``acdc_df`` with missing CCA columns initialized."""
    updated_acdc_df = acdc_df.copy()
    for column in cca_colnames:
        if column not in updated_acdc_df.columns:
            updated_acdc_df[column] = fill_value
    return updated_acdc_df


def build_base_cell_cycle_annotations(
    cell_ids,
    *,
    with_tree_cols: bool = False,
    base_values: dict | None = None,
    tree_values: dict | None = None,
) -> pd.DataFrame:
    """Return a base CCA table for ``cell_ids``."""
    base_values = (
        DEFAULT_CELL_CYCLE_ANNOTATION_VALUES
        if base_values is None else base_values
    )
    tree_values = (
        DEFAULT_LINEAGE_TREE_ANNOTATION_VALUES
        if tree_values is None else tree_values
    )

    cell_ids = list(cell_ids)
    row_data = dict(base_values)
    if with_tree_cols:
        row_data = {**row_data, **tree_values}

    cca_df = pd.DataFrame([row_data.copy() for _ in cell_ids], index=cell_ids)
    if with_tree_cols:
        cca_df['Cell_ID_tree'] = cell_ids
    cca_df.index.name = 'Cell_ID'
    return cca_df


def base_cell_cycle_annotation_status(
    base_values: dict | None = None,
) -> pd.Series:
    """Return one base CCA status row."""
    base_values = (
        DEFAULT_CELL_CYCLE_ANNOTATION_VALUES
        if base_values is None else base_values
    )
    return pd.Series(dict(base_values))


def add_base_cell_cycle_annotation(
    cca_df: pd.DataFrame,
    cell_id: int,
    *,
    base_values: dict | None = None,
) -> pd.DataFrame:
    """Return ``cca_df`` with one base CCA row added or reset."""
    if int(cell_id) <= 0:
        return cca_df

    base_values = (
        DEFAULT_CELL_CYCLE_ANNOTATION_VALUES
        if base_values is None else base_values
    )
    if cca_df.empty:
        return build_base_cell_cycle_annotations(
            [cell_id],
            base_values=base_values,
        )

    updated_cca_df = cca_df.copy()
    for column, value in base_values.items():
        updated_cca_df.at[cell_id, column] = value
    return updated_cca_df


from .cell_cycle_deletions import (  # noqa: E402
    CcaDeletedIdsPropagationResult,
    CcaDeletedIdsResult,
    CcaDeletedRelativeStatusesResult,
    CcaRelativeRestoreResult,
    apply_cca_deleted_ids_to_frame,
    apply_deleted_cell_cycle_ids_to_frame,
    delete_cca_ids,
    deleted_relative_cca_status,
    propagate_deleted_cell_cycle_ids,
    restore_cca_relative_statuses,
    restore_deleted_relative_cell_cycle_statuses,
)

def last_annotated_cca_by_cell(
    annotated_cca_dfs,
) -> pd.DataFrame:
    """Return last annotated CCA status for each cell across frame tables."""
    annotated_cca_dfs = list(annotated_cca_dfs)
    if not annotated_cca_dfs:
        return pd.DataFrame()

    keys = range(len(annotated_cca_dfs))
    names = ['frame_i', 'Cell_ID']
    annotated_cca_df = (
        pd.concat(annotated_cca_dfs, keys=keys, names=names)
        .reset_index()
        .set_index(['Cell_ID', 'frame_i'])
        .sort_index()
    )
    return annotated_cca_df.groupby(level=0).last()


def overlay_last_annotated_cca(
    base_cca_df: pd.DataFrame,
    last_annotated_cca_df: pd.DataFrame,
    cca_colnames,
) -> pd.DataFrame:
    """Overlay last known CCA rows onto a base CCA frame."""
    updated_cca_df = base_cca_df.copy()
    if last_annotated_cca_df.empty:
        return updated_cca_df

    idx = last_annotated_cca_df.index.intersection(updated_cca_df.index)
    updated_cca_df.loc[idx, cca_colnames] = last_annotated_cca_df.loc[idx]
    return updated_cca_df


def has_cell_cycle_annotations(acdc_df: pd.DataFrame | None) -> bool:
    """Return whether an ACDC frame table contains CCA annotations."""
    return acdc_df is not None and 'cell_cycle_stage' in acdc_df.columns


from .cell_cycle_auto import (  # noqa: E402
    AutoCcaAssignmentResult,
    AutoCcaFrameInitResult,
    AutoCcaRepeatFrameResult,
    apply_auto_cca_assignments,
    apply_auto_bud_assignment,
    auto_cca_assignments_from_cost,
    auto_cca_candidate_mother_ids,
    auto_cca_cost_matrix_from_contours,
    auto_cca_cost_matrix_from_distances,
    auto_cca_repeat_frame_state,
    merge_current_with_found_cca_rows,
    nearest_point_2d_yx,
    prepare_auto_cca_current_frame,
    uncorrected_new_ids_for_auto_cca,
)


def last_annotated_cell_cycle_frame_index(acdc_dfs) -> int:
    """Return the last frame index with CCA annotations.

    This preserves the GUI's legacy first-frame behavior: if the first frame is
    missing or unannotated, frame index 0 is returned.
    """
    acdc_dfs = list(acdc_dfs)
    if not acdc_dfs:
        return 0

    last_seen_i = 0
    for frame_i, acdc_df in enumerate(acdc_dfs):
        last_seen_i = frame_i
        if not has_cell_cycle_annotations(acdc_df):
            break
    else:
        return last_seen_i

    if last_seen_i == 0 or last_seen_i + 1 == len(acdc_dfs):
        return last_seen_i
    return last_seen_i - 1


def extract_cell_cycle_annotations(
    acdc_df: pd.DataFrame | None,
    cca_colnames,
    *,
    dropna: bool = True,
) -> pd.DataFrame | None:
    """Return the CCA columns from an ACDC frame table, if present."""
    if not has_cell_cycle_annotations(acdc_df):
        return None

    cca_df = acdc_df[list(cca_colnames)].copy()
    if dropna:
        cca_df = cca_df.dropna()
    return cca_df


def concat_cell_cycle_annotations(
    frame_records,
    cca_colnames,
    *,
    acdc_key: str = 'acdc_df',
    size_t: int | None = None,
) -> pd.DataFrame | None:
    """Return consecutive per-frame CCA tables as one MultiIndex table."""
    cca_dfs = []
    keys = []

    if size_t is None:
        records = enumerate(frame_records)
    else:
        records = ((frame_i, frame_records[frame_i]) for frame_i in range(size_t))

    for frame_i, frame_record in records:
        acdc_df = frame_record[acdc_key]
        cca_df = extract_cell_cycle_annotations(
            acdc_df,
            cca_colnames,
        )
        if cca_df is None:
            break

        cca_dfs.append(cca_df)
        keys.append(frame_i)

    if not cca_dfs:
        return None

    return pd.concat(cca_dfs, keys=keys, names=['frame_i'])


def split_concat_cell_cycle_annotations(
    global_cca_df: pd.DataFrame | None,
    *,
    size_t: int | None = None,
    frame_level: str = 'frame_i',
) -> list[tuple[int, pd.DataFrame]]:
    """Return per-frame CCA tables from a concatenated CCA table."""
    if global_cca_df is None:
        return []

    if size_t is None:
        frame_indices = global_cca_df.index.get_level_values(frame_level).unique()
    else:
        frame_indices = range(size_t)

    frame_tables = []
    for frame_i in frame_indices:
        try:
            cca_df = global_cca_df.xs(
                frame_i,
                level=frame_level,
                drop_level=True,
            )
        except KeyError:
            break

        frame_tables.append((int(frame_i), cca_df.copy()))

    return frame_tables


def remove_cell_cycle_annotations(
    acdc_df: pd.DataFrame | None,
    cca_colnames,
) -> CcaFrameRemovalResult:
    """Return an ACDC frame table without CCA columns."""
    if acdc_df is None:
        return CcaFrameRemovalResult(
            acdc_df=None,
            removed=False,
            missing_frame=True,
        )
    if not has_cell_cycle_annotations(acdc_df):
        return CcaFrameRemovalResult(acdc_df=acdc_df, removed=False)

    return CcaFrameRemovalResult(
        acdc_df=acdc_df.drop(columns=cca_colnames, errors='ignore'),
        removed=True,
    )


def remove_future_cell_cycle_annotations(
    frame_records,
    cca_colnames,
    from_frame_i: int,
    *,
    size_t: int | None = None,
    concatenated_acdc_df: pd.DataFrame | None = None,
    acdc_key: str = 'acdc_df',
) -> CcaFutureRemovalResult:
    """Return future frame-table CCA removals from ``from_frame_i`` onward."""
    stop_frame_i = None
    acdc_dfs_by_frame = {}
    cache_frame_indices = []
    removed_frame_indices = []
    stop_at = len(frame_records) if size_t is None else int(size_t)

    for frame_i in range(int(from_frame_i), stop_at):
        cache_frame_indices.append(frame_i)
        acdc_df = frame_records[frame_i][acdc_key]
        result = remove_cell_cycle_annotations(acdc_df, cca_colnames)
        if result.missing_frame:
            stop_frame_i = frame_i
            break
        if not result.removed:
            continue

        acdc_dfs_by_frame[frame_i] = result.acdc_df
        removed_frame_indices.append(frame_i)

    truncated_acdc_df = concatenated_acdc_df
    if concatenated_acdc_df is not None:
        frames = concatenated_acdc_df.index.get_level_values(0)
        if from_frame_i in frames:
            truncated_acdc_df = concatenated_acdc_df.loc[:from_frame_i]

    return CcaFutureRemovalResult(
        acdc_dfs_by_frame=acdc_dfs_by_frame,
        cache_frame_indices=cache_frame_indices,
        removed_frame_indices=removed_frame_indices,
        concatenated_acdc_df=truncated_acdc_df,
        stopped_at_frame_i=stop_frame_i,
    )


def store_cell_cycle_annotations(
    acdc_df: pd.DataFrame | None,
    cca_df: pd.DataFrame | None,
    cca_colnames,
) -> pd.DataFrame | None:
    """Return ``acdc_df`` with ``cca_df`` annotations merged in."""
    if acdc_df is None or cca_df is None:
        return acdc_df

    if has_cell_cycle_annotations(acdc_df):
        updated_acdc_df = acdc_df.copy()
        updated_acdc_df[list(cca_colnames)] = cca_df[list(cca_colnames)]
        return updated_acdc_df

    metadata_df = acdc_df.drop(cca_df.columns, axis=1, errors='ignore')
    return metadata_df.join(cca_df, how='left')


from .cell_cycle_frames import (  # noqa: E402
    CcaFrameResolutionResult,
    CcaFrameStoreResult,
    CcaMissingFramesInitResult,
    normalize_loaded_cell_cycle_frame_annotations,
    prepare_cell_cycle_checker_annotations,
    prepare_missing_cell_cycle_frame_annotations,
    resolve_cell_cycle_annotations,
    store_cell_cycle_frame_annotations,
)


def apply_manual_cca_changes(
    cca_df: pd.DataFrame,
    changes,
) -> pd.DataFrame:
    """Return ``cca_df`` with manual CCA table changes applied."""
    updated_cca_df = cca_df.copy()
    for cell_id, changes_for_id in changes.items():
        if cell_id not in updated_cca_df.index:
            continue
        for column, (_old_value, new_value) in changes_for_id.items():
            updated_cca_df.at[cell_id, column] = new_value
    return updated_cca_df


def missing_cell_cycle_annotation_items(
    positions_frame_records,
    cca_colnames,
    *,
    is_snapshot: bool = False,
) -> list[MissingCcaAnnotationItem]:
    """Return frames whose CCA annotation columns contain missing values."""
    missing_items = []
    for position_i, frame_records in enumerate(positions_frame_records):
        for frame_i, frame_record in enumerate(frame_records):
            acdc_df = frame_record['acdc_df']
            if not has_cell_cycle_annotations(acdc_df):
                continue

            cca_df = acdc_df[list(cca_colnames)]
            if not cca_df.isnull().values.any():
                continue

            missing_items.append(
                MissingCcaAnnotationItem(
                    position_i=position_i,
                    frame_i=None if is_snapshot else frame_i,
                    cca_df=cca_df,
                )
            )

    return missing_items


def s_phase_relative_ids_gone(
    previous_cca_df: pd.DataFrame,
    current_ids,
) -> list[int]:
    """Return S-phase relative IDs that disappeared while their pair remains."""
    current_ids = set(current_ids)
    disappeared_ids = []
    for cc_series in previous_cca_df.itertuples():
        if cc_series.cell_cycle_stage != 'S':
            continue

        cell_id = cc_series.Index
        relative_id = cc_series.relative_ID
        if relative_id == -1:
            continue
        if relative_id not in current_ids and cell_id in current_ids:
            disappeared_ids.append(relative_id)

    return disappeared_ids


def mark_current_relative_after_disappearance(
    cca_df: pd.DataFrame,
    cell_id: int,
    division_frame_i: int,
) -> pd.DataFrame:
    """Return current CCA with surviving relative marked as divided."""
    updated_cca_df = cca_df.copy()
    updated_cca_df.at[cell_id, 'generation_num'] += 1
    updated_cca_df.at[cell_id, 'division_frame_i'] = division_frame_i
    updated_cca_df.at[cell_id, 'relationship'] = 'mother'
    return updated_cca_df


def mark_disappeared_before_division_frame(
    cca_df: pd.DataFrame | None,
    gone_id: int,
    relative_id: int,
    generation_num: int,
) -> CcaDisappearedBeforeDivisionFrameResult:
    """Mark one past CCA frame while generation continuity holds."""
    if cca_df is None:
        return CcaDisappearedBeforeDivisionFrameResult(
            cca_df=None,
            should_store=False,
            stop=True,
        )

    try:
        if cca_df.at[relative_id, 'generation_num'] != generation_num:
            return CcaDisappearedBeforeDivisionFrameResult(
                cca_df=cca_df,
                should_store=False,
                stop=True,
            )
    except Exception:
        return CcaDisappearedBeforeDivisionFrameResult(
            cca_df=cca_df,
            should_store=False,
            stop=True,
        )

    updated_cca_df = cca_df.copy()
    updated_cca_df.at[gone_id, 'disappears_before_division'] = 1
    updated_cca_df.at[relative_id, 'daughter_disappears_before_division'] = 1
    return CcaDisappearedBeforeDivisionFrameResult(
        cca_df=updated_cca_df,
        should_store=True,
        stop=False,
    )


def propagate_s_phase_disappearance_divisions(
    previous_cca_df: pd.DataFrame,
    current_cca_df: pd.DataFrame | None,
    current_frame_i: int,
    current_ids,
    *,
    past_cca_frames=(),
    disappeared_ids=None,
) -> CcaSPhaseDisappearancePropagationResult:
    """Return CCA updates for S-phase cells whose relatives disappeared."""
    current_frame_i = int(current_frame_i)
    previous_frame_i = current_frame_i - 1
    if disappeared_ids is None:
        disappeared_ids = s_phase_relative_ids_gone(
            previous_cca_df,
            current_ids,
        )
    else:
        disappeared_ids = list(disappeared_ids)

    previous_update = previous_cca_df.copy()
    current_update = None if current_cca_df is None else current_cca_df.copy()
    past_cca_frames = list(past_cca_frames)
    updated_cca_dfs_by_frame = {}
    automatically_divided_ids = []

    for gone_id in disappeared_ids:
        relative_id = previous_update.at[gone_id, 'relative_ID']
        generation_num = previous_update.at[relative_id, 'generation_num']

        previous_result = mark_disappeared_before_division_frame(
            previous_update,
            gone_id,
            relative_id,
            generation_num,
        )
        if previous_result.should_store:
            previous_update = previous_result.cca_df

        annotate_division(
            previous_update,
            gone_id,
            relative_id,
            frame_i=previous_frame_i,
        )
        updated_cca_dfs_by_frame[previous_frame_i] = previous_update

        if current_update is not None:
            current_update = mark_current_relative_after_disappearance(
                current_update,
                relative_id,
                previous_frame_i,
            )

        automatically_divided_ids.append(relative_id)

        for past_frame_i, past_cca_df in past_cca_frames:
            past_update = updated_cca_dfs_by_frame.get(
                past_frame_i,
                past_cca_df,
            )
            result = mark_disappeared_before_division_frame(
                past_update,
                gone_id,
                relative_id,
                generation_num,
            )
            if result.stop:
                break
            if result.should_store:
                updated_cca_dfs_by_frame[past_frame_i] = result.cca_df

    return CcaSPhaseDisappearancePropagationResult(
        previous_cca_df=previous_update,
        current_cca_df=current_update,
        updated_cca_dfs_by_frame=updated_cca_dfs_by_frame,
        disappeared_ids=disappeared_ids,
        automatically_divided_ids=automatically_divided_ids,
    )


def reset_cca_future_flags(cca_df: pd.DataFrame) -> pd.DataFrame:
    """Clear future-cycle flags that no longer apply to the current CCA frame."""
    updated_cca_df = cca_df.copy()
    s_phase_mask = updated_cca_df.cell_cycle_stage == 'S'
    updated_cca_df.loc[s_phase_mask, 'will_divide'] = 0

    mothers_mask = (
        (updated_cca_df.relationship == 'mother')
        & s_phase_mask
    )
    bud_mask = updated_cca_df.relationship == 'bud'

    updated_cca_df.loc[
        mothers_mask, 'daughter_disappears_before_division'
    ] = 0
    updated_cca_df.loc[bud_mask, 'disappears_before_division'] = 0
    return updated_cca_df


def reset_will_divide_for_generations(
    global_cca_df: pd.DataFrame,
    cell_generation_ids,
) -> pd.DataFrame:
    """Return concatenated CCA table with selected ``will_divide`` values reset."""
    updated_cca_df = global_cca_df.copy()
    generation_index_df = (
        updated_cca_df.reset_index()
        .set_index(['Cell_ID', 'generation_num'])
    )
    generation_index_df.loc[cell_generation_ids, 'will_divide'] = 0
    return (
        generation_index_df.reset_index()
        .set_index(['frame_i', 'Cell_ID'])
    )


def will_divide_without_next_generation_ids(
    global_cca_df: pd.DataFrame,
) -> list[tuple[int, int]]:
    """Return ``(Cell_ID, generation_num)`` pairs with stale ``will_divide``."""
    global_cca_will_divide = global_cca_df[global_cca_df['will_divide'] > 0]
    global_cca_will_divide = global_cca_will_divide.reset_index()

    cell_generation_index = (
        global_cca_df.reset_index()
        .set_index(['Cell_ID', 'generation_num'])
        .index
    )

    next_gen_will_divide_df = (
        global_cca_will_divide[['Cell_ID', 'generation_num']].copy()
    )
    next_gen_will_divide_df['generation_num'] += 1
    next_gen_will_divide_index = (
        next_gen_will_divide_df.reset_index()
        .set_index(['Cell_ID', 'generation_num'])
        .index
    )

    wrong_next_generation_ids = (
        next_gen_will_divide_index.difference(cell_generation_index)
        .to_frame()
        .to_numpy()
    )
    if wrong_next_generation_ids.size == 0:
        return []

    wrong_next_generation_ids[:, -1] -= 1
    return [
        (cell_id, generation_num)
        for cell_id, generation_num in wrong_next_generation_ids
    ]


def fix_will_divide_without_next_generation(
    acdc_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return ``acdc_df`` with stale ``will_divide`` values reset to 0."""
    if 'cell_cycle_stage' not in acdc_df.columns:
        return acdc_df

    required_cols = ['frame_i', 'Cell_ID', 'generation_num', 'will_divide']

    cca_df_mask = ~acdc_df['cell_cycle_stage'].isna()
    cca_df = acdc_df[cca_df_mask].reset_index()[required_cols]

    cell_generation_ids = will_divide_without_next_generation_ids(cca_df)
    if not cell_generation_ids:
        return acdc_df

    cca_df = reset_will_divide_for_generations(cca_df, cell_generation_ids)
    updated_acdc_df = acdc_df.reset_index().set_index(['frame_i', 'Cell_ID'])

    updated_acdc_df.loc[cca_df.index, 'will_divide'] = cca_df['will_divide']
    return updated_acdc_df


def bud_known_history_status(
    cell_id: int,
    past_cca_frames,
    base_status: pd.Series,
) -> pd.Series | None:
    """Return restored known-history status for a bud absent in past frames."""
    for frame_i, cca_df in past_cca_frames:
        if cca_df is None:
            continue
        if cell_id in cca_df.index:
            continue

        bud_status = base_status.copy()
        bud_status['cell_cycle_stage'] = 'S'
        bud_status['generation_num'] = 0
        bud_status['relationship'] = 'bud'
        bud_status['emerg_frame_i'] = frame_i + 1
        bud_status['is_history_known'] = True
        return bud_status

    return None


def relative_status_before_bud_emergence(
    bud_id: int,
    current_mother_id: int,
    past_cca_frames,
    base_mother_status: pd.Series,
    base_bud_status: pd.Series,
) -> pd.Series:
    """Return a mother's status from before ``bud_id`` emerged."""
    for frame_i, cca_df in past_cca_frames:
        if cca_df is None:
            continue
        if bud_id in cca_df.index:
            continue

        if current_mother_id in cca_df.index:
            return cca_df.loc[current_mother_id].copy()

        bud_status = base_bud_status.copy()
        bud_status['cell_cycle_stage'] = 'S'
        bud_status['generation_num'] = 0
        bud_status['relationship'] = 'bud'
        bud_status['emerg_frame_i'] = frame_i + 1
        bud_status['is_history_known'] = True
        return bud_status

    return base_mother_status.copy()


def assign_bud_to_mother(
    cca_df: pd.DataFrame,
    bud_id: int,
    mother_id: int,
    *,
    corrected_frame_i: int | None = None,
    update_mother: bool = True,
    update_mother_only_if_g1: bool = False,
    mother_generation_num: int | None = None,
    mother_relationship: str | None = 'mother',
    previous_mother_id: int | None = None,
    previous_mother_status: pd.Series | None = None,
    reset_previous_mother: bool = False,
) -> pd.DataFrame:
    """Return ``cca_df`` with ``bud_id`` assigned to ``mother_id``."""
    updated_cca_df = cca_df.copy()

    if reset_previous_mother and previous_mother_id in updated_cca_df.index:
        updated_cca_df.at[previous_mother_id, 'relative_ID'] = -1
        updated_cca_df.at[previous_mother_id, 'generation_num'] = 2
        updated_cca_df.at[previous_mother_id, 'cell_cycle_stage'] = 'G1'

    updated_cca_df.at[bud_id, 'relative_ID'] = mother_id
    updated_cca_df.at[bud_id, 'generation_num'] = 0
    updated_cca_df.at[bud_id, 'relationship'] = 'bud'
    updated_cca_df.at[bud_id, 'cell_cycle_stage'] = 'S'
    if corrected_frame_i is not None:
        updated_cca_df.at[bud_id, 'corrected_on_frame_i'] = corrected_frame_i

    should_update_mother = update_mother and mother_id in updated_cca_df.index
    if should_update_mother and update_mother_only_if_g1:
        should_update_mother = (
            updated_cca_df.at[mother_id, 'cell_cycle_stage'] == 'G1'
        )
    if should_update_mother:
        updated_cca_df.at[mother_id, 'relative_ID'] = bud_id
        updated_cca_df.at[mother_id, 'cell_cycle_stage'] = 'S'
        if mother_generation_num is not None:
            updated_cca_df.at[mother_id, 'generation_num'] = mother_generation_num
        if mother_relationship is not None:
            updated_cca_df.at[mother_id, 'relationship'] = mother_relationship

    if (
            previous_mother_status is not None
            and previous_mother_id in updated_cca_df.index
    ):
        updated_cca_df.loc[previous_mother_id] = previous_mother_status

    return updated_cca_df


def future_bud_division(
    bud_id: int,
    future_cca_frames,
) -> FutureBudDivisionResult | None:
    """Return first future frame where ``bud_id`` is already in G1."""
    for frame_i, cca_df in future_cca_frames:
        if frame_i == 0:
            continue
        if cca_df is None:
            return None
        if bud_id not in cca_df.index:
            return None

        cell_cycle_stage = cca_df.at[bud_id, 'cell_cycle_stage']
        if cell_cycle_stage == 'G1':
            return FutureBudDivisionResult(
                frame_i=frame_i,
                mother_id=cca_df.at[bud_id, 'relative_ID'],
            )

    return None


def mother_not_g1_before_bud_emergence_frame(
    mother_id: int,
    bud_id: int,
    wrong_bud_id: int,
    past_cca_frames,
) -> int | None:
    """Return first frame without required mother G1 before bud emergence."""
    for frame_i, cca_df in past_cca_frames:
        if bud_id in cca_df.index:
            continue

        if cca_df.at[mother_id, 'cell_cycle_stage'] == 'G1':
            return None

        bud_id_previous_cycle = cca_df.at[mother_id, 'relative_ID']
        if bud_id_previous_cycle != wrong_bud_id:
            return frame_i + 1

        break

    return None


def dead_or_excluded_mother_pairs(
    cca_df: pd.DataFrame,
    acdc_df: pd.DataFrame,
    frame_i: int,
) -> list[MotherBudPair]:
    """Return new bud pairings where the mother is dead or excluded."""
    buds_df = cca_df[
        (cca_df.relationship == 'bud')
        & (cca_df.emerg_frame_i == frame_i)
    ]
    if buds_df.empty:
        return []

    mother_ids = buds_df.relative_ID.to_list()
    mothers_df = acdc_df.loc[mother_ids]
    excluded_df = mothers_df[
        (mothers_df.is_cell_dead > 0)
        | (mothers_df.is_cell_excluded > 0)
    ]

    return [
        MotherBudPair(bud_id=bud_id, mother_id=mother_id)
        for mother_id, bud_id in zip(
            excluded_df.index.to_list(),
            excluded_df.relative_ID.to_list(),
        )
    ]


def evaluate_mother_future_eligibility_frame(
    cca_df: pd.DataFrame | None,
    bud_id: int,
    mother_id: int,
    frame_i: int,
    g1_duration: int,
    last_cca_frame_i: int,
) -> MotherEligibilityFrameResult:
    """Check one future frame for a proposed mother-bud assignment."""
    if cca_df is None:
        return MotherEligibilityFrameResult(
            issue=None,
            stop=True,
            g1_duration=g1_duration,
        )

    if bud_id not in cca_df.index:
        return MotherEligibilityFrameResult(
            issue=None,
            stop=True,
            g1_duration=g1_duration,
        )

    is_still_bud = cca_df.at[bud_id, 'relationship'] == 'bud'
    if not is_still_bud:
        return MotherEligibilityFrameResult(
            issue=None,
            stop=True,
            g1_duration=g1_duration,
        )

    next_g1_duration = g1_duration + 1
    cell_cycle_stage = cca_df.at[mother_id, 'cell_cycle_stage']
    if cell_cycle_stage == 'G1':
        return MotherEligibilityFrameResult(
            issue=None,
            stop=False,
            g1_duration=next_g1_duration,
        )

    issue = MotherEligibilityIssue(
        mother_id=mother_id,
        bud_id=bud_id,
        frame_i=frame_i,
        reason='not_G1_in_the_future',
        blocks_assignment=(
            g1_duration == 1
            and frame_i != last_cca_frame_i
        ),
    )
    return MotherEligibilityFrameResult(
        issue=issue,
        stop=False,
        g1_duration=next_g1_duration,
    )


def evaluate_mother_past_eligibility_frame(
    cca_df: pd.DataFrame,
    bud_id: int,
    mother_id: int,
    frame_i: int,
) -> MotherEligibilityFrameResult:
    """Check one past frame for a proposed mother-bud assignment."""
    is_bud_existing = bud_id in cca_df.index
    is_mother_existing = mother_id in cca_df.index

    if not is_mother_existing:
        return MotherEligibilityFrameResult(issue=None, stop=True)

    cell_cycle_stage = cca_df.at[mother_id, 'cell_cycle_stage']
    if cell_cycle_stage != 'G1' and is_bud_existing:
        issue = MotherEligibilityIssue(
            mother_id=mother_id,
            bud_id=bud_id,
            frame_i=frame_i,
            reason='not_G1_in_the_past',
            blocks_assignment=True,
        )
        return MotherEligibilityFrameResult(issue=issue, stop=True)

    if not is_bud_existing:
        issue = None
        if cell_cycle_stage != 'G1':
            issue = MotherEligibilityIssue(
                mother_id=mother_id,
                bud_id=bud_id,
                frame_i=frame_i,
                reason='single_frame_G1_duration',
                blocks_assignment=True,
            )
        return MotherEligibilityFrameResult(issue=issue, stop=True)

    return MotherEligibilityFrameResult(issue=None, stop=False)


def apply_mother_bud_pairing(
    cca_df: pd.DataFrame,
    bud_id: int,
    mother_id: int,
    corrected_frame_i: int,
    *,
    set_mother_s_if_g1: bool = True,
) -> pd.DataFrame:
    """Return ``cca_df`` with reciprocal mother-bud IDs corrected."""
    updated_cca_df = cca_df.copy()
    updated_cca_df.at[bud_id, 'relative_ID'] = mother_id
    updated_cca_df.at[mother_id, 'relative_ID'] = bud_id
    updated_cca_df.at[bud_id, 'corrected_on_frame_i'] = corrected_frame_i
    updated_cca_df.at[mother_id, 'corrected_on_frame_i'] = corrected_frame_i

    if (
            set_mother_s_if_g1
            and updated_cca_df.at[mother_id, 'cell_cycle_stage'] == 'G1'
    ):
        updated_cca_df.at[mother_id, 'cell_cycle_stage'] = 'S'

    return updated_cca_df


def wrong_bud_id_for_mother(cca_df: pd.DataFrame, mother_id: int) -> int | None:
    """Return mother's current bud ID if it is a bud row in ``cca_df``."""
    try:
        relative_id = cca_df.at[mother_id, 'relative_ID']
    except Exception:
        return None
    if relative_id not in cca_df.index:
        return None
    if cca_df.at[relative_id, 'relationship'] != 'bud':
        return None
    return int(relative_id)


def mother_status_before_wrong_bud(
    mother_id: int,
    wrong_bud_id: int,
    past_cca_frames,
    base_status: pd.Series,
) -> pd.Series:
    """Return mother's status from before ``wrong_bud_id`` emerged."""
    for cca_df in past_cca_frames:
        if cca_df is None:
            continue
        if wrong_bud_id not in cca_df.index:
            return cca_df.loc[mother_id].copy()
    return base_status.copy()


def restore_mother_status_for_wrong_bud_frame(
    cca_df: pd.DataFrame,
    mother_id: int,
    wrong_bud_id: int,
    mother_status: pd.Series,
    corrected_frame_i: int,
) -> CcaMotherStatusRestoreResult:
    """Restore mother status if ``wrong_bud_id`` is present in ``cca_df``."""
    if wrong_bud_id not in cca_df.index:
        return CcaMotherStatusRestoreResult(cca_df=cca_df, restored=False)

    updated_cca_df = cca_df.copy()
    updated_cca_df.loc[mother_id] = mother_status
    updated_cca_df.at[mother_id, 'corrected_on_frame_i'] = corrected_frame_i
    return CcaMotherStatusRestoreResult(cca_df=updated_cca_df, restored=True)


def restore_mother_status_until_g1(
    cca_df: pd.DataFrame,
    mother_id: int,
    mother_status: pd.Series,
    corrected_frame_i: int,
) -> CcaMotherStatusRestoreResult:
    """Restore mother status unless the mother is already back in G1."""
    if cca_df.at[mother_id, 'cell_cycle_stage'] == 'G1':
        return CcaMotherStatusRestoreResult(cca_df=cca_df, restored=False)

    updated_cca_df = cca_df.copy()
    updated_cca_df.loc[mother_id] = mother_status
    updated_cca_df.at[mother_id, 'corrected_on_frame_i'] = corrected_frame_i
    return CcaMotherStatusRestoreResult(cca_df=updated_cca_df, restored=True)


def mark_will_divide_frame(
    cca_df: pd.DataFrame,
    cell_id: int,
    relative_id: int,
    generation_num: int | None = None,
) -> CcaWillDivideFrameResult:
    """Mark ``cell_id`` and ``relative_id`` as will-divide for one frame."""
    if cell_id not in cca_df.index:
        return CcaWillDivideFrameResult(
            cca_df=cca_df,
            generation_num=generation_num,
            should_store=False,
            stop=True,
        )

    if generation_num is None:
        generation_num = cca_df.at[cell_id, 'generation_num']
    if cca_df.at[cell_id, 'generation_num'] != generation_num:
        return CcaWillDivideFrameResult(
            cca_df=cca_df,
            generation_num=generation_num,
            should_store=False,
            stop=True,
        )

    updated_cca_df = cca_df.copy()
    updated_cca_df.at[cell_id, 'will_divide'] = 1
    updated_cca_df.at[relative_id, 'will_divide'] = 1
    return CcaWillDivideFrameResult(
        cca_df=updated_cca_df,
        generation_num=generation_num,
        should_store=True,
        stop=False,
    )


def division_undo_blocking_frame(
    cell_id: int,
    relative_id: int,
    current_frame_i: int,
    current_cca_df: pd.DataFrame,
    future_cca_frames=(),
    past_cca_frames=(),
) -> int | None:
    """Return frame index blocking division undo, or ``None`` if allowed."""
    if current_cca_df.at[relative_id, 'cell_cycle_stage'] == 'S':
        return current_frame_i

    for frame_i, cca_df in future_cca_frames:
        if cca_df is None:
            break
        if relative_id not in cca_df.index:
            continue
        if cca_df.at[relative_id, 'cell_cycle_stage'] == 'S':
            return frame_i

    for frame_i, cca_df in past_cca_frames:
        if cca_df is None:
            break
        if cell_id not in cca_df.index or relative_id not in cca_df.index:
            break
        if cca_df.at[cell_id, 'cell_cycle_stage'] == 'S':
            break
        if cca_df.at[relative_id, 'cell_cycle_stage'] == 'S':
            return frame_i

    return None


def annotate_division(
    cca_df: pd.DataFrame,
    cell_id: int,
    relative_id: int,
    frame_i: int,
) -> bool:
    """Annotate a division between two related cells in ``cca_df``."""
    cca_df.at[cell_id, 'cell_cycle_stage'] = 'G1'
    cca_df.at[relative_id, 'cell_cycle_stage'] = 'G1'

    if frame_i > 0:
        cell_generation = cca_df.at[cell_id, 'generation_num']
        cca_df.at[cell_id, 'generation_num'] += 1
        cca_df.at[cell_id, 'division_frame_i'] = frame_i
        relative_generation = cca_df.at[relative_id, 'generation_num']
        cca_df.at[relative_id, 'generation_num'] = relative_generation + 1
        cca_df.at[relative_id, 'division_frame_i'] = frame_i
        if cell_generation < relative_generation:
            cca_df.at[cell_id, 'relationship'] = 'mother'
        else:
            cca_df.at[relative_id, 'relationship'] = 'mother'
    else:
        cca_df.at[cell_id, 'generation_num'] = 2
        cca_df.at[relative_id, 'generation_num'] = 2
        cca_df.at[cell_id, 'division_frame_i'] = -1
        cca_df.at[relative_id, 'division_frame_i'] = -1
        cca_df.at[cell_id, 'relationship'] = 'mother'
        cca_df.at[relative_id, 'relationship'] = 'mother'

    return True


def undo_division_annotation(
    cca_df: pd.DataFrame,
    cell_id: int,
    relative_id: int,
) -> bool:
    """Undo a division annotation between two related cells in ``cca_df``."""
    cca_df.at[cell_id, 'cell_cycle_stage'] = 'S'
    cell_generation = cca_df.at[cell_id, 'generation_num']
    cca_df.at[cell_id, 'generation_num'] -= 1
    cca_df.at[cell_id, 'division_frame_i'] = -1

    cca_df.at[relative_id, 'cell_cycle_stage'] = 'S'
    relative_generation = cca_df.at[relative_id, 'generation_num']
    cca_df.at[relative_id, 'generation_num'] -= 1
    cca_df.at[relative_id, 'division_frame_i'] = -1

    if cell_generation < relative_generation:
        cca_df.at[cell_id, 'relationship'] = 'bud'
    else:
        cca_df.at[relative_id, 'relationship'] = 'bud'

    cca_df.at[cell_id, 'will_divide'] = 0
    cca_df.at[relative_id, 'will_divide'] = 0
    return True


def undo_bud_mother_assignment(
    cca_df: pd.DataFrame,
    cell_id: int,
) -> bool:
    """Undo a bud/mother assignment for ``cell_id`` and its relative if present."""
    relative_id = cca_df.at[cell_id, 'relative_ID']
    cell_cycle_stage = cca_df.at[cell_id, 'cell_cycle_stage']
    if cell_cycle_stage == 'G1':
        return False

    cca_df.at[cell_id, 'relative_ID'] = -1
    cca_df.at[cell_id, 'generation_num'] = 2
    cca_df.at[cell_id, 'cell_cycle_stage'] = 'G1'
    cca_df.at[cell_id, 'relationship'] = 'mother'

    if relative_id in cca_df.index:
        cca_df.at[relative_id, 'relative_ID'] = -1
        cca_df.at[relative_id, 'generation_num'] = 2
        cca_df.at[relative_id, 'cell_cycle_stage'] = 'G1'
        cca_df.at[relative_id, 'relationship'] = 'mother'

    return True


def toggle_history_knowledge(
    cca_df: pd.DataFrame,
    cell_id: int,
    *,
    status_when_emerged: pd.Series | None = None,
) -> bool:
    """Toggle whether ``cell_id`` has known history in ``cca_df``."""
    is_history_known = cca_df.at[cell_id, 'is_history_known']
    if is_history_known:
        cca_df.at[cell_id, 'is_history_known'] = False
        cca_df.at[cell_id, 'cell_cycle_stage'] = 'G1'
        cca_df.at[cell_id, 'generation_num'] += 2
        cca_df.at[cell_id, 'emerg_frame_i'] = -1
        cca_df.at[cell_id, 'relative_ID'] = -1
        cca_df.at[cell_id, 'relationship'] = 'mother'
    else:
        if status_when_emerged is None:
            raise ValueError(
                'status_when_emerged is required to restore known history'
            )
        cca_df.loc[cell_id] = status_when_emerged

    return True


from .cell_cycle_history import (  # noqa: E402
    CcaHistoryKnowledgePropagationResult,
    apply_history_knowledge_to_frame,
    known_history_status_for_bud,
    propagate_history_knowledge,
)
from .cell_cycle_divisions import (  # noqa: E402
    CcaBudMotherAssignmentPropagationResult,
    CcaBudMotherChangeEligibilityResult,
    CcaManualDivisionPropagationResult,
    CcaMotherAssignmentEligibilityResult,
    CcaMotherBudPairingsResult,
    CcaSwapMothersEligibilityResult,
    CcaSwapMothersFutureDivisionResult,
    CcaSwapMothersPairingPlan,
    CcaSwapMothersPastRestoreResult,
    CcaSwapMothersPropagationResult,
    CcaWillDividePropagationResult,
    apply_mother_bud_pairings,
    previous_relative_status_before_bud_emergence,
    bud_mother_change_eligibility,
    mother_assignment_eligibility,
    propagate_bud_mother_assignment,
    propagate_manual_division_annotation,
    propagate_swap_mothers_assignment,
    propagate_swap_mothers_future_division,
    propagate_will_divide,
    restore_swap_mothers_past_status,
    swap_mothers_eligibility,
    swap_mothers_pairing_plan,
)
