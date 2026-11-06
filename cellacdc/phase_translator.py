"""Ordered, per-cell translation of time point boundaries into phases.

Intervals include their starting boundary and exclude their ending boundary.
Empty phase names leave an interval unassigned. Checked annotations are
cell-wide requirements, not boundaries. Each boundary must occur exactly once,
in the configured order; an ineligible rule falls through to the next rule.

Rules are dictionaries with these fields:

- ``name``: unique rule name, used in the configuration and warnings.
- ``predictor``: custom annotation predictor settings; type ``none`` means
  no formula requirement. Otherwise its boolean results are consolidated
  across each cell's history using the predictor's All/None/Most setting.
- ``timepoints``: ordered names of boolean ``timepoint_<name>`` columns.
- ``annotations``: column names that must each be truthy in at least one
  row of the cell; they do not define interval boundaries.
- ``phases``: one name per interval, so N boundaries require N + 1 names.
  For boundaries at frames 2 and 5, names ``['G1', 'S', 'G2']`` assign
  G1 before frame 2, S at frames 2 through 4, and G2 from frame 5 onward.

Evaluation returns a new position table with a shared ``phase`` string column
and boolean ``phase_<name>`` columns. It does not modify or save the input table.
"""
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .custom_annotation_predictors import CustomAnnotationPredictor
from .timepoint_predictors import CellTable, PREDICTOR_NONE, TIMEPOINT_PREFIX


def validate_rules(rules: list[dict[str, Any]]) -> None:
    """Validate serialized rule structure and reject input/output name clashes.

    An empty rule list is valid for a saved, unconfigured translator, but cannot
    be evaluated. Input availability and per-cell boundary order are checked
    against the measurements later, during evaluation.
    """
    if not isinstance(rules, list):
        raise ValueError('Phase translator rules must be a list.')
    names = set()
    for rule in rules:
        if not isinstance(rule, dict):
            raise ValueError('Invalid phase translator rule.')
        name = rule.get('name')
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError('Phase rule names must be nonempty and unique.')
        names.add(name)
        CustomAnnotationPredictor.validate(rule.get('predictor'))
        for key in ('timepoints', 'annotations'):
            values = rule.get(key)
            if (
                not isinstance(values, list)
                or any(not isinstance(value, str) or not value for value in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError(f'Invalid {key} in phase rule "{name}".')
        if any(not col.startswith(TIMEPOINT_PREFIX) for col in rule['timepoints']):
            raise ValueError(f'Invalid time point boundary in phase rule "{name}".')
        phases = rule.get('phases')
        if (
            not isinstance(phases, list)
            or len(phases) != len(rule['timepoints']) + 1
            or any(not isinstance(phase, str) for phase in phases)
            or not any(phase.strip() for phase in phases)
            or any(phase != phase.strip() for phase in phases)
        ):
            raise ValueError(
                f'Enter at least one phase name in rule "{name}", with one '
                'interval before, between and after its boundaries.'
            )
    inputs = {
        column for rule in rules
        for column in rule['timepoints'] + rule['annotations']
    }
    if inputs.intersection(output_columns(rules)):
        raise ValueError('Phase output columns cannot be required inputs.')


def output_columns(rules: list[dict[str, Any]]) -> set[str]:
    """Return output names, sharing a boolean column for repeated phase names."""
    return {'phase'} | {
        f'phase_{phase}' for rule in rules for phase in rule['phases'] if phase
    }


def existing_output_columns(df: pd.DataFrame) -> set[str]:
    """Infer translator outputs when re-evaluating a previously exported CSV.

    Runtime column ownership is not stored in CSV files. Recognize a boolean
    column only when it exactly matches the corresponding string phase values,
    and require at least one such column before recognizing ``phase`` itself.
    This is value-based inference, not proof of the columns' origin.
    """
    if 'phase' not in df:
        return set()
    values = df['phase'].fillna('')
    if not values.map(lambda value: isinstance(value, str)).all():
        return set()
    columns = {'phase'}
    for column in df.columns:
        if column.startswith('phase_') and pd.api.types.is_bool_dtype(df[column]):
            phase = column[len('phase_'):]
            if phase and df[column].equals(values.eq(phase).rename(column)):
                columns.add(column)
    return columns if len(columns) > 1 else set()


@dataclass
class PhaseEvaluation:
    """Evaluation result for the viewer to commit after successful completion.

    ``table`` retains the source layout and all non-translator measurements.
    ``columns`` tracks translator-owned outputs for subsequent evaluations.
    ``warnings`` reports skipped rules with missing columns and cells left
    unmatched because of duplicate boundaries; these do not abort evaluation.
    """
    table: pd.DataFrame
    columns: set[str]
    warnings: list[str]


def evaluate(
        df: pd.DataFrame, rules: list[dict[str, Any]],
        IDs: Iterable[int] | None = None, SizeT: int | None = None,
        owned_columns: Iterable[str] = ()
    ) -> PhaseEvaluation:
    """Assign phases using the first matching rule for each selected cell.

    Parameters
    ----------
    df
        Entire measurements table of one position, including all frames and
        cells. Accepts a (frame_i, Cell_ID) index or those columns directly.
        The original row order and index are preserved in the returned table.
    rules
        Nonempty rule list in descending priority (see the module docstring).
    IDs
        Cells whose phases should be recalculated; None selects all cells.
        The full df remains available to predictors that use other-cell inputs.
    SizeT
        Optional video length passed to the predictor, which clips its
        annotated frames to [0, SizeT). It does not truncate the output table.
    owned_columns
        Previously generated output names tracked by the viewer for this
        position. These may be replaced or removed, unlike unrelated columns.

    Returns
    -------
    PhaseEvaluation
        Updated table, retained output names and nonfatal warnings. Unmatched
        selected cells have empty phases; unselected cells keep their phases.
        A matching rule owns the whole cell, including unnamed intervals:
        blank phase names do not fall through to a lower-priority rule.

    Raises
    ------
    ValueError
        Invalid rules or target cells, conflicting output column names, or
        predictor evaluation failures. No changes are made to df on failure.
    """
    validate_rules(rules)
    if not rules:
        raise ValueError('Add at least one phase rule before applying evaluation.')
    # Flatten for row lookup without losing positional alignment with df.
    flat = CellTable.flatten(df)
    available_ids = set(flat['Cell_ID'].unique())
    target_ids = available_ids if IDs is None else set(IDs)
    if not target_ids:
        raise ValueError('No cells with measurement rows were selected.')
    missing_ids = target_ids.difference(available_ids)
    if missing_ids:
        raise ValueError(
            'No measurement rows for selected cells: '
            + ', '.join(str(ID) for ID in sorted(missing_ids)) + '.'
        )
    target = flat['Cell_ID'].isin(target_ids).to_numpy()
    columns = output_columns(rules)
    owned = set(owned_columns) | existing_output_columns(df)
    result = df.copy()
    if 'phase' not in result:
        result['phase'] = ''
    result['phase'] = result['phase'].fillna('').astype(object)
    phase_values = result['phase'].to_numpy(copy=True)
    # Reset only the requested cells. Existing phases on other cells also
    # determine which old boolean outputs must survive this evaluation.
    phase_values[target] = ''
    retained = columns | {
        f'phase_{phase}' for phase in phase_values if phase
    }
    collisions = retained.intersection(df.columns).difference(owned)
    if collisions:
        raise ValueError(
            'Phase outputs would overwrite existing measurements: '
            + ', '.join(sorted(collisions))
        )
    pending = target_ids.copy()
    # Positional row indices let assignments preserve even an unsorted index.
    cell_rows = flat.groupby('Cell_ID', sort=False).indices
    frames = flat['frame_i'].to_numpy()
    duplicate_cells = set()
    warnings = []
    # Iterate over each rule and apply it to the eligible cells, breaking
    # early if all Cells have had a phase assigned, or if rules are exhausted.
    for rule in rules:
        if not pending:
            break
        required = rule['timepoints'] + rule['annotations']
        missing = set(required).difference(flat.columns)
        if missing:
            warnings.append(
                f'Rule "{rule["name"]}" skipped: missing inputs '
                + ', '.join(sorted(missing)) + '.'
            )
            continue
        eligible = {}
        # Iterate over each cell in the subset of pending cells to determine eligibility.
        for ID, cell in CellTable.iter_cells(CellTable.subset(flat, pending)):
            boundaries = []
            duplicate = False # check for duplicate time points within one cell
            for column in rule['timepoints']:
                marked_frames = cell.loc[
                    cell[column].fillna(False).astype(bool), 'frame_i'
                ]
                if len(marked_frames) > 1:
                    duplicate = True
                    duplicate_cells.add(ID)
                if len(marked_frames) == 1:
                    boundaries.append(int(marked_frames.iloc[0]))
            # Missing or repeated boundaries are ambiguous. Do not sort the
            # configured sequence: reversing it would change interval meanings.
            if duplicate or len(boundaries) != len(rule['timepoints']):
                continue
            if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
                continue # skip if time points are not in correct order
            if not all(cell[col].fillna(False).astype(bool).any()
                       for col in rule['annotations']):
                continue # skip if required annotations are not present
            eligible[ID] = boundaries # store the valid boundaries for this cell
        if not eligible:
            continue
        predictor = rule['predictor']
        # Restrict target cells, not the source table: other-cell formulas
        # still need access to measurements outside the eligible set.
        matches = (
            set(eligible) if predictor['type'] == PREDICTOR_NONE else
            set(CustomAnnotationPredictor.predict(
                df, predictor, IDs=eligible, SizeT=SizeT
            ))
        ) # further restrict matches to eligible cells based on the predicotr 
        # (defined formula)
        for ID in matches:
            # These are positional indices into the full position table, not
            # frame numbers. frames[rows] gives this cell's actual frame numbers.
            rows = cell_rows[ID]
            # eligible[ID] contains this cell's strictly increasing boundaries.
            # searchsorted returns one interval index per frame by counting
            # boundaries <= that frame. With boundaries [2, 5], frames
            # [0, 1, 2, 3, 4, 5, 6] map to intervals [0, 0, 1, 1, 1, 2, 2].
            # side='right' includes a boundary frame in the new interval:
            # interval 0 is before 2, interval 1 is [2, 5), and 2 is from 5 on.
            intervals = np.searchsorted(
                eligible[ID], frames[rows], side='right'
            )
            # Use those indices to look up the configured phase names, e.g.
            # ['G1', 'S', 'G2'] becomes ['G1', 'G1', 'S', 'S', 'S', 'G2', 'G2'].
            # Assign only this cell's rows; an empty name leaves its interval
            # unassigned. dtype=object holds strings without fixed-width limits.
            phase_values[rows] = np.asarray(rule['phases'], dtype=object)[intervals]
        # Remove cells that have been successfully matched from the pending set.
        pending.difference_update(matches)
    unmatched_duplicates = duplicate_cells.intersection(pending)
    # A later fallback can resolve a duplicate-boundary cell without warning.
    if unmatched_duplicates:
        warnings.append(
            'No phase rule could be applied to cells with duplicate time points: '
            + ', '.join(str(ID) for ID in sorted(unmatched_duplicates)) + '.'
        )
    result['phase'] = phase_values
    # Keep outputs still used by untouched cells, but remove obsolete outputs
    # after a batch evaluation or when their last assigned cell is re-evaluated.
    result = result.drop(columns=sorted(owned.difference(retained)), errors='ignore')
    for column in sorted(retained.difference({'phase'})):
        result[column] = result['phase'].eq(column[len('phase_'):])
    return PhaseEvaluation(result, retained, warnings)
