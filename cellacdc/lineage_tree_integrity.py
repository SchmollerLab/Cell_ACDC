"""Consistency checks for the lineage tree columns of acdc_output tables.

Older versions of the normal-division lineage tree (``CellACDC_normal_division``
tracker, GUI lineage tree mode) could save inconsistent lineage annotations.
The typical case is a mother cell that kept its ID after division and was
then annotated as its own parent (``parent_ID_tree == Cell_ID``). Propagating
such a loop made pandas lookups return a Series instead of a scalar, and the
stringified Series (e.g. ``'Cell_ID\\n64    8.0\\n64    8.0\\nName:
root_ID_tree, dtype: float64'``) ended up in the CSV. Loading the table then
fails with ``ValueError: could not convert string to float``.

:func:`check_lineage_tree` finds such problems and returns a
:class:`LineageTreeReport`. Nothing is modified. Entry points:

- ``check_lineage_tree(acdc_df)``: a loaded table (indexed by ``frame_i`` /
  ``Cell_ID`` or with them as columns).
- ``check_lineage_tree_file(csv_path)``: reads the CSV with plain pandas, so
  corrupted values are reported instead of raising.
- ``check_lineage_tree_files(path)``: every ``*acdc_output*.csv`` of a folder
  tree (or a list of files/folders). The raw tracker tables
  ``*acdc_output_auto.csv`` are skipped unless ``include_auto=True`` or
  given explicitly as files.
- Command line: ``python -m cellacdc.lineage_tree_integrity <path> [...]
  [--out report.txt|report.csv]``.

Expected annotations (normal division, one node per ``Cell_ID``):

- One row per ``frame_i`` x ``Cell_ID``; lineage columns hold integers.
- The lineage values of a cell (``parent_ID_tree``, ``generation_num_tree``,
  ``root_ID_tree``, sisters and, in pure lineage tree tables,
  ``emerg_frame_i`` / ``division_frame_i``) are constant over its frames.
- Unknown history (``parent_ID_tree == -1``): ``generation_num_tree == 1``,
  ``root_ID_tree == Cell_ID``, no sisters. ``root_ID_tree == -1`` also means
  "the cell is its own root" (default of ``base_cca_tree_dict``, kept for
  root cells by ``myutils.df_ctc_to_acdc_df``, e.g. Trackastra output).
- Known parent: the parent is another cell that exists before the daughter
  appears (and normally disappears when it divides);
  ``generation_num_tree`` is the parent's + 1 and ``root_ID_tree`` the
  parent's root. Following parents never loops back to a cell.
- Sisters (``sister_ID_tree`` and ``sister_ID_tree_<i>``, -1 = none) are
  exactly the other cells with the same parent. A sister may be left out in
  the frames after it disappeared (the lineage tree code does that), which
  is not reported. Each parent has 2 daughters
  (``max_daughters``) appearing in the same frame.
- Pure lineage tree tables (no budding cell cycle columns ``cell_cycle_stage``
  / ``relationship`` / ``relative_ID``): ``is_history_known`` is True exactly
  when the parent is known, ``emerg_frame_i`` and ``division_frame_i`` equal
  the frame the cell appears (-1 is accepted for cells with unknown history).
  In budding tables these columns follow the cell cycle annotation and are
  checked by ``core.CcaIntegrityChecker`` instead.

Frames where no cell has lineage values are reported once (``info``) and
otherwise ignored. Rows with ``parent_ID_tree == generation_num_tree == 0``
hold the GUI fill value of ``guiWin.store_data`` (cell added after the
lineage was annotated) and count as not annotated, like the GUI does. Rows
whose ``parent_ID_tree`` is empty or invalid are left out of the structural
checks; empty or invalid generation/root/sister values only skip the checks
of those values. Tables whose ``Cell_ID_tree`` differs from
``Cell_ID`` (lineage tree built from budding annotations by
``core.LineageTree``, one node per cell cycle) only get the value checks.

Issues are grouped per cell and check (one issue covers consecutive frames
with the same values). ``first_frame_i`` / ``last_frame_i`` of the issues are
0-based ``frame_i`` values as in the table; the text report uses the 1-based
"frame n." of the GUI.
"""

import argparse
import os
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .timepoint_predictors import parse_flags

TREE_COLS = (
    'generation_num_tree', 'parent_ID_tree', 'root_ID_tree', 'sister_ID_tree'
)
OTHER_INT_COLS = (
    'Cell_ID_tree', 'generation_num', 'emerg_frame_i', 'division_frame_i'
)
BUDDING_CCA_COLS = ('cell_cycle_stage', 'relationship', 'relative_ID')

SEVERITIES = ('error', 'warning', 'info')

# check name: (severity, description, hint)
CHECKS = {
    'missing_index_values': (
        'error',
        'Rows with an empty or non-numeric frame_i or Cell_ID.',
        'These rows were ignored by the other checks.'
    ),
    'duplicated_rows': (
        'error',
        'Several rows with the same frame_i and Cell_ID.',
        'Only the first row of each frame_i/Cell_ID was checked.'
    ),
    'invalid_values': (
        'error',
        'Non-numeric values in a lineage column (e.g. a stringified pandas '
        'Series written by an old lineage tree bug). The table cannot be '
        'loaded.',
        'Restore the values from a backup or re-annotate the lineage of '
        'these cells.'
    ),
    'non_integer_values': (
        'error',
        'Non-integer numbers in a lineage column (IDs, generation numbers '
        'and frames are integers).',
        ''
    ),
    'unreadable_file': (
        'error',
        'The file cannot be read or checked (empty, locked, not downloaded '
        'or not an acdc_output table).',
        'Check that the file opens and has frame_i and Cell_ID columns.'
    ),
    'fill_value_rows': (
        'warning',
        'Lineage values are 0, the GUI fill value for a cell added to the '
        'frame after the lineage was annotated (never annotated).',
        'Annotate the lineage of these cells in the GUI.'
    ),
    'missing_values': (
        'warning',
        'Empty lineage values in a frame where other cells are annotated.',
        'These rows were ignored by the structural checks.'
    ),
    'unannotated_frames': (
        'info',
        'Frames where no cell has lineage values (not annotated yet).',
        ''
    ),
    'budding_lineage_tree': (
        'info',
        'Cell_ID_tree differs from Cell_ID: lineage tree built from budding '
        'cell cycle annotations. Only the values were checked.',
        ''
    ),
    'value_changes_over_time': (
        'warning',
        'A lineage value of a cell changes between frames (it should be '
        'constant over the frames of the cell).',
        'Usually a lineage correction that was not propagated to all '
        'frames, a cell ID that was reused for another cell, or a cell that '
        'disappeared for some frames (segmentation gap) and got a new '
        'lineage (unknown history) when it reappeared.'
    ),
    'self_parent': (
        'error',
        'A cell is its own parent.',
        'Usually the mother kept its ID after dividing. Give the daughter '
        'that kept the mother ID a new ID, or set the correct parent.'
    ),
    'lineage_cycle': (
        'error',
        'Following the parents of these cells loops back to them.',
        ''
    ),
    'parent_not_found': (
        'error',
        'The parent ID does not exist in any frame of the table.',
        ''
    ),
    'parent_after_daughter': (
        'error',
        'The parent appears only in or after the frame where the daughter '
        'appears.',
        ''
    ),
    'parent_present_after_division': (
        'warning',
        'The parent still exists in or after the frame where the daughter '
        'appears (with normal division the mother disappears).',
        'Often the mother kept its ID after dividing, or the parent is '
        'wrong.'
    ),
    'parent_gap': (
        'warning',
        'The parent disappears more than one frame before the daughter '
        'appears.',
        ''
    ),
    'generation_num_tree_mismatch': (
        'error',
        'generation_num_tree is not the parent\'s + 1 (1 for cells with '
        'unknown history).',
        ''
    ),
    'root_ID_tree_mismatch': (
        'error',
        'root_ID_tree is not the parent\'s root (the cell itself for cells '
        'with unknown history).',
        ''
    ),
    'invalid_sister': (
        'error',
        'A listed sister is the cell itself or its parent, does not exist, '
        'or has a different parent (cells with unknown history have no '
        'sisters).',
        ''
    ),
    'missing_sister': (
        'warning',
        'Another daughter of the same parent is not listed as sister '
        '(sisters may be left out in frames after they disappeared).',
        ''
    ),
    'single_daughter': (
        'warning',
        'The parent has only one daughter.',
        'Fine if the other daughter left the field of view, otherwise a '
        'parent assignment is missing or wrong.'
    ),
    'too_many_daughters': (
        'warning',
        'The parent has more daughters than expected (max_daughters).',
        ''
    ),
    'daughters_born_in_different_frames': (
        'warning',
        'The daughters of the parent appear in different frames.',
        ''
    ),
    'is_history_known_mismatch': (
        'info',
        'is_history_known does not match parent_ID_tree (True exactly when '
        'the parent is known).',
        'Seen in tables saved by the GUI: True in the frame where the cell '
        'appears (and in frames where other cells divide), False otherwise.'
    ),
    'zero_sister_IDs': (
        'info',
        'Sister columns contain 0, the GUI fill value for sister columns '
        'added later. Treated as no sister.',
        ''
    ),
    'emerg_frame_i_mismatch': (
        'warning',
        'emerg_frame_i is not the frame where the cell appears (-1 is '
        'accepted for cells with unknown history).',
        ''
    ),
    'division_frame_i_mismatch': (
        'warning',
        'division_frame_i differs from emerg_frame_i.',
        ''
    ),
}


@dataclass
class LineageTreeIssue:
    """One inconsistency, usually of one cell over consecutive frames.

    ``first_frame_i`` / ``last_frame_i`` are 0-based (``frame_i``). ``Cell_ID``
    is None for issues that do not belong to a single cell.
    """
    check: str
    severity: str
    message: str
    Cell_ID: int | None = None
    first_frame_i: int | None = None
    last_frame_i: int | None = None
    n_rows: int | None = None
    column: str | None = None
    value: str | None = None
    expected: str | None = None


class LineageTreeReport:
    """Result of :func:`check_lineage_tree`.

    Attributes
    ----------
    issues : list of LineageTreeIssue
        Sorted by severity, check (in ``CHECKS`` order), Cell_ID and frame.
    source : str or None
        File path of the checked table, if any.
    n_rows, n_cells : int
        Size of the checked table.
    frames : tuple of int or None
        First and last ``frame_i`` of the table.
    """

    ISSUE_COLUMNS = list(LineageTreeIssue.__dataclass_fields__)

    def __init__(
            self, issues, source=None, n_rows=0, n_cells=0, frames=None
        ):
        order = {name: i for i, name in enumerate(CHECKS)}
        severity_order = {name: i for i, name in enumerate(SEVERITIES)}
        self.issues = sorted(
            issues, key=lambda issue: (
                severity_order[issue.severity],
                order.get(issue.check, len(order)),
                -1 if issue.Cell_ID is None else issue.Cell_ID,
                -1 if issue.first_frame_i is None else issue.first_frame_i,
                issue.column or '',
            )
        )
        self.source = source
        self.n_rows = n_rows
        self.n_cells = n_cells
        self.frames = frames

    def __len__(self):
        return len(self.issues)

    def __repr__(self):
        counts = self.counts()
        return (
            f'<LineageTreeReport {self.source or ""} '
            f'errors={counts["error"]} warnings={counts["warning"]} '
            f'info={counts["info"]}>'
        )

    def __str__(self):
        return self.to_text()

    @property
    def has_errors(self):
        return any(issue.severity == 'error' for issue in self.issues)

    @property
    def ok(self):
        """True when there are no errors and no warnings."""
        return not any(
            issue.severity in ('error', 'warning') for issue in self.issues
        )

    def counts(self):
        """Number of issues per severity."""
        counts = dict.fromkeys(SEVERITIES, 0)
        for issue in self.issues:
            counts[issue.severity] += 1
        return counts

    def filter(self, severity=None, check=None, Cell_ID=None):
        """Issues matching all given criteria (each a value or a list)."""
        def _match(value, wanted):
            if wanted is None:
                return True
            if isinstance(wanted, (str, int, np.integer)):
                return value == wanted
            return value in wanted
        return [
            issue for issue in self.issues
            if _match(issue.severity, severity)
            and _match(issue.check, check)
            and _match(issue.Cell_ID, Cell_ID)
        ]

    def cell_IDs(self, severity=('error', 'warning')):
        """Sorted IDs of the cells with issues of the given severities."""
        return sorted({
            issue.Cell_ID for issue in self.filter(severity=severity)
            if issue.Cell_ID is not None
        })

    def to_dataframe(self):
        """Issues as a DataFrame (one row per issue)."""
        df = pd.DataFrame(
            [asdict(issue) for issue in self.issues],
            columns=self.ISSUE_COLUMNS
        )
        for col in ('Cell_ID', 'first_frame_i', 'last_frame_i', 'n_rows'):
            df[col] = df[col].astype('Int64')
        if self.source is not None:
            df.insert(0, 'source', self.source)
        return df

    def to_text(
            self, max_issues_per_check=20,
            more_hint='to_text(None) or to_dataframe()'
        ):
        """Human-readable report (frames as 1-based "frame n.").

        Parameters
        ----------
        max_issues_per_check : int or None
            Issues listed per check; None lists all of them.
        more_hint : str
            How to see the issues left out, shown after their number.
        """
        lines = []
        title = 'Lineage tree check'
        if self.source:
            title = f'{title}: {self.source}'
        lines.append(title)
        lines.append('=' * min(len(title), 79))
        size = f'{self.n_rows} rows, {self.n_cells} cells'
        if self.frames is not None:
            size = (
                f'{size}, frame n. {self.frames[0] + 1}-{self.frames[1] + 1}'
            )
        lines.append(size)
        counts = self.counts()
        if not self.issues:
            lines.append('No inconsistencies found.')
            return '\n'.join(lines)
        lines.append(
            f'Found {counts["error"]} error(s), {counts["warning"]} '
            f'warning(s), {counts["info"]} info(s).'
        )
        affected = self.cell_IDs()
        if affected:
            lines.append(
                f'Cells with errors or warnings ({len(affected)}): '
                f'{_format_IDs(affected)}'
            )
        for severity in SEVERITIES:
            issues = [i for i in self.issues if i.severity == severity]
            if not issues:
                continue
            lines.append('')
            lines.append(f'{severity.upper()}S')
            checks = list(dict.fromkeys(issue.check for issue in issues))
            for check in checks:
                check_issues = [i for i in issues if i.check == check]
                _, description, hint = CHECKS.get(check, (None, '', ''))
                lines.append(f'  {check} ({len(check_issues)}): {description}')
                if hint:
                    lines.append(f'    Hint: {hint}')
                shown = check_issues
                if max_issues_per_check is not None:
                    shown = check_issues[:max_issues_per_check]
                for issue in shown:
                    lines.append(f'    - {issue.message}')
                hidden = len(check_issues) - len(shown)
                if hidden > 0:
                    lines.append(
                        f'    ... and {hidden} more ({more_hint})'
                    )
        return '\n'.join(lines)

    def save(self, path, max_issues_per_check=None):
        """Write the issues as CSV (``.csv``) or the text report."""
        if str(path).lower().endswith('.csv'):
            self.to_dataframe().to_csv(path, index=False)
        else:
            with open(path, 'w', encoding='utf-8') as file:
                file.write(self.to_text(max_issues_per_check))
                file.write('\n')


def _format_IDs(IDs, max_shown=30):
    IDs = list(IDs)
    text = ', '.join(str(ID) for ID in IDs[:max_shown])
    if len(IDs) > max_shown:
        text = f'{text}, ... ({len(IDs) - max_shown} more)'
    return text


def _frames_text(first, last):
    if first is None:
        return ''
    if last is None or last == first:
        return f'frame n. {first + 1}'
    return f'frame n. {first + 1}-{last + 1}'


def _short_repr(value, max_len=60):
    text = repr(value) if isinstance(value, str) else str(value)
    text = text.replace('\n', '\\n')
    if len(text) > max_len:
        text = f'{text[:max_len - 3]}...'
    return text


def _fmt(value):
    """Integer-looking text for numbers, '{...}' for sister sets."""
    if isinstance(value, (set, frozenset, tuple, list)):
        values = sorted(int(v) for v in value)
        return '{' + ', '.join(str(v) for v in values) + '}' if values else '{}'
    if value is None:
        return 'None'
    try:
        if pd.isna(value):
            return 'NaN'
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return str(int(number))
    return str(number)


def _runs(frames, values):
    """Consecutive runs of equal values as (value, first, last, n)."""
    runs = []
    for frame_i, value in zip(frames, values):
        if runs and runs[-1][0] == value:
            runs[-1][2] = frame_i
            runs[-1][3] += 1
        else:
            runs.append([value, frame_i, frame_i, 1])
    return [tuple(run) for run in runs]


def _prepare_table(acdc_df, issues):
    """Flat copy with numeric frame_i/Cell_ID and unique rows."""
    df = acdc_df
    index_names = [name for name in df.index.names if name is not None]
    if index_names:
        in_columns = all(name in df.columns for name in index_names)
        df = df.reset_index(drop=in_columns)
    else:
        df = df.reset_index(drop=True)
    missing = [col for col in ('frame_i', 'Cell_ID') if col not in df.columns]
    if missing:
        raise ValueError(
            f'The table has no {" and ".join(missing)} column or index level.'
        )
    df = df.copy()
    for col in ('frame_i', 'Cell_ID'):
        numbers = pd.to_numeric(df[col], errors='coerce')
        invalid = numbers.isna() | (numbers % 1 != 0)
        if invalid.any():
            issues.append(LineageTreeIssue(
                check='missing_index_values',
                severity=CHECKS['missing_index_values'][0],
                message=(
                    f'{int(invalid.sum())} row(s) with invalid {col} '
                    f'(e.g. {_short_repr(df.loc[invalid, col].iloc[0])})'
                ),
                n_rows=int(invalid.sum()), column=col,
                value=_short_repr(df.loc[invalid, col].iloc[0])
            ))
        df[col] = numbers
        df = df[~invalid]
    df['frame_i'] = df['frame_i'].astype(int)
    df['Cell_ID'] = df['Cell_ID'].astype(int)
    df = df.sort_values(['Cell_ID', 'frame_i'], kind='stable')

    duplicated = df.duplicated(['frame_i', 'Cell_ID'], keep=False)
    if duplicated.any():
        for ID, group in df[duplicated].groupby('Cell_ID'):
            frames = sorted(group['frame_i'].unique())
            for value, first, last, n in _runs(
                    frames, np.cumsum(np.diff(frames, prepend=frames[0]) > 1)
                ):
                issues.append(LineageTreeIssue(
                    check='duplicated_rows',
                    severity=CHECKS['duplicated_rows'][0],
                    message=(
                        f'Cell {ID}, {_frames_text(first, last)}: several '
                        'rows per frame'
                    ),
                    Cell_ID=int(ID), first_frame_i=int(first),
                    last_frame_i=int(last),
                    n_rows=int(group['frame_i'].between(first, last).sum())
                ))
        df = df[~df.duplicated(['frame_i', 'Cell_ID'], keep='first')]
    return df.reset_index(drop=True)


def _add_grouped_issues(issues, df, mask, check, column, make_message):
    """One issue per cell and run of consecutive frames in ``mask``."""
    if not mask.any():
        return
    sub = df.loc[mask, ['Cell_ID', 'frame_i']].copy()
    sub['_raw'] = df.loc[mask, column] if column in df.columns else None
    for ID, group in sub.groupby('Cell_ID', sort=True):
        all_frames = df.loc[df['Cell_ID'] == ID, 'frame_i'].to_numpy()
        frames = group['frame_i'].to_numpy()
        # Split where a frame of the cell in between is not in the mask
        positions = np.searchsorted(all_frames, frames)
        breaks = np.diff(positions, prepend=positions[0]) > 1
        run_ids = np.cumsum(breaks)
        for run_id in np.unique(run_ids):
            run = group[run_ids == run_id]
            first = int(run['frame_i'].iloc[0])
            last = int(run['frame_i'].iloc[-1])
            raw = run['_raw'].iloc[0]
            issues.append(LineageTreeIssue(
                check=check, severity=CHECKS[check][0],
                message=make_message(ID, first, last, raw),
                Cell_ID=int(ID), first_frame_i=first, last_frame_i=last,
                n_rows=len(run), column=column, value=_short_repr(raw)
            ))


def _parse_values(df, columns, issues):
    """Numeric copy of ``columns``; invalid values reported and set to NaN."""
    values = pd.DataFrame(index=df.index)
    for col in columns:
        raw = df[col]
        invalid = pd.Series(False, index=raw.index)
        if pd.api.types.is_bool_dtype(raw.dtype):
            numbers = raw.astype(float)
        elif pd.api.types.is_numeric_dtype(raw.dtype):
            numbers = pd.to_numeric(raw, errors='coerce').astype(float)
        else:
            # object, 'string' and pandas 3 'str' dtypes
            numbers = pd.to_numeric(
                raw.astype(object), errors='coerce'
            ).astype(float)
            text = (
                raw.astype(object).where(raw.notna(), '')
                .astype(str).str.strip()
            )
            invalid = pd.Series(
                (raw.notna() & numbers.isna() & (text != '')).to_numpy(bool),
                index=raw.index
            )
        non_integer = numbers.notna() & (numbers % 1 != 0)
        _add_grouped_issues(
            issues, df, invalid, 'invalid_values', col,
            lambda ID, first, last, raw, col=col: (
                f'Cell {ID}, {_frames_text(first, last)}: {col} is '
                f'{_short_repr(raw)}'
            )
        )
        _add_grouped_issues(
            issues, df, non_integer, 'non_integer_values', col,
            lambda ID, first, last, raw, col=col: (
                f'Cell {ID}, {_frames_text(first, last)}: {col} is '
                f'{_fmt(raw)}'
            )
        )
        values[col] = numbers.mask(invalid | non_integer)
    return values


def _strongly_connected_components(graph):
    """Tarjan's algorithm (iterative). ``graph``: {node: set(nodes)}."""
    index_of, lowlink, on_stack = {}, {}, set()
    stack, components, counter = [], [], 0
    for root in graph:
        if root in index_of:
            continue
        work = [(root, iter(graph.get(root, ())))]
        index_of[root] = lowlink[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, children = work[-1]
            advanced = False
            for child in children:
                if child not in index_of:
                    index_of[child] = lowlink[child] = counter
                    counter += 1
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, iter(graph.get(child, ()))))
                    advanced = True
                    break
                if child in on_stack:
                    lowlink[node] = min(lowlink[node], index_of[child])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[node])
            if lowlink[node] == index_of[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                components.append(component)
    return components


def check_lineage_tree(acdc_df, max_daughters=2, source=None):
    """Check the lineage tree columns of an acdc_output table.

    Parameters
    ----------
    acdc_df : pandas.DataFrame
        The table, indexed by ``frame_i`` / ``Cell_ID`` or with them as
        columns. Not modified.
    max_daughters : int, optional
        Expected number of daughters per division (default 2).
    source : str, optional
        Shown in the report (e.g. the file path).

    Returns
    -------
    LineageTreeReport
        See the module docstring for the checks.
    """
    issues = []
    df = _prepare_table(acdc_df, issues)
    n_cells = int(df['Cell_ID'].nunique())
    frames = None
    if len(df):
        frames = (int(df['frame_i'].min()), int(df['frame_i'].max()))

    def report():
        return LineageTreeReport(
            issues, source=source, n_rows=len(df), n_cells=n_cells,
            frames=frames
        )

    if 'parent_ID_tree' not in df.columns:
        return report()

    sister_cols = [
        col for col in df.columns
        if col == 'sister_ID_tree' or str(col).startswith('sister_ID_tree_')
    ]
    tree_cols = [
        col for col in TREE_COLS if col in df.columns
        and col != 'sister_ID_tree'
    ]
    int_cols = [
        *tree_cols, *sister_cols,
        *[col for col in OTHER_INT_COLS if col in df.columns]
    ]
    values = _parse_values(df, int_cols, issues)

    # Rows with lineage annotations: all core tree values present. Extra
    # sister columns added later by the GUI may be empty in older rows.
    core_cols = [*tree_cols]
    if 'sister_ID_tree' in df.columns:
        core_cols.append('sister_ID_tree')
    raw_present = df[core_cols].notna() & (
        df[core_cols].astype(object).where(df[core_cols].notna(), '').astype(str)
        .apply(lambda s: s.str.strip()) != ''
    )
    # GUI fill value (store_data reindexes with fill_value=0): not annotated
    fill_rows = values['parent_ID_tree'] == 0
    if 'generation_num_tree' in values.columns:
        fill_rows &= values['generation_num_tree'] == 0
    if 'root_ID_tree' in values.columns:
        fill_rows &= (
            (values['root_ID_tree'] == 0) | values['root_ID_tree'].isna()
        )
    raw_present = raw_present & ~fill_rows.to_numpy()[:, None]
    annotated = values['parent_ID_tree'].notna() & ~fill_rows
    frame_has_annotation = (
        raw_present.any(axis=1).groupby(df['frame_i']).transform('any')
    )
    unannotated_frames = sorted(
        set(df['frame_i']) - set(df.loc[frame_has_annotation, 'frame_i'])
    )
    if unannotated_frames:
        runs = _runs(
            unannotated_frames,
            np.cumsum(np.diff(unannotated_frames, prepend=unannotated_frames[0]) > 1)
        )
        ranges = ', '.join(_frames_text(first, last) for _, first, last, _ in runs)
        issues.append(LineageTreeIssue(
            check='unannotated_frames',
            severity=CHECKS['unannotated_frames'][0],
            message=f'{len(unannotated_frames)} frame(s) without lineage '
                    f'values: {ranges}',
            first_frame_i=int(unannotated_frames[0]),
            last_frame_i=int(unannotated_frames[-1]),
            n_rows=int((~frame_has_annotation).sum())
        ))
    _add_grouped_issues(
        issues, df, frame_has_annotation & fill_rows, 'fill_value_rows',
        'parent_ID_tree',
        lambda ID, first, last, raw: (
            f'Cell {ID}, {_frames_text(first, last)}: lineage values are 0 '
            '(GUI fill value, never annotated)'
        )
    )
    for col in core_cols:
        missing = frame_has_annotation & ~raw_present[col] & ~fill_rows
        _add_grouped_issues(
            issues, df, missing, 'missing_values', col,
            lambda ID, first, last, raw, col=col: (
                f'Cell {ID}, {_frames_text(first, last)}: {col} is empty'
            )
        )

    if 'Cell_ID_tree' in values.columns:
        differs = (
            values['Cell_ID_tree'].notna()
            & (values['Cell_ID_tree'] != df['Cell_ID'])
            & (values['Cell_ID_tree'] != -1)
        )
        if differs.any():
            issues.append(LineageTreeIssue(
                check='budding_lineage_tree',
                severity=CHECKS['budding_lineage_tree'][0],
                message=(
                    f'Cell_ID_tree differs from Cell_ID in '
                    f'{int(differs.sum())} row(s); structural checks skipped'
                ),
                n_rows=int(differs.sum()), column='Cell_ID_tree'
            ))
            return report()

    is_pure_tree = not any(col in df.columns for col in BUDDING_CCA_COLS)
    _check_structure(
        df, values, annotated, sister_cols, tree_cols, is_pure_tree,
        max_daughters, issues
    )
    return report()


def _check_structure(
        df, values, annotated, sister_cols, tree_cols, is_pure_tree,
        max_daughters, issues
    ):
    first_frames = df.groupby('Cell_ID')['frame_i'].min()
    last_frames = df.groupby('Cell_ID')['frame_i'].max()
    frames_of = {
        ID: frames.to_numpy() for ID, frames in df.groupby('Cell_ID')['frame_i']
    }
    all_IDs = set(first_frames.index)

    rows = pd.DataFrame({
        'Cell_ID': df['Cell_ID'], 'frame_i': df['frame_i']
    })
    for col in tree_cols:
        rows[col] = values[col]
    if sister_cols:
        sisters = values[sister_cols].fillna(-1).to_numpy()
        # 0 is the GUI's fill value for sister columns added later: no sister
        rows_with_zero = (sisters == 0).any(axis=1)
        # A set per row: sister_ID_tree_0 repeats sister_ID_tree
        sisters_key = [
            tuple(sorted({int(s) for s in row if s > 0})) for row in sisters
        ]
        # Columns listing each sister (to name them in the issues)
        sister_sources = [
            tuple(sorted(
                (int(s), col) for s, col in zip(row, sister_cols) if s > 0
            )) for row in sisters
        ]
        main_sister_col = (
            'sister_ID_tree' if 'sister_ID_tree' in sister_cols
            else sister_cols[0]
        )
        sisters_valid = values[main_sister_col].notna().to_numpy()
    else:
        sisters_key = [()] * len(rows)
        sister_sources = [()] * len(rows)
        rows_with_zero = np.zeros(len(rows), dtype=bool)
        sisters_valid = np.zeros(len(rows), dtype=bool)
    rows['sisters'] = sisters_key
    rows['sister_sources'] = sister_sources
    rows['sisters_valid'] = sisters_valid
    rows['zero_sister'] = rows_with_zero
    extra_cols = []
    if is_pure_tree:
        for col in ('emerg_frame_i', 'division_frame_i'):
            if col in values.columns:
                rows[col] = values[col]
                extra_cols.append(col)
        if 'is_history_known' in df.columns:
            rows['is_history_known'] = parse_flags(df['is_history_known'])
            extra_cols.append('is_history_known')
    rows = rows[annotated.to_numpy()]
    if rows.empty:
        return

    # Segments: consecutive rows of a cell with the same lineage values
    key_cols = [
        *tree_cols, 'sisters', *extra_cols, 'zero_sister', 'sister_sources',
        'sisters_valid'
    ]
    # GUI save artifacts (reported once per cell instead) and bookkeeping
    no_change_check_cols = (
        'is_history_known', 'zero_sister', 'sister_sources', 'sisters_valid'
    )
    # Missing values compare equal to each other
    keys = rows[key_cols].astype(object).where(rows[key_cols].notna(), '<NA>')
    changed = (rows['Cell_ID'] != rows['Cell_ID'].shift()).to_numpy()
    for col in key_cols:
        changed |= (keys[col] != keys[col].shift()).to_numpy()
    rows['_segment'] = np.cumsum(changed)
    segments = rows.groupby('_segment', sort=False).agg(
        Cell_ID=('Cell_ID', 'first'),
        first=('frame_i', 'min'),
        last=('frame_i', 'max'),
        n=('frame_i', 'size'),
        **{col: (col, 'first') for col in key_cols}
    )

    # Ordinal of each segment within its cell, to merge the issues of
    # consecutive segments that differ only in values of other checks
    segments['ordinal'] = segments.groupby('Cell_ID').cumcount()
    segment_issues = []
    last_issue_of = {}

    def add(
            check, segment, text, column=None, value=None, expected=None,
            merge_gaps=False
        ):
        """Record an issue of a segment, merged with the same issue of the
        previous segment of the cell (of any earlier one if merge_gaps)."""
        key = (
            check, int(segment.Cell_ID), column,
            None if value is None else _fmt(value),
            None if expected is None else _fmt(expected), text
        )
        previous = last_issue_of.get(key)
        adjacent = (
            previous is not None
            and previous['ordinal'] == segment.ordinal - 1
        )
        if previous is not None and (adjacent or merge_gaps):
            previous['gaps'] = previous['gaps'] or not adjacent
            previous['last'] = int(segment.last)
            previous['n'] += int(segment.n)
            previous['ordinal'] = segment.ordinal
            return
        pending = {
            'key': key, 'ordinal': segment.ordinal,
            'first': int(segment.first), 'last': int(segment.last),
            'n': int(segment.n), 'gaps': False
        }
        segment_issues.append(pending)
        last_issue_of[key] = pending

    # Daughters per parent (from every segment)
    daughters_of = {}
    for segment in segments.itertuples():
        parent_ID = int(segment.parent_ID_tree)
        if parent_ID > 0 and parent_ID != segment.Cell_ID:
            daughters_of.setdefault(parent_ID, set()).add(int(segment.Cell_ID))
    parents_of = {}
    for parent_ID, daughters in daughters_of.items():
        for daughter in daughters:
            parents_of.setdefault(daughter, set()).add(parent_ID)

    # Every parent_ID_tree value of each cell (itself and -1 included)
    parent_values_of = {}
    for segment in segments.itertuples():
        parent_values_of.setdefault(int(segment.Cell_ID), set()).add(
            int(segment.parent_ID_tree)
        )

    def describe_parents(ID):
        texts = []
        for parent in sorted(parent_values_of.get(ID, set())):
            if parent == ID:
                texts.append('itself')
            elif parent == -1:
                texts.append('unknown')
            else:
                texts.append(str(parent))
        return '/'.join(texts) if texts else 'no lineage values'

    def gone_sisters(segment):
        """Sisters that no longer exist in the frames of the segment. The
        lineage tree may stop listing them (sister_ID_tree -1)."""
        parent_ID = int(segment.parent_ID_tree)
        if parent_ID <= 0 or parent_ID == segment.Cell_ID:
            return set()
        return {
            sister for sister in daughters_of.get(parent_ID, set())
            if sister != segment.Cell_ID
            and last_frames[sister] < segment.first
        }

    # Values changing over time
    for ID, cell_segments in segments.groupby('Cell_ID', sort=True):
        if len(cell_segments) < 2:
            continue
        for col in key_cols:
            if col in no_change_check_cols:
                continue
            col_runs = []
            for segment in cell_segments.itertuples():
                value = getattr(segment, col)
                if col != 'sisters' and pd.isna(value):
                    # Empty or invalid values are reported elsewhere
                    continue
                if col == 'sisters' and not segment.sisters_valid:
                    continue
                if col_runs and col_runs[-1][0] == value:
                    col_runs[-1][2] = segment.last
                    col_runs[-1][3] += segment.n
                else:
                    col_runs.append([value, segment.first, segment.last, segment.n])
            if len(col_runs) < 2:
                continue
            if col == 'sisters':
                listed = {
                    frozenset(set(segment.sisters) | gone_sisters(segment))
                    for segment in cell_segments.itertuples()
                }
                if len(listed) == 1:
                    # Only sisters that no longer exist were dropped
                    continue
            column = 'sister_ID_tree' if col == 'sisters' else col
            cell_frames = frames_of[ID]
            parts = []
            for i, (value, first, last, _) in enumerate(col_runs):
                if i > 0:
                    previous_last = col_runs[i - 1][2]
                    present = cell_frames[
                        (cell_frames > previous_last) & (cell_frames < first)
                    ]
                    n_absent = first - previous_last - 1 - len(present)
                    if n_absent > 0:
                        parts.append(
                            f'[cell absent in {n_absent} frame(s) between '
                            f'frame n. {previous_last + 1} and {first + 1}]'
                        )
                parts.append(f'{_fmt(value)} ({_frames_text(first, last)})')
            changes = ', '.join(parts)
            issues.append(LineageTreeIssue(
                check='value_changes_over_time',
                severity=CHECKS['value_changes_over_time'][0],
                message=f'Cell {ID}: {column} changes: {changes}',
                Cell_ID=int(ID), first_frame_i=int(col_runs[0][1]),
                last_frame_i=int(col_runs[-1][2]),
                n_rows=int(sum(run[3] for run in col_runs)), column=column,
                value=' -> '.join(_fmt(run[0]) for run in col_runs)
            ))

    has_gen = 'generation_num_tree' in segments.columns
    has_root = 'root_ID_tree' in segments.columns
    parent_rows = rows.set_index(['Cell_ID', 'frame_i']).sort_index()

    def parent_reference(parent_ID, daughter_first):
        """Parent values in its last annotated frame before the daughter."""
        try:
            parent = parent_rows.loc[parent_ID]
        except KeyError:
            return None
        before = parent[parent.index < daughter_first]
        return before.iloc[-1] if len(before) else parent.iloc[0]

    for segment in segments.itertuples():
        ID = int(segment.Cell_ID)
        parent_ID = int(segment.parent_ID_tree)
        cell_first = int(first_frames[ID])
        known_parent = parent_ID > 0
        sisters = set(segment.sisters)

        if is_pure_tree and 'is_history_known' in segments.columns:
            if bool(segment.is_history_known) != known_parent:
                add(
                    'is_history_known_mismatch', segment,
                    f'is_history_known is '
                    f'{bool(segment.is_history_known)} with parent_ID_tree '
                    f'{parent_ID}',
                    'is_history_known', segment.is_history_known, known_parent,
                    merge_gaps=True
                )
        if segment.zero_sister:
            add(
                'zero_sister_IDs', segment,
                'sister columns contain 0 (treated as no sister)',
                'sister_ID_tree', merge_gaps=True
            )
        if is_pure_tree and 'emerg_frame_i' in segments.columns:
            emerg = segment.emerg_frame_i
            if pd.notna(emerg):
                valid = (
                    emerg == cell_first
                    or (emerg == -1 and not known_parent)
                )
                if not valid:
                    add(
                        'emerg_frame_i_mismatch', segment,
                        f'emerg_frame_i is {_fmt(emerg)} (frame n. '
                        f'{_fmt(emerg + 1)}), but the cell appears at '
                        f'{_frames_text(cell_first, None)}',
                        'emerg_frame_i', emerg, cell_first
                    )
            if 'division_frame_i' in segments.columns:
                division = segment.division_frame_i
                if pd.notna(division) and pd.notna(emerg) and division != emerg:
                    add(
                        'division_frame_i_mismatch', segment,
                        f'division_frame_i is '
                        f'{_fmt(division)}, emerg_frame_i is {_fmt(emerg)}',
                        'division_frame_i', division, emerg
                    )

        if not known_parent:
            if parent_ID != -1:
                add(
                    'parent_not_found', segment,
                    f'invalid parent_ID_tree {parent_ID} '
                    '(-1 = unknown parent)',
                    'parent_ID_tree', parent_ID
                )
            if (
                has_gen and pd.notna(segment.generation_num_tree)
                and segment.generation_num_tree != 1
            ):
                add(
                    'generation_num_tree_mismatch', segment,
                    f'generation_num_tree is '
                    f'{_fmt(segment.generation_num_tree)}, expected 1 '
                    '(unknown parent)',
                    'generation_num_tree', segment.generation_num_tree, 1
                )
            if (
                has_root and pd.notna(segment.root_ID_tree)
                and segment.root_ID_tree not in (ID, -1)
            ):
                add(
                    'root_ID_tree_mismatch', segment,
                    f'root_ID_tree is '
                    f'{_fmt(segment.root_ID_tree)}, expected the cell itself '
                    '(unknown parent)',
                    'root_ID_tree', segment.root_ID_tree, ID
                )
            if sisters and segment.sisters_valid:
                columns = sorted({col for _, col in segment.sister_sources})
                add(
                    'invalid_sister', segment,
                    f'sisters {_fmt(sisters)} listed in '
                    f'{", ".join(columns)}, but the parent is unknown',
                    ','.join(columns), sisters, set()
                )
            continue

        if parent_ID == ID:
            add(
                'self_parent', segment,
                'parent_ID_tree is the cell itself',
                'parent_ID_tree', parent_ID
            )
            continue

        if parent_ID not in all_IDs:
            add(
                'parent_not_found', segment,
                f'parent {parent_ID} does not exist in '
                'any frame',
                'parent_ID_tree', parent_ID
            )
        else:
            parent_first = int(first_frames[parent_ID])
            parent_last = int(last_frames[parent_ID])
            if parent_first >= cell_first:
                add(
                    'parent_after_daughter', segment,
                    f'parent {parent_ID} first appears '
                    f'at {_frames_text(parent_first, None)}, the cell at '
                    f'{_frames_text(cell_first, None)}',
                    'parent_ID_tree', parent_ID
                )
            else:
                parent_frames = frames_of[parent_ID]
                after = parent_frames[parent_frames >= cell_first]
                if len(after):
                    add(
                        'parent_present_after_division', segment,
                        f'parent {parent_ID} still '
                        f'exists at {_frames_text(int(after[0]), int(after[-1]))} '
                        f'({len(after)} frame(s)), the cell appears at '
                        f'{_frames_text(cell_first, None)}',
                        'parent_ID_tree', parent_ID
                    )
                else:
                    if parent_last < cell_first - 1:
                        add(
                            'parent_gap', segment,
                            f'parent {parent_ID} last '
                            f'exists at {_frames_text(parent_last, None)}, '
                            f'the cell appears at '
                            f'{_frames_text(cell_first, None)}',
                            'parent_ID_tree', parent_ID
                        )
            reference = parent_reference(parent_ID, cell_first)
            if reference is not None:
                expected_gen = None
                if has_gen and pd.notna(reference['generation_num_tree']):
                    expected_gen = reference['generation_num_tree'] + 1
                if expected_gen is not None and pd.notna(
                        segment.generation_num_tree):
                    if segment.generation_num_tree != expected_gen:
                        add(
                            'generation_num_tree_mismatch', segment,
                            f'generation_num_tree is '
                            f'{_fmt(segment.generation_num_tree)}, expected '
                            f'{_fmt(expected_gen)} (parent {parent_ID} has '
                            f'{_fmt(reference["generation_num_tree"])})',
                            'generation_num_tree',
                            segment.generation_num_tree, expected_gen
                        )
                expected_root = None
                if has_root and pd.notna(reference['root_ID_tree']):
                    # -1: the parent is its own root
                    expected_root = reference['root_ID_tree']
                    if expected_root == -1:
                        expected_root = parent_ID
                if expected_root is not None and pd.notna(
                        segment.root_ID_tree):
                    if segment.root_ID_tree != expected_root:
                        add(
                            'root_ID_tree_mismatch', segment,
                            f'root_ID_tree is '
                            f'{_fmt(segment.root_ID_tree)}, expected '
                            f'{_fmt(expected_root)} (root of parent '
                            f'{parent_ID})',
                            'root_ID_tree', segment.root_ID_tree,
                            expected_root
                        )

        if not sister_cols or not segment.sisters_valid:
            continue
        expected_sisters = daughters_of.get(parent_ID, set()) - {ID}
        for sister in sorted(sisters):
            if sister == ID:
                reason = 'is the cell itself'
            elif sister == parent_ID:
                reason = 'is the parent of the cell'
            elif sister not in all_IDs:
                reason = 'does not exist in any frame'
            elif parent_ID not in parents_of.get(sister, set()):
                reason = (
                    f'has parent {describe_parents(sister)}, not {parent_ID}'
                )
            else:
                continue
            columns = sorted(
                col for listed, col in segment.sister_sources
                if listed == sister
            )
            add(
                'invalid_sister', segment,
                f'{", ".join(columns)} lists {sister}, which {reason}',
                ','.join(columns), sister
            )
        missing = expected_sisters - sisters - gone_sisters(segment)
        if missing:
            add(
                'missing_sister', segment,
                f'cell(s) {_fmt(missing)} also have parent {parent_ID} '
                f'but are not listed as sisters (listed: {_fmt(sisters)})',
                'sister_ID_tree', sisters, expected_sisters
            )

    for pending in segment_issues:
        check, ID, column, value, expected, text = pending['key']
        frames_text = _frames_text(pending['first'], pending['last'])
        if pending['gaps']:
            frames_text = f'{frames_text} ({pending["n"]} frames)'
        issues.append(LineageTreeIssue(
            check=check, severity=CHECKS[check][0],
            message=f'Cell {ID}, {frames_text}: {text}',
            Cell_ID=ID, first_frame_i=pending['first'],
            last_frame_i=pending['last'], n_rows=pending['n'],
            column=column, value=value, expected=expected
        ))

    # Parent graph: cycles and number of daughters
    graph = {daughter: parents for daughter, parents in parents_of.items()}
    for component in _strongly_connected_components(graph):
        if len(component) < 2:
            continue
        cycle = sorted(int(ID) for ID in component)
        issues.append(LineageTreeIssue(
            check='lineage_cycle', severity=CHECKS['lineage_cycle'][0],
            message=f'Cells {_format_IDs(cycle)} are ancestors of each other',
            Cell_ID=cycle[0], column='parent_ID_tree',
            value=_fmt(set(cycle))
        ))

    for parent_ID, daughters in sorted(daughters_of.items()):
        if parent_ID not in all_IDs:
            continue
        births = {
            daughter: int(first_frames[daughter]) for daughter in daughters
        }
        frames = sorted(set(births.values()))
        daughters_text = _fmt(daughters)
        base = dict(
            Cell_ID=int(parent_ID), first_frame_i=frames[0],
            last_frame_i=frames[-1], column='parent_ID_tree',
            value=daughters_text
        )
        if len(daughters) == 1:
            issues.append(LineageTreeIssue(
                check='single_daughter',
                severity=CHECKS['single_daughter'][0],
                message=(
                    f'Cell {parent_ID}: only daughter {daughters_text} '
                    f'(appears at {_frames_text(frames[0], None)})'
                ), **base
            ))
        elif len(daughters) > max_daughters:
            issues.append(LineageTreeIssue(
                check='too_many_daughters',
                severity=CHECKS['too_many_daughters'][0],
                message=(
                    f'Cell {parent_ID}: {len(daughters)} daughters '
                    f'{daughters_text}'
                ), **base
            ))
        if len(frames) > 1:
            births_text = ', '.join(
                f'{daughter} at {_frames_text(frame_i, None)}'
                for daughter, frame_i in sorted(births.items())
            )
            issues.append(LineageTreeIssue(
                check='daughters_born_in_different_frames',
                severity=CHECKS['daughters_born_in_different_frames'][0],
                message=f'Cell {parent_ID}: daughters {births_text}',
                **base
            ))


def read_acdc_output_csv(csv_path):
    """Read an acdc_output CSV without casting (corrupted values stay text)."""
    return pd.read_csv(csv_path, low_memory=False)


def check_lineage_tree_file(csv_path, max_daughters=2):
    """Check the lineage tree columns of an acdc_output CSV file."""
    df = read_acdc_output_csv(csv_path)
    return check_lineage_tree(
        df, max_daughters=max_daughters, source=str(csv_path)
    )


def find_acdc_output_files(path, include_auto=False):
    """``*acdc_output*.csv`` files in ``path`` (recursively) or ``path``.

    The raw tracker tables ``*acdc_output_auto.csv`` (never loaded by
    Cell-ACDC) are skipped unless ``include_auto``.
    """
    if os.path.isfile(path):
        return [os.path.normpath(path)]
    found = []
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for file in sorted(files):
            name = file.lower()
            if 'acdc_output' not in name or not name.endswith('.csv'):
                continue
            if not include_auto and name.endswith('acdc_output_auto.csv'):
                continue
            found.append(os.path.normpath(os.path.join(root, file)))
    return found


def check_lineage_tree_files(
        paths, max_daughters=2, only_with_tree=True, include_auto=False
    ):
    """Check every acdc_output CSV of the given files/folders.

    Parameters
    ----------
    paths : str or iterable of str
        Files or folders (searched recursively for ``*acdc_output*.csv``).
    max_daughters : int, optional
        See :func:`check_lineage_tree`.
    only_with_tree : bool, optional
        Skip tables without a ``parent_ID_tree`` column (default True).
    include_auto : bool, optional
        Also check ``*acdc_output_auto.csv`` files found in folders.

    Returns
    -------
    list of LineageTreeReport
        One report per file. Files that cannot be read or checked get a
        report with one ``unreadable_file`` error.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    reports = []
    for path in paths:
        for csv_path in find_acdc_output_files(
                os.fspath(path), include_auto=include_auto
            ):
            try:
                df = read_acdc_output_csv(csv_path)
                if only_with_tree and 'parent_ID_tree' not in df.columns:
                    continue
                report = check_lineage_tree(
                    df, max_daughters=max_daughters, source=csv_path
                )
            except Exception as error:
                report = LineageTreeReport([LineageTreeIssue(
                    check='unreadable_file',
                    severity=CHECKS['unreadable_file'][0],
                    message=f'{type(error).__name__}: {error}'
                )], source=csv_path)
            reports.append(report)
    return reports


def reports_to_text(
        reports, max_issues_per_check=20,
        more_hint='to_text(None) or to_dataframe()'
    ):
    """Join the text of several reports, with a summary of all files."""
    reports = list(reports)
    parts = [
        report.to_text(max_issues_per_check, more_hint=more_hint)
        for report in reports
    ]
    if len(reports) > 1:
        not_ok = [report for report in reports if not report.ok]
        summary = [
            f'Checked {len(reports)} file(s): {len(not_ok)} with errors or '
            'warnings.'
        ]
        for report in not_ok:
            counts = report.counts()
            summary.append(
                f'  {report.source}: {counts["error"]} error(s), '
                f'{counts["warning"]} warning(s)'
            )
        parts.append('\n'.join(summary))
    return '\n\n'.join(parts)


def reports_to_dataframe(reports):
    """Issues of several reports in one DataFrame (with a source column)."""
    frames = [report.to_dataframe() for report in reports]
    if not frames:
        return pd.DataFrame(columns=['source', *LineageTreeReport.ISSUE_COLUMNS])
    return pd.concat(frames, ignore_index=True)


def main(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(
        prog='python -m cellacdc.lineage_tree_integrity',
        description=(
            'Report inconsistencies in the lineage tree columns of '
            'acdc_output CSV files.'
        )
    )
    parser.add_argument(
        'paths', nargs='+',
        help='acdc_output CSV files or folders (searched recursively).'
    )
    parser.add_argument(
        '--out', help='Write the report to this file (.csv: issue table).'
    )
    parser.add_argument('--max-daughters', type=int, default=2)
    parser.add_argument(
        '--include-auto', action='store_true',
        help='Also check the raw tracker tables *acdc_output_auto.csv found '
             'in folders.'
    )
    parser.add_argument(
        '--max-issues-per-check', type=int, default=20,
        help='Issues listed per check in the printed report (0: all).'
    )
    args = parser.parse_args(None if argv is None else list(argv))
    missing = [path for path in args.paths if not os.path.exists(path)]
    for path in missing:
        print(f'Path not found: {path}', file=sys.stderr)
    if missing:
        return 2
    reports = check_lineage_tree_files(
        args.paths, max_daughters=args.max_daughters,
        include_auto=args.include_auto
    )
    max_issues = args.max_issues_per_check or None
    if not reports:
        print('No acdc_output CSV files with lineage tree columns found.')
        return 0
    more_hint = (
        'run with --max-issues-per-check 0, or save all of them with '
        '--out report.txt / report.csv'
    )
    print(reports_to_text(reports, max_issues, more_hint=more_hint))
    if args.out:
        if args.out.lower().endswith('.csv'):
            reports_to_dataframe(reports).to_csv(args.out, index=False)
        else:
            with open(args.out, 'w', encoding='utf-8') as file:
                file.write(reports_to_text(reports, None))
                file.write('\n')
        print(f'\nReport saved to {args.out}')
    return 1 if any(report.has_errors for report in reports) else 0


if __name__ == '__main__':
    sys.exit(main())
