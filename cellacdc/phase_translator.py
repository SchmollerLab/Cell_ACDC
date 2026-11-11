"""Ordered, per-cell translation of time point boundaries into phases.

Rules are listed in descending priority and each rule has its own
eligibility criteria. Every cell is checked against the rules in order and
gets its phases from the first rule it is eligible for; a cell that is not
eligible for a rule moves on to the next one. A cell is eligible for a rule
when all of these hold, checked in this order:

1. Each of the rule's boundary time points is marked exactly once in the
   cell, and their frames increase in the configured order. A boundary
   marked several times makes the cell ineligible for that rule; if no rule
   matches such a cell, a warning lists it.
2. Each of the rule's required annotations is truthy in at least one row of
   the cell. Annotations are requirements only, not boundaries.
3. The rule's eligibility formula (``predictor``) holds for the cell. It is
   evaluated only for cells that passed 1 and 2.

The first matching rule owns the whole cell: an interval with an empty phase
name stays unassigned and is not filled by a lower-priority rule. Intervals
include their starting boundary and exclude their ending boundary.

Rules are dictionaries with these fields:

- ``name``: unique rule name, used in the configuration and warnings.
- ``predictor``: the rule's eligibility formula ("Rule eligibility formula"
  in the rule editor), stored as custom annotation predictor settings. It
  can use its own main-cell filter and other-cell measurements, so different
  rules can select cells differently. Its boolean results are consolidated
  across each cell's history with the predictor's All/None/Most setting.
  Type ``none`` ("Always eligible") skips this check.
- ``timepoints``: ordered names of boolean ``timepoint_<name>`` columns, the
  interval boundaries.
- ``annotations``: required annotation columns (ticked in the rule editor)
  that must each be truthy in at least one row of the cell.
- ``phases``: one name per interval, so N boundaries require N + 1 names.
  For boundaries at frames 2 and 5, names ``['G1', 'S', 'G2']`` assign
  G1 before frame 2, S at frames 2 through 4, and G2 from frame 5 onward.
  A rule without boundaries has one interval covering the whole cell, which
  makes it a fallback (e.g. ``['Arrested']`` with a formula).

Evaluation returns a new position table with a shared ``phase`` string column
and boolean ``phase_<name>`` columns. It does not modify or save the input table.
Unassigned rows have an empty phase (``''``).

Output ownership is not stored in files: ``existing_output_columns`` recognizes
outputs of a saved or exported table by their values, so they can be
re-evaluated. Empty phases read back from CSV are missing values, which
``load._load_acdc_df_file`` fills with 0 for tables with cell cycle columns;
missing values and these zeros count as unassigned. CSV reading also turns
phase names such as ``'1'`` or ``'NA'`` into numbers or missing values; these
are recovered from the boolean ``phase_<name>`` column that is True in the
row. When the main GUI adds cells, it reindexes frames with ``fill_value=0``
and saves the ``phase_<name>`` columns as ``'True'``/``'False'``/``'0'``
strings; such bool-like marker columns are recognized by their parsed
values. Call ``normalize_loaded_outputs`` on a freshly loaded table to
restore unassigned phases to ``''``, recovered names to strings and
recognized marker columns to bool.

``phase_names`` lists the distinct phase names of the rules and
``phase_colors`` resolves their display colors (``#rrggbb``): user overrides
first, otherwise the colorblind-friendly Okabe-Ito palette by name order
(``DEFAULT_PHASE_COLORS``: no black and vermillion last, because red is
reserved for the current frame line), then evenly spaced hues that skip red.
"""
import colorsys
import re
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .custom_annotation_predictors import CustomAnnotationPredictor
from .timepoint_predictors import (
    CellTable, PREDICTOR_NONE, TIMEPOINT_PREFIX, parse_flags
)


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


def phase_names(rules: list[dict[str, Any]]) -> list[str]:
    """Return the nonempty phase names of all rules, ordered by first use."""
    names = {}
    for rule in rules:
        for phase in rule.get('phases', []):
            if phase:
                names.setdefault(phase, None)
    return list(names)


HEX_COLOR = re.compile(r'#[0-9a-fA-F]{6}')


# Okabe-Ito palette without black, vermillion last (the current frame line
# drawn over the phase band is red)
DEFAULT_PHASE_COLORS = (
    '#56b4e9', '#e69f00', '#009e73', '#f0e442', '#0072b2', '#cc79a7',
    '#d55e00',
)


def default_phase_color(index: int, count: int) -> str:
    """Return the default ``#rrggbb`` color of the index-th of count phases.

    The first phases get the colorblind-friendly ``DEFAULT_PHASE_COLORS``,
    further phases evenly spaced hues between 30 and 330 degrees (at least 9
    hues), so no default is red and adding a phase does not change the
    colors of the others.
    """
    if index < len(DEFAULT_PHASE_COLORS):
        return DEFAULT_PHASE_COLORS[index]
    index -= len(DEFAULT_PHASE_COLORS)
    hues = max(count - len(DEFAULT_PHASE_COLORS), 9)
    hue = (30 + 300 * ((index % hues) + 0.5) / hues) / 360
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return '#{:02x}{:02x}{:02x}'.format(
        round(r * 255), round(g * 255), round(b * 255)
    )


def phase_colors(
        names: Iterable[str], overrides: dict[str, str] | None = None
    ) -> dict[str, str]:
    """Return ``{name: '#rrggbb'}`` for names, preferring valid overrides.

    Defaults depend on each name's position in ``names``; overrides that are
    not ``#rrggbb`` strings are ignored.
    """
    names = list(dict.fromkeys(names))
    overrides = overrides or {}
    colors = {}
    for index, name in enumerate(names):
        color = overrides.get(name)
        if not isinstance(color, str) or not HEX_COLOR.fullmatch(color):
            color = default_phase_color(index, len(names))
        colors[name] = color.lower()
    return colors


def validate_phase_colors(colors: Any) -> None:
    """Validate a serialized ``{phase name: '#rrggbb'}`` mapping."""
    if not isinstance(colors, dict) or any(
            not isinstance(name, str) or not name
            or not isinstance(color, str) or not HEX_COLOR.fullmatch(color)
            for name, color in colors.items()):
        raise ValueError(
            'Phase colors must map phase names to colors like "#ff0000".'
        )


try:
    from pandas._libs.parsers import STR_NA_VALUES as _CSV_NA_TOKENS
except ImportError: # private pandas API; these are its default NA tokens
    _CSV_NA_TOKENS = {
        '', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan',
        '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'None',
        'n/a', 'nan', 'null'
    }
_CSV_BOOL_TOKENS = {
    'True': True, 'TRUE': True, 'true': True,
    'False': False, 'FALSE': False, 'false': False,
}

# Kinds of the values of an object ``phase`` column (see ``_value_kind``).
_KIND_OTHER, _KIND_STR, _KIND_BOOL, _KIND_NUMBER = range(4)


def _value_kind(value: Any) -> int:
    """Classify one non-missing value of an object ``phase`` column."""
    if isinstance(value, str):
        return _KIND_STR
    if isinstance(value, (bool, np.bool_)):
        return _KIND_BOOL
    if isinstance(value, (int, float, np.integer, np.floating)):
        return _KIND_NUMBER
    return _KIND_OTHER


def _phase_markers(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Return ``{column: bool flags}`` of the bool-like ``phase_<name>``
    columns with a nonempty name.

    Columns are parsed with ``parse_flags(strict=True)``: markers the main
    GUI saved as ``'True'``/``'False'``/``'0'`` strings are recognized, while
    other ``phase_*`` columns, e.g. channel measurements such as
    ``phase_contr_mean``, are not markers.
    """
    markers = {}
    for column in df.columns:
        if (isinstance(column, str) and column.startswith('phase_')
                and len(column) > len('phase_')):
            flags = parse_flags(df[column], strict=True)
            if flags is not None:
                markers[column] = flags
    return markers


def _recover_names(
        df: pd.DataFrame, rows: np.ndarray, values: np.ndarray,
        missing: np.ndarray, zero: np.ndarray, kinds: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
    """Recover phase names that CSV reading turned into non-string values.

    The translator writes the phase of each row twice: as a string in the
    ``phase`` column and as a True value in one boolean ``phase_<name>``
    marker column. After the table is saved to ``acdc_output.csv`` and loaded
    again, ``pandas.read_csv`` converts some names in ``phase``, while the
    marker columns still say which phase each row has:

    =====================================  ===================================
    Phase name written                     Value in ``phase`` after loading
    =====================================  ===================================
    ``'G1'`` (any other text)              ``'G1'`` (unchanged, not passed
                                           here)
    ``'2'``, ``'1.5'`` (numeric text)      the number ``2``, ``1.5``
    ``'NA'``, ``'None'``, ``'nan'``, ...   missing value (pandas NA token,
    (``_CSV_NA_TOKENS``)                   see ``read_csv(na_values=...)``)
    ``'True'``, ``'false'``, ...           the boolean ``True``, ``False``
    ``''`` (no phase)                      missing value
    =====================================  ===================================

    ``load._load_acdc_df_file`` then fills missing values with 0 when the
    table has cell cycle columns, and a second save and load turns that 0
    into the string ``'0'``.

    A row is recovered only when both sources agree:

    1. Exactly one bool-like ``phase_<name>`` column is True in the row (see
       ``_phase_markers``). Rows with no True marker (unassigned, or rows the
       main GUI added with ``fill_value=0``) or with several True markers are
       ambiguous and are not recovered.
    2. Reading ``<name>`` back from CSV can give the row's value:

       - ``<name>`` is a pandas NA token (e.g. ``'NA'``): the value is
         missing or the loader's 0.
       - ``<name>`` is numeric text (e.g. ``'1'``): the value is a number
         equal to ``float(<name>)``. For the name ``'0'`` the loader's 0,
         including the string ``'0'`` after another save, also matches.
       - ``<name>`` is a boolean token (``_CSV_BOOL_TOKENS``): the value is
         that boolean.

    Requiring both conditions means a leftover marker cannot invent a phase
    for a row whose value does not fit it, and a value cannot be renamed
    after a marker it does not belong to.

    Parameters
    ----------
    df : pandas.DataFrame
        Table holding the ``phase_<name>`` marker columns. Only the markers
        are read; ``df['phase']`` is passed in through ``values``.
    rows : numpy.ndarray
        Integer positions (not index labels) in ``df`` of the rows to
        recover, i.e. the rows of ``phase`` that are not plain strings.
    values : numpy.ndarray
        Object array of the ``phase`` values at ``rows``.
    missing : numpy.ndarray
        Boolean mask (aligned with ``rows``) of missing values.
    zero : numpy.ndarray
        Boolean mask (aligned with ``rows``) of the loader's 0: the number 0
        or the string ``'0'``.
    kinds : numpy.ndarray
        ``_KIND_*`` code of each value (see ``_value_kind``); missing values
        have ``_KIND_OTHER``.

    Returns
    -------
    recovered : numpy.ndarray
        Boolean mask (aligned with ``rows``) of the rows whose name was
        recovered.
    names : numpy.ndarray
        Object array (aligned with ``rows``) with the recovered phase name
        where ``recovered`` is True; the other entries are undefined.

    Examples
    --------
    Rows of a reloaded table with phases named ``'1'``, ``'2'`` and ``'NA'``
    (only the True marker of each row shown):

    ===========  ==================  ==========================
    ``phase``    True marker         Result
    ===========  ==================  ==========================
    ``2.0``      ``phase_2``         recovered as ``'2'``
    ``0``        ``phase_NA``        recovered as ``'NA'``
    ``0``        none                not recovered (unassigned)
    ``2.0``      ``phase_1``         not recovered (mismatch)
    ===========  ==================  ==========================

    Notes
    -----
    The caller ``_phase_strings`` sets recovered rows to their name, turns
    unrecovered missing values and zeros into ``''`` (unassigned), and
    returns None if any other value remains unexplained, so that an
    unrelated ``phase`` column is never treated as translator output.

    The work is vectorised per marker column (there are only a few) rather
    than per row, so this stays fast on large tables.
    """
    recovered = np.zeros(len(rows), dtype=bool)
    names = np.empty(len(rows), dtype=object)
    markers = _phase_markers(df)
    if not markers or not len(rows):
        return recovered, names
    # rows x markers matrix: which phase_<name> column is True in each row
    flags = np.column_stack(
        [marker.to_numpy(dtype=bool)[rows] for marker in markers.values()]
    )
    # Condition 1: exactly one True marker; marker_index says which one
    single = flags.sum(axis=1) == 1
    marker_index = flags.argmax(axis=1)
    # Numeric values as float, to compare them with numeric phase names
    numbers = np.full(len(rows), np.nan)
    is_number = kinds == _KIND_NUMBER
    if is_number.any():
        numbers[is_number] = values[is_number].astype(float)
    is_bool = kinds == _KIND_BOOL
    for index, column in enumerate(markers):
        # Rows whose only True marker is this column
        selected = single & (marker_index == index)
        if not selected.any():
            continue
        name = column[len('phase_'):]
        try:
            number = float(name)
        except ValueError:
            number = None
        # Condition 2: the value is what reading `name` from CSV can give.
        # NA token ('NA', 'None', ...): read as missing, then maybe the 0 fill
        is_na_token = name in _CSV_NA_TOKENS
        matches = (missing | zero) & is_na_token
        if number is not None:
            # Numeric text ('1', '2.5'): read as the equal number
            matches |= is_number & (numbers == number)
            # A string '0' is the loader's 0 written by another save.
            matches |= zero & (number == 0)
        # Boolean token ('True', 'false', ...): read as that boolean
        boolean = _CSV_BOOL_TOKENS.get(name)
        if boolean is not None and is_bool.any():
            matches[is_bool] |= values[is_bool].astype(bool) == boolean
        matches &= selected
        recovered |= matches
        names[matches] = name
    return recovered, names


def _phase_strings(df: pd.DataFrame) -> pd.Series | None:
    """Return ``df['phase']`` as strings, with unassigned values set to ``''``.

    Empty phases written to CSV are read back as missing values, which
    ``load._load_acdc_df_file`` fills with 0 when the table has cell cycle
    columns (``'0'`` after another save). CSV reading also turns phase names
    such as ``'1'`` or ``'NA'`` into numbers or missing values. Such rows get
    their name back from the boolean ``phase_<name>`` columns when exactly one
    of them is True and matches the value (see ``_recover_names``); otherwise
    missing values and zeros count as unassigned. Returns None when any other
    non-string value remains, i.e. ``phase`` is not a translator output.

    String, numeric and boolean columns take vectorised paths; only mixed
    object columns (e.g. strings and the loader's 0) are classified per value.
    """
    values = df['phase']
    size = len(values)
    missing = values.isna().to_numpy()
    zero = np.zeros(size, dtype=bool)
    kind = pd.api.types.infer_dtype(values, skipna=True)
    if kind in ('string', 'empty'):
        strings = values.to_numpy(dtype=object, copy=True)
        kinds = np.where(missing, _KIND_OTHER, _KIND_STR)
        present = ~missing
        zero[present] = strings[present] == '0'
    elif pd.api.types.is_bool_dtype(values.dtype):
        strings = values.to_numpy(dtype=object, copy=True)
        kinds = np.where(missing, _KIND_OTHER, _KIND_BOOL)
    elif pd.api.types.is_numeric_dtype(values.dtype):
        numbers = values.to_numpy(dtype=float, na_value=np.nan)
        strings = numbers.astype(object)
        kinds = np.where(missing, _KIND_OTHER, _KIND_NUMBER)
        zero = numbers == 0
    else:
        strings = values.to_numpy(dtype=object, copy=True)
        kinds = np.full(size, _KIND_OTHER)
        present = ~missing
        kinds[present] = [_value_kind(value) for value in strings[present]]
        is_str = kinds == _KIND_STR
        zero[is_str] = strings[is_str] == '0'
        is_number = kinds == _KIND_NUMBER
        if is_number.any():
            zero[is_number] = strings[is_number].astype(float) == 0
    unresolved = missing | zero | (kinds != _KIND_STR)
    if unresolved.any():
        rows = np.flatnonzero(unresolved)
        recovered, names = _recover_names(
            df, rows, strings[rows], missing[rows], zero[rows], kinds[rows]
        )
        strings[rows[recovered]] = names[recovered]
        blank = (missing[rows] | zero[rows]) & ~recovered
        if not (recovered | blank).all():
            return None
        strings[rows[blank]] = ''
    return pd.Series(strings, index=values.index, name='phase', dtype=object)


def _recognized_outputs(
        df: pd.DataFrame, strings: pd.Series | None,
        known: Iterable[str] = ()
    ) -> set[str]:
    """Return the outputs recognized from normalized phase ``strings``.

    Bool-like ``phase_<name>`` columns (see ``_phase_markers``) are compared
    by their parsed flags. Columns in ``known`` are skipped (their
    comparison is the expensive part); see ``existing_output_columns``.
    """
    if strings is None:
        return set()
    known = set(known)
    columns = {
        column for column, flags in _phase_markers(df).items()
        if column not in known and flags.equals(
            strings.eq(column[len('phase_'):]).rename(column)
        )
    }
    return columns | {'phase'} if columns else set()


def existing_output_columns(df: pd.DataFrame) -> set[str]:
    """Infer translator outputs when re-evaluating a previously exported CSV.

    Runtime column ownership is not stored in CSV files. Recognize a boolean
    column only when it exactly matches the corresponding string phase values,
    and require at least one such column before recognizing ``phase`` itself.
    Missing phases and the 0 that the acdc_output loader fills in for them
    count as unassigned (``''``); phase names that CSV reading turned into
    numbers or missing values are recovered (see ``_phase_strings``).
    Marker columns that the main GUI saved as True/False/0 strings are
    compared by their parsed values (see ``_phase_markers``).
    This is value-based inference, not proof of the columns' origin.
    """
    if 'phase' not in df:
        return set()
    return _recognized_outputs(df, _phase_strings(df))


def normalize_loaded_outputs(df: pd.DataFrame) -> set[str]:
    """Restore unassigned phases of a table read from an acdc_output file.

    Modifies df in place: when ``existing_output_columns`` recognizes
    translator outputs, missing or loader-filled 0 values of ``phase`` become
    ``''`` and phase names read back as numbers or missing values (e.g.
    ``'1'``, ``'NA'``) are restored (see ``_phase_strings``). Recognized
    ``phase_<name>`` columns that the main GUI saved as True/False/0 strings
    (after adding cells) become bool columns again. Any other ``phase``
    or ``phase_*`` column, e.g. an unrelated numeric measurement, is left
    untouched. Returns the recognized output column names (empty when
    nothing was changed).
    """
    if 'phase' not in df:
        return set()
    strings = _phase_strings(df)
    columns = _recognized_outputs(df, strings)
    if 'phase' in columns:
        df['phase'] = strings.to_numpy()
        markers = _phase_markers(df[sorted(columns.difference({'phase'}))])
        for column, flags in markers.items():
            if df[column].dtype != bool:
                df[column] = flags.to_numpy()
    return columns


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
    """Assign phases to each selected cell from the first rule it is eligible
    for (boundaries, required annotations, then the rule's own eligibility
    formula; see the module docstring).

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
    # Only the rule inputs are needed here (predictors receive df itself), so
    # avoid copying every measurement of the position.
    inputs = {
        column for rule in rules
        for column in rule['timepoints'] + rule['annotations']
    }
    flat = CellTable.flatten(
        df.loc[:, df.columns.isin(inputs | {'frame_i', 'Cell_ID'})]
    )
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
    owned = set(owned_columns)
    # Normalize the phase strings once; they serve both to recognize outputs
    # of a saved table and as the starting phases. Boolean columns already
    # owned (with 'phase') need no value comparison.
    phase_strings = _phase_strings(df) if 'phase' in df else None
    owned |= _recognized_outputs(
        df, phase_strings, known=owned if 'phase' in owned else ()
    )
    result = df.copy()
    if 'phase' not in result:
        result['phase'] = ''
    # Recognized outputs reloaded from acdc_output may hold NaN or the
    # loader's 0 for unassigned cells; keep those cells unassigned ('').
    if 'phase' in owned and phase_strings is not None:
        result['phase'] = phase_strings.to_numpy()
    else:
        result['phase'] = result['phase'].fillna('').astype(object)
    phase_values = result['phase'].to_numpy(copy=True)
    # Reset only the requested cells. Existing phases on other cells also
    # determine which old boolean outputs must survive this evaluation.
    phase_values[target] = ''
    retained = columns | {
        f'phase_{phase}' for phase in pd.unique(phase_values) if phase
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
    cell_ids = flat['Cell_ID'].to_numpy()
    # Cache of per-column facts used by the eligibility checks. For a boundary
    # column: how many rows each target cell has marked and its first marked
    # frame (marked_frames). For an annotation column: which target cells have
    # it set at least once (present_cells). These facts depend only on the
    # column, not on the rule, so each column is scanned once for all target
    # cells and reused by every rule that lists it. Which columns a rule uses
    # and how it checks them (exactly once, order, required annotations, its
    # formula) is still decided per rule in the loop below.
    marked_cache = {}
    present_cache = {}

    def marked_frames(column):
        """Return ({ID: number of marked rows}, {ID: first marked frame})."""
        if column not in marked_cache:
            rows = parse_flags(flat[column]).to_numpy() & target
            marked = pd.Series(frames[rows]).groupby(cell_ids[rows], sort=False)
            marked_cache[column] = (
                marked.size().to_dict(), marked.first().to_dict()
            )
        return marked_cache[column]

    def present_cells(column):
        """Return the target cells with a truthy value in at least one row."""
        if column not in present_cache:
            rows = parse_flags(flat[column]).to_numpy() & target
            present_cache[column] = set(cell_ids[rows].tolist())
        return present_cache[column]

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
        marks = [marked_frames(column) for column in rule['timepoints']]
        annotated = [present_cells(column) for column in rule['annotations']]
        # Check the eligibility of each pending cell, in Cell_ID order.
        for ID in sorted(pending):
            counts = [count.get(ID, 0) for count, _ in marks]
            if any(count > 1 for count in counts):
                # Duplicate time points within one cell are ambiguous.
                duplicate_cells.add(ID)
                continue
            # Missing boundaries are ambiguous too. Do not sort the configured
            # sequence: reversing it would change interval meanings.
            if any(count != 1 for count in counts):
                continue
            boundaries = [int(first[ID]) for _, first in marks]
            if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
                continue # skip if time points are not in correct order
            if not all(ID in cells for cells in annotated):
                continue # skip if required annotations are not present
            eligible[int(ID)] = boundaries # store the valid boundaries for this cell
        if not eligible:
            continue
        # Step 3: the rule's own eligibility formula, evaluated only for the
        # cells that passed the boundary and annotation checks. Restrict the
        # target cells, not the source table: other-cell formulas still need
        # measurements outside the eligible set.
        predictor = rule['predictor']
        matches = (
            set(eligible) if predictor['type'] == PREDICTOR_NONE else
            set(CustomAnnotationPredictor.predict(
                df, predictor, IDs=eligible, SizeT=SizeT
            ))
        )
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
