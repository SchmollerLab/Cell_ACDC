"""Time point identifiers of the measurements viewer and their predictors.

A time point identifier (e.g., "Bud emergence") marks one or more frames of 
each cell. Time points can be added manually or predicted on a per-cell basis 
with either:

- an equation using the columns of the acdc_df (evaluated as in 
  ``Combine measurements``) and a rule to convert the per-frame result into 
  frames (e.g., first frame where the equation is true). Equations can use 
  the per-cell functions of ``CELL_FUNCTIONS`` (e.g., 
  ``cell_area_pxl > 0.5 * MAX(cell_area_pxl)`` or ``DIFF(cell_area_pxl) > 50``), 
  computed over the frames of each cell, ``any``/``all`` to reduce conditions
  across those frames, and suffixed ``_other`` inputs to aggregate selected
  cells within a frame window, or
- a Python file defining ``predictor(df)``. ``df`` is the acdc_df of a 
  single cell (one row per frame, with ``frame_i`` and ``Cell_ID`` columns, 
  sorted by ``frame_i``). It returns a frame index (``frame_i``, 0-based), a 
  list of frame indices, or None if the cell has no time point. If the 
  function accepts the keyword argument ``entire_df`` (or ``**kwargs``), it 
  also receives the acdc_df of all cells.

Custom filters use an equation. Candidate rows are sorted by frame.
Results must be bool or numeric 0/1 (scalar or one per
row). ``condition_mode`` selects candidate cells matching ``any``/``all``
candidate frames, or just matching rows (``where_true``). Aggregated values
are stored at the enquiring cell's frame t.
Equations expose the enquiring cell's values at t as ``this_<column>``.
Time point predictor equations can also restrict target cells with a
main-cell filter evaluated across each cell's frames (``any`` or ``all``).
Custom time scopes evaluate an inclusive window (e.g. ``t - 5`` to ``t + 2``)
for each enquiring-cell frame t, applying filters within that window and
storing the aggregate at t. Missing frames are excluded, not padded.
Use ``t - 1`` for both bounds to select the previous frame, e.g. a symmetric
mother before division. Budding relatives use the same frame (``t``).
``SHIFT(column, 1)`` instead shifts values by one observed row of a cell.

Class-based API
---------------
The classes group stateless operations; no instances are required except for
``TimepointIdentifier``, which holds one marker's configuration:

- ``TimepointIdentifier`` serializes and validates marker definitions.
- ``TimepointPredictor`` validates settings, describes predictors and runs them.
- ``CellEquation`` evaluates expressions using the shared ``CELL_FUNCTIONS``.
- ``CellFilter`` validates main/other-cell settings and supplies filter presets.
- ``OtherCellMeasurements`` prepares filtered, windowed aggregate inputs.
- ``CellTable`` normalizes table layouts, iterates cells and subsets target IDs.
- ``FrameRange`` parses and validates inclusive bounds relative to t.
- ``PredictorDependencies`` orders equation predictors by their input columns
  and rejects circular dependencies.

For example::

    frames = TimepointPredictor.predict(
        acdc_df,
        {'type': 'equation', 'equation': 'area > 0.5 * MAX(area)',
         'mode': 'first_true'},
        SizeT=100,
    )
    lower, upper = FrameRange.bounds('t - 1', 't', t=5)

Shared configuration keys, UI labels and templates remain module constants.
Set CELLACDC_TIMEPOINT_DEBUG=1 before launching, or set DEBUG=True in this
module, to log operation timings and repeat the active Python stack every
10 seconds during a slow operation. Diagnostics go to stderr, not Qt widgets.
The historical per-cell constants implementation is retained as commented
code, not an active API. Its old function names are preserved for reference;
restoring it would require migrating those calls to the classes above.
"""
import ast
import importlib.util
import inspect
import itertools
# import keyword
import operator
import os
import re
import sys
import threading
import time
import traceback
import warnings
# from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache, wraps
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any, overload

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, DTypeLike, NDArray
from ._equations import equation_variable_spans

DEBUG = False
DEBUG_STACK_INTERVAL = 10.0
_debug_state = threading.local()


def _debug_enabled() -> bool:
    return DEBUG or os.environ.get(
        'CELLACDC_TIMEPOINT_DEBUG', ''
    ).lower() in ('1', 'true', 'yes')


def debug_trace(func):
    """Trace timepoint operations without depending on the GUI event loop."""
    signature = inspect.signature(func)
    parameters = signature.parameters.values()
    max_positional = None
    if not any(p.kind == p.VAR_POSITIONAL for p in parameters):
        max_positional = sum(
            p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            for p in parameters
        )

    @wraps(func)
    def traced(*args, **kwargs):
        # Qt passes signal arguments (e.g. clicked's checked) to *args slots.
        if max_positional is not None:
            args = args[:max_positional]
        if not _debug_enabled():
            return func(*args, **kwargs)
        depth = getattr(_debug_state, 'depth', 0)
        thread_id = threading.get_ident()
        label = f'{func.__qualname__} [thread={thread_id}, depth={depth}]'
        details = []
        for name, value in signature.bind_partial(*args, **kwargs).arguments.items():
            if isinstance(value, pd.DataFrame):
                details.append(f'{name}.shape={value.shape}')
            elif name in (
                'equation', 'formula', 'condition', 'mode', 'combine',
                'SizeT', 'other_cells', 'main_cell_filter',
                'enforce_dtype_output',
            ):
                details.append(f'{name}={value!r}')
        print(f'[timepoint-debug] BEGIN {label} {" ".join(details)}',
              file=sys.stderr, flush=True)
        start = time.perf_counter()
        stop = threading.Event()
        watcher = None
        if depth == 0:
            def dump_stack():
                while not stop.wait(DEBUG_STACK_INTERVAL):
                    print(
                        f'[timepoint-debug] STILL RUNNING {label} '
                        f'elapsed={time.perf_counter() - start:.3f}s',
                        file=sys.stderr, flush=True
                    )
                    frame = sys._current_frames().get(thread_id)
                    if frame is not None:
                        traceback.print_stack(frame, file=sys.stderr)
                        sys.stderr.flush()
                    del frame

            watcher = threading.Thread(
                target=dump_stack, name='timepoint-debug-watchdog', daemon=True
            )
            watcher.start()
        _debug_state.depth = depth + 1
        outcome = 'END'
        try:
            result = func(*args, **kwargs)
            if func.__name__ == 'constant_main_any':
                print(
                    '[timepoint-debug] constant_main_any '
                    + ('FALLBACK' if result is None
                       else f'FAST PATH eligible_cells={len(result)}'),
                    file=sys.stderr, flush=True
                )
            return result
        except BaseException:
            outcome = 'FAILED'
            traceback.print_exc(file=sys.stderr)
            raise
        finally:
            stop.set()
            if watcher is not None:
                watcher.join()
            _debug_state.depth = depth
            print(
                f'[timepoint-debug] {outcome} {label} '
                f'elapsed={time.perf_counter() - start:.3f}s',
                file=sys.stderr, flush=True
            )

    return traced


DEFAULT_SHORTCUTS = tuple('QWERTYUIOP')
DEFAULT_COLORS = (
    (255, 165, 0), (0, 200, 255), (50, 205, 50), (255, 0, 255),
    (255, 215, 0), (148, 103, 189), (0, 128, 128), (255, 105, 180),
    (160, 82, 45), (128, 128, 0),
)

PREDICTOR_NONE = 'none'
PREDICTOR_EQUATION = 'equation'
PREDICTOR_FILE = 'file'
PREDICTOR_TYPES = (PREDICTOR_NONE, PREDICTOR_EQUATION, PREDICTOR_FILE)

# Pass as ``enforce_dtype_output`` to accept only booleans (no numeric 0/1)
STRICT_BOOL = 'strict_bool'

EQUATION_MODES = {
    'first_true': 'First frame where the equation is true',
    'last_true': 'Last frame where the equation is true',
    'rising_true': 'Every frame where the equation becomes true',
    'max': 'Frame of the maximum value',
    'min': 'Frame of the minimum value',
}

PREDICTOR_FILE_TEMPLATE = '''def predictor(df, entire_df=None):
    """df: acdc_df of a single cell (columns include frame_i and Cell_ID).
    entire_df: acdc_df of all cells (e.g., to use the measurements of other 
    cells). Remove the argument if not needed.
    Return a frame_i (0-based), a list of frame_i, or None."""
    above = df.loc[df['cell_area_pxl'] > 500, 'frame_i']
    return None if above.empty else int(above.iloc[0])
'''

_uid_counter = itertools.count(1)

# For each cell, ``equation`` is evaluated on the rows of its
# source cells (see ``SOURCE_CELLS``, e.g. the cell itself or its mother) 
# and must reduce to a single value per cell (or per frame of the cell if 
# ``per_frame`` is True). Constants are added as columns to the acdc_df used 
# by the predictors. The equation can use CELL_FUNCTIONS, previously defined 
# constants and the number of frames `SizeT`.
# CONSTANT_VARIABLES = {
#     'SizeT': 'Number of frames of the position',
# }

SOURCE_THIS = 'this'
SOURCE_SAME_FRAME = 'same_frame'
SOURCE_MOTHER = 'mother'
SOURCE_SISTER = 'sister'
SOURCE_DAUGHTER = 'daughter'
SOURCE_CUSTOM = 'custom'
FILTER_FRAME_MODES = {
    'any': 'Any',
    'all': 'All',
    'where_true': 'Only where true',
}
# CONSTANT_TIME_SCOPES = {
#     'video': 'Entire video',
#     'frame': 'Per frame (overlapping frames)',
#     'custom': 'Custom frame range',
# }
CELL_FILTER_FILE_TEMPLATE = '''\
def filter_cells(df, enquiring_df):
    """Aligned overlapping frames of candidate and enquiring cell.

    Return a bool (or 0/1), or one bool/0/1 per row.
    """
    # example: exclude the enquiring cell itself
    return df["Cell_ID"].to_numpy() != enquiring_df["Cell_ID"].to_numpy()
'''
# Relatives determined from parent_ID_tree/sister_ID_tree or relative_ID
CELL_RELATIONS = ('mother', 'sister', 'daughter')
SOURCE_CELLS = {
    SOURCE_THIS: 'This cell',
    SOURCE_MOTHER: 'Mother cell',
    SOURCE_SISTER: 'Sister cell(s)',
    SOURCE_DAUGHTER: 'Daughter cell(s)',
    SOURCE_SAME_FRAME: 'Other cells in the same frame',
    SOURCE_CUSTOM: 'Custom...',
}
# Suffix of the constant names created from presets
# SOURCE_SUFFIXES = {
#     SOURCE_THIS: '', SOURCE_MOTHER: '_mother', SOURCE_SISTER: '_sister',
#     SOURCE_DAUGHTER: '_daughter', SOURCE_SAME_FRAME: '_others',
#     SOURCE_CUSTOM: '_custom',
# }
# Prefix of the columns of the cell itself in conditions/equations of 
# other cells in the same frame (e.g., ``Cell_ID == this_relative_ID``)
THIS_PREFIX = 'this_'

_OTHER_FUNCTION_PATTERN = re.compile(
    r'\b(?P<function>MAX|MIN|MEAN|MEDIAN|SUM|STD|COUNT)_OTHER\s*'
    r'\(\s*(?P<column>`[^`]+`|[A-Za-z_]\w*)\s*\)',
    re.IGNORECASE
)
_OTHER_MEASUREMENT_PATTERN = re.compile(
    r'\b(?P<function>MAX|MIN|MEAN|AVERAGE|MEDIAN|SUM|STD|COUNT|NUNIQUE|'
    r'FIRST|LAST)\s*\(\s*(?P<column>`[^`]+_other`|[A-Za-z_]\w*_other)\s*\)',
    re.IGNORECASE
)
_OTHER_AGGREGATIONS = {
    'MAX': 'MAX',
    'MIN': 'MIN',
    'MEAN': 'MEAN',
    'AVERAGE': 'AVERAGE',
    'MEDIAN': 'MEDIAN',
    'SUM': 'SUM',
    'STD': 'STD',
    'COUNT': 'COUNT',
    'NUNIQUE': 'NUNIQUE',
    'FIRST': 'FIRST',
    'LAST': 'LAST',
}


class CellTable:
    """DataFrame layout shared by predictors and equation editors.

    Accept tables indexed by (frame_i, Cell_ID) or containing those columns.
    Flattening and iteration return new layouts without modifying the input;
    iteration sorts each cell's rows by frame_i. Subsetting limits target
    cells without changing the table used for other-cell measurements.
    """

    @staticmethod
    def flatten(acdc_df: pd.DataFrame) -> pd.DataFrame:
        """Return a flat copy with frame_i and Cell_ID available as columns."""
        if 'Cell_ID' in acdc_df.index.names:
            return acdc_df.reset_index()
        if 'frame_i' in acdc_df.index.names:
            return acdc_df.reset_index()
        return acdc_df.reset_index(drop=True)

    @staticmethod
    def iter_cells(acdc_df: pd.DataFrame) -> Iterator[tuple[int, pd.DataFrame]]:
        """Iterate over cells in the ACDC DataFrame, yielding (Cell_ID, df_ID)."""
        df = acdc_df
        if list(df.index.names) == ['frame_i', 'Cell_ID']:
            df = df.reset_index()
        df = df.sort_values(['Cell_ID', 'frame_i'])
        for ID, df_ID in df.groupby('Cell_ID', sort=True):
            yield int(ID), df_ID.reset_index(drop=True)

    @staticmethod
    def subset(
            acdc_df: pd.DataFrame, IDs: Iterable[int] | None
        ) -> pd.DataFrame:
        """Select target IDs, or return the original table when IDs is None."""
        if IDs is None:
            return acdc_df
        if 'Cell_ID' in acdc_df.index.names:
            cellIDs = acdc_df.index.get_level_values('Cell_ID')
        else:
            cellIDs = acdc_df['Cell_ID']
        return acdc_df[np.asarray(pd.Series(cellIDs).isin(list(IDs)))]


class FrameRange:
    """Parse inclusive frame windows relative to the target frame t.

    Bounds allow integer arithmetic, parentheses and t, not arbitrary Python.
    For example, bounds('t - 1', 't', 5) returns (4, 5).
    Bounds are clipped to nonnegative frames and, when SizeT is supplied, to
    the video's last frame. Missing frames are handled by callers, never padded.
    Parsed syntax trees are cached; array_bounds evaluates whole frame arrays.
    """

    @staticmethod
    @lru_cache(maxsize=128)
    def _parse(expression: str) -> ast.AST:
        return ast.parse(expression, mode='eval').body

    @staticmethod
    @overload
    def value(expression: str, t: int) -> int: ...

    @staticmethod
    @overload
    def value(expression: str, t: NDArray[np.object_]) -> int | NDArray[np.object_]: ...

    @staticmethod
    def value(
            expression: str, t: int | NDArray[np.object_]
        ) -> int | NDArray[np.object_]:
        """Evaluate a frame-bound expression using a restricted integer grammar.

        The expression is parsed in ``eval`` mode to produce an AST; it is
        never passed to Python's ``eval`` or executed as user code. The AST is
        then recursively interpreted, accepting only integer literals, the
        name ``t``, parentheses, unary ``+``/``-``, and binary ``+``, ``-``,
        ``*``, ``//`` and ``%``. Each occurrence of ``t`` is replaced by the
        supplied frame index before its surrounding operation is calculated.

        For example, with ``t=10``, ``(t - 2) // 2`` evaluates to ``4``.
        Calls, attributes, other names, decimals, booleans, and all other AST
        forms are rejected with ``ValueError``. Syntax errors, arithmetic
        errors (such as division by zero), and excessive AST recursion are
        also reported as ``ValueError``.

        An object-dtype integer array for ``t`` evaluates all its frames at once;
        expressions without ``t`` still return a scalar.
        """
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError('Frame range bounds cannot be empty.')
        operations = {
            ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
        }

        def evaluate(node: ast.AST) -> int | NDArray[np.object_]:
            if isinstance(node, ast.Constant) and type(node.value) is int:
                return node.value
            if isinstance(node, ast.Name) and node.id == 't':
                return t if isinstance(t, np.ndarray) else int(t)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = evaluate(node.operand)
                return -value if isinstance(node.op, ast.USub) else value
            if isinstance(node, ast.BinOp) and type(node.op) in operations:
                return operations[type(node.op)](evaluate(node.left), evaluate(node.right))
            raise ValueError(
                'Use integers, t, parentheses and +, -, *, // or % in frame bounds.'
            )

        try:
            return evaluate(FrameRange._parse(expression))
        except (SyntaxError, ArithmeticError, RecursionError) as err:
            raise ValueError(f'Invalid frame range expression "{expression}": {err}') from err

    @staticmethod
    def bounds(
            start: str, end: str, t: int, SizeT: int | None = None
        ) -> tuple[int, int]:
        """Return inclusive bounds intersected with the available video frames.

        A reversed expression range raises ``ValueError``. Otherwise, frames
        before zero and after ``SizeT - 1`` are excluded. If the range lies
        entirely outside the available frames, the returned bounds remain
        reversed so consumers using an inclusive range select no rows.
        """
        lower, upper = FrameRange.value(start, t), FrameRange.value(end, t)
        if lower > upper:
            raise ValueError(
                f'Frame range start ({lower}) exceeds end ({upper}) at t={t}.'
            )
        lower = max(lower, 0)
        if SizeT is not None:
            if SizeT < 0:
                raise ValueError('Video frame count cannot be negative.')
            upper = min(upper, SizeT - 1)
        return lower, upper

    @staticmethod
    def array_bounds(
            start: str, end: str, frames: ArrayLike, SizeT: int | None = None
        ) -> tuple[NDArray[np.object_], NDArray[np.object_]]:
        """Return inclusive bounds for all frames without per-frame AST walks."""
        # Python integers preserve scalar overflow and division-error semantics.
        times = np.asarray(frames, dtype=object)
        lower = np.broadcast_to(
            np.asarray(FrameRange.value(start, times), dtype=object), times.shape
        )
        upper = np.broadcast_to(
            np.asarray(FrameRange.value(end, times), dtype=object), times.shape
        )
        reversed_range = lower > upper
        if reversed_range.any():
            i = np.flatnonzero(reversed_range)[0]
            raise ValueError(
                f'Frame range start ({lower[i]}) exceeds end ({upper[i]}) '
                f'at t={times[i]}.'
            )
        lower = np.maximum(lower, 0)
        if SizeT is not None:
            if SizeT < 0:
                raise ValueError('Video frame count cannot be negative.')
            upper = np.minimum(upper, SizeT - 1)
        return lower, upper


class CellEquation:
    """Evaluate measurement expressions and their per-cell functions.

    evaluate() uses pandas DataFrame.eval with CELL_FUNCTIONS and optional
    variables such as SizeT. Callers supply rows in temporal order and choose
    their scope: one cell, or candidate rows within a frame window.
    SHIFT moves observed rows, not absolute video frames. Dtype enforcement
    validates results without lossy casts; boolean output accepts only bool
    or numeric 0/1, with no missing values.
    """

    @staticmethod
    def _numeric(x: ArrayLike) -> NDArray[np.float64]:
        values = np.atleast_1d(np.asarray(x)).ravel()
        if values.dtype.kind in 'biuf':
            return values.astype(float, copy=False)
        return pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(float)

    @staticmethod
    def _nan_aggregate(
            func: Callable[[NDArray[np.float64]], float | np.float64]
        ) -> Callable[[ArrayLike], float]:
        def aggregate(x: ArrayLike) -> float:
            values = CellEquation._numeric(x)
            finite = values[np.isfinite(values)]
            if not finite.size:
                return np.nan
            return float(func(finite))
        return aggregate

    @staticmethod
    def _first(x: ArrayLike) -> float:
        values = CellEquation._numeric(x)
        finite = values[np.isfinite(values)]
        return float(finite[0]) if len(finite) else np.nan

    @staticmethod
    def _last(x: ArrayLike) -> float:
        values = CellEquation._numeric(x)
        finite = values[np.isfinite(values)]
        return float(finite[-1]) if len(finite) else np.nan

    @staticmethod
    def _shift(x: ArrayLike, n: int = 1) -> NDArray[np.float64]:
        values = CellEquation._numeric(x)
        n = int(n)
        shifted = np.full_like(values, np.nan)
        if n == 0:
            return values
        if abs(n) < len(values):
            if n > 0:
                shifted[n:] = values[:-n]
            else:
                shifted[:n] = values[-n:]
        return shifted

    @staticmethod
    def _diff(x: ArrayLike) -> NDArray[np.float64]:
        return CellEquation._numeric(x) - CellEquation._shift(x, 1)

    @staticmethod
    def _count(x: ArrayLike) -> int:
        return int(np.isfinite(CellEquation._numeric(x)).sum())

    @staticmethod
    def _any(x: ArrayLike) -> bool:
        return bool(pd.Series(np.asarray(x).ravel()).fillna(False).astype(bool).any())

    @staticmethod
    def _all(x: ArrayLike) -> bool:
        return bool(pd.Series(np.asarray(x).ravel()).fillna(False).astype(bool).all())

    @staticmethod
    def _nunique(x: ArrayLike) -> int:
        return int(pd.Series(np.atleast_1d(np.asarray(x)).ravel()).nunique())

    @staticmethod
    def resolver():
        """Return the function namespace passed to DataFrame.eval."""
        return {name: func for name, (func, _) in CELL_FUNCTIONS.items()}

    @staticmethod
    def enforce_dtype(
            values, dtype: DTypeLike | pd.api.extensions.ExtensionDtype,
            length: int
        ):
        """Validate output without lossy casts; bool also accepts numeric 0/1,
        while ``STRICT_BOOL`` accepts only booleans (missing values -> False)."""
        if isinstance(dtype, str) and dtype == STRICT_BOOL:
            if np.ndim(values) == 0:
                return bool(CellFilter.strict_boolean_values(values, 1)[0])
            mask = CellFilter.strict_boolean_values(values, length)
            if isinstance(values, pd.Series):
                return pd.Series(mask, index=values.index, name=values.name)
            return mask
        expected = pd.api.types.pandas_dtype(dtype)
        if pd.api.types.is_bool_dtype(expected):
            mask = CellFilter.boolean_values(values, length)
            if isinstance(values, pd.Series):
                return pd.Series(mask, index=values.index, name=values.name)
            return mask if np.ndim(values) else bool(np.asarray(values).item())
        actual = values.dtype if hasattr(values, 'dtype') else np.asarray(values).dtype
        if not pd.api.types.is_dtype_equal(actual, expected):
            raise ValueError(
                f'The equation must return dtype {expected}; got {actual}.'
            )
        return values

    @staticmethod
    @debug_trace
    def evaluate(
            df: pd.DataFrame, equation: str,
            variables: Mapping[str, object] | None = None, *,
            enforce_dtype_output: (
                DTypeLike | pd.api.extensions.ExtensionDtype | None
            ) = None
        ):
        """Evaluate ``equation`` on the rows of ``df`` with the functions of 
        ``CELL_FUNCTIONS`` (e.g., ``area > 0.5 * MAX(area)``) and the 
        additional ``variables`` (e.g., ``{'SizeT': 10}``) available.

        ``enforce_dtype_output`` validates the result's dtype without silently
        casting it. Bool accepts bool or numeric 0/1, as required by cell filters.

        Returns the scalar or per-row result from pandas. This method neither
        sorts nor groups rows: use CellTable.iter_cells() for per-cell scopes.
        Invalid expressions and incompatible output dtypes raise exceptions.
        """
        resolvers = [CellEquation.resolver()]
        if variables:
            resolvers.append(dict(variables))
        if _debug_enabled():
            print(f"{equation =}, {variables =}", file=sys.stderr, flush=True)
        if '`' not in equation and '@' not in equation:
            # DataFrame.eval builds a resolver for every column; wide
            # measurement tables made each evaluation needlessly expensive.
            used = CellEquation.names(equation)
            if 'index' not in used and 'columns' not in used:
                keep = df.columns.isin(list(used))
                if not keep.all():
                    df = df.loc[:, keep]
        values = df.eval(equation, resolvers=resolvers)
        if enforce_dtype_output is not None:
            values = CellEquation.enforce_dtype(
                values, enforce_dtype_output, len(df)
            )
        return values

    @staticmethod
    def names(equation: str | None) -> set[str]:
        """Input names, excluding string literals, function names and keywords."""
        return {name for name, _, _ in equation_variable_spans(equation or '')}


# Functions available in predictor equations, evaluated on the rows of a 
# single cell (one row per frame, sorted by frame_i). NaNs are ignored.
CELL_FUNCTIONS = {
    'MIN': (CellEquation._nan_aggregate(np.min), 'Minimum over all frames of the cell'),
    'MAX': (CellEquation._nan_aggregate(np.max), 'Maximum over all frames of the cell'),
    'MEAN': (CellEquation._nan_aggregate(np.mean), 'Mean over all frames of the cell'),
    'AVERAGE': (CellEquation._nan_aggregate(np.mean), 'Same as MEAN'),
    'MEDIAN': (CellEquation._nan_aggregate(np.median), 'Median over all frames of the cell'),
    'SUM': (CellEquation._nan_aggregate(np.sum), 'Sum over all frames of the cell'),
    'STD': (CellEquation._nan_aggregate(np.std), 'Standard deviation over all frames of the cell'),
    'FIRST': (CellEquation._first, 'Value at the first frame of the cell'),
    'LAST': (CellEquation._last, 'Value at the last frame of the cell'),
    'COUNT': (CellEquation._count, 'Number of frames of the cell with a value'),
    'NUNIQUE': (
        CellEquation._nunique, 
        'Number of unique values, e.g. NUNIQUE(Cell_ID): number of cells'
    ),
    'DIFF': (CellEquation._diff, 'Change from the previous frame (NaN at the first frame)'),
    'SHIFT': (
        CellEquation._shift, 
        'SHIFT(x, n): value n frames earlier (negative n: later). Default n=1'
    ),
    'any': (CellEquation._any, 'True if the condition is true in at least one frame of the cell'),
    'all': (CellEquation._all, 'True only if the condition is true in every frame of the cell'),
    'ANY': (CellEquation._any, 'True if the condition is true in at least one frame of the cell'),
    'ALL': (CellEquation._all, 'True only if the condition is true in every frame of the cell'),
}


class PredictorDependencies:
    """Order predictors by the table columns read by equations and filters."""

    @staticmethod
    def inputs(predictor: Mapping[str, Any]) -> dict[str, bool]:
        """Map input columns to whether they require predictions on other cells."""
        if predictor.get('type') != PREDICTOR_EQUATION:
            return {}
        inputs: dict[str, bool] = {}

        def add_inputs(expression, candidate=False):
            spans = set(equation_variable_spans(expression))
            legacyOther = {
                match['column'].strip('`')
                for match in _OTHER_FUNCTION_PATTERN.finditer(expression)
                if (
                    match['column'].strip('`'), *match.span('column')
                ) in spans
            }
            for name, _, _ in spans:
                otherCell = candidate
                if candidate and name.startswith(THIS_PREFIX):
                    name = name[len(THIS_PREFIX):]
                    otherCell = False
                elif not candidate and name.endswith('_other'):
                    name = name[:-len('_other')]
                    otherCell = True
                elif name in legacyOther:
                    otherCell = True
                inputs[name] = inputs.get(name, False) or otherCell

        add_inputs(predictor.get('equation', ''))
        add_inputs((predictor.get('main_cell_filter') or {}).get('condition', ''))
        other = predictor.get('other_cells')
        if other is not None and any(inputs.values()):
            add_inputs(other.get('condition', ''), candidate=True)
        return inputs

    @staticmethod
    def order(
            predictors: Mapping[str, Mapping[str, Any]],
            targets: Iterable[str] | None = None,
        ) -> list[str]:
        """Stable dependency-first order, including prerequisites of targets."""
        dependencies = {
            column: PredictorDependencies.inputs(predictor)
            for column, predictor in predictors.items()
        }
        ordered: list[str] = []
        visiting: list[str] = []
        visited: set[str] = set()

        def visit(column: str) -> None:
            if column in visiting:
                cycle = visiting[visiting.index(column):] + [column]
                raise ValueError(
                    'Circular predictor dependency: ' + ' -> '.join(cycle)
                )
            if column in visited:
                return
            visiting.append(column)
            for dependency in predictors:
                if dependency in dependencies[column]:
                    visit(dependency)
            visiting.pop()
            visited.add(column)
            ordered.append(column)

        for column in predictors if targets is None else targets:
            visit(column)
        return ordered


class CellFilter:
    """Validate filter settings, build presets and check boolean masks.

    Main-cell filters retain cells matching any/all of their observed frames.
    Other-cell filters support any/all candidate frames or matching rows only.
    other_presets() returns visible equations and inclusive frame bounds:
    budding relatives use t, symmetric mothers t - 1, daughters t + 1.
    In candidate equations, this_<column> denotes the target's value at t.
    Validators return settings with defaults and raise ValueError for invalid
    input. They do not mutate the supplied settings.
    """

    @staticmethod
    def default_other() -> dict[str, str]:
        """Return fresh default other-cell settings for the current frame."""
        return {
            'source': SOURCE_SAME_FRAME,
            'condition': '',
            'condition_mode': 'where_true',
            'range_start': 't',
            'range_end': 't',
        }

    @staticmethod
    def validate_other(other_cells: dict[str, str]) -> dict[str, str]:
        """Return validated other-cell settings merged with defaults."""
        if not isinstance(other_cells, dict):
            raise ValueError('Invalid other-cell settings.')
        settings = CellFilter.default_other()
        settings.update(other_cells)
        if settings['source'] not in (
                SOURCE_SAME_FRAME, 'mother', 'sister', 'daughter'
            ):
            raise ValueError('Invalid other-cell source.')
        if not isinstance(settings['condition'], str):
            raise ValueError('Invalid other-cell filter condition.')
        if settings['condition_mode'] not in FILTER_FRAME_MODES:
            raise ValueError('Invalid other-cell filter mode.')
        for field in ('range_start', 'range_end'):
            if not isinstance(settings[field], str) or not settings[field].strip():
                raise ValueError(f'Other-cell {field.replace("_", " ")} cannot be empty.')
        FrameRange.bounds(settings['range_start'], settings['range_end'], 0)
        return settings

    @staticmethod
    def default_main() -> dict[str, str]:
        """Return fresh main-cell settings that retain all target cells."""
        return {'condition': '', 'condition_mode': 'any'}

    @staticmethod
    def validate_main(main_cell_filter: dict[str, str]) -> dict[str, str]:
        """Return validated main-cell settings merged with defaults."""
        if not isinstance(main_cell_filter, dict):
            raise ValueError('Invalid main-cell filter settings.')
        settings = CellFilter.default_main()
        settings.update(main_cell_filter)
        if not isinstance(settings['condition'], str):
            raise ValueError('Invalid main-cell filter condition.')
        if settings['condition_mode'] not in ('any', 'all'):
            raise ValueError('Invalid main-cell filter mode.')
        return settings

    @staticmethod
    def other_presets(
            acdc_df: pd.DataFrame | None
        ) -> dict[str, tuple[str, str, str]]:
        """Return (filter equation, start, end) presets relative to frame t.

        Relations and other-cell identity use segmentation Cell_ID only.
        """
        columns = CellTable.flatten(acdc_df).columns if acdc_df is not None else ()
        presets = {SOURCE_SAME_FRAME: ('', 't', 't')}
        if {'relative_ID', 'relationship'}.issubset(columns):
            presets.update({
                SOURCE_MOTHER: (
                    '(Cell_ID == this_relative_ID) & '
                    '(this_relationship == "bud")', 't', 't'
                ),
                SOURCE_DAUGHTER: (
                    '(relative_ID == this_Cell_ID) & '
                    '(relationship == "bud")', 't', 't'
                ),
            })
        elif 'parent_ID_tree' in columns:
            presets.update({
                SOURCE_MOTHER: (
                    'Cell_ID == this_parent_ID_tree', 't - 1', 't - 1'
                ),
                SOURCE_SISTER: (
                    '(parent_ID_tree > 0) & '
                    '(parent_ID_tree == this_parent_ID_tree)', 't', 't'
                ),
                SOURCE_DAUGHTER: (
                    'parent_ID_tree == this_Cell_ID', 't + 1', 't + 1'
                ),
            })
        return presets

    @staticmethod
    def boolean_values(values: ArrayLike, length: int) -> NDArray[np.bool_]:
        """Validate and broadcast filter results without truthiness coercion."""
        array = np.asarray(values)
        if array.ndim > 1 or (array.ndim == 1 and len(array) != length):
            raise ValueError('A cell filter must return a scalar or one value per frame.')
        if not (pd.api.types.is_bool_dtype(array.dtype)
                or pd.api.types.is_numeric_dtype(array.dtype)):
            raise ValueError('A cell filter must return bool or numeric 0/1 values.')
        if not np.isin(array, [0, 1]).all():
            raise ValueError('A cell filter must return only bool or 0/1 (no missing values).')
        return np.broadcast_to(array, length).astype(bool)

    @staticmethod
    def strict_boolean_values(values: ArrayLike, length: int) -> NDArray[np.bool_]:
        """Validate results that must be booleans: a scalar bool (broadcast to
        ``length``) or one bool per row. Numeric (including 0/1) and other 
        values are rejected; missing values (e.g. ``pd.NA`` of nullable 
        booleans) are treated as False."""
        if isinstance(values, pd.DataFrame):
            raise ValueError(
                'The result must be a single boolean per row, not a table.'
            )
        array = (
            values.array if isinstance(values, (pd.Series, pd.Index)) 
            else values
        )
        if np.ndim(array) == 0:
            item = np.asarray(array).item() if hasattr(array, 'dtype') else array
            if not isinstance(item, (bool, np.bool_)):
                raise ValueError(
                    'The result must be a boolean (True/False); got '
                    f'{type(item).__name__} {item!r}.'
                )
            return np.full(length, bool(item))
        if not hasattr(array, 'dtype'):
            array = np.asarray(array)
        if array.ndim != 1 or len(array) != length:
            raise ValueError(
                'The result must be a single boolean or one boolean per row '
                f'({length}); got shape {np.shape(array)}.'
            )
        dtype = array.dtype
        if pd.api.types.is_bool_dtype(dtype):
            return pd.array(array, dtype='boolean').fillna(False).to_numpy(
                dtype=bool
            )
        if not pd.api.types.is_object_dtype(dtype):
            raise ValueError(
                f'The result must be boolean (True/False); got dtype {dtype}.'
            )
        objects = np.asarray(array, dtype=object)
        isMissing = pd.isna(objects)
        isBool = np.fromiter(
            (isinstance(value, (bool, np.bool_)) for value in objects),
            dtype=bool, count=len(objects)
        )
        invalid = ~(isBool | isMissing)
        if invalid.any():
            value = objects[invalid.argmax()]
            raise ValueError(
                'The result must be boolean (True/False); got '
                f'{type(value).__name__} {value!r}.'
            )
        result = np.zeros(len(objects), dtype=bool)
        result[isBool] = objects[isBool].astype(bool)
        return result

    @staticmethod
    def evaluate(
            acdc_df: pd.DataFrame, condition: str | None,
            variables: Mapping[str, object] | None = None
        ) -> NDArray[np.bool_]:
        """Boolean mask (aligned with the rows of ``acdc_df``) where 
        ``condition`` is true, evaluated on the rows of each cell (hence 
        ``CELL_FUNCTIONS`` can be used)."""
        df = CellTable.flatten(acdc_df)
        mask = pd.Series(False, index=df.index)
        if not condition or not condition.strip():
            mask[:] = True
            return mask.to_numpy()
        for ID, df_ID in df.groupby('Cell_ID', sort=False):
            df_ID = df_ID.sort_values('frame_i')
            try:
                values = CellEquation.evaluate(
                    df_ID.reset_index(drop=True), condition, variables=variables,
                    enforce_dtype_output=bool
                )
            except Exception as err:
                raise ValueError(
                    f'Evaluating the condition "{condition}" failed for Cell_ID '
                    f'{ID}: {err}'
                ) from err
            mask[df_ID.index] = CellFilter.boolean_values(values, len(df_ID))
        return mask.to_numpy()

    @staticmethod
    def _this_columns(
            df: pd.DataFrame, *equations: str | None
        ) -> list[str]:
        names = set().union(*(CellEquation.names(eq) for eq in equations))
        return [
            name[len(THIS_PREFIX):] for name in sorted(names)
            if name.startswith(THIS_PREFIX) and name[len(THIS_PREFIX):] in df.columns
        ]

    @staticmethod
    def load_function(filepath: str):
        """Load a callback accepting aligned candidate and enquiring tables."""
        spec = importlib.util.spec_from_file_location(
            '_acdc_cell_filter', filepath
        )
        if spec is None or spec.loader is None:
            raise ImportError(f'Cannot import "{filepath}".')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        func = getattr(module, 'filter_cells', None)
        if not callable(func):
            raise ValueError('The filter file must define filter_cells(df, enquiring_df).')
        inspect.signature(func).bind(pd.DataFrame(), pd.DataFrame())
        return func

    # @staticmethod
    # def _relation_pairs(
    #         lineage: tuple[
    #             Mapping[int, int], Mapping[int, Iterable[int]],
    #             Mapping[int, Iterable[int]]
    #         ],
    #         relation: str
    #     ) -> pd.DataFrame:
    #     mothers, sisters, daughters = lineage
    #     if relation == 'mother':
    #         pairs = list(mothers.items())
    #     else:
    #         related = sisters if relation == 'sister' else daughters
    #         pairs = [(ID, other) for ID, others in related.items() for other in others]
    #     return pd.DataFrame(pairs, columns=['Cell_ID', '_other_ID'], dtype=int)


class OtherCellMeasurements:
    """Resolve other-cell aggregates without per-cell constants.

    prepare_inputs() recognizes MEAN(area_other) and legacy MEAN_OTHER(area)
    calls, selects candidates in each target row's inclusive frame window,
    applies filters, and replaces aggregate calls with temporary columns.
    It returns (table, equation), preserving input row alignment and leaving
    the original table unchanged. Invalid inputs raise contextual errors.
    Frame windows are located in a sorted candidate table with binary search;
    targets at the same frame reuse window offsets, not target-specific filters.
    Direct *_other inputs use the first distinct nonmissing value in the
    prepared table. When evaluating equations, distinct candidate-row values
    are tried as alternatives and boolean results are combined with any().
    Empty candidate sets yield zero.
    Row-local filter expressions run once per target window; expressions with
    function calls retain per-candidate-cell scope for temporal operations.
    Main-cell Any filters with constant own-cell inputs can compare unique
    candidate rows across the union of their frame windows once per cell.
    Their bounds, merged row offsets and empty-window checks use whole arrays.
    """

    @staticmethod
    @debug_trace
    def constant_main_any(
            acdc_df: pd.DataFrame, equation: str,
            other_cells: dict[str, str] | None, SizeT: int | None,
            IDs: Iterable[int] | None
        ) -> list[int] | None:
        """Return eligible IDs, or None when temporal evaluation is required.

        Fast path for main-cell filters in ``any`` mode such as
        ``any(Cell_ID == parent_ID_tree_other)``. The general path evaluates
        the filter once per (cell, frame, candidate row) combination. Here,
        if every own-cell input is constant over the cell's frames, the
        result of "is the condition true in at least one frame" is the same
        as: collect the distinct candidate ``*_other`` values over all frames
        of the cell, and test the equation once against each of them. A cell
        is eligible when any comparison is true.

        Returning None is always safe: the caller then falls back to the
        general (slower) implementation with identical semantics.
        """
        # Only plain, parseable expressions can be handled.
        try:
            tree = ast.parse(equation, mode='eval')
        except SyntaxError:
            return None
        body = tree.body
        # any(<expr>) over frames is what this fast path computes anyway, so
        # the wrapper is stripped and <expr> is evaluated row-wise instead.
        if (
            isinstance(body, ast.Call) and isinstance(body.func, ast.Name)
            and body.func.id in ('any', 'ANY')
            and len(body.args) == 1 and not body.keywords
        ):
            body = body.args[0]
        expression = ast.unparse(body)
        # Function calls (MEAN, SHIFT, ...) depend on the frames grouped
        # together, which this approach does not preserve.
        if not OtherCellMeasurements._is_rowwise_condition(expression):
            return None
        names = CellEquation.names(expression)
        # direct maps equation names (area_other) to table columns (area).
        direct = {name: name[:-6] for name in names if name.endswith('_other')}
        if not direct or other_cells is None:
            return None
        settings = CellFilter.validate_other(other_cells)
        try:
            # Same condition as prepare_inputs, including lineage presets.
            condition = OtherCellMeasurements._candidate_condition(
                settings, acdc_df
            )
        except ValueError:
            return None  # the general path reports the error
        # An other-cell condition must filter single rows independently
        # (where_true); any/all-per-candidate-cell needs temporal grouping.
        if condition and (
            settings['condition_mode'] != 'where_true'
            or not OtherCellMeasurements._is_rowwise_condition(condition)
        ):
            return None
        flat = CellTable.flatten(acdc_df)
        variables = {'SizeT': SizeT} if SizeT is not None else None
        own = names - direct.keys() - (variables or {}).keys()
        # Bail out when inputs are missing, when an *_other name clashes with
        # a real column, or when the source columns contain NaN (NaN handling
        # of the general path is not replicated here).
        if (
            not own.issubset(flat.columns)
            or any(name in flat or column not in flat for name, column in direct.items())
            or flat[list(direct.values())].isna().any().any()
        ):
            return None
        this_columns = CellFilter._this_columns(flat, condition)
        cells = list(CellTable.iter_cells(CellTable.subset(flat, IDs)))
        required = own | set(this_columns)
        # The shortcut is only valid if the enquiring cell's own values do
        # not change over time.
        if any(
            cell[list(required)].nunique(dropna=False).gt(1).any()
            for _, cell in cells
        ):
            return None
        # Candidate rows are looked up by frame via binary search on this
        # sorted table.
        sorted_df = flat.sort_values(['frame_i', 'Cell_ID'])
        frames = sorted_df['frame_i'].to_numpy()
        eligible = []
        ambiguous = set()
        for ID, cell in cells:
            # Determine the valid frame range for the current cell.
            lower, upper = FrameRange.array_bounds(
                settings['range_start'], settings['range_end'],
                cell['frame_i'].to_numpy(dtype=int), SizeT
            )
            left = np.searchsorted(frames, lower, side='left')
            right = np.searchsorted(frames, upper, side='right')
            valid = (lower <= upper) & (left < right)
            left, right = left[valid], right[valid]
            # Arbitrary bound expressions need not preserve temporal order.
            order = np.argsort(left)
            left, right = left[order], np.maximum.accumulate(right[order])
            if left.size:
                starts = np.flatnonzero(np.r_[True, left[1:] > right[:-1]])
                ends = np.r_[starts[1:] - 1, len(left) - 1]
                left, right = left[starts], right[ends]
            lengths = right - left
            offsets = np.cumsum(lengths) - lengths
            positions = np.arange(lengths.sum()) + np.repeat(left - offsets, lengths)
            # Apply the other-cell condition once for the whole cell using
            # its (constant) own values; the cell itself is excluded.
            candidates = OtherCellMeasurements._select_candidates(
                sorted_df.iloc[positions], cell.iloc[0], condition,
                settings['condition_mode'], this_columns, variables,
                rowwise=True
            )
            candidate_frames = candidates['frame_i'].to_numpy()
            # A frame without candidates evaluates the direct input as 0 in
            # the general path, so a 0 comparison row is added to match.
            has_empty_window = np.any(
                (lower > upper)
                | (np.searchsorted(candidate_frames, lower, side='left')
                   == np.searchsorted(candidate_frames, upper, side='right'))
            )
            # One row per distinct candidate value (or value combination),
            # with own-cell values broadcast, so a single vectorized
            # evaluation replaces all per-frame evaluations.
            comparisons = candidates[list(direct.values())].drop_duplicates()
            comparisons = comparisons.rename(
                columns={column: name for name, column in direct.items()}
            ).reset_index(drop=True)
            if has_empty_window:
                comparisons.loc[len(comparisons)] = [0] * len(direct)
            for name in own:
                comparisons[name] = cell[name].iloc[0]
            # Remember inputs with several distinct values to warn once.
            ambiguous.update(
                name for name in direct if comparisons[name].nunique() > 1
            )
            try:
                values = CellEquation.evaluate(
                    comparisons, expression, variables, enforce_dtype_output=bool
                )
                if CellFilter.boolean_values(values, len(comparisons)).any():
                    eligible.append(ID)
            except Exception as err:
                raise ValueError(
                    f'Main-cell filter failed for Cell_ID {ID}: {err}'
                ) from err
        # Same single warning as the general path in prepare_inputs.
        if ambiguous:
            warnings.warn(
                f'Other-cell input(s) {", ".join(sorted(ambiguous))} have multiple '
                'distinct values; evaluating the equation against each candidate row.',
                UserWarning, stacklevel=2
            )
        return eligible

    @staticmethod
    def _candidate_condition(
            settings: dict[str, str], acdc_df: pd.DataFrame
        ) -> str:
        """User condition combined with the lineage/budding source preset."""
        condition = settings['condition'].strip()
        if settings['source'] == SOURCE_SAME_FRAME:
            return condition
        presets = CellFilter.other_presets(acdc_df)
        if settings['source'] not in presets:
            raise ValueError(
                'The selected other-cell source requires lineage tree or '
                'budding relationship annotations.'
            )
        sourceCondition = presets[settings['source']][0]
        return (
            f'({sourceCondition}) & ({condition})'
            if condition else sourceCondition
        )

    @staticmethod
    def _is_rowwise_condition(condition: str) -> bool:
        """Only batch expressions whose values cannot depend on row grouping."""
        try:
            tree = ast.parse(condition, mode='eval')
        except SyntaxError:
            # Pandas-only syntax (for example backticks) keeps the general path.
            return False
        allowed = (
            ast.Expression, ast.Compare, ast.BinOp, ast.UnaryOp, ast.BoolOp,
            ast.Name, ast.Load, ast.Constant, ast.operator, ast.unaryop,
            ast.boolop, ast.cmpop,
        )
        return all(
            isinstance(node, allowed)
            and not (isinstance(node, ast.Name) and node.id in ('index', 'columns'))
            for node in ast.walk(tree)
        )

    @staticmethod
    def _select_candidates(
            window: pd.DataFrame, target: pd.Series, condition: str,
            mode: str, this_columns: list[str],
            variables: Mapping[str, object] | None, *,
            rowwise: bool = False
        ) -> pd.DataFrame:
        """Filter one target's candidates, preserving temporal aggregate order."""
        candidates = window[window['Cell_ID'] != target['Cell_ID']]
        if not condition or candidates.empty:
            return candidates
        if rowwise:
            equality = re.fullmatch(
                r'([A-Za-z_]\w*)\s*==\s*this_([A-Za-z_]\w*)', condition
            )
            if (
                equality is not None
                and equality[1] in candidates
                and equality[2] in this_columns
                and not equality[1].startswith(THIS_PREFIX)
                and equality[1] not in CELL_FUNCTIONS
                and (not variables or equality[1] not in variables)
                and not pd.api.types.is_extension_array_dtype(candidates[equality[1]])
            ):
                mask = candidates[equality[1]].eq(target[equality[2]]).to_numpy()
            else:
                evaluation = candidates.reset_index(drop=True).copy()
                for column in this_columns:
                    evaluation[f'{THIS_PREFIX}{column}'] = target[column]
                try:
                    mask = CellFilter.boolean_values(CellEquation.evaluate(
                        evaluation, condition, variables, enforce_dtype_output=bool
                    ), len(candidates))
                except Exception as err:
                    raise ValueError(
                        f'Other-cell filter failed at frame {int(target["frame_i"])} '
                        f'for enquiring cell {int(target["Cell_ID"])} '
                        f'with condition {condition!r}: {err}'
                    ) from err
            if mode != 'where_true':
                mask = pd.Series(mask).groupby(
                    candidates['Cell_ID'].to_numpy(), sort=False
                ).transform(mode).to_numpy()
            return candidates.loc[mask]
        selected = []
        for candidateID, rows in candidates.groupby('Cell_ID', sort=False):
            evaluation = rows.reset_index(drop=True).copy()
            for column in this_columns:
                evaluation[f'{THIS_PREFIX}{column}'] = target[column]
            try:
                mask = CellFilter.boolean_values(CellEquation.evaluate(
                    evaluation, condition, variables, enforce_dtype_output=bool
                ), len(rows))
            except Exception as err:
                raise ValueError(
                    f'Other-cell filter failed at frame {int(target["frame_i"])} '
                    f'for enquiring cell {int(target["Cell_ID"])}, '
                    f'candidate {candidateID} with condition {condition!r}: {err}'
                ) from err
            if mode == 'where_true':
                selected.append(rows.loc[mask])
            elif (mode == 'any' and mask.any()) or (mode == 'all' and mask.all()):
                selected.append(rows)
        return (
            pd.concat(selected).sort_values(['frame_i', 'Cell_ID'])
            if selected else candidates.iloc[:0]
        )

    @staticmethod
    @debug_trace
    def evaluate_with_alternatives(
            df: pd.DataFrame, equation: str,
            alternatives: dict | None,
            variables: Mapping[str, object] | None = None, *,
            combine: str = 'any',
            enforce_dtype_output: (
                DTypeLike | pd.api.extensions.ExtensionDtype | None
            ) = None
        ):
        """Evaluate against aligned direct-input alternatives for each row.

        ``alternatives`` comes from prepare_inputs(return_alternatives=True):
        ``columns`` are the direct ``*_other`` input names and ``by_target``
        maps (frame_i, Cell_ID) to a list of value tuples, one per distinct
        candidate value (aligned across several inputs). Each row is
        evaluated once per tuple and the results are reduced per row:
        ``combine='any'`` (filters) is true if any alternative is true,
        'max'/'min' (predictors) take the extreme numeric result.

        All (row, alternative) pairs are expanded and evaluated together.
        When no function call touches an ``*_other`` input, functions are
        evaluated once on the original cell timeline before expansion.
        Calls involving direct alternatives retain their expanded-row scope.
        """
        # No alternatives supplied: evaluate the original table directly.
        if not alternatives:
            return CellEquation.evaluate(
                df, equation, variables=variables,
                enforce_dtype_output=enforce_dtype_output
            )

        columns = alternatives['columns']
        by_target = alternatives['by_target']
        # Classify the equation (see strategies in the docstring).
        try:
            tree = ast.parse(equation, mode='eval')
        except SyntaxError:
            tree = None
        has_function = tree is None or any(
            isinstance(node, ast.Call) for node in ast.walk(tree)
        )
        has_direct_function = tree is None or any(
            isinstance(node, ast.Call)
            and any(
                isinstance(child, ast.Name) and child.id in columns
                for child in ast.walk(node)
            )
            for node in ast.walk(tree)
        )

        def reduce_grouped(positions, array):
            # Vectorized reduce_options: ``positions`` maps each expanded
            # result back to its source row; rows without results become NaN.
            series = pd.Series(array)
            if combine in ('max', 'min'):
                reduced = pd.to_numeric(series, errors='coerce').groupby(
                    positions
                ).agg(combine)
            else:
                truth = series.dropna().astype(bool).reindex(
                    series.index, fill_value=False
                )
                reduced = truth.groupby(positions).any()
            fill_value = (
                False if has_function and not has_direct_function
                and combine not in ('max', 'min') else np.nan
            )
            return pd.Series(
                reduced.reindex(range(len(df)), fill_value=fill_value).to_numpy(),
                index=df.index
            )

        def frame_values(name):
            # frame_i / Cell_ID may be columns or index levels.
            if name in df.columns:
                return df[name].to_numpy()
            return df.index.get_level_values(name).to_numpy()

        # Target row key and own values (fallback when a row has no
        # alternatives, i.e. the value already in the table is used).
        keys = list(zip(
            frame_values('frame_i').astype(int).tolist(),
            frame_values('Cell_ID').astype(int).tolist(),
        ))
        ownValues = [df[column].to_numpy() for column in columns]

        def target_options(position):
            """Return options for the zero-based df row offset, not its index label."""
            options = by_target.get(keys[position])
            if options is None:
                options = [tuple(values[position] for values in ownValues)]
            return options

        options = [target_options(position) for position in range(len(df))]
        counts = np.fromiter(map(len, options), dtype=np.intp, count=len(df))
        # positions[i] is the zero-based row offset in df for expanded row i,
        # not a frame number or index label: counts [2, 1] -> [0, 0, 1].
        positions = np.repeat(np.arange(len(df)), counts)
        if not positions.size:
            if has_function and not has_direct_function:
                # No comparisons: return the empty reduction for each source row.
                return pd.Series(
                    np.nan if combine in ('max', 'min') else False,
                    index=df.index
                )
            # Other equation types retain plain evaluation when nothing expands.
            return CellEquation.evaluate(
                df, equation, variables=variables,
                enforce_dtype_output=enforce_dtype_output
            )

        source = df
        if tree is not None and has_function and not has_direct_function:
            # Own-cell calls do not depend on alternatives. Compute them before
            # repeating rows so MEAN/SHIFT/etc. see the original cell timeline.
            source = df.copy()
            cached_calls = {}
            # Temporary names must not shadow measurements or equation variables.
            reserved = (
                set(df.columns) | CellEquation.names(equation)
                | set(variables or {})
            )

            class CacheCalls(ast.NodeTransformer):
                """Replace whole calls with cached scalar or per-row columns."""

                def visit_Call(self, node: ast.Call):
                    # Evaluate the whole outer call, including nested calls.
                    expression = ast.unparse(node)
                    if expression not in cached_calls:
                        name = f'_acdc_cached_call_{len(cached_calls)}'
                        while name in reserved:
                            name += '_'
                        reserved.add(name)
                        cached_calls[expression] = name
                        values = CellEquation.evaluate(
                            df, expression, variables=variables
                        )
                        source[name] = np.asarray(values)
                    return ast.copy_location(
                        ast.Name(id=cached_calls[expression], ctx=ast.Load()), node
                    )

            equation = ast.unparse(CacheCalls().visit(tree))

        # Flatten options in the same row-major order as positions, keeping
        # multiple *_other values from each candidate tuple aligned.
        direct_values = pd.DataFrame.from_records(
            itertools.chain.from_iterable(options), columns=columns
        )
        # One take instead of concatenating a single-row frame per option.
        evaluation = source.iloc[positions].reset_index(drop=True)
        for column in columns:
            evaluation[column] = direct_values[column].to_numpy()
        values = CellEquation.evaluate(
            evaluation, equation, variables=variables,
            enforce_dtype_output=enforce_dtype_output
        )
        array = np.asarray(values)
        if array.ndim == 0:
            array = np.repeat(array, len(evaluation))
        return reduce_grouped(positions, array)

    @staticmethod
    @debug_trace
    def prepare_inputs(
            acdc_df: pd.DataFrame, equation: str,
            other_cells: dict[str, str] | None, SizeT: int | None,
            IDs: Iterable[int] | None = None, *,
            max_targets: int | None = None,
            return_alternatives: bool = False
        ) -> tuple[pd.DataFrame, str] | tuple[pd.DataFrame, str, dict | None]:
        """Resolve other-cell aggregates for target IDs using all candidate cells.

        The returned table contains only target rows, in their original order
        and with their original index. With IDs=None, all rows are targets.
        max_targets keeps only the first target rows (for quick previews),
        while still using the full table as candidates.
        With return_alternatives=True, a third value holds the distinct
        direct ``*_other`` options per target for evaluate_with_alternatives
        (None without direct inputs). They are deliberately not stored in
        DataFrame.attrs: pandas deep-copies attrs on every derived frame.
        Targets share frame windows; equality-filtered windows are located
        by binary search within each value's pre-indexed candidate rows.
        """
        # Two syntaxes: legacy MEAN_OTHER(col) and MEAN(col_other). Direct
        # inputs are bare ``col_other`` names outside such calls.
        legacyCalls = list(_OTHER_FUNCTION_PATTERN.finditer(equation))
        measurementCalls = list(_OTHER_MEASUREMENT_PATTERN.finditer(equation))
        calls = legacyCalls + measurementCalls
        directInputs = {
            name: name[:-len('_other')]
            for name, start, end in equation_variable_spans(equation)
            if name.endswith('_other')
            and not any(call.start() <= start and end <= call.end() for call in calls)
        }
        # No other-cell input: only subset the targets.
        if not calls and not directInputs:
            targets = CellTable.subset(acdc_df, IDs)
            if max_targets is not None:
                targets = targets.iloc[:max_targets]
            if return_alternatives:
                return targets, equation, None
            return targets, equation
        if other_cells is None:
            raise ValueError(
                'Configure Other-cell measurements before using *_OTHER(...) '
                'or suffixed _other inputs.'
            )
        settings = CellFilter.validate_other(other_cells)
        flatDf = CellTable.flatten(acdc_df)
        columns = flatDf.columns
        for name, column in directInputs.items():
            if column not in columns:
                raise ValueError(f'Unknown other-cell input column "{column}".')
            if name in columns:
                raise ValueError(
                    f'Other-cell input "{name}" conflicts with an existing column.'
                )
        # replacements: (function, column) -> temporary column holding the
        # aggregate; the equation is rewritten to use it at the end.
        replacements = {}
        for matches, suffixed in ((legacyCalls, False), (measurementCalls, True)):
            for match in matches:
                function = match.group('function').upper()
                column = match.group('column').strip('`')
                if suffixed:
                    column = column[:-len('_other')]
                if column not in columns:
                    raise ValueError(f'Unknown other-cell input column "{column}".')
                if (function in ('MEAN', 'AVERAGE', 'SUM', 'STD')
                        and not pd.api.types.is_numeric_dtype(flatDf[column])):
                    raise ValueError(
                        f'{function}({column}_other) requires a numerical input; '
                        f'"{column}" is not numerical.'
                    )
                key = (function, column)
                if key not in replacements:
                    name = f'_acdc_other_{function.lower()}_{len(replacements)}'
                    while name in columns:
                        name += '_'
                    replacements[key] = name

        condition = OtherCellMeasurements._candidate_condition(settings, acdc_df)
        variables = {'SizeT': SizeT} if SizeT is not None else None
        # this_<col> columns referenced by the condition (enquiring cell).
        this_columns = CellFilter._this_columns(flatDf, condition)
        # Row-wise conditions can be evaluated for many targets at once.
        rowwise = bool(condition) and OtherCellMeasurements._is_rowwise_condition(
            condition
        )
        aggregates = [
            (column, name, CELL_FUNCTIONS[_OTHER_AGGREGATIONS[function]][0])
            for (function, column), name in replacements.items()
        ]
        targets = CellTable.subset(acdc_df, IDs)
        if max_targets is not None:
            targets = targets.iloc[:max_targets]
        neededColumns = {
            'frame_i', 'Cell_ID', *this_columns, *directInputs.values(),
            *(column for column, _, _ in aggregates),
        }
        if condition:
            if '`' in condition:
                neededColumns.update(columns)
            else:
                neededColumns.update(CellEquation.names(condition))
        # Copying all measurement columns per target dominated large tables,
        # so candidates keep only the columns that are actually needed.
        # sortedDf is sorted by frame so windows are contiguous slices.
        sortedDf = flatDf[
            [column for column in columns if column in neededColumns]
        ].sort_values(['frame_i', 'Cell_ID'])
        frames = sortedDf['frame_i'].to_numpy()
        sortedIDs = sortedDf['Cell_ID'].to_numpy()
        # Collect columns used by direct inputs or aggregates, removing
        # duplicates while preserving their order (dict.fromkeys).
        valueColumns = list(dict.fromkeys([
            *directInputs.values(), *(column for column, _, _ in aggregates)
        ]))
        # Map each column name to its frame-sorted NumPy values so candidate
        # windows can be sliced without repeated pandas column lookups.
        sortedValues = {
            column: sortedDf[column].to_numpy() for column in valueColumns
        }
        # Index row-wise equality candidates by value to avoid window scans.
        # any/all must also see nonmatching frames of each candidate cell.
        equalityIndex = None
        equality = re.fullmatch(
            r'([A-Za-z_]\w*)\s*==\s*this_([A-Za-z_]\w*)', condition
        )
        if (
            equality is not None
            and settings['condition_mode'] == 'where_true'
            and equality[1] in sortedDf
            and equality[2] in this_columns
            and not equality[1].startswith(THIS_PREFIX)
            and equality[1] not in CELL_FUNCTIONS
            and (not variables or equality[1] not in variables)
            and not pd.api.types.is_extension_array_dtype(
                sortedDf[equality[1]]
            )
        ):
            equalityIndex = {
                value: (rows, rows['frame_i'].to_numpy())
                for value, rows in sortedDf.groupby(
                    equality[1], sort=False
                )
            }
        # Target values as arrays; results are filled by target position.
        flatTargets = CellTable.flatten(targets)
        nTargets = len(flatTargets)
        targetColumns = list(dict.fromkeys(['frame_i', 'Cell_ID', *this_columns]))
        targetArrays = {
            column: flatTargets[column].to_numpy() for column in targetColumns
        }
        targetIDs = targetArrays['Cell_ID']
        results = {name: [None] * nTargets for name in replacements.values()}
        results.update({name: [None] * nTargets for name in directInputs})
        directAlternatives = {}
        ambiguousInputs = set()

        def emit(position, t, candidateValues):
            # Store aggregates and direct inputs of one target given the
            # values of its (already filtered) candidate rows.
            empty = not len(next(iter(candidateValues.values()), ()))
            for column, name, aggregate in aggregates:
                results[name][position] = (
                    0 if empty else aggregate(candidateValues[column])
                )
            uniques = {}
            for name, column in directInputs.items():
                values = np.asarray(candidateValues[column])
                values = np.unique(values[~pd.isna(values)])
                uniques[name] = values
                if len(values) > 1:
                    ambiguousInputs.add(name)
                results[name][position] = values[0] if len(values) else 0
            if not directInputs:
                return
            # Alternatives for evaluate_with_alternatives: distinct non-NaN
            # candidate values, 0 when there are no candidates at all.
            if empty:
                options = [tuple(0 for _ in directInputs)]
            elif len(directInputs) == 1:
                options = [(value,) for value in next(iter(uniques.values()))]
                if not options:
                    options = [(0,)]
            else:
                # Several inputs: keep value combinations aligned per row.
                candidateFrame = pd.DataFrame({
                    column: candidateValues[column]
                    for column in directInputs.values()
                })
                for column in candidateFrame:
                    if candidateFrame[column].notna().any():
                        candidateFrame = candidateFrame.dropna(subset=[column])
                    else:
                        candidateFrame[column] = 0
                candidateFrame = candidateFrame.drop_duplicates()
                options = [
                    tuple(option)
                    for option in candidateFrame.itertuples(index=False, name=None)
                ]
            directAlternatives[(t, int(targetIDs[position]))] = options

        def emit_positions(position, t, rowPositions):
            # Candidates given as row offsets into sortedDf.
            emit(position, t, {
                column: values[rowPositions]
                for column, values in sortedValues.items()
            } or {'Cell_ID': sortedIDs[rowPositions]})

        # Group targets by frame: they share the same candidate window.
        targetFrames = pd.Series(targetArrays['frame_i'].astype(int))
        framePositions = targetFrames.groupby(targetFrames, sort=False).indices

        batchCondition = bool(condition) and rowwise and equalityIndex is None
        mode = settings['condition_mode']
        for t, group in framePositions.items():
            # Candidate window of this frame as [left, right) offsets.
            lower, upper = FrameRange.bounds(
                settings['range_start'], settings['range_end'], t, SizeT
            )
            left = int(np.searchsorted(frames, lower, side='left'))
            right = max(left, int(np.searchsorted(frames, upper, side='right')))
            windowIDs = sortedIDs[left:right]
            # Case 1: no condition, candidates are the window minus the
            # target cell itself.
            if not condition:
                for position in group:
                    emit_positions(position, t, left + np.flatnonzero(
                        windowIDs != targetIDs[position]
                    ))
                continue
            # Case 2: row-wise condition. Evaluate once for all
            # (target, candidate) pairs of the frame instead of once per
            # target, in chunks to bound memory (about 2M rows per chunk).
            if batchCondition:
                windowSize = right - left
                if windowSize == 0:
                    for position in group:
                        emit_positions(position, t, np.arange(0))
                    continue
                window = sortedDf.iloc[left:right]
                chunkSize = max(1, 2_000_000 // windowSize)
                for chunkStart in range(0, len(group), chunkSize):
                    chunk = group[chunkStart:chunkStart + chunkSize]
                    evaluation = window.iloc[
                        np.tile(np.arange(windowSize), len(chunk))
                    ].reset_index(drop=True)
                    for column in this_columns:
                        evaluation[f'{THIS_PREFIX}{column}'] = np.repeat(
                            targetArrays[column][chunk], windowSize
                        )
                    try:
                        mask = CellFilter.boolean_values(CellEquation.evaluate(
                            evaluation, condition, variables,
                            enforce_dtype_output=bool
                        ), len(evaluation))
                    except Exception as err:
                        raise ValueError(
                            f'Other-cell filter failed at frame {t} '
                            f'with condition {condition!r}: {err}'
                        ) from err
                    # Mask layout: one block of windowSize candidates per target.
                    # any/all modes decide per (target, candidate cell).
                    if mode != 'where_true':
                        mask = pd.Series(mask).groupby(
                            [
                                np.repeat(np.arange(len(chunk)), windowSize),
                                evaluation['Cell_ID'].to_numpy(),
                            ],
                            sort=False,
                        ).transform(mode).to_numpy(dtype=bool)
                    # Rows are targets, columns are candidate rows.
                    mask = mask.reshape(len(chunk), windowSize)
                    # Broadcasting excludes each target's own candidate rows.
                    mask &= windowIDs[None, :] != targetIDs[chunk, None]
                    for position, rowMask in zip(chunk, mask):
                        emit_positions(
                            int(position), t, left + np.flatnonzero(rowMask)
                        )
                continue
            # Case 3: general per-target selection (function calls, backticks,
            # or the ``col == this_col`` equality index).
            window = sortedDf.iloc[left:right]
            for position in group:
                target = {
                    column: values[position]
                    for column, values in targetArrays.items()
                }
                candidateWindow = window
                if equalityIndex is not None:
                    targetValue = target[equality[2]]
                    indexed = (
                        None if pd.isna(targetValue)
                        else equalityIndex.get(targetValue)
                    )
                    if indexed is None:
                        candidateWindow = window.iloc[:0]
                    else:
                        indexedRows, indexedFrames = indexed
                        indexedLeft = np.searchsorted(
                            indexedFrames, lower, side='left'
                        )
                        indexedRight = max(indexedLeft, np.searchsorted(
                            indexedFrames, upper, side='right'
                        ))
                        candidateWindow = indexedRows.iloc[indexedLeft:indexedRight]
                rows = OtherCellMeasurements._select_candidates(
                    candidateWindow, target, condition, mode, this_columns, variables,
                    rowwise=rowwise
                )
                emit(position, t, {
                    column: rows[column].to_numpy() for column in valueColumns
                } or {'Cell_ID': rows['Cell_ID'].to_numpy()})
        if ambiguousInputs:
            names = ', '.join(sorted(ambiguousInputs))
            warnings.warn(
                f'Other-cell input(s) {names} have multiple distinct values '
                'for one or more enquiring cells; evaluating the equation '
                'against each candidate row.',
                UserWarning,
                stacklevel=2,
            )
        acdc_df = targets.copy()
        for name, values in results.items():
            acdc_df[name] = values
        alternatives = None
        if directInputs:
            alternatives = {
                'columns': tuple(directInputs),
                'by_target': directAlternatives,
            }

        # constants = []
        # replacements = {}

        # def addConstant(match, *, suffixed):
        #     function = match.group('function').upper()
        #     column = match.group('column').strip('`')
        #     if suffixed:
        #         if not column.endswith('_other'):
        #             return
        #         column = column[:-len('_other')]
        #     if column not in columns:
        #         raise ValueError(
        #             f'Unknown other-cell input column "{column}".'
        #         )
        #     if (function in ('MEAN', 'AVERAGE', 'SUM', 'STD')
        #             and not pd.api.types.is_numeric_dtype(flatDf[column])):
        #         raise ValueError(
        #             f'{function}({column}_other) requires a numerical input; '
        #             f'"{column}" is not numerical.'
        #         )
        #     key = (function, column)
        #     if key not in replacements:
        #         name = f'_acdc_other_{function.lower()}_{len(replacements)}'
        #         while name in columns:
        #             name += '_'
        #         equationColumn = (
        #             column if column.isidentifier() else f'`{column}`'
        #         )
        #         constant = default_constant(
        #             name, f'{_OTHER_AGGREGATIONS[function]}({equationColumn})',
        #             settings['source']
        #         )
        #         constant.update(
        #             condition=settings['condition'],
        #             condition_mode=settings['condition_mode'],
        #             time_scope='custom',
        #             range_start=settings['range_start'],
        #             range_end=settings['range_end'],
        #             per_frame=True,
        #         )
        #         constants.append(constant)
        #         replacements[key] = name

        # for match in legacyCalls:
        #     addConstant(match, suffixed=False)
        # for match in measurementCalls:
        #     addConstant(match, suffixed=True)

        # values = compute_cell_constants(acdc_df, constants, SizeT=SizeT)
        # acdc_df = add_constants_columns(acdc_df, values)

        def replacement(match: re.Match[str], *, suffixed: bool) -> str:
            column = match.group('column').strip('`')
            if suffixed:
                column = column[:-len('_other')]
            key = (
                match.group('function').upper(),
                column
            )
            return replacements[key]

        equation = _OTHER_FUNCTION_PATTERN.sub(
            lambda match: replacement(match, suffixed=False), equation
        )
        equation = _OTHER_MEASUREMENT_PATTERN.sub(
            lambda match: replacement(match, suffixed=True), equation
        )
        if return_alternatives:
            return acdc_df, equation, alternatives
        return acdc_df, equation


class TimepointPredictor:
    """Validate and execute equation-based or file-based predictors.

    predict() is the main entry point. It returns {Cell_ID: [frame_i, ...]},
    with sorted unique 0-based frames, omitting cells without predictions and
    discarding frames outside [0, SizeT) when SizeT is supplied. IDs limits
    target cells only; all input cells remain available for other-cell inputs.

    from_equation() and from_function() expose the lower-level execution and
    retain empty lists. File callbacks receive a copy of each cell's sorted
    rows and optionally entire_df. Failures propagate as contextual errors.
    """

    @staticmethod
    def validate(predictor: dict[str, Any]) -> None:
        """Validate predictor type, equation mode and optional filters."""
        if not isinstance(predictor, dict):
            raise ValueError('Invalid time point predictor.')
        kind = predictor.get('type')
        if kind not in PREDICTOR_TYPES:
            raise ValueError(f'Invalid time point predictor type: {kind}.')
        if kind == PREDICTOR_EQUATION:
            if not isinstance(predictor.get('equation'), str) or not predictor['equation'].strip():
                raise ValueError('The predictor equation cannot be empty.')
            if predictor.get('mode') not in EQUATION_MODES:
                raise ValueError('Invalid predictor equation mode.')
            if predictor.get('other_cells') is not None:
                CellFilter.validate_other(predictor['other_cells'])
            if predictor.get('main_cell_filter') is not None:
                CellFilter.validate_main(predictor['main_cell_filter'])
        elif kind == PREDICTOR_FILE:
            if not isinstance(predictor.get('filepath'), str) or not predictor['filepath']:
                raise ValueError('The predictor file path cannot be empty.')

    @staticmethod
    def describe(predictor: dict[str, Any]) -> str:
        """Return the user-facing summary of a predictor configuration."""
        kind = predictor.get('type', PREDICTOR_NONE)
        if kind == PREDICTOR_EQUATION:
            description = (
                f'{EQUATION_MODES[predictor["mode"]]}: {predictor["equation"]}'
            )
            otherCells = predictor.get('other_cells')
            if otherCells is not None:
                settings = CellFilter.validate_other(otherCells)
                description += (
                    f' (uses {SOURCE_CELLS[settings["source"]]}, '
                    f'frames {settings["range_start"]} to '
                    f'{settings["range_end"]}'
                )
                if settings['condition']:
                    description += f', filter: {settings["condition"]}'
                description += ')'
            return description
        if kind == PREDICTOR_FILE:
            return f'predictor(df) from "{predictor["filepath"]}"'
        return 'None (manual annotation)'

    @staticmethod
    def load_function(filepath: str):
        """Import ``predictor`` from the Python file at ``filepath``."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f'Predictor file not found: "{filepath}".')
        spec = importlib.util.spec_from_file_location(
            '_acdc_timepoint_predictor', filepath
        )
        if spec is None or spec.loader is None:
            raise ImportError(f'Cannot import "{filepath}".')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        func = getattr(module, 'predictor', None)
        if not callable(func):
            raise AttributeError(
                f'"{filepath}" does not define a function called "predictor(df)".'
            )
        try:
            inspect.signature(func).bind(
                pd.DataFrame(), **({'entire_df': pd.DataFrame()}
                                   if TimepointPredictor.accepts_entire_df(func) else {})
            )
        except TypeError as err:
            raise AttributeError(
                f'"predictor" in "{filepath}" must accept df and optionally entire_df.'
            ) from err
        return func

    @staticmethod
    def normalize_frames(value) -> list[int]:
        """Convert a predictor output (None, int or iterable of int) to a 
        sorted list of unique frame indices."""
        if value is None:
            return []
        if isinstance(value, (pd.Series, pd.Index, np.ndarray)):
            value = value.tolist()
        if not isinstance(value, (list, tuple, set)):
            value = [value]
        frames = set()
        for item in value:
            if item is None or (isinstance(item, float) and np.isnan(item)):
                continue
            if isinstance(item, (bool, np.bool_)):
                raise TypeError('Predicted time points must be integers.')
            if isinstance(item, (float, np.floating)) and not float(item).is_integer():
                raise TypeError(f'Predicted time point {item} is not an integer.')
            try:
                frames.add(int(item))
            except (TypeError, ValueError):
                raise TypeError(
                    f'Predicted time point {item!r} is not an integer.'
                ) from None
        return sorted(frames)

    @staticmethod
    def _frames_from_values(
            values: ArrayLike, frames: NDArray[np.integer[Any]], mode: str
        ) -> list[int]:
    
        """Convert aligned per-frame values into frames using an equation mode."""
        if mode in ('max', 'min'):
            values = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(
                dtype=float
            )
            if not np.isfinite(values).any():
                return []
            values = np.where(np.isfinite(values), values, np.nan)
            idx = np.nanargmax(values) if mode == 'max' else np.nanargmin(values)
            return [int(frames[idx])]
    
        values = pd.Series(values)
        isTrue = (values.notna() & values.astype(bool)).to_numpy()
        if mode == 'first_true':
            return [int(frames[isTrue.argmax()])] if isTrue.any() else []
        if mode == 'last_true':
            return [int(frames[np.flatnonzero(isTrue)[-1]])] if isTrue.any() else []
        becomesTrue = isTrue & ~np.concatenate(([False], isTrue[:-1]))
        return [int(frame) for frame in frames[becomesTrue]]

    @staticmethod
    def accepts_entire_df(func: object) -> bool:
        """Whether the callback accepts entire_df explicitly or via **kwargs."""
        try:
            parameters = inspect.signature(func).parameters.values()
        except (TypeError, ValueError):
            return False
        return any(
            param.name == 'entire_df' or param.kind == param.VAR_KEYWORD 
            for param in parameters
        )

    @staticmethod
    def main_cell_filter_IDs(
            acdc_df: pd.DataFrame,
            main_cell_filter: dict[str, str] | None,
            other_cells: dict[str, str] | None = None,
            SizeT: int | None = None, IDs: Iterable[int] | None = None
        ) -> list[int] | None:
        """Return the cells of ``IDs`` (None means all cells) passing the
        main-cell filter. A cell is eligible if the condition holds in any/all
        of its frames. Returns ``IDs`` (as a list, or None) if there is no
        filter condition."""
        if IDs is not None:
            IDs = list(IDs)
        if main_cell_filter is None:
            return IDs
        settings = CellFilter.validate_main(main_cell_filter)
        if not settings['condition'].strip():
            return IDs
        eligibleIDs = []
        variables = {'SizeT': SizeT} if SizeT is not None else None
        # Fastest path: constant own values with mode "any" (see
        # constant_main_any). None means "not applicable".
        constantIDs = (
            OtherCellMeasurements.constant_main_any(
                acdc_df, settings['condition'], other_cells, SizeT, IDs
            )
            if settings['condition_mode'] == 'any' else None
        )
        if constantIDs is not None:
            return constantIDs
        # General path: resolve other-cell inputs, then evaluate.
        filter_df, condition, alternatives = (
            OtherCellMeasurements.prepare_inputs(
                acdc_df, settings['condition'], other_cells, SizeT,
                IDs=IDs, return_alternatives=True
            )
        )
        batchedIDs = None
        if (
            not filter_df.empty
            and OtherCellMeasurements._is_rowwise_condition(condition)
        ):
            # Row-wise filters need one evaluation for all cells;
            # per-cell results are then reduced with any/all.
            # On any failure fall back to the per-cell loop below,
            # which reports errors with the offending Cell_ID.
            try:
                flatFilter = CellTable.flatten(filter_df)
                values = OtherCellMeasurements.evaluate_with_alternatives(
                    flatFilter, condition, alternatives,
                    variables=variables, enforce_dtype_output=bool
                )
                mask = CellFilter.boolean_values(values, len(flatFilter))
                matches = pd.Series(mask).groupby(
                    flatFilter['Cell_ID'].to_numpy(), sort=True
                ).agg(settings['condition_mode'])
                batchedIDs = [
                    int(ID) for ID in matches.index[matches.to_numpy()]
                ]
            except Exception:
                batchedIDs = None
        if batchedIDs is not None:
            return batchedIDs
        # Per-cell fallback (temporal functions, errors).
        for ID, df_ID in CellTable.iter_cells(filter_df):
            try:
                values = OtherCellMeasurements.evaluate_with_alternatives(
                    df_ID, condition, alternatives,
                    variables=variables, enforce_dtype_output=bool
                )
                mask = CellFilter.boolean_values(values, len(df_ID))
            except Exception as err:
                raise ValueError(
                    f'Main-cell filter failed for Cell_ID {ID}: {err}'
                ) from err
            matches = (
                mask.any() if settings['condition_mode'] == 'any'
                else mask.all()
            )
            if matches:
                eligibleIDs.append(ID)
        return eligibleIDs

    @staticmethod
    @debug_trace
    def from_equation(
            acdc_df: pd.DataFrame, equation: str, mode: str,
            SizeT: int | None = None, IDs: Iterable[int] | None = None,
            other_cells: dict[str, str] | None = None,
            main_cell_filter: dict[str, str] | None = None
        ) -> dict[int, list[int]]:
        """Evaluate ``equation`` with ``DataFrame.eval`` on each cell's rows and 
        convert the per-frame results into frames according to ``mode``.
    
        If ``IDs`` is not None, only these cells are predicted. Main-cell
        filters select whole eligible cells, including when their equations
        use other-cell inputs from the configured source/window. All observed
        frames of retained cells remain available for prediction. Other-cell
        aggregates are computed only for eligible target rows but still use
        the full input table as candidates. Unlike predict(), this method
        retains empty lists and does not clip predicted frames to SizeT.

        Steps: (1) optionally reduce IDs to the cells passing the main-cell
        filter; (2) resolve other-cell inputs for the remaining targets;
        (3) evaluate the equation per cell and convert the per-frame results
        into frames according to ``mode``.
        """
        if mode not in EQUATION_MODES:
            raise ValueError(f'Invalid predictor equation mode: {mode}.')
        # Step 1: main-cell filter. A cell is eligible if the condition holds
        # in any/all of its frames; retained cells keep all their frames.
        IDs = TimepointPredictor.main_cell_filter_IDs(
            acdc_df, main_cell_filter, other_cells=other_cells, SizeT=SizeT,
            IDs=IDs
        )
        # Step 2: only the (filtered) targets get other-cell values; all rows
        # of the original table remain candidates.
        acdc_df, equation, alternatives = OtherCellMeasurements.prepare_inputs(
            acdc_df, equation, other_cells, SizeT, IDs=IDs,
            return_alternatives=True
        )
        variables = {'SizeT': SizeT} if SizeT is not None else None
        # Step 3: per-cell evaluation (rows sorted by frame_i); 'mode' is
        # also used to reduce alternatives (e.g. max/min predictors).
        timepoints = {}
        for ID, df_ID in CellTable.iter_cells(acdc_df):
            try:
                values = OtherCellMeasurements.evaluate_with_alternatives(
                    df_ID, equation, alternatives,
                    variables=variables, combine=mode
                )
            except Exception as err:
                raise ValueError(
                    f'Evaluating the equation failed for Cell_ID {ID}: {err}'
                ) from err
            # Scalar results (e.g. constants) apply to every frame.
            if not isinstance(values, pd.Series):
                values = pd.Series(np.broadcast_to(values, len(df_ID)))
            frames = df_ID['frame_i'].to_numpy()
            timepoints[ID] = TimepointPredictor._frames_from_values(values.to_numpy(), frames, mode)
        return timepoints

    @staticmethod
    def from_function(
            acdc_df: pd.DataFrame, func, IDs: Iterable[int] | None = None
        ) -> dict[int, list[int]]:
        """Call ``func(df)`` on each cell's rows (with ``entire_df``, the table 
        of all cells, if ``func`` accepts it). If ``IDs`` is not None, only 
        these cells are predicted."""
        kwargs = {}
        if TimepointPredictor.accepts_entire_df(func):
            kwargs['entire_df'] = (
                CellTable.flatten(acdc_df).sort_values(['Cell_ID', 'frame_i'])
                .reset_index(drop=True)
            )
        timepoints = {}
        for ID, df_ID in CellTable.iter_cells(CellTable.subset(acdc_df, IDs)):
            try:
                timepoints[ID] = TimepointPredictor.normalize_frames(func(df_ID.copy(), **kwargs))
            except Exception as err:
                raise ValueError(
                    f'predictor(df) failed for Cell_ID {ID}: {err}'
                ) from err
        return timepoints

    @staticmethod
    @debug_trace
    def predict(
            acdc_df: pd.DataFrame | None, predictor: dict[str, Any],
            SizeT: int | None = None, IDs: Iterable[int] | None = None
        ) -> dict[int, list[int]]:
        """Predict time points per cell.

        If ``IDs`` is not None, only these cells are predicted, while all 
        cells of ``acdc_df`` are available as other cells.

        Parameters
        ----------
        acdc_df : pandas.DataFrame or None
            Measurements indexed by (frame_i, Cell_ID), or with those columns.
        predictor : dict
            Configuration with type 'none', 'equation' or 'file'. Equations
            require equation and mode; files require filepath. Optional
            main_cell_filter and other_cells are validated by CellFilter.
        SizeT : int or None
            Video frame count, also exposed to equations as SizeT.
        IDs : iterable of int or None
            Target cell IDs. None evaluates every cell.

        Returns
        -------
        dict
            ``{Cell_ID: sorted list of frame_i}`` (cells without time points are 
            omitted). Frames outside ``[0, SizeT)`` are discarded.

        Raises
        ------
        ValueError
            Invalid settings or per-cell equation/callback execution failure.
            File loading errors propagate with the failing path.
        """
        TimepointPredictor.validate(predictor)
        kind = predictor['type']
        if kind == PREDICTOR_NONE or acdc_df is None or acdc_df.empty:
            return {}
        if kind == PREDICTOR_EQUATION:
            timepoints = TimepointPredictor.from_equation(
                acdc_df, predictor['equation'], predictor['mode'],
                SizeT=SizeT, IDs=IDs,
                other_cells=predictor.get('other_cells'),
                main_cell_filter=predictor.get('main_cell_filter')
            )
        else:
            func = TimepointPredictor.load_function(predictor['filepath'])
            timepoints = TimepointPredictor.from_function(acdc_df, func, IDs=IDs)
    
        predicted = {}
        for ID, frames in timepoints.items():
            if SizeT is not None:
                frames = [frame for frame in frames if 0 <= frame < SizeT]
            if frames:
                predicted[ID] = frames
        return predicted


@dataclass
class TimepointIdentifier:
    """User-defined marker with a stable runtime UID and serialized predictor.

    to_dict() stores display settings and predictor configuration, not the UID.
    from_dict() validates stored data and assigns a fresh UID on import.
    """
    name: str
    color: tuple
    shortcut: str
    predictor: dict[str, Any] = field(default_factory=lambda: {'type': PREDICTOR_NONE})
    uid: int = field(default_factory=lambda: next(_uid_counter))
    autoApply: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible display and predictor settings, excluding UID."""
        return {
            'name': self.name,
            'color': [int(c) for c in self.color[:3]],
            'shortcut': self.shortcut,
            'predictor': dict(self.predictor),
            'autoApply': self.autoApply,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'TimepointIdentifier':
        """Build a marker with a fresh UID from validated serialized settings."""
        cls.validate_dict(data)
        return cls(
            name=data['name'], color=tuple(data['color']),
            shortcut=data['shortcut'], predictor=dict(data['predictor']),
            autoApply=data.get('autoApply', False)
        )

    @staticmethod
    def validate_dict(data: dict[str, Any]) -> None:
        """Validate a serialized identifier, including its predictor."""
        if not isinstance(data, dict):
            raise ValueError('Invalid time point identifier.')
        if not isinstance(data.get('name'), str) or not data['name'].strip():
            raise ValueError('Invalid time point identifier name.')
        color = data.get('color')
        if (not isinstance(color, (list, tuple)) or len(color) != 3
                or any(type(c) is not int or not 0 <= c <= 255 for c in color)):
            raise ValueError('Invalid time point identifier color.')
        if not isinstance(data.get('shortcut'), str):
            raise ValueError('Invalid time point identifier shortcut.')
        TimepointPredictor.validate(data.get('predictor'))


# {label: (name template, equation template)}, `{col}` is a selected column
# CONSTANT_PRESETS = {
#     'Maximum value': ('max_{col}', 'MAX({col})'),
#     'Minimum value': ('min_{col}', 'MIN({col})'),
#     'Mean value': ('mean_{col}', 'MEAN({col})'),
#     'Median value': ('median_{col}', 'MEDIAN({col})'),
#     'Value at first frame': ('first_{col}', 'FIRST({col})'),
#     'Value at last frame': ('last_{col}', 'LAST({col})'),
#     'Was at beginning': ('was_at_beginning', 'FIRST(frame_i) == 0'),
#     'Was at end': ('was_at_end', 'LAST(frame_i) == SizeT - 1'),
#     'Number of frames': ('num_frames', 'NUNIQUE(frame_i)'),
#     'First frame': ('first_frame_i', 'FIRST(frame_i)'),
#     'Last frame': ('last_frame_i', 'LAST(frame_i)'),
#     'Number of source cells': ('num_cells', 'NUNIQUE(Cell_ID)'),
# }

# def preset_needs_column(label):
#     return '{col}' in CONSTANT_PRESETS[label][1]

# def default_constant(name='constant', equation='', source=SOURCE_THIS):
#     return {
#         'name': name, 'equation': equation, 'source': source, 
#         'condition': '', 'per_frame': False, 'condition_mode': 'where_true',
#         'condition_type': 'equation', 'condition_filepath': '',
#         'time_scope': 'video', 'range_start': 't - 5', 'range_end': 't',
#     }

# def normalize_constant(constant):
#     """Constant dict with all keys (older constants only have 'name' and 
#     'equation')."""
#     normalized = default_constant()
#     normalized.update({
#         key: constant[key] for key in normalized if key in constant
#     })
#     if ('time_scope' not in constant
#             or (constant['time_scope'] == 'video' and normalized['per_frame'])):
#         normalized['time_scope'] = 'frame' if normalized['per_frame'] else 'video'
#     normalized['per_frame'] = normalized['time_scope'] != 'video'
#     return normalized

# def constant_from_preset(label, col=None, source=SOURCE_THIS):
#     nameTemplate, equationTemplate = CONSTANT_PRESETS[label]
#     if '{col}' in equationTemplate and not col:
#         raise ValueError(f'The preset "{label}" requires a column.')
#     if source not in SOURCE_CELLS:
#         raise ValueError(f'Invalid source cells: {source}.')
#     name = f'{nameTemplate.format(col=col)}{SOURCE_SUFFIXES[source]}'
#     return default_constant(name, equationTemplate.format(col=col), source)

# def constant_description(constant):
#     constant = normalize_constant(constant)
#     description = f'{constant["equation"]} of {SOURCE_CELLS[constant["source"]]}'
#     if constant['condition']:
#         description = f'{description} where {constant["condition"]}'
#     if constant['time_scope'] == 'custom':
#         description = (
#             f'{description} (range {constant["range_start"]} to '
#             f'{constant["range_end"]}, inclusive)'
#         )
#     elif constant['per_frame']:
#         description = f'{description} (per frame)'
#     return description

# def validate_constant_name(name, existingNames=(), columns=()):
#     if not isinstance(name, str) or not name.isidentifier():
#         raise ValueError(
#             f'Invalid constant name "{name}". Use only letters, numbers and '
#             'underscores, not starting with a number.'
#         )
#     if keyword.iskeyword(name) or name in ('True', 'False', 'None'):
#         raise ValueError(f'"{name}" is a reserved Python keyword.')
#     if name in CELL_FUNCTIONS or name in CONSTANT_VARIABLES:
#         raise ValueError(f'"{name}" is the name of a built-in function/variable.')
#     if name.startswith(THIS_PREFIX):
#         raise ValueError(
#             f'Constant names cannot start with "{THIS_PREFIX}" (reserved for '
#             'the columns of the cell itself).'
#         )
#     if name in existingNames:
#         raise ValueError(f'The constant name "{name}" is used more than once.')
#     if name in columns:
#         raise ValueError(
#             f'"{name}" is already a column of the measurements table.'
#         )

# def validate_constants(constants, columns=()):
#     if not isinstance(constants, (list, tuple)):
#         raise ValueError('Invalid per-cell constants.')
#     names = []
#     for constant in constants:
#         if not isinstance(constant, dict):
#             raise ValueError('Invalid per-cell constant.')
#         name = constant.get('name')
#         validate_constant_name(name, names, columns)
#         equation = constant.get('equation')
#         if not isinstance(equation, str) or not equation.strip():
#             raise ValueError(
#                 f'The equation of the constant "{name}" cannot be empty.'
#             )
#         source = constant.get('source', SOURCE_THIS)
#         if source not in SOURCE_CELLS:
#             raise ValueError(f'Invalid source cells of the constant "{name}".')
#         condition = constant.get('condition', '')
#         if not isinstance(condition, str):
#             raise ValueError(f'Invalid condition of the constant "{name}".')
#         conditionType = constant.get('condition_type', 'equation')
#         if conditionType not in ('equation', 'file'):
#             raise ValueError(f'Invalid filter type of the constant "{name}".')
#         filepath = constant.get('condition_filepath', '')
#         if not isinstance(filepath, str):
#             raise ValueError(f'Invalid filter file of the constant "{name}".')
#         if conditionType == 'file' and not filepath.strip():
#             raise ValueError(f'The constant "{name}" requires a filter file.')
#         if (source == SOURCE_CUSTOM and conditionType == 'equation'
#                 and not condition.strip()):
#             raise ValueError(
#                 f'The constant "{name}" requires a condition to select the '
#                 f'other cells (e.g., Cell_ID == {THIS_PREFIX}relative_ID).'
#             )
#         if type(constant.get('per_frame', False)) is not bool:
#             raise ValueError(f'Invalid "per frame" option of the constant "{name}".')
#         if constant.get('condition_mode', 'where_true') not in FILTER_FRAME_MODES:
#             raise ValueError(f'Invalid filter frame rule of the constant "{name}".')
#         scope = constant.get('time_scope')
#         if scope is not None and scope not in CONSTANT_TIME_SCOPES:
#             raise ValueError(f'Invalid time scope of the constant "{name}".')
#         if scope == 'custom':
#             frame_range_bounds(
#                 constant.get('range_start', 't - 5'),
#                 constant.get('range_end', 't'), t=0
#             )
#         names.append(name)


# def _scalar(value, name, ID):
#     if isinstance(value, (pd.Series, pd.Index, np.ndarray, list, tuple)):
#         values = pd.Series(np.asarray(value).ravel())
#         unique = values.dropna().unique()
#         if len(unique) > 1:
#             raise ValueError(
#                 f'The constant "{name}" has more than one value for Cell_ID '
#                 f'{ID}. Reduce the equation to a single value per cell, e.g. '
#                 'with MAX, MIN, MEAN, FIRST or LAST.'
#             )
#         value = unique[0] if len(unique) else np.nan
#     if isinstance(value, np.generic):
#         value = value.item()
#     return value


# def _valid_IDs(values):
#     values = pd.to_numeric(pd.Series(values), errors='coerce')
#     return values[values > 0].astype(int)

# def lineage_source(acdc_df):
#     """Columns used to determine the relationships between cells: 'tree' 
#     (``parent_ID_tree`` and ``sister_ID_tree``), 'relative_ID' (budding 
#     yeast cell cycle annotations) or None."""
#     df = _flat_df(acdc_df)
#     if 'parent_ID_tree' in df.columns and len(_valid_IDs(df['parent_ID_tree'])):
#         return 'tree'
#     if {'relative_ID', 'relationship'}.issubset(df.columns):
#         return 'relative_ID'
#     if 'parent_ID_tree' in df.columns:
#         return 'tree'
#     return None


# def cell_lineage(acdc_df):
#     """Relationships between the cells of ``acdc_df``.

#     The mother cell is taken from ``parent_ID_tree`` (most frequent valid 
#     value of the cell) or, if the tree columns are not present (or not 
#     annotated), from ``relative_ID`` of the frames where the cell is a bud. 
#     Sisters are cells with the same mother (plus ``sister_ID_tree*``).

#     Returns
#     -------
#     tuple of dicts
#         ``(mothers, sisters, daughters)``: ``{Cell_ID: mother ID}``, 
#         ``{Cell_ID: set of sister IDs}`` and ``{Cell_ID: set of daughter IDs}``
#     """
#     df = _flat_df(acdc_df).sort_values(['Cell_ID', 'frame_i'])
#     source = lineage_source(df)
#     if source is None:
#         raise ValueError(
#             'Determining mother, sister and daughter cells requires the '
#             'lineage tree columns ("parent_ID_tree", "sister_ID_tree") or the '
#             'cell cycle annotations ("relative_ID", "relationship").'
#         )
#     mothers = {}
#     sisters = defaultdict(set)
#     if source == 'tree':
#         for ID, parentIDs in df.groupby('Cell_ID')['parent_ID_tree']:
#             parentIDs = _valid_IDs(parentIDs)
#             if len(parentIDs):
#                 mothers[int(ID)] = int(parentIDs.mode().iloc[0])
#         sisterCols = [
#             col for col in df.columns if str(col).startswith('sister_ID_tree')
#         ]
#         for col in sisterCols:
#             for ID, sisterIDs in df.groupby('Cell_ID')[col]:
#                 for sisterID in set(_valid_IDs(sisterIDs)) - {int(ID)}:
#                     sisters[int(ID)].add(sisterID)
#                     sisters[sisterID].add(int(ID))
#     else:
#         buds = df[df['relationship'] == 'bud']
#         for ID, relativeIDs in buds.groupby('Cell_ID')['relative_ID']:
#             relativeIDs = _valid_IDs(relativeIDs)
#             if len(relativeIDs):
#                 mothers[int(ID)] = int(relativeIDs.iloc[0])
    
#     daughters = defaultdict(set)
#     for ID, motherID in mothers.items():
#         daughters[motherID].add(ID)
#     for siblings in daughters.values():
#         for ID in siblings:
#             sisters[ID].update(siblings - {ID})
#     return mothers, dict(sisters), dict(daughters)


# def _source_rows(df, constant, lineage, variables):
#     """Rows of the source cells of each cell, with the ID of the cell 
#     in the ``_target`` column."""
#     source = constant['source']
#     condition = constant['condition'].strip()
#     hasFilter = bool(condition) or constant['condition_type'] == 'file'
#     if (source in (SOURCE_THIS, *CELL_RELATIONS)
#             and constant['condition_type'] == 'equation'
#             and not _this_columns(df, constant['equation'], condition)):
#         rows = df
#         if condition:
#             mask = pd.Series(evaluate_cells_condition(df, condition, variables), index=df.index)
#             mode = constant['condition_mode']
#             if mode in ('any', 'all'):
#                 mask = mask.groupby(df['Cell_ID']).transform(mode)
#             rows = df[mask]
#         if source == SOURCE_THIS:
#             return rows.assign(_target=rows['Cell_ID'].to_numpy())
#         pairs = _relation_pairs(lineage, source).rename(
#             columns={'Cell_ID': '_target', '_other_ID': 'Cell_ID'}
#         )
#         return pairs.merge(rows, on='Cell_ID')
    
#     # Other cells in the same frame of each cell
#     thisCols = list(df.columns)
#     targets = pd.DataFrame({
#         'frame_i': df['frame_i'].to_numpy(), '_target': df['Cell_ID'].to_numpy()
#     })
#     for col in thisCols:
#         targets[f'{THIS_PREFIX}{col}'] = df[col].to_numpy()
#     pairs = targets.merge(df, on='frame_i')
#     if source == SOURCE_THIS:
#         pairs = pairs[pairs['Cell_ID'] == pairs['_target']]
#     elif source in CELL_RELATIONS:
#         relations = _relation_pairs(lineage, source).rename(
#             columns={'Cell_ID': '_target', '_other_ID': 'Cell_ID'}
#         )
#         pairs = pairs.merge(relations, on=['_target', 'Cell_ID'])
#     else:
#         pairs = pairs[pairs['Cell_ID'] != pairs['_target']]
#     if not hasFilter or pairs.empty:
#         return pairs
#     func = (
#         load_cell_filter_function(constant['condition_filepath'])
#         if constant['condition_type'] == 'file' else None
#     )
#     selected = []
#     for (target, candidate), rows in pairs.groupby(['_target', 'Cell_ID'], sort=False):
#         rows = rows.sort_values('frame_i')
#         evaluation = rows.reset_index(drop=True).copy()
#         try:
#             if func is None:
#                 result = eval_cell_equation(
#                     evaluation, condition, variables, enforce_dtype_output=bool
#                 )
#             else:
#                 candidateDf = evaluation[list(df.columns)].copy()
#                 enquirerDf = evaluation[
#                     [f'{THIS_PREFIX}{col}' for col in df.columns]
#                 ].copy()
#                 enquirerDf.columns = df.columns
#                 result = func(candidateDf, enquirerDf)
#             mask = boolean_filter_values(result, len(rows))
#         except Exception as err:
#             raise ValueError(
#                 f'Filter of constant "{constant["name"]}" failed for '
#                 f'enquiring cell {target}, candidate {candidate}: {err}'
#             ) from err
#         mode = constant['condition_mode']
#         if mode == 'where_true':
#             selected.append(rows.loc[mask])
#         elif (mode == 'any' and mask.any()) or (mode == 'all' and mask.all()):
#             selected.append(rows)
#     return pd.concat(selected, ignore_index=True) if selected else pairs.iloc[:0]

# def _evaluate_rows(rows, constant, variables, ID):
#     name = constant['name']
#     try:
#         value = eval_cell_equation(rows, constant['equation'], variables=variables)
#     except Exception as err:
#         raise ValueError(
#             f'Evaluating the constant "{name}" failed for Cell_ID {ID}: {err}'
#         ) from err
#     return _scalar(value, name, ID)

# def _evaluate_constant(df, constant, lineage, variables):
#     if constant['time_scope'] == 'custom':
#         index = pd.MultiIndex.from_frame(df[['frame_i', 'Cell_ID']])
#         values = pd.Series(index=index, dtype=object, name=constant['name'])
#         # Each window has its own filtering result, including Any/All decisions.
#         aggregate = dict(constant, time_scope='video', per_frame=False)
#         for t in sorted(df['frame_i'].unique()):
#             lower, upper = frame_range_bounds(
#                 constant['range_start'], constant['range_end'], t
#             )
#             window = df[df['frame_i'].between(lower, upper)]
#             # Preserve the target even for future-only/past-only empty windows.
#             rows = _source_rows(window, aggregate, lineage, variables)
#             for ID in df.loc[df['frame_i'] == t, 'Cell_ID']:
#                 sourceRows = rows[rows['_target'] == ID].sort_values(
#                     ['frame_i', 'Cell_ID']
#                 ).reset_index(drop=True)
#                 values.loc[(t, ID)] = _evaluate_rows(
#                     sourceRows, aggregate, variables, ID
#                 )
#         return values.infer_objects()
#     rows = _source_rows(df, constant, lineage, variables)
#     keys = ['_target', 'frame_i'] if constant['per_frame'] else ['_target']
#     sortColumns = keys + [col for col in ('frame_i', 'Cell_ID') if col not in keys]
#     rows = rows.sort_values(sortColumns)
#     # Value of cells without source cells (e.g., NaN or 0 cells)
#     try:
#         emptyValue = _evaluate_rows(
#             rows.iloc[0:0].reset_index(drop=True), constant, variables, None
#         )
#     except Exception:
#         emptyValue = np.nan
#     results = {}
#     for key, rows_ID in rows.groupby(keys, sort=False):
#         results[key] = _evaluate_rows(
#             rows_ID.reset_index(drop=True), constant, variables, key[0]
#         )
#     if constant['per_frame']:
#         index = pd.MultiIndex.from_frame(df[['frame_i', 'Cell_ID']])
#         values = [
#             results.get((ID, frame_i), emptyValue) 
#             for frame_i, ID in index
#         ]
#         return pd.Series(values, index=index, name=constant['name']).infer_objects()
#     return {
#         int(ID): results.get((ID,), emptyValue) 
#         for ID in df['Cell_ID'].unique()
#     }

# def _constant_column(df, values):
#     """Values of a constant aligned with the rows of the flat ``df``"""
#     if isinstance(values, pd.Series):
#         index = pd.MultiIndex.from_arrays(
#             [df['frame_i'].to_numpy(), df['Cell_ID'].to_numpy()]
#         )
#         return values.reindex(index).to_numpy()
#     return df['Cell_ID'].map(values).to_numpy()

# def compute_cell_constants(acdc_df, constants, SizeT=None):
#     """Evaluate the per-cell constants.

#     Constants are evaluated in order, hence they can use previously
#     defined constants. Custom scopes aggregate an inclusive window for
#     each target frame, applying the cell filter within that window.

#     Returns
#     -------
#     dict
#         ``{constant name: values}`` with values ``{Cell_ID: value}`` or, 
#         for constants evaluated per frame, a Series indexed by 
#         ``(frame_i, Cell_ID)``.
#     """
#     validate_constants(constants)
#     constants = [normalize_constant(constant) for constant in constants]
#     values = {constant['name']: {} for constant in constants}
#     if not constants or acdc_df is None or acdc_df.empty:
#         return values
#     df = (
#         _flat_df(acdc_df).sort_values(['Cell_ID', 'frame_i'])
#         .reset_index(drop=True)
#     )
#     variables = {'SizeT': SizeT} if SizeT is not None else None
#     lineage = None
#     for constant in constants:
#         if constant['source'] in CELL_RELATIONS and lineage is None:
#             lineage = cell_lineage(df)
#         values[constant['name']] = _evaluate_constant(
#             df, constant, lineage, variables
#         )
#         df[constant['name']] = _constant_column(df, values[constant['name']])
#     return values

# def add_constants_columns(acdc_df, constantValues):
#     """Return a copy of ``acdc_df`` with one column per constant (NaN for 
#     cells without value)."""
#     if acdc_df is None or not constantValues:
#         return acdc_df
#     acdc_df = acdc_df.copy()
#     df = _flat_df(acdc_df)
#     for name, values in constantValues.items():
#         acdc_df[name] = _constant_column(df, values)
#     return acdc_df
