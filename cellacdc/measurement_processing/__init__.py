"""Measurement recipes, independent of the image preprocessing registry.

Steps operate per frame by default (one measurement across that frame's cells).
``is_timelapse`` steps receive one cell's measurement in frame order.
``is_bulk_processing`` steps receive the entire selected measurement DataFrame
once and must return a DataFrame with the same index and columns. Bulk and
timelapse are mutually exclusive. Recipes preserve row order and missing values;
time windows count observations, not elapsed time or absent frames.

Like segmentation models, each processor lives in its own package containing
an ``acdcProcess.py`` module with a ``Model`` class. The module's ``name`` is
the label stored in recipes. Public processor folders are discovered locally;
no model server or separate environment is needed. A processor folder that
fails to import, lacks a non-empty string ``name`` or a ``Model`` with a
static ``run``, or reuses the ``name`` of a processor loaded earlier
(``SmoothMeasurement`` first, then folder name order) is skipped with a
``RuntimeWarning`` and never blocks startup; recipes and configurations that
reference it are rejected as unknown methods.

Unlike segmentation models, ``Model`` is never instantiated: ``Model.run``
must be a ``@staticmethod`` (it is called on the class; a ``run(self, ...)``
instance method is skipped at load time). It takes the measurement as its
required first argument. Annotate
optional parameters and document them in a NumPy-style ``Parameters`` section:
the processing window uses the same argument parser as models to build typed
controls and parameter help buttons. Class docstrings provide step-level help.
Per-frame and timelapse processors receive a one-dimensional NumPy array;
bulk processors receive a DataFrame. Return the same shape (and, for bulk
processing, the same index and columns).

``apply_recipe(..., max_frame_i=n)`` restricts processing to the analyzed
frames of a position (the measurements viewer passes the last analyzed frame
when frames after it are ignored): every step, including timelapse windows
and the pooled statistics of bulk steps, sees only the rows with
``frame_i <= n``. The returned table still has every row of the input in the
same order; output columns are NaN on the rows after ``n`` (and on all rows
when no row is analyzed, which is then not an error).
"""
import inspect
import warnings
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

def _warn_skipped_processor(folder_name, reason):
    warnings.warn(
        f'Skipping measurement processor "{folder_name}": {reason}',
        RuntimeWarning, stacklevel=3
    )


def _load_processors(folder=None, package=None):
    """Discover processor packages in ``folder`` (default: this package).

    ``package`` is the importable name of ``folder`` (default: this
    package). ``SmoothMeasurement`` is loaded first, then the other folders
    in name order. A processor that fails to import, has no non-empty string
    ``name`` or lacks a ``@staticmethod`` ``Model.run`` (``apply_recipe``
    calls it on the class) is skipped with a ``RuntimeWarning``, as
    is one that reuses the ``name`` of a processor loaded before it (the
    first one wins), so a broken user plugin never blocks Cell-ACDC startup.
    """
    root = Path(__file__).parent if folder is None else Path(folder)
    package = __name__ if package is None else package
    mapper = {}
    loaded_from = {}
    folders = sorted(
        root.iterdir(),
        key=lambda folder: (folder.name != 'SmoothMeasurement', folder.name)
    )
    for folder in folders:
        if folder.name.startswith('_') or not (folder / 'acdcProcess.py').is_file():
            continue
        try:
            module = import_module(
                f'.{folder.name}.acdcProcess', package=package
            )
            name = module.name
            if not isinstance(name, str) or not name:
                raise TypeError('`name` must be a non-empty string')
            processor = module.Model
            function = processor.run
            if not isinstance(
                    inspect.getattr_static(processor, 'run'), staticmethod):
                raise TypeError('`Model.run` must be a @staticmethod')
        except Exception as err:
            _warn_skipped_processor(
                folder.name, f'{type(err).__name__}: {err}'
            )
            continue
        if name in mapper:
            _warn_skipped_processor(
                folder.name,
                f'duplicate name "{name}" (already used by '
                f'"{loaded_from[name]}")'
            )
            continue
        loaded_from[name] = folder.name
        mapper[name] = {
            'class': processor,
            'function': function,
            'function_name': folder.name,
            'docstring': processor.__doc__,
        }
    return mapper


MEASUREMENT_PROCESSING_MAPPER = _load_processors()


def _group_row_positions(index, is_timelapse):
    """Return the row positions of each group processed by a non-bulk step.

    Per-frame steps group rows by ``frame_i`` and keep their row order;
    timelapse steps group rows by ``Cell_ID`` and sort each cell's positions
    by frame (stable, so ties keep row order). Computed once per step and
    reused for every column.
    """
    level = 'Cell_ID' if is_timelapse else 'frame_i'
    groups = pd.Series(0, index=index).groupby(level=level, sort=False).indices
    if not is_timelapse:
        return list(groups.values())
    frames = index.get_level_values('frame_i').to_numpy()
    return [
        indices[np.argsort(frames[indices], kind='stable')]
        for indices in groups.values()
    ]


def _step_processor(step, result):
    """Return the processor of ``step`` after checking it can run on ``result``.

    Raises ``KeyError`` for an unknown method and ``ValueError`` for a
    bulk+timelapse processor or non-numeric input to a numeric processor.
    """
    processor = MEASUREMENT_PROCESSING_MAPPER[step['method']]['class']
    if processor.is_timelapse and processor.is_bulk_processing:
        raise ValueError('Bulk and timelapse processing are mutually exclusive.')
    if processor.numeric_input and any(not pd.api.types.is_numeric_dtype(dtype)
                                     for dtype in result.dtypes):
        raise ValueError('Processor requires numeric measurements.')
    return processor


def _run_recipe(result, recipe):
    """Run ``recipe`` on the selected-measurement table ``result``."""
    for step in recipe:
        processor = _step_processor(step, result)
        kwargs = step['kwargs']
        if processor.is_bulk_processing:
            processed = processor.run(result.copy(), **kwargs)
            if (not isinstance(processed, pd.DataFrame)
                    or not processed.index.equals(result.index)
                    or not processed.columns.equals(result.columns)):
                raise ValueError('Bulk processing must preserve measurement rows and columns.')
            result = processed
            continue

        groups = _group_row_positions(result.index, processor.is_timelapse)
        for column in result.columns:
            values = result[column].to_numpy(dtype=float, na_value=np.nan).copy()
            for indices in groups:
                processed = np.asarray(processor.run(values[indices].copy(), **kwargs))
                if processed.shape != indices.shape:
                    raise ValueError('Measurement processing must preserve the number of values.')
                values[indices] = processed
            result[column] = values
    return result


def apply_recipe(df, cols, recipe, new_colnames, max_frame_i=None):
    """Apply a recipe to selected numeric columns without modifying the input.

    Parameters
    ----------
    df : pandas.DataFrame
        Measurements indexed by unique (``frame_i``, ``Cell_ID``) pairs.
    cols : list of str
        Selected measurement columns of ``df``.
    recipe : list of dict
        Steps ``{'method': <processor name>, 'kwargs': {...}}`` applied in
        order.
    new_colnames : list of str
        One new (not yet existing) output column name per selected column.
    max_frame_i : int or None, optional
        Last analyzed frame. ``None`` (default) processes every row. With an
        int, the recipe runs only on the rows with ``frame_i <= max_frame_i``
        (timelapse, per-frame and bulk steps never see later rows); the
        output columns are NaN on the later rows. If no row is analyzed, all
        output values are NaN instead of raising "Select at least one
        measurement with data" (no processor runs, but the column checks
        on the full ``df`` and the recipe checks still apply).

    Returns
    -------
    pandas.DataFrame
        ``df`` (same rows, same order) joined with the output columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError('df must be a pandas DataFrame.')
    if list(df.index.names) != ['frame_i', 'Cell_ID']:
        raise ValueError('df must be indexed by frame_i and Cell_ID.')
    if not df.index.is_unique:
        raise ValueError('Measurement frame and cell pairs must be unique.')
    masked = max_frame_i is not None
    if not len(df.columns) or (df.empty and not masked):
        raise ValueError('Select at least one measurement with data.')
    if not recipe:
        raise ValueError('The measurement-processing recipe cannot be empty.')
    cols = list(cols)
    new_colnames = list(new_colnames)
    if not cols:
        raise ValueError('Select at least one measurement with data.')
    if len(new_colnames) != len(cols):
        raise ValueError(
            'Provide one output column name per selected measurement.'
        )
    if len(set(new_colnames)) != len(new_colnames):
        raise ValueError('Output column names must be unique.')
    existing = [name for name in new_colnames if name in df.columns]
    if existing:
        raise ValueError(
            f'Output columns already exist: {", ".join(map(str, existing))}'
        )

    selected = df[cols]
    if masked:
        analyzed = df.index.get_level_values('frame_i') <= max_frame_i
        selected = selected.loc[analyzed]
    if selected.empty:
        # Only reachable with `max_frame_i` (no analyzed row): all NaN. The
        # steps are still checked (unknown method, non-numeric input) so an
        # invalid recipe is rejected as without masking. Only the first
        # step is checked against the input dtypes, later steps see floats.
        for i, step in enumerate(recipe):
            _step_processor(step, selected if i == 0 else selected.iloc[:, :0])
        result = pd.DataFrame(np.nan, index=df.index, columns=selected.columns)
    else:
        result = _run_recipe(selected.copy(), recipe)

    # merge new column names (rows missing from `result` get NaN)
    result = result.rename(columns=dict(zip(cols, new_colnames)))
    df = df.join(result, how='left')
    return df
