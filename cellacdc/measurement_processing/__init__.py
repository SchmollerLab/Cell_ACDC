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
no model server or separate environment is needed.

``Model.run`` takes the measurement as its required first argument. Annotate
optional parameters and document them in a NumPy-style ``Parameters`` section:
the processing window uses the same argument parser as models to build typed
controls and parameter help buttons. Class docstrings provide step-level help.
Per-frame and timelapse processors receive a one-dimensional NumPy array;
bulk processors receive a DataFrame. Return the same shape (and, for bulk
processing, the same index and columns).
"""
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

def _load_processors():
    mapper = {}
    folders = sorted(
        Path(__file__).parent.iterdir(),
        key=lambda folder: (folder.name != 'SmoothMeasurement', folder.name)
    )
    for folder in folders:
        if folder.name.startswith('_') or not (folder / 'acdcProcess.py').is_file():
            continue
        module = import_module(f'.{folder.name}.acdcProcess', package=__name__)
        if module.name in mapper:
            raise ValueError(f'Duplicate measurement processor name: {module.name}')
        processor = module.Model
        mapper[module.name] = {
            'class': processor,
            'function': processor.run,
            'function_name': folder.name,
            'docstring': processor.__doc__,
        }
    return mapper


MEASUREMENT_PROCESSING_MAPPER = _load_processors()


def apply_recipe(measurements, recipe):
    """Apply a recipe to selected numeric columns without modifying the input."""
    if not isinstance(measurements, pd.DataFrame):
        raise ValueError('Measurements must be a pandas DataFrame.')
    if list(measurements.index.names) != ['frame_i', 'Cell_ID']:
        raise ValueError('Measurements must be indexed by frame_i and Cell_ID.')
    if not measurements.index.is_unique:
        raise ValueError('Measurement frame and cell pairs must be unique.')
    if measurements.empty or not len(measurements.columns):
        raise ValueError('Select at least one measurement with data.')
    if not recipe:
        raise ValueError('The measurement-processing recipe cannot be empty.')

    result = measurements.copy()
    for step in recipe:
        processor = MEASUREMENT_PROCESSING_MAPPER[step['method']]['class']
        if processor.is_timelapse and processor.is_bulk_processing:
            raise ValueError('Bulk and timelapse processing are mutually exclusive.')
        if processor.numeric_input and any(not pd.api.types.is_numeric_dtype(np.dtype)
                                         for dtype in result.dtypes):
            raise ValueError('Processor requires numeric measurements.')
        kwargs = step['kwargs']
        if processor.is_bulk_processing:
            processed = processor.run(result.copy(), **kwargs)
            if (not isinstance(processed, pd.DataFrame)
                    or not processed.index.equals(result.index)
                    or not processed.columns.equals(result.columns)):
                raise ValueError('Bulk processing must preserve measurement rows and columns.')
            result = processed
            continue

        level = 'Cell_ID' if processor.is_timelapse else 'frame_i'
        groups = result.groupby(level=level, sort=False).indices.values()
        for column in result.columns:
            values = result[column].to_numpy(dtype=float, na_value=np.nan).copy()
            for indices in groups:
                if processor.is_timelapse:
                    frames = result.index.get_level_values('frame_i')
                    indices = indices[np.argsort(frames.take(indices), kind='stable')]
                processed = np.asarray(processor.run(values[indices].copy(), **kwargs))
                if processed.shape != indices.shape:
                    raise ValueError('Measurement processing must preserve the number of values.')
                values[indices] = processed
            result[column] = values
    return result
