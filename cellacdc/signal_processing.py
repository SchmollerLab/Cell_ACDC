"""Measurement recipes, independent of the image preprocessing registry.

Steps operate per frame by default (one measurement across that frame's cells).
``is_timelapse`` steps receive one cell's measurement in frame order.
``is_bulk_processing`` steps receive the entire selected measurement DataFrame
once and must return a DataFrame with the same index and columns. Bulk and
timelapse are mutually exclusive. Recipes preserve row order and missing values;
time windows count observations, not elapsed time or absent frames.
"""
import numpy as np
import pandas as pd


class Dummy:
    """Return measurements unchanged (per-frame processing)."""

    is_timelapse = False
    is_bulk_processing = False

    @staticmethod
    def run(signal):
        return signal.copy()


class SmoothSignal:
    """Centered moving average per cell, ignoring missing observations.

    Missing input values remain missing. At the ends, use the available
    observations. ``window`` must be a positive number of observations.
    """

    is_timelapse = True
    is_bulk_processing = False

    @staticmethod
    def run(signal, window: int = 3):
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError('The smoothing window must be a positive integer.')
        values = pd.Series(signal, dtype=float)
        result = values.rolling(window, center=True, min_periods=1).mean()
        return result.mask(values.isna()).to_numpy()


SIGNAL_PROCESSING_MAPPER = {
    name: {
        'class': step,
        'function': step.run,
        'function_name': step.__name__,
        'docstring': step.__doc__,
    }
    for name, step in (('Smooth signal', SmoothSignal), ('Dummy', Dummy))
}


def apply_recipe(measurements, recipe):
    """Apply a recipe to selected numeric columns without modifying the input."""
    if list(measurements.index.names) != ['frame_i', 'Cell_ID']:
        raise ValueError('Measurements must be indexed by frame_i and Cell_ID.')
    if not measurements.index.is_unique:
        raise ValueError('Measurement frame and cell pairs must be unique.')
    if measurements.empty or not len(measurements.columns):
        raise ValueError('Select at least one measurement with data.')
    if not recipe:
        raise ValueError('The signal-processing recipe cannot be empty.')
    if any(not pd.api.types.is_numeric_dtype(dtype)
           for dtype in measurements.dtypes):
        raise ValueError('Signal processing requires numeric measurements.')

    result = measurements.copy()
    for step in recipe:
        processor = SIGNAL_PROCESSING_MAPPER[step['method']]['class']
        if processor.is_timelapse and processor.is_bulk_processing:
            raise ValueError('Bulk and timelapse processing are mutually exclusive.')
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
                    raise ValueError('Signal processing must preserve the number of values.')
                values[indices] = processed
            result[column] = values
    return result
