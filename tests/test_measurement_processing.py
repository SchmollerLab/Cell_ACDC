import numpy as np
import pandas as pd
import pytest

from cellacdc import measurement_processing
from cellacdc.measurement_processing.Dummy.acdcProcess import Model as Dummy
from cellacdc.measurement_processing.SmoothMeasurement.acdcProcess import (
    Model as SmoothMeasurement
)

SUFFIX = 'processed'


def recipe(method='Smooth measurement', **kwargs):
    return [{'method': method, 'kwargs': kwargs}]


def output_names(cols):
    return [f'{col}_{SUFFIX}' for col in cols]


def process(df, steps, cols=None):
    """Apply `steps` to `cols` (default: all columns), appending outputs."""
    cols = list(df.columns) if cols is None else list(cols)
    return measurement_processing.apply_recipe(
        df, cols, steps, output_names(cols)
    )


def test_registry_loads_processors_from_model_style_packages():
    registry = measurement_processing.MEASUREMENT_PROCESSING_MAPPER
    assert list(registry) == [
        'Smooth measurement', 'Dummy', 'Min-max normalization',
        'Ramp detection'
    ]
    assert registry['Smooth measurement']['class'] is SmoothMeasurement
    assert registry['Dummy']['class'] is Dummy
    assert registry['Smooth measurement']['function'] is SmoothMeasurement.run
    assert registry['Smooth measurement']['function_name'] == 'SmoothMeasurement'
    assert registry['Smooth measurement']['docstring'] == SmoothMeasurement.__doc__
    # Processor folders are subpackages; their Model is the registered class
    assert measurement_processing.SmoothMeasurement.acdcProcess.Model is SmoothMeasurement
    assert measurement_processing.Dummy.acdcProcess.Model is Dummy
    for entry in registry.values():
        processor = entry['class']
        assert entry['function'] is processor.run
        assert not (processor.is_timelapse and processor.is_bulk_processing)


@pytest.fixture
def measurements():
    return pd.DataFrame({
        'frame_i': [2, 0, 1, 0, 2, 1],
        'Cell_ID': [1, 1, 1, 2, 2, 2],
        'value': [9, 0, 0, 100, 100, 100],
        'other': [90, 0, 0, 10, 10, 10],
    }).set_index(['frame_i', 'Cell_ID'])


def test_smoothing_sorts_per_cell_without_changing_input(measurements):
    original = measurements.copy()
    result = process(measurements, recipe(window=3))
    assert list(result.columns) == (
        ['value', 'other'] + output_names(['value', 'other'])
    )
    np.testing.assert_allclose(
        result[f'value_{SUFFIX}'], [4.5, 0, 3, 100, 100, 100]
    )
    np.testing.assert_allclose(
        result[f'other_{SUFFIX}'], [45, 0, 30, 10, 10, 10]
    )
    # Input columns are kept unchanged next to the appended outputs
    pd.testing.assert_frame_equal(result[['value', 'other']], original)
    pd.testing.assert_index_equal(result.index, measurements.index)
    pd.testing.assert_frame_equal(measurements, original)


def test_only_selected_columns_are_processed(measurements):
    result = process(measurements, recipe(window=3), cols=['value'])
    assert list(result.columns) == ['value', 'other', f'value_{SUFFIX}']
    np.testing.assert_allclose(
        result[f'value_{SUFFIX}'], [4.5, 0, 3, 100, 100, 100]
    )
    pd.testing.assert_series_equal(result['other'], measurements['other'])


def test_integer_measurements_keep_fractional_results(measurements):
    result = process(measurements, recipe(window=3))
    assert result.iloc[0][f'value_{SUFFIX}'] == 4.5
    assert result[f'value_{SUFFIX}'].dtype.kind == 'f'


def test_smoothing_short_tracks_missing_frames_and_values():
    df = pd.DataFrame({
        'frame_i': [0, 2, 5, 0],
        'Cell_ID': [1, 1, 1, 2],
        'value': [2, np.nan, 8, np.nan],
    }).set_index(['frame_i', 'Cell_ID'])
    result = process(df, recipe(window=9))
    np.testing.assert_allclose(result[f'value_{SUFFIX}'], [5, np.nan, 5, np.nan])


@pytest.mark.parametrize('window', [0, -1, 2.5, True])
def test_invalid_smoothing_window(measurements, window):
    with pytest.raises(ValueError, match='positive integer'):
        process(measurements, recipe(window=window))


def test_recipe_steps_are_sequential(measurements):
    steps = recipe(window=3) + recipe('Dummy') + recipe(window=3)
    result = process(measurements, steps)
    np.testing.assert_allclose(
        result[f'value_{SUFFIX}'], [3.75, 1.5, 2.5, 100, 100, 100]
    )


def register(monkeypatch, processor):
    monkeypatch.setitem(
        measurement_processing.MEASUREMENT_PROCESSING_MAPPER,
        'Test', {'class': processor}
    )


def test_per_frame_processing_does_not_mix_frames(measurements, monkeypatch):
    class Center(Dummy):
        @staticmethod
        def run(values):
            return values - values.mean()

    register(monkeypatch, Center)
    result = process(measurements, recipe('Test'))
    np.testing.assert_allclose(
        result[f'value_{SUFFIX}'], [-45.5, -50, -50, 50, 45.5, 50]
    )


def test_bulk_processing_receives_selected_table_once(measurements, monkeypatch):
    calls = []

    class Bulk(Dummy):
        is_bulk_processing = True

        @staticmethod
        def run(table):
            calls.append(table.copy())
            return table + 1

    register(monkeypatch, Bulk)
    result = process(measurements, recipe('Test'), cols=['value'])
    assert len(calls) == 1
    # Only the selected columns are passed to the bulk processor
    pd.testing.assert_frame_equal(calls[0], measurements[['value']])
    pd.testing.assert_series_equal(
        result[f'value_{SUFFIX}'],
        (measurements['value'] + 1).rename(f'value_{SUFFIX}')
    )


def test_bulk_and_timelapse_are_exclusive(measurements, monkeypatch):
    class Invalid(SmoothMeasurement):
        is_bulk_processing = True

    register(monkeypatch, Invalid)
    with pytest.raises(ValueError, match='mutually exclusive'):
        process(measurements, recipe('Test'))


@pytest.mark.parametrize('bulk', [False, True])
def test_processors_must_preserve_shape(measurements, monkeypatch, bulk):
    class Invalid(Dummy):
        is_bulk_processing = bulk

        @staticmethod
        def run(values):
            return values[:-1]

    register(monkeypatch, Invalid)
    with pytest.raises(ValueError, match='preserve'):
        process(measurements, recipe('Test'))


def test_empty_recipe_is_rejected(measurements):
    with pytest.raises(ValueError, match='cannot be empty'):
        process(measurements, [])
