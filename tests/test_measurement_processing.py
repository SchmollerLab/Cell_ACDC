import numpy as np
import pandas as pd
import pytest

from cellacdc import measurement_processing


def recipe(method='Smooth measurement', **kwargs):
    return [{'method': method, 'kwargs': kwargs}]


def test_registry_loads_processors_from_model_style_packages():
    from cellacdc.measurement_processing.Dummy.acdcProcess import Model as Dummy
    from cellacdc.measurement_processing.SmoothMeasurement.acdcProcess import (
        Model as SmoothMeasurement
    )

    registry = measurement_processing.MEASUREMENT_PROCESSING_MAPPER
    assert list(registry) == ['Smooth measurement', 'Dummy']
    assert registry['Smooth measurement']['class'] is SmoothMeasurement
    assert registry['Dummy']['class'] is Dummy
    assert registry['Smooth measurement']['function'] is SmoothMeasurement.run
    assert registry['Smooth measurement']['function_name'] == 'SmoothMeasurement'
    assert registry['Smooth measurement']['docstring'] == SmoothMeasurement.__doc__
    assert measurement_processing.SmoothMeasurement is SmoothMeasurement
    assert measurement_processing.Dummy is Dummy


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
    result = measurement_processing.apply_recipe(measurements, recipe(window=3))
    np.testing.assert_allclose(result['value'], [4.5, 0, 3, 100, 100, 100])
    np.testing.assert_allclose(result['other'], [45, 0, 30, 10, 10, 10])
    pd.testing.assert_index_equal(result.index, measurements.index)
    pd.testing.assert_frame_equal(measurements, original)


def test_integer_measurements_keep_fractional_results(measurements):
    result = measurement_processing.apply_recipe(measurements, recipe(window=3))
    assert result.iloc[0]['value'] == 4.5
    assert result['value'].dtype.kind == 'f'


def test_smoothing_short_tracks_missing_frames_and_values():
    df = pd.DataFrame({
        'frame_i': [0, 2, 5, 0],
        'Cell_ID': [1, 1, 1, 2],
        'value': [2, np.nan, 8, np.nan],
    }).set_index(['frame_i', 'Cell_ID'])
    result = measurement_processing.apply_recipe(df, recipe(window=9))
    np.testing.assert_allclose(result['value'], [5, np.nan, 5, np.nan])


@pytest.mark.parametrize('window', [0, -1, 2.5, True])
def test_invalid_smoothing_window(measurements, window):
    with pytest.raises(ValueError, match='positive integer'):
        measurement_processing.apply_recipe(measurements, recipe(window=window))


def test_recipe_steps_are_sequential(measurements):
    steps = recipe(window=3) + recipe('Dummy') + recipe(window=3)
    result = measurement_processing.apply_recipe(measurements, steps)
    np.testing.assert_allclose(result['value'], [3.75, 1.5, 2.5, 100, 100, 100])


def register(monkeypatch, processor):
    monkeypatch.setitem(
        measurement_processing.MEASUREMENT_PROCESSING_MAPPER,
        'Test', {'class': processor}
    )


def test_per_frame_processing_does_not_mix_frames(measurements, monkeypatch):
    class Center(measurement_processing.Dummy):
        @staticmethod
        def run(values):
            return values - values.mean()

    register(monkeypatch, Center)
    result = measurement_processing.apply_recipe(measurements, recipe('Test'))
    np.testing.assert_allclose(result['value'], [-45.5, -50, -50, 50, 45.5, 50])


def test_bulk_processing_receives_selected_table_once(measurements, monkeypatch):
    calls = []

    class Bulk(measurement_processing.Dummy):
        is_bulk_processing = True

        @staticmethod
        def run(table):
            calls.append(table.copy())
            return table + 1

    register(monkeypatch, Bulk)
    result = measurement_processing.apply_recipe(measurements, recipe('Test'))
    assert len(calls) == 1
    pd.testing.assert_frame_equal(calls[0], measurements)
    pd.testing.assert_frame_equal(result, measurements + 1)


def test_bulk_and_timelapse_are_exclusive(measurements, monkeypatch):
    class Invalid(measurement_processing.SmoothMeasurement):
        is_bulk_processing = True

    register(monkeypatch, Invalid)
    with pytest.raises(ValueError, match='mutually exclusive'):
        measurement_processing.apply_recipe(measurements, recipe('Test'))


@pytest.mark.parametrize('bulk', [False, True])
def test_processors_must_preserve_shape(measurements, monkeypatch, bulk):
    class Invalid(measurement_processing.Dummy):
        is_bulk_processing = bulk

        @staticmethod
        def run(values):
            return values[:-1]

    register(monkeypatch, Invalid)
    with pytest.raises(ValueError, match='preserve'):
        measurement_processing.apply_recipe(measurements, recipe('Test'))


def test_empty_recipe_is_rejected(measurements):
    with pytest.raises(ValueError, match='cannot be empty'):
        measurement_processing.apply_recipe(measurements, [])
