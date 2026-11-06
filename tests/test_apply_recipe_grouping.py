"""Grouping of rows by `measurement_processing.apply_recipe`.

Timelapse steps receive each cell's values in frame order, per-frame steps
receive each frame's values in row order, and results are written back to the
original rows, for every selected column. The per-step groups are shared by
all columns, so these tests use several cells, columns, shuffled rows and
missing frames.
"""
import numpy as np
import pandas as pd
import pytest

from cellacdc.measurement_processing import (
    MEASUREMENT_PROCESSING_MAPPER, apply_recipe
)


class _Recorder:
    """Order-sensitive processor that records every call."""

    numeric_input = True
    is_bulk_processing = False
    calls = []

    @classmethod
    def run(cls, measurement, offset=0.0):
        cls.calls.append(measurement.copy())
        # Depends on the order of the input: reveals any reordering.
        return np.nancumsum(measurement) + np.arange(len(measurement)) + offset


class TimelapseRecorder(_Recorder):
    is_timelapse = True
    calls = []


class FrameRecorder(_Recorder):
    is_timelapse = False
    calls = []


class WrongLength:
    is_timelapse = True
    is_bulk_processing = False
    numeric_input = True

    @staticmethod
    def run(measurement):
        return measurement[:-1] if len(measurement) > 1 else measurement


@pytest.fixture
def processors(monkeypatch):
    TimelapseRecorder.calls = []
    FrameRecorder.calls = []
    for name, processor in (('_timelapse_recorder', TimelapseRecorder),
                            ('_frame_recorder', FrameRecorder),
                            ('_wrong_length', WrongLength)):
        monkeypatch.setitem(
            MEASUREMENT_PROCESSING_MAPPER, name, {'class': processor}
        )


def step(method, **kwargs):
    return {'method': method, 'kwargs': kwargs}


def shuffled_measurements(seed=0, n_cells=4, n_frames=8, columns=('a', 'b', 'c')):
    """Rows in random order; each cell misses some frames; values have NaNs."""
    rng = np.random.default_rng(seed)
    rows = [
        (frame, cell)
        for cell in range(1, n_cells + 1)
        for frame in range(n_frames)
        if (frame + cell) % 3  # every cell misses different frames
    ]
    rows = [rows[i] for i in rng.permutation(len(rows))]
    index = pd.MultiIndex.from_tuples(rows, names=['frame_i', 'Cell_ID'])
    data = {}
    for column in columns:
        values = rng.normal(size=len(rows))
        values[rng.random(len(rows)) < 0.15] = np.nan
        data[column] = values
    data['untouched'] = rng.normal(size=len(rows))
    return pd.DataFrame(data, index=index)


def expected_per_group(df, columns, level, sort_by_frame, func):
    """Reference: apply `func` group by group with pandas, one column at a time."""
    expected = df[columns].astype(float).copy()
    for column in columns:
        for _, group in df[column].astype(float).groupby(level=level, sort=False):
            if sort_by_frame:
                group = group.sort_index(level='frame_i', kind='stable')
            expected.loc[group.index, column] = func(group.to_numpy())
    return expected


def test_timelapse_groups_each_cell_in_frame_order(processors):
    df = shuffled_measurements()
    columns = ['a', 'b', 'c']
    result = apply_recipe(
        df, columns, [step('_timelapse_recorder')], ['A', 'B', 'C']
    )

    # One call per cell and column, each with that cell's frame-sorted values.
    n_cells = df.index.get_level_values('Cell_ID').nunique()
    assert len(TimelapseRecorder.calls) == n_cells * len(columns)
    calls = iter(TimelapseRecorder.calls)
    for column in columns:
        for cell_id in pd.unique(df.index.get_level_values('Cell_ID')):
            cell = df.xs(cell_id, level='Cell_ID')[column].sort_index()
            np.testing.assert_array_equal(next(calls), cell.to_numpy())

    expected = expected_per_group(
        df, columns, 'Cell_ID', True,
        lambda values: np.nancumsum(values) + np.arange(len(values))
    )
    for column, new_column in zip(columns, ['A', 'B', 'C']):
        np.testing.assert_array_equal(
            result[new_column].to_numpy(), expected[column].to_numpy()
        )


def test_timelapse_result_does_not_depend_on_row_order(processors):
    df = shuffled_measurements(seed=3)
    columns = ['a', 'b', 'c']
    recipe = [step('Smooth measurement', window=3), step('_timelapse_recorder')]
    new_columns = ['A', 'B', 'C']

    shuffled = apply_recipe(df, columns, recipe, new_columns)
    ordered = apply_recipe(df.sort_index(), columns, recipe, new_columns)

    # Rows keep the input order, and values follow their (frame, cell) row.
    assert shuffled.index.equals(df.index)
    pd.testing.assert_frame_equal(
        shuffled.sort_index(), ordered, check_exact=True
    )


def test_missing_frames_are_skipped_not_filled(processors):
    df = shuffled_measurements(seed=1, n_cells=3, n_frames=10, columns=('a',))
    apply_recipe(df, ['a'], [step('_timelapse_recorder')], ['A'])
    for call, cell_id in zip(
            TimelapseRecorder.calls,
            pd.unique(df.index.get_level_values('Cell_ID'))):
        present = df.xs(cell_id, level='Cell_ID').index
        assert len(call) == len(present) < 10


def test_per_frame_groups_keep_row_order(processors):
    df = shuffled_measurements(seed=2)
    columns = ['a', 'b', 'c']
    result = apply_recipe(
        df, columns, [step('_frame_recorder', offset=1.5)], ['A', 'B', 'C']
    )

    n_frames = df.index.get_level_values('frame_i').nunique()
    assert len(FrameRecorder.calls) == n_frames * len(columns)
    calls = iter(FrameRecorder.calls)
    frames = df.index.get_level_values('frame_i')
    for column in columns:
        for frame in pd.unique(frames):
            # Rows of a frame in their original (unsorted) order.
            np.testing.assert_array_equal(
                next(calls), df[column].to_numpy()[frames == frame]
            )

    expected = expected_per_group(
        df, columns, 'frame_i', False,
        lambda values: np.nancumsum(values) + np.arange(len(values)) + 1.5
    )
    for column, new_column in zip(columns, ['A', 'B', 'C']):
        np.testing.assert_array_equal(
            result[new_column].to_numpy(), expected[column].to_numpy()
        )


def test_multi_step_recipe_matches_steps_applied_one_by_one(processors):
    df = shuffled_measurements(seed=4)
    df['ints'] = np.arange(len(df)) % 5
    columns = ['a', 'ints', 'b']
    recipe = [
        step('_timelapse_recorder'),
        step('_frame_recorder', offset=-2.0),
        step('Min-max normalization'),
        step('Smooth measurement', window=5),
    ]
    result = apply_recipe(df, columns, recipe, ['A', 'I', 'B'])

    current = df[columns]
    processed_columns = [f'{column}_processed' for column in columns]
    for item in recipe:
        current = apply_recipe(current, columns, [item], processed_columns)
        current = current[processed_columns].set_axis(columns, axis=1)
    for column, new_column in zip(columns, ['A', 'I', 'B']):
        np.testing.assert_array_equal(
            result[new_column].to_numpy(), current[column].to_numpy()
        )
    # Integer input columns are processed as floats.
    assert result['I'].dtype == float
    # Unselected columns and the input are untouched.
    pd.testing.assert_series_equal(result['untouched'], df['untouched'])
    assert 'A' not in df.columns


def test_wrong_output_length_still_raises(processors):
    df = shuffled_measurements(seed=5)
    with pytest.raises(ValueError, match='preserve the number of values'):
        apply_recipe(df, ['a', 'b'], [step('_wrong_length')], ['A', 'B'])


_SMOOTH = [{'method': 'Smooth measurement', 'kwargs': {'window': 3}}]


def _two_column_df():
    index = pd.MultiIndex.from_tuples(
        [(0, 1), (1, 1), (0, 2)], names=['frame_i', 'Cell_ID']
    )
    return pd.DataFrame({'a': [1., 2., 3.], 'b': [4., 5., 6.]}, index=index)


@pytest.mark.parametrize('cols, new_colnames, match', [
    ([], [], 'at least one measurement'),
    (['a', 'b'], ['a_smooth'], 'one output column name per'),
    (['a'], ['a_smooth', 'b_smooth'], 'one output column name per'),
    (['a', 'b'], ['out', 'out'], 'must be unique'),
    (['a'], ['b'], 'already exist: b'),
    (['a'], ['a'], 'already exist: a'),
])
def test_invalid_output_columns_are_rejected(cols, new_colnames, match):
    df = _two_column_df()
    original = df.copy()
    with pytest.raises(ValueError, match=match):
        apply_recipe(df, cols, _SMOOTH, new_colnames)
    pd.testing.assert_frame_equal(df, original)


def test_valid_output_columns_are_appended():
    df = _two_column_df()
    result = apply_recipe(df, ['a', 'b'], _SMOOTH, ['a_s', 'b_s'])
    assert list(result.columns) == ['a', 'b', 'a_s', 'b_s']
    pd.testing.assert_frame_equal(result[['a', 'b']], df)
