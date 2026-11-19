"""`measurement_processing.apply_recipe(..., max_frame_i=n)`.

With ``max_frame_i`` only the analyzed rows (``frame_i <= n``) are processed:
every step (timelapse, per-frame, bulk pooling) sees only them, and the
outputs on later rows are NaN. The row set and order of the table never
change. ``max_frame_i=None`` keeps the unmasked behaviour.
"""
import numpy as np
import pandas as pd
import pytest
from qtpy.QtCore import QMutex, QWaitCondition
from qtpy.QtWidgets import QApplication

from cellacdc import measurement_processing, workers
from cellacdc.measurement_processing import apply_recipe

MAX_FRAME = 5


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


class PooledZScore:
    """Bulk step standardizing with statistics pooled over all rows."""

    is_timelapse = False
    is_bulk_processing = True
    numeric_input = True
    calls = []

    @classmethod
    def run(cls, table):
        cls.calls.append(table.copy())
        return (table - table.mean()) / table.std()


@pytest.fixture
def pooled(monkeypatch):
    PooledZScore.calls = []
    monkeypatch.setitem(
        measurement_processing.MEASUREMENT_PROCESSING_MAPPER, '_pooled',
        {'class': PooledZScore}
    )
    return PooledZScore


def step(method, **kwargs):
    return {'method': method, 'kwargs': kwargs}


def timelapse(seed=0, n_cells=4, n_frames=12):
    """Shuffled rows, some missing frames and NaNs, ramp-like values."""
    rng = np.random.default_rng(seed)
    rows = [
        (frame, cell)
        for cell in range(1, n_cells + 1)
        for frame in range(n_frames)
        if (frame + cell) % 5
    ]
    rows = [rows[i] for i in rng.permutation(len(rows))]
    index = pd.MultiIndex.from_tuples(rows, names=['frame_i', 'Cell_ID'])
    frames = index.get_level_values('frame_i').to_numpy()
    cells = index.get_level_values('Cell_ID').to_numpy()
    a = frames * cells + rng.normal(scale=0.1, size=len(rows))
    a[rng.random(len(rows)) < 0.1] = np.nan
    return pd.DataFrame({
        'a': a,
        'b': rng.normal(size=len(rows)) + 10 * (frames > MAX_FRAME),
        'label': [f'x{i}' for i in range(len(rows))],
    }, index=index)


RECIPES = {
    'smooth': [step('Smooth measurement', window=3)],
    'ramp': [step('Ramp detection', gap_frames=1, min_length=3)],
    'dummy': [step('Dummy'), step('Smooth measurement', window=5)],
    'pooled': [step('_pooled'), step('Smooth measurement', window=3)],
    'mixed': [
        step('Smooth measurement', window=3), step('_pooled'),
        step('Min-max normalization', min_value=-1.0, max_value=1.0),
    ],
}


@pytest.mark.parametrize('name', list(RECIPES))
def test_masked_outputs_match_truncated_table(name, dummy_processor, pooled):
    df = timelapse()
    original = df.copy()
    recipe = RECIPES[name]
    result = apply_recipe(df, ['a', 'b'], recipe, ['A', 'B'],
                          max_frame_i=MAX_FRAME)

    # Same rows in the same order, input untouched, other columns kept.
    assert result.index.equals(df.index)
    pd.testing.assert_frame_equal(df, original)
    pd.testing.assert_frame_equal(result[df.columns], df)

    frames = df.index.get_level_values('frame_i')
    later = frames > MAX_FRAME
    assert later.any() and (~later).any()
    assert result.loc[later, ['A', 'B']].isna().all().all()

    truncated = df[~later]
    expected = apply_recipe(truncated, ['a', 'b'], recipe, ['A', 'B'])
    pd.testing.assert_frame_equal(
        result.loc[~later, ['A', 'B']], expected[['A', 'B']]
    )


def test_masking_changes_timelapse_results_near_the_boundary():
    # The smoothing window must not reach into ignored frames.
    df = timelapse(seed=1)
    recipe = RECIPES['smooth']
    masked = apply_recipe(df, ['b'], recipe, ['B'], max_frame_i=MAX_FRAME)
    full = apply_recipe(df, ['b'], recipe, ['B'])
    at_boundary = df.index.get_level_values('frame_i') == MAX_FRAME
    assert not np.allclose(
        masked.loc[at_boundary, 'B'], full.loc[at_boundary, 'B']
    )


def test_bulk_pooling_sees_only_analyzed_rows(pooled):
    df = timelapse(seed=2)
    result = apply_recipe(df, ['a', 'b'], [step('_pooled')], ['A', 'B'],
                          max_frame_i=MAX_FRAME)
    call, = pooled.calls
    analyzed = df.index.get_level_values('frame_i') <= MAX_FRAME
    pd.testing.assert_frame_equal(call, df.loc[analyzed, ['a', 'b']])
    # Pooled statistics of the analyzed rows: mean 0, std 1 there.
    np.testing.assert_allclose(result.loc[analyzed, ['A', 'B']].mean(), 0,
                               atol=1e-12)
    np.testing.assert_allclose(result.loc[analyzed, ['A', 'B']].std(), 1)


def test_none_and_late_max_frame_keep_unmasked_behaviour(dummy_processor,
                                                          pooled):
    df = timelapse(seed=3)
    recipe = RECIPES['mixed']
    unmasked = apply_recipe(df, ['a', 'b'], recipe, ['A', 'B'])
    pd.testing.assert_frame_equal(
        apply_recipe(df, ['a', 'b'], recipe, ['A', 'B'], max_frame_i=None),
        unmasked
    )
    last = int(df.index.get_level_values('frame_i').max())
    pd.testing.assert_frame_equal(
        apply_recipe(df, ['a', 'b'], recipe, ['A', 'B'], max_frame_i=last),
        unmasked
    )


@pytest.mark.parametrize('max_frame_i', [-1, np.int64(-3)])
def test_no_analyzed_row_gives_all_nan_outputs(max_frame_i, pooled):
    df = timelapse(seed=4)
    result = apply_recipe(df, ['a', 'b'], RECIPES['pooled'], ['A', 'B'],
                          max_frame_i=max_frame_i)
    assert pooled.calls == []
    assert result.index.equals(df.index)
    assert list(result.columns) == [*df.columns, 'A', 'B']
    assert result[['A', 'B']].isna().all().all()


def test_empty_table_raises_only_when_unmasked():
    df = timelapse().iloc[:0]
    with pytest.raises(ValueError, match='at least one measurement with data'):
        apply_recipe(df, ['a'], RECIPES['smooth'], ['A'])
    result = apply_recipe(df, ['a'], RECIPES['smooth'], ['A'], max_frame_i=3)
    assert result.empty and list(result.columns) == [*df.columns, 'A']


@pytest.mark.parametrize('cols, new_colnames, recipe, error, match', [
    (['a'], ['b'], RECIPES['smooth'], ValueError, 'already exist: b'),
    (['a', 'b'], ['A'], RECIPES['smooth'], ValueError, 'one output column'),
    ([], [], RECIPES['smooth'], ValueError, 'at least one measurement'),
    (['a'], ['A'], [], ValueError, 'cannot be empty'),
    (['missing'], ['M'], RECIPES['smooth'], KeyError, 'missing'),
])
def test_validation_uses_full_table_even_without_analyzed_rows(
        cols, new_colnames, recipe, error, match):
    df = timelapse()
    with pytest.raises(error, match=match):
        apply_recipe(df, cols, recipe, new_colnames, max_frame_i=-1)


@pytest.mark.parametrize('max_frame_i', [MAX_FRAME, -1])
def test_masked_invalid_recipe_still_rejected(max_frame_i):
    # Also when no row is analyzed (no processor runs).
    df = timelapse()
    with pytest.raises(ValueError, match='requires numeric'):
        apply_recipe(df, ['label'], RECIPES['smooth'], ['L'],
                     max_frame_i=max_frame_i)
    with pytest.raises(KeyError, match='unknown'):
        apply_recipe(df, ['a'], [step('unknown')], ['A'],
                     max_frame_i=max_frame_i)


def test_cell_only_after_last_analyzed_frame_gets_nan():
    df = timelapse(seed=6)
    late = pd.DataFrame(
        {'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0], 'label': ['l'] * 3},
        index=pd.MultiIndex.from_tuples(
            [(f, 99) for f in (MAX_FRAME + 1, MAX_FRAME + 2, MAX_FRAME + 3)],
            names=['frame_i', 'Cell_ID']
        )
    )
    df = pd.concat([late.iloc[:1], df, late.iloc[1:]])
    for recipe in (RECIPES['smooth'], RECIPES['ramp']):
        result = apply_recipe(df, ['a', 'b'], recipe, ['A', 'B'],
                              max_frame_i=MAX_FRAME)
        assert result.index.equals(df.index)
        assert result.loc[(slice(None), 99), ['A', 'B']].isna().all().all()


def _run_worker(df, recipe, **setup_kwargs):
    worker = workers.MeasurementProcessingWorkerGUI(QMutex(), QWaitCondition())
    done, failed = [], []
    worker.sigDone.connect(done.append)
    worker.signals.critical.connect(failed.append)
    worker.setupJob(
        df=df, cols=['a', 'b'], recipe=recipe, new_colnames=['A', 'B'],
        **setup_kwargs
    )
    worker.runJob()
    worker.deleteLater()
    assert failed == []
    result, = done
    return result


def test_worker_passes_max_frame_i(app):
    df = timelapse(seed=5)
    recipe = RECIPES['smooth']
    masked = _run_worker(df, recipe, max_frame_i=MAX_FRAME)
    pd.testing.assert_frame_equal(
        masked,
        apply_recipe(df, ['a', 'b'], recipe, ['A', 'B'], max_frame_i=MAX_FRAME)
    )
    later = df.index.get_level_values('frame_i') > MAX_FRAME
    assert masked.loc[later, ['A', 'B']].isna().all().all()
    # Default: unmasked.
    pd.testing.assert_frame_equal(
        _run_worker(df, recipe),
        apply_recipe(df, ['a', 'b'], recipe, ['A', 'B'])
    )
