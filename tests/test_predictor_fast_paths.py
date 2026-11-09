"""Parity of the vectorized predictor paths with the per-target/per-cell ones.

OtherCellMeasurements.prepare_inputs stores whole blocks of targets with
array operations (``_BLOCKWISE_EMIT``) and TimepointPredictor /
CustomAnnotationPredictor.from_equation evaluate row-wise equations once for
all cells (``TimepointPredictor._evaluate_all_cells``). Both must give
exactly the results of the target-by-target and cell-by-cell reference
paths, which are forced here by disabling the fast paths.
"""
import warnings
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from cellacdc import timepoint_predictors as tp
from cellacdc.custom_annotation_predictors import CustomAnnotationPredictor


def random_table(seed, kind='tree', nCells=12, nFrames=7):
    """Random table with tracking gaps, NaN/inf, ties, a backtick-quoted
    column, int/bool/object columns and lineage annotations."""
    rng = np.random.default_rng(seed)
    rows = [
        (t, ID) for t in range(nFrames) for ID in range(1, nCells + 1)
        if rng.random() > 0.2
    ]
    df = pd.DataFrame(rows, columns=['frame_i', 'Cell_ID'])
    n = len(df)
    # Rounded values: ties for NUNIQUE and distinct direct values.
    area = rng.normal(1, 0.5, n).round(1)
    area[rng.random(n) < 0.15] = np.nan
    area[rng.random(n) < 0.04] = np.inf
    area[rng.random(n) < 0.04] = -np.inf
    # + 0.0 turns the -0.0 of rounding into 0.0 (-0.0 disables block
    # emission, see test_block_emission_is_used_and_skips_negative_zero).
    df['area'] = area + 0.0
    df['GFP mean'] = rng.random(n).round(2)
    df['n_spots'] = rng.integers(0, 4, n)
    df['flag'] = rng.random(n) < 0.5
    df['stage'] = rng.choice(['G1', 'S', 'G2'], n)
    if kind == 'tree':
        parents = {
            ID: (int(rng.integers(1, ID)) if ID > 3 else -1)
            for ID in range(1, nCells + 1)
        }
        parents[7] = np.nan
        df['parent_ID_tree'] = df['Cell_ID'].map(parents).astype(float)
    else:
        df['relative_ID'] = rng.choice([-1, *range(1, nCells + 1)], n)
        df['relationship'] = rng.choice(['mother', 'bud'], n)
    return df.sample(frac=1, random_state=seed).set_index(['frame_i', 'Cell_ID'])


def assert_alternatives_equal(actual, expected):
    if expected is None:
        assert actual is None
        return
    assert actual['columns'] == expected['columns']
    assert actual['by_target'].keys() == expected['by_target'].keys()
    for key, options in expected['by_target'].items():
        actualOptions = actual['by_target'][key]
        assert actualOptions == options
        # Same scalar types: they decide the dtypes of the expanded rows.
        assert [tuple(map(type, option)) for option in actualOptions] == [
            tuple(map(type, option)) for option in options
        ]


def prepare_both(monkeypatch, *args, **kwargs):
    """prepare_inputs with and without block emission (and warnings)."""
    results = []
    for blockwise in (True, False):
        with monkeypatch.context() as patch, warnings.catch_warnings(
                record=True) as caught:
            warnings.simplefilter('always')
            patch.setattr(tp.OtherCellMeasurements, '_BLOCKWISE_EMIT', blockwise)
            output = tp.OtherCellMeasurements.prepare_inputs(*args, **kwargs)
        results.append((output, [str(warning.message) for warning in caught]))
    return results


AGGREGATES = (
    'MIN(area_other) + MAX(area_other) + MEAN(area_other) '
    '+ AVERAGE(`GFP mean_other`) + MEDIAN(area_other) + SUM(area_other) '
    '+ STD(`GFP mean_other`) + COUNT(area_other) + FIRST(area_other) '
    '+ LAST(area_other) + NUNIQUE(area_other) + NUNIQUE(n_spots_other) '
    '+ MAX(flag_other) + MEAN_OTHER(n_spots)'
)
SOURCES = [
    ('tree', tp.SOURCE_SAME_FRAME), ('tree', tp.SOURCE_MOTHER),
    ('tree', tp.SOURCE_SISTER), ('tree', tp.SOURCE_DAUGHTER),
    ('budding', tp.SOURCE_MOTHER), ('budding', tp.SOURCE_DAUGHTER),
]
CONDITIONS = [
    ('', 'where_true'), ('area > this_area', 'where_true'),
    ('area > this_area', 'any'), ('`GFP mean` > 0.5', 'all'),
]


@pytest.mark.parametrize('kind,source', SOURCES)
@pytest.mark.parametrize('condition,mode', CONDITIONS)
@pytest.mark.parametrize('seed,frames', [(0, None), (1, ('t - 2', 't + 1'))])
def test_block_emission_matches_per_target_emission(
        monkeypatch, kind, source, condition, mode, seed, frames
    ):
    df = random_table(seed, kind)
    _, start, end = (
        ('', 't', 't') if source == tp.SOURCE_SAME_FRAME
        else tp.CellFilter.other_presets(df)[source]
    )
    start, end = frames or (start, end)
    settings = dict(
        tp.CellFilter.default_other(), source=source, condition=condition,
        condition_mode=mode, range_start=start, range_end=end
    )
    equations = (
        AGGREGATES, 'area > MEAN(area_other) + `GFP mean_other`',
        'flag_other', '`GFP mean_other` + n_spots_other',
    )
    for equation in equations:
        (actual, actualWarnings), (expected, expectedWarnings) = prepare_both(
            monkeypatch, df, equation, settings, 7, return_alternatives=True
        )
        pd.testing.assert_frame_equal(actual[0], expected[0], check_exact=True)
        assert actual[1] == expected[1]
        assert_alternatives_equal(actual[2], expected[2])
        assert actualWarnings == expectedWarnings


def test_block_emission_is_used_and_skips_negative_zero(monkeypatch):
    df = random_table(3)
    function, description = tp.CELL_FUNCTIONS['MIN']
    perTarget = Mock(wraps=function)
    monkeypatch.setitem(tp.CELL_FUNCTIONS, 'MIN', (perTarget, description))
    settings = dict(tp.CellFilter.default_other(), source=tp.SOURCE_SAME_FRAME)
    tp.OtherCellMeasurements.prepare_inputs(
        df, 'MIN(area_other) > 1', settings, 7
    )
    perTarget.assert_not_called()  # whole blocks of targets at once
    # -0.0 == 0.0: which one np.unique/MIN keep depends on the algorithm,
    # so such columns keep the per-target path, with the same results.
    df['area'] = np.where(np.arange(len(df)) % 2, -0.0, 0.0)
    (actual, _), (expected, _) = prepare_both(
        monkeypatch, df, 'MIN(area_other) + area_other', settings, 7,
        return_alternatives=True
    )
    pd.testing.assert_frame_equal(actual[0], expected[0], check_exact=True)
    assert_alternatives_equal(actual[2], expected[2])
    assert perTarget.called


def test_block_reductions_match_numpy_per_row():
    # emit_block reduces blocks of targets with the same number of finite
    # values row-wise; NumPy must give each row its 1D result bit for bit.
    rng = np.random.default_rng(0)
    for size in (1, 2, 3, 8, 9, 129, 1000, 2001):
        block = rng.normal(size=(5, size)) * 1e6 + rng.integers(-3, 3, (5, size))
        for func in set(tp._NAN_AGGREGATES.values()):
            rows = func(block, axis=1)
            for row, value in zip(block, rows):
                assert value.tobytes() == func(row.copy()).tobytes()


def predict_both(monkeypatch, predict, *args, **kwargs):
    """Results (or error messages) with and without the all-cells path."""
    batch = Mock(wraps=tp.TimepointPredictor._evaluate_all_cells)
    results = []
    for fast in (True, False):
        with monkeypatch.context() as patch:
            patch.setattr(
                tp.TimepointPredictor, '_evaluate_all_cells',
                batch if fast else staticmethod(lambda *a, **k: None)
            )
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    results.append(predict(*args, **kwargs))
                except Exception as err:
                    results.append((type(err), str(err)))
    return results, batch


EQUATIONS = [
    ('area > 1', None),
    ('`GFP mean` * 2 - area', None),
    ('(area > 0.8) & flag', None),
    ('stage == "S"', None),
    ('SizeT > 3', None),
    ('n_spots', None),
    ('area > area_other', tp.SOURCE_MOTHER),
    ('area > area_other', tp.SOURCE_SAME_FRAME),
    ('area_other > `GFP mean`', tp.SOURCE_DAUGHTER),
    ('area / area_other', tp.SOURCE_MOTHER),
    ('area - area_other', tp.SOURCE_SAME_FRAME),
    ('area - `GFP mean_other`', tp.SOURCE_SISTER),
    ('abs(area - area_other) < 0.3', tp.SOURCE_MOTHER),
    ('area > MEAN(area_other)', tp.SOURCE_SAME_FRAME),
    ('MAX(area_other) - area', tp.SOURCE_DAUGHTER),
    ('area > MAX(area)', None),  # per-cell function: per-cell path
    ('area_other', tp.SOURCE_SISTER),
    ('stage in ["S", "G2"]', None),
    ('stage not in ("S",)', None),
    # Column on the right: isin over the evaluated rows, per-cell path.
    ('n_spots in Cell_ID', None),
    ('n_spots not in Cell_ID', None),
]


@pytest.mark.parametrize('seed', [0, 4])
@pytest.mark.parametrize('equation,source', EQUATIONS)
@pytest.mark.parametrize('mode', list(tp.EQUATION_MODES))
def test_timepoint_all_cells_path_matches_per_cell(
        monkeypatch, seed, equation, source, mode
    ):
    df = random_table(seed)
    otherCells = None
    if source is not None:
        otherCells = dict(tp.CellFilter.default_other(), source=source)
        if source != tp.SOURCE_SAME_FRAME:
            _, start, end = tp.CellFilter.other_presets(df)[source]
            otherCells.update(range_start=start, range_end=end)
    (actual, expected), batch = predict_both(
        monkeypatch, tp.TimepointPredictor.from_equation, df, equation, mode,
        SizeT=7, IDs=[1, 2, 4, 5, 7, 8, 9, 11, 12], other_cells=otherCells
    )
    assert actual == expected
    assert batch.called


@pytest.mark.parametrize('equation,source', [
    ('area > 1', None),
    ('flag', None),
    ('stage == "S"', None),
    ('SizeT > 3', None),
    ('area > area_other', tp.SOURCE_MOTHER),
    ('area_other > `GFP mean`', tp.SOURCE_SAME_FRAME),
    ('(area > MEAN(area_other)) | flag', tp.SOURCE_DAUGHTER),
    ('area > MAX(area) / 2', None),
    # Invalid results: the per-cell path reports the Cell_ID.
    ('area', None),
    ('area_other', tp.SOURCE_MOTHER),
])
@pytest.mark.parametrize('consolidation', ['all', 'none', 'most'])
def test_annotation_all_cells_path_matches_per_cell(
        monkeypatch, equation, source, consolidation
    ):
    df = random_table(5)
    otherCells = None
    if source is not None:
        otherCells = dict(tp.CellFilter.default_other(), source=source)
    (actual, expected), _ = predict_both(
        monkeypatch, CustomAnnotationPredictor.from_equation, df, equation,
        SizeT=7, other_cells=otherCells, consolidation=consolidation
    )
    assert actual == expected


def test_all_cells_path_keeps_dtype_sensitive_alternatives_per_cell(monkeypatch):
    # Cells without a mother get the int 0 of an empty candidate set; on
    # their own it is int64, while all cells together make it float64.
    # ``0 ** -1`` fails for int64 but gives inf for float64, so the
    # per-cell path is used. Plain comparisons (0 and 0.0 compare equally)
    # are batched.
    df = random_table(6)
    otherCells = dict(tp.CellFilter.default_other(), source=tp.SOURCE_MOTHER)
    _, start, end = tp.CellFilter.other_presets(df)[tp.SOURCE_MOTHER]
    otherCells.update(range_start=start, range_end=end)
    for equation, batched in (
            ('area_other ** -1', False), ('area > area_other', True)):
        prepared, prepared_equation, alternatives = (
            tp.OtherCellMeasurements.prepare_inputs(
                df, equation, otherCells, 7, return_alternatives=True
            )
        )
        batch = tp.TimepointPredictor._evaluate_all_cells(
            prepared, prepared_equation, alternatives, {'SizeT': 7},
            combine='max'
        )
        assert (batch is not None) == batched
        (actual, expected), _ = predict_both(
            monkeypatch, tp.TimepointPredictor.from_equation, df, equation,
            'max', SizeT=7, other_cells=otherCells
        )
        assert actual == expected
    assert isinstance(expected, dict)
    assert actual == expected


def test_column_membership_is_not_row_wise(monkeypatch):
    # pandas evaluates ``a in b`` with a column b as a.isin(b): on all cells
    # together the result would depend on the other cells.
    for equation, rowwise in (
            ('Cell_ID in relative_ID', False),
            ('Cell_ID not in relative_ID', False),
            ('(area > 1) & (Cell_ID in `relative ID`)', False),
            ('Cell_ID in [1, 2]', True),
            ('Cell_ID not in (1, -2)', True),
            ('stage in {"S", "G2"}', True)):
        assert tp.CellEquation.is_row_wise(equation) is rowwise
        if not rowwise:
            # (Other-cell conditions with literal containers keep the
            # general path too: they only batch plain comparisons.)
            assert not tp.OtherCellMeasurements._is_rowwise_condition(
                equation)
    df = pd.DataFrame({
        'frame_i': [0, 1, 0, 1], 'Cell_ID': [1, 1, 2, 2],
        'relative_ID': [2, 2, 1, 1],
    }).set_index(['frame_i', 'Cell_ID'])
    for equation, expected in (
            ('Cell_ID in relative_ID', {1: [], 2: []}),
            ('Cell_ID not in relative_ID', {1: [0], 2: [0]})):
        (actual, reference), _ = predict_both(
            monkeypatch, tp.TimepointPredictor.from_equation, df, equation,
            'rising_true'
        )
        assert actual == reference == expected


def test_other_cell_condition_with_column_membership():
    # Per target, this_relative_ID holds one value, so ``in`` behaves like
    # ``==``; batching the targets must not turn it into isin over all
    # targets' values.
    df = random_table(2, kind='budding')
    results = {}
    for condition in (
            'Cell_ID in this_relative_ID', 'Cell_ID == this_relative_ID',
            'Cell_ID not in this_relative_ID', 'Cell_ID != this_relative_ID'):
        otherCells = dict(
            tp.CellFilter.default_other(), source=tp.SOURCE_SAME_FRAME,
            condition=condition
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            prepared, equation = tp.OtherCellMeasurements.prepare_inputs(
                df, 'COUNT(area_other) + MEAN(area_other)', otherCells, 7
            )
        results[condition] = prepared.eval(equation)
    pd.testing.assert_series_equal(
        results['Cell_ID in this_relative_ID'],
        results['Cell_ID == this_relative_ID']
    )
    pd.testing.assert_series_equal(
        results['Cell_ID not in this_relative_ID'],
        results['Cell_ID != this_relative_ID']
    )


@pytest.mark.parametrize('index', [True, False])
def test_sorted_cells_match_iter_cells(index):
    df = tp.CellTable.flatten(random_table(7))
    df.loc[df.index[:3], 'Cell_ID'] = np.nan  # rows without Cell_ID
    df = pd.concat([df, df.iloc[5:8]])  # duplicated rows
    if index:
        df = df.dropna(subset=['Cell_ID']).set_index(['frame_i', 'Cell_ID'])
    table, cells = tp.CellTable.sorted_cells(df)
    expected = list(tp.CellTable.iter_cells(df))
    assert [ID for ID, _, _ in cells] == [ID for ID, _ in expected]
    for (_, start, stop), (_, df_ID) in zip(cells, expected):
        pd.testing.assert_frame_equal(
            table.iloc[start:stop].reset_index(drop=True), df_ID
        )


def nullable_bool_table():
    """Two cells over frames 0-4 with a nullable boolean column holding NA
    (e.g. ``is_history_known`` of partially annotated cell cycles)."""
    df = pd.DataFrame({
        'frame_i': [0, 1, 2, 3, 4] * 2,
        'Cell_ID': [1] * 5 + [2] * 5,
        'area': [5.0, 6, 7, 8, 9, 5, 6, 7, 8, 9],
        'known': pd.array(
            [pd.NA, True, pd.NA, True, False,
             pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
            dtype='boolean'
        ),
    })
    return df.set_index(['frame_i', 'Cell_ID'])


@pytest.mark.parametrize('equation,rowWise', [
    ('known', True),
    ('known & (area > 1)', True),
    # Per-cell function: only the per-cell path applies.
    ('known & (area >= MIN(area))', False),
])
@pytest.mark.parametrize('mode,expected', [
    ('first_true', {1: [1], 2: []}),
    ('last_true', {1: [3], 2: []}),
    ('rising_true', {1: [1, 3], 2: []}),
])
def test_missing_boolean_results_count_as_false(
        monkeypatch, equation, rowWise, mode, expected
    ):
    (actual, perCell), batch = predict_both(
        monkeypatch, tp.TimepointPredictor.from_equation,
        nullable_bool_table(), equation, mode, SizeT=5
    )
    assert actual == expected
    assert perCell == expected
    assert batch.called
    # The row-wise equations really take the all-cells path.
    batchResult = tp.TimepointPredictor._evaluate_all_cells(
        nullable_bool_table(), equation, None, None, combine=mode
    )
    assert (batchResult is not None) == rowWise


@pytest.mark.parametrize('values', [
    pd.array([pd.NA, True, False, True], dtype='boolean').to_numpy(),
    np.array([np.nan, 1.0, 0.0, 1.0]),
    np.array([None, True, False, True], dtype=object),
])
def test_frames_from_values_treats_missing_as_false(values):
    frames = np.arange(4)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert tp.TimepointPredictor._frames_from_values(
            values, frames, 'first_true'
        ) == [1]
        assert tp.TimepointPredictor._frames_from_values(
            values, frames, 'rising_true'
        ) == [1, 3]


def spy_eval_columns(monkeypatch):
    """Record the columns of each frame DataFrame.eval receives."""
    seen = []
    original = pd.DataFrame.eval

    def spy(self, *args, **kwargs):
        seen.append(list(self.columns))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, 'eval', spy)
    return seen


def test_quoted_equation_narrows_columns_like_unquoted(monkeypatch):
    rng = np.random.default_rng(0)
    values = rng.random(20)
    wide = {f'other_{i}': rng.random(20) for i in range(30)}
    quotedDf = pd.DataFrame({'mKO2-hCdt1_fucci': values, **wide})
    plainDf = pd.DataFrame({'mCherry_fucci': values, **wide})
    seen = spy_eval_columns(monkeypatch)

    quoted = tp.CellEquation.evaluate(
        quotedDf, '`mKO2-hCdt1_fucci` > 0.5*MAX(`mKO2-hCdt1_fucci`)'
    )
    plain = tp.CellEquation.evaluate(
        plainDf, 'mCherry_fucci > 0.5*MAX(mCherry_fucci)'
    )
    np.testing.assert_array_equal(np.asarray(quoted), np.asarray(plain))
    np.testing.assert_array_equal(
        np.asarray(quoted), values > 0.5*values.max()
    )
    assert seen == [['mKO2-hCdt1_fucci'], ['mCherry_fucci']]


def test_quoted_equation_keeps_all_columns_if_tokenizing_fails(monkeypatch):
    df = pd.DataFrame({
        "a'b": [1.0, 2.0], 'GFP mean': [1.0, 2.0], 'area': [3.0, 4.0],
        'other': [0.0, 0.0],
    })
    # The apostrophe in the quoted name stops tokenizing after ``area``, so
    # CellEquation.names misses both quoted names: the whole table is passed
    # and pandas reports the same error as without narrowing.
    equation = "area + `a'b` + `GFP mean`"
    assert tp.CellEquation.names(equation) == {'area'}
    with pytest.raises(Exception) as expected:
        df.eval(equation)
    seen = spy_eval_columns(monkeypatch)
    with pytest.raises(type(expected.value)):
        tp.CellEquation.evaluate(df, equation)
    assert seen == [list(df.columns)]
