import ast
import threading
import warnings
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from cellacdc import timepoint_predictors as tp
from cellacdc._equations import equation_variable_spans


@pytest.fixture(scope='module')
def qapp():
    # Module-scoped so the QApplication exists (and stays alive) before the
    # function-scoped dialog watchdog of conftest.py creates its QTimer
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_timepoint_debug_disabled_is_silent(monkeypatch, capsys):
    monkeypatch.setattr(tp, 'DEBUG', False)
    monkeypatch.delenv('CELLACDC_TIMEPOINT_DEBUG', raising=False)

    @tp.debug_trace
    def operation():
        return 42

    assert operation() == 42
    assert capsys.readouterr().err == ''


def test_timepoint_debug_dumps_blocked_stack_and_stops(monkeypatch, capsys):
    monkeypatch.setenv('CELLACDC_TIMEPOINT_DEBUG', '1')
    monkeypatch.setattr(tp, 'DEBUG_STACK_INTERVAL', 0.01)
    dumped = threading.Event()
    print_stack = tp.traceback.print_stack

    def record_stack(*args, **kwargs):
        print_stack(*args, **kwargs)
        dumped.set()

    monkeypatch.setattr(tp.traceback, 'print_stack', record_stack)

    @tp.debug_trace
    def blocked_operation():
        assert dumped.wait(5), 'Debug watchdog did not run'
        return 42

    assert blocked_operation() == 42
    output = capsys.readouterr().err
    assert 'BEGIN' in output
    assert 'STILL RUNNING' in output
    assert 'blocked_operation' in output
    assert 'elapsed=' in output
    assert 'END' in output
    assert not any(
        thread.name == 'timepoint-debug-watchdog'
        for thread in threading.enumerate()
    )


@pytest.mark.parametrize('debug', [False, True])
def test_timepoint_debug_trace_accepts_qt_clicked_slot(
        qapp, monkeypatch, debug
    ):
    monkeypatch.setattr(tp, 'DEBUG', debug)
    monkeypatch.delenv('CELLACDC_TIMEPOINT_DEBUG', raising=False)
    from qtpy.QtWidgets import QPushButton

    calls = []

    class Target:
        @tp.debug_trace
        def test_cb(self):
            calls.append(self)

    target = Target()
    button = QPushButton()
    button.clicked.connect(target.test_cb)
    button.click()
    target.test_cb(True)
    assert calls == [target, target]
    button.deleteLater()
    qapp.processEvents()


def test_evaluate_passes_only_referenced_columns_to_eval(monkeypatch):
    df = pd.DataFrame({
        'frame_i': [0, 1], 'Cell_ID': [1, 1], 'area': [1., 3.],
        **{f'unused_{i}': [0., 0.] for i in range(50)},
    }).set_index(['frame_i', 'Cell_ID'])
    seen = []
    evaluate = pd.DataFrame.eval

    def record(self, *args, **kwargs):
        seen.append(list(self.columns))
        return evaluate(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, 'eval', record)
    values = tp.CellEquation.evaluate(df, 'area > MEAN(area) + frame_i')
    assert seen == [['area']]
    assert values.tolist() == [False, False]
    assert tp.CellEquation.evaluate(df, '`area` > 2').tolist() == [False, True]
    assert len(seen[-1]) == 51


def test_timepoint_debug_logs_failure_and_preserves_exception(monkeypatch, capsys):
    monkeypatch.setattr(tp, 'DEBUG', True)

    @tp.debug_trace
    def failing_operation():
        raise ValueError('diagnostic test failure')

    with pytest.raises(ValueError, match='diagnostic test failure'):
        failing_operation()
    assert 'FAILED' in capsys.readouterr().err
    assert tp._debug_state.depth == 0
    assert not any(
        thread.name == 'timepoint-debug-watchdog'
        for thread in threading.enumerate()
    )


@pytest.fixture
def measurements():
    df = pd.DataFrame({
        'frame_i': np.repeat([0, 1, 3, 4], 3),
        'Cell_ID': np.tile([1, 2, 3], 4),
        'area': [10., 4., 7., 8., np.nan, 12., 15., 6., 3., 9., 20., 11.],
        'stage': ['G1', 'S', 'G1'] * 4,
    })
    return df.sample(frac=1, random_state=42).set_index(['frame_i', 'Cell_ID'])


@pytest.mark.parametrize('values', [
    np.array([], dtype=float),
    np.array([True, False]),
    np.array([0, 2**32 - 1], dtype=np.uint32),
    np.array([0, 2**64 - 1], dtype=np.uint64),
    np.array([-3, 0, 12], dtype=np.int64),
    np.array([1., np.nan, np.inf, -np.inf], dtype=np.float32),
    np.array([[1., 2.], [3., 4.]]),
    np.array(['2', 'invalid', '3.5']),
    np.array([1, None, pd.NA, '4'], dtype=object),
    pd.Series([1, pd.NA, 3], dtype='Int64'),
    3.5,
])
def test_numeric_conversion_matches_pandas(values):
    expected = pd.to_numeric(
        pd.Series(np.atleast_1d(np.asarray(values)).ravel()), errors='coerce'
    ).to_numpy(float)
    result = tp.CellEquation._numeric(values)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.dtype(float)


def test_numeric_arrays_skip_pandas_conversion(monkeypatch):
    convert = Mock(wraps=pd.to_numeric)
    monkeypatch.setattr(pd, 'to_numeric', convert)
    result = tp.CellEquation._numeric(np.array([1., np.nan, 3.]))
    np.testing.assert_array_equal(result, [1., np.nan, 3.])
    convert.assert_not_called()


def full_scan_rows(df, target, settings, sizeT):
    """Reference the original full-table window scan and candidate filtering."""
    lower, upper = tp.FrameRange.bounds(
        settings['range_start'], settings['range_end'], int(target['frame_i']),
        sizeT
    )
    candidates = df[
        df['frame_i'].between(lower, upper)
        & (df['Cell_ID'] != target['Cell_ID'])
    ].sort_values(['frame_i', 'Cell_ID'])
    selected = []
    for _, rows in candidates.groupby('Cell_ID', sort=False):
        evaluation = rows.reset_index(drop=True).copy()
        for column in df:
            evaluation[f'this_{column}'] = target[column]
        mask = (
            tp.CellFilter.boolean_values(tp.CellEquation.evaluate(
                evaluation, settings['condition'], {'SizeT': sizeT},
                enforce_dtype_output=bool
            ), len(rows))
            if settings['condition'] else np.ones(len(rows), dtype=bool)
        )
        mode = settings['condition_mode']
        if mode == 'where_true':
            selected.append(rows.loc[mask])
        elif (mode == 'any' and mask.any()) or (mode == 'all' and mask.all()):
            selected.append(rows)
    return (
        pd.concat(selected).sort_values(['frame_i', 'Cell_ID'])
        if selected else candidates.iloc[:0]
    )


@pytest.mark.parametrize('mode', ['where_true', 'any', 'all'])
@pytest.mark.parametrize('condition', [
    '', 'area > this_area', 'area > MEAN(area)',
    '(DIFF(area) > 0) | (area > this_area)',
])
@pytest.mark.parametrize('start,end', [
    ('t - 2', 't + 1'), ('t - 8', 't - 6'), ('t + 10', 't + 20'),
])
def test_indexed_windows_match_full_scan(measurements, mode, condition, start, end):
    settings = dict(
        tp.CellFilter.default_other(), condition=condition,
        condition_mode=mode, range_start=start, range_end=end
    )
    original = measurements.copy(deep=True)
    equation = (
        'MEAN(area_other) + MAX_OTHER(area) + FIRST(area_other) '
        '+ LAST(area_other) + COUNT(area_other) + NUNIQUE(stage_other) '
        '+ MEAN(area_other)'
    )
    prepared, rewritten = tp.OtherCellMeasurements.prepare_inputs(
        measurements, equation, settings, 4, IDs=iter([1, 3])
    )
    targets = measurements[measurements.index.get_level_values('Cell_ID').isin([1, 3])]
    assert prepared.index.equals(targets.index)
    assert len(prepared.columns) == len(targets.columns) + 6
    flat = tp.CellTable.flatten(measurements)
    rows = [
        full_scan_rows(flat, target, settings, 4)
        for _, target in tp.CellTable.flatten(targets).iterrows()
    ]
    for function, column in (
            ('MEAN', 'area'), ('MAX', 'area'), ('FIRST', 'area'),
            ('LAST', 'area'), ('COUNT', 'area'), ('NUNIQUE', 'stage')
        ):
        name, = [
            name for name in prepared
            if name.startswith(f'_acdc_other_{function.lower()}_')
        ]
        aggregate = tp.CELL_FUNCTIONS[function][0]
        expected = [
            0 if selected.empty else aggregate(selected[column])
            for selected in rows
        ]
        np.testing.assert_allclose(prepared[name].to_numpy(), expected, equal_nan=True)
    assert rewritten.count('_acdc_other_mean_') == 2
    pd.testing.assert_frame_equal(measurements, original)


@pytest.mark.parametrize('flat', [False, True])
@pytest.mark.parametrize('eligible_ID,expected,count', [
    (1, {1: [4]}, 4), (99, {}, 0),
])
def test_only_eligible_targets_are_aggregated(
        measurements, monkeypatch, flat, eligible_ID, expected, count
    ):
    if flat:
        measurements = measurements.reset_index()
        measurements.index = np.arange(len(measurements)) * 7
    prepare = Mock(wraps=tp.OtherCellMeasurements.prepare_inputs)
    monkeypatch.setattr(tp.OtherCellMeasurements, 'prepare_inputs', prepare)
    result = tp.TimepointPredictor.from_equation(
        measurements, 'MEAN(area_other)', 'max',
        SizeT=5, IDs=iter([1, 2]),
        main_cell_filter={
            'condition': f'Cell_ID == {eligible_ID}', 'condition_mode': 'all'
        },
        other_cells=tp.CellFilter.default_other()
    )
    assert result == expected
    targetIDs = list(prepare.call_args.kwargs['IDs'])
    assert targetIDs == [1] * (count > 0)
    prepared, _ = tp.OtherCellMeasurements.prepare_inputs(
        measurements, 'MEAN(area_other)', tp.CellFilter.default_other(),
        5, IDs=targetIDs
    )
    assert len(prepared) == count


@pytest.mark.parametrize('equation', ['area', 'MEAN(area_other)'])
def test_prepared_flat_target_rows_preserve_index(measurements, equation):
    flat = measurements.reset_index()
    flat.index = np.arange(len(flat)) * 7
    original = flat.copy(deep=True)
    prepared, _ = tp.OtherCellMeasurements.prepare_inputs(
        flat, equation, tp.CellFilter.default_other(), 5, IDs=iter([1])
    )
    pd.testing.assert_frame_equal(prepared[flat.columns], flat[flat['Cell_ID'] == 1])
    pd.testing.assert_frame_equal(flat, original)


def test_window_setup_is_reused_but_target_conditions_are_not(
        measurements, monkeypatch
    ):
    bounds = Mock(wraps=tp.FrameRange.bounds)
    columns = Mock(wraps=tp.CellFilter._this_columns)
    monkeypatch.setattr(tp.FrameRange, 'bounds', bounds)
    monkeypatch.setattr(tp.CellFilter, '_this_columns', columns)
    settings = dict(tp.CellFilter.default_other(), condition='area > this_area')
    prepared, _ = tp.OtherCellMeasurements.prepare_inputs(
        measurements, 'COUNT(area_other)', settings, 5
    )
    # One validation at t=0, then one lookup for each distinct target frame.
    assert bounds.call_count == 5
    columns.assert_called_once()
    assert prepared.loc[(0, 1), '_acdc_other_count_0'] == 0
    assert prepared.loc[(0, 2), '_acdc_other_count_0'] == 2
    assert prepared.loc[(0, 3), '_acdc_other_count_0'] == 1


def test_frame_bound_syntax_is_parsed_once_per_expression(measurements, monkeypatch):
    tp.FrameRange._parse.cache_clear()
    parse = Mock(wraps=ast.parse)
    monkeypatch.setattr(ast, 'parse', parse)
    settings = dict(
        tp.CellFilter.default_other(), range_start='t - 2', range_end='t + 1'
    )
    try:
        tp.OtherCellMeasurements.prepare_inputs(
            measurements, 'MEAN(area_other)', settings, 5
        )
        assert parse.call_count == 2
    finally:
        tp.FrameRange._parse.cache_clear()


def test_cached_frame_expression_still_evaluates_current_frame():
    assert tp.FrameRange.value('10 // t', 2) == 5
    assert tp.FrameRange.value('10 // t', 5) == 2
    with pytest.raises(ValueError, match='Invalid frame range expression'):
        tp.FrameRange.value('10 // t', 0)


@pytest.mark.parametrize('start,end', [
    ('t + 1', 't + 1'), ('t - 2', 't + 2'), ('0', '4'),
    ('4 - t', '5 - t'), ('t % 3', 't % 3 + 1'),
    ('(t * 3) // 2', '(t * 3) // 2 + 2'), ('-5', '-1'),
    ('t + 100000000000000000000', 't + 100000000000000000001'),
])
@pytest.mark.parametrize('sizeT', [None, 0, 5])
def test_array_frame_bounds_match_scalar(start, end, sizeT):
    frames = np.array([4, 0, 1, 3, 1])
    lower, upper = tp.FrameRange.array_bounds(start, end, frames, sizeT)
    expected = [tp.FrameRange.bounds(start, end, int(t), sizeT) for t in frames]
    np.testing.assert_array_equal(lower, [pair[0] for pair in expected])
    np.testing.assert_array_equal(upper, [pair[1] for pair in expected])


@pytest.mark.parametrize('start,end,sizeT,match', [
    ('t', '3 - t', 5, r'start \(3\) exceeds end \(0\) at t=3'),
    ('0', '10 // (t - 1)', 5, 'Invalid frame range expression'),
    ('0', 't', -1, 'Video frame count cannot be negative'),
    ('0', 't / 2', 5, 'Use integers'),
])
def test_array_frame_bounds_preserve_errors(start, end, sizeT, match):
    with pytest.raises(ValueError, match=match):
        tp.FrameRange.array_bounds(start, end, np.array([0, 1, 3]), sizeT)


@pytest.mark.parametrize('equation,expected', [
    (None, set()),
    ('MAX(area) > this_area', {'area', 'this_area'}),
    ('stage == "this_area"', {'stage'}),
    ('stage == "escaped \\"this_area\\""', {'stage'}),
    ('stage == """this_area\nthis_stage"""', {'stage'}),
    ('`this_cell area` > `cell area`', {'this_cell area', 'cell area'}),
    ('area > 0 # this_not_an_input', {'area'}),
    ('(area > 0) and True', {'area'}),
    ('MEAN(area', {'area'}),
    ('area + `unfinished', {'area'}),
])
def test_equation_names_use_shared_tokens(equation, expected):
    assert tp.CellEquation.names(equation) == expected


def test_shared_tokens_preserve_replacement_spans():
    equation = 'MAX(`cell area`) > this_area and stage == "area"'
    spans = equation_variable_spans(equation)
    assert [name for name, _, _ in spans] == ['cell area', 'this_area', 'stage']
    assert [equation[start:end] for _, start, end in spans] == [
        '`cell area`', 'this_area', 'stage'
    ]


def test_measurement_replacement_uses_shared_tokens():
    from cellacdc import measurements

    assert measurements.equation_variable_spans is equation_variable_spans
    equation = 'MAX(`cell area`) > this_area and stage == "cell area"'
    assert measurements.replace_equation_variables(
        equation, {'cell area': 'new area', 'this_area': 'this_new_area'}
    ) == 'MAX(`new area`) > this_new_area and stage == "cell area"'


@pytest.mark.parametrize('values,expected', [
    ([], 0), ([np.nan], 0), ([5.], 5.), ([5., 5.], 5.),
    ([np.nan, 5.], 5.),
])
def test_direct_other_input_collapses_single_or_constant_values(values, expected):
    df = pd.DataFrame({
        'frame_i': [0] * (len(values) + 1),
        'Cell_ID': list(range(1, len(values) + 2)),
        'area': [10.] + values,
    }).set_index(['frame_i', 'Cell_ID'])
    prepared, equation = tp.OtherCellMeasurements.prepare_inputs(
        df, 'area_other', tp.CellFilter.default_other(), 1, IDs=[1]
    )
    assert tp.CellEquation.evaluate(
        tp.CellTable.flatten(prepared), equation
    ).tolist() == [expected]


def test_direct_other_input_warns_and_compares_each_unique_value():
    df = pd.DataFrame({
        'frame_i': [0, 0, 0, 0], 'Cell_ID': [1, 2, 3, 4],
        'area': [10., 4., 12., 15.]
    })
    with pytest.warns(
            UserWarning, match='area_other.*multiple.*candidate row'
        ) as caught:
        result = tp.TimepointPredictor.from_equation(
            df, 'area_other > area', 'first_true', SizeT=1, IDs=[1, 2],
            other_cells=tp.CellFilter.default_other()
        )
    assert result == {1: [0], 2: [0]}
    assert len(caught) == 1


def test_direct_other_input_comparisons_evaluate_unique_values_together(
        monkeypatch
    ):
    df = pd.DataFrame({
        'frame_i': [0], 'Cell_ID': [1], 'area': [3.], 'area_other': [0.]
    })
    alternatives = {
        'columns': ('area_other',),
        'by_target': {(0, 1): [(1.,), (2.,), (4.,)]},
    }
    evaluate = Mock(wraps=tp.CellEquation.evaluate)
    monkeypatch.setattr(tp.CellEquation, 'evaluate', evaluate)

    result = tp.OtherCellMeasurements.evaluate_with_alternatives(
        df, 'area_other > area', alternatives
    )

    assert result.tolist() == [True]
    evaluate.assert_called_once()


@pytest.mark.parametrize('equation', [
    'area_other > MEAN(area)',
    'area_other + SUM(area)',
    'area_other - SHIFT(area)',
    'area_other + DIFF(area)',
    'area_other + SHIFT(area, -1)',
    'area_other + MEAN(SHIFT(area))',
    'MEAN(area) + MEAN(area)',
    '(area_other > MEAN(area)) & (stage_other == "S")',
])
@pytest.mark.parametrize('combine', ['any', 'max', 'min'])
@pytest.mark.parametrize('indexed', [False, True])
def test_alternatives_preserve_original_function_timeline(equation, combine, indexed):
    df = pd.DataFrame({
        'frame_i': [0, 1, 3, 7], 'Cell_ID': [1] * 4,
        'area': [2., 8., np.nan, 4.], 'area_other': [0., 1., 2., 3.],
        'stage_other': ['G1', 'S', 'G1', 'S'],
    }, index=[8, 3, 8, 2])
    if indexed:
        df = df.set_index(['frame_i', 'Cell_ID'])
    original = df.copy(deep=True)
    alternatives = {
        'columns': ('area_other', 'stage_other'),
        'by_target': {
            (0, 1): [(1., 'S'), (10., 'G1')],
            (1, 1): [],
            (3, 1): [(3., 'S'), (np.nan, 'G1'), (20., 'S')],
        },
    }
    expected = []
    for position, t in enumerate([0, 1, 3, 7]):
        options = alternatives['by_target'].get(
            (t, 1), [(df['area_other'].iloc[position], df['stage_other'].iloc[position])]
        )
        evaluated = []
        for option in options:
            variant = df.copy()
            for column, value in zip(alternatives['columns'], option):
                variant.iloc[position, variant.columns.get_loc(column)] = value
            values = np.asarray(tp.CellEquation.evaluate(variant, equation))
            evaluated.append(values.item() if values.ndim == 0 else values[position])
        series = pd.Series(evaluated, dtype=object)
        if combine == 'any':
            expected.append(bool(series.dropna().astype(bool).any()))
        else:
            expected.append(pd.to_numeric(series, errors='coerce').agg(combine))
    result = tp.OtherCellMeasurements.evaluate_with_alternatives(
        df, equation, alternatives, combine=combine
    )
    pd.testing.assert_series_equal(result, pd.Series(expected, index=df.index))
    pd.testing.assert_frame_equal(df, original)


def test_alternatives_cache_repeated_calls_once_and_preserve_name_collisions(monkeypatch):
    df = pd.DataFrame({
        'frame_i': np.arange(100), 'Cell_ID': 1,
        'area': np.arange(100, dtype=float), 'area_other': 0.,
        '_acdc_cached_call_0': 10.,
    })
    alternatives = {
        'columns': ('area_other',),
        'by_target': {(t, 1): [(0.,), (100.,)] for t in range(100)},
    }
    original = df.copy(deep=True)
    evaluate = Mock(wraps=tp.CellEquation.evaluate)
    monkeypatch.setattr(tp.CellEquation, 'evaluate', evaluate)
    result = tp.OtherCellMeasurements.evaluate_with_alternatives(
        df, 'area_other > MEAN(area) + MEAN(area) + _acdc_cached_call_0 - offset',
        alternatives, variables={'offset': 20.}, enforce_dtype_output=bool
    )
    assert result.tolist() == [True] * 100
    assert evaluate.call_count == 2  # One unique call, then all comparisons.
    assert len(evaluate.call_args_list[0].args[0]) == 100
    assert len(evaluate.call_args_list[1].args[0]) == 200
    pd.testing.assert_frame_equal(df, original)


@pytest.mark.parametrize('combine,expected', [('any', False), ('max', np.nan), ('min', np.nan)])
def test_function_alternatives_all_empty(combine, expected):
    df = pd.DataFrame({'frame_i': [0], 'Cell_ID': [1], 'area': [2.], 'area_other': [0.]})
    result = tp.OtherCellMeasurements.evaluate_with_alternatives(
        df, 'area_other + SUM(area)',
        {'columns': ('area_other',), 'by_target': {(0, 1): []}},
        combine=combine
    )
    pd.testing.assert_series_equal(result, pd.Series([expected]))


def test_direct_function_alternatives_keep_expanded_scope():
    df = pd.DataFrame({
        'frame_i': [0, 1], 'Cell_ID': [1, 1], 'area_other': [0., 0.],
    })
    alternatives = {
        'columns': ('area_other',),
        'by_target': {(0, 1): [(1.,), (3.,)], (1, 1): [(5.,)]},
    }
    result = tp.OtherCellMeasurements.evaluate_with_alternatives(
        df, 'area_other + SUM(area_other)', alternatives, combine='max'
    )
    assert result.tolist() == [12., 14.]


def test_direct_other_alternatives_are_returned_not_stored_in_attrs():
    df = pd.DataFrame({
        'frame_i': [0, 0, 0], 'Cell_ID': [1, 2, 3], 'area': [10., 4., 12.]
    })
    with pytest.warns(UserWarning):
        prepared, equation, alternatives = (
            tp.OtherCellMeasurements.prepare_inputs(
                df, 'area_other > area', tp.CellFilter.default_other(), 1,
                return_alternatives=True
            )
        )
    # pandas deep-copies attrs on every derived frame; a per-row payload there
    # made each copy/reset_index cost hundreds of ms on large tables.
    assert prepared.attrs == {}
    assert alternatives['columns'] == ('area_other',)
    assert alternatives['by_target'][(0, 1)] == [(4.,), (12.,)]
    values = tp.OtherCellMeasurements.evaluate_with_alternatives(
        prepared, equation, alternatives
    )
    assert values.tolist() == [True, True, False]
    assert tp.OtherCellMeasurements.prepare_inputs(
        df, 'area > 0', None, 1, return_alternatives=True
    )[2] is None


def test_direct_other_inputs_keep_candidate_row_alignment():
    df = pd.DataFrame({
        'frame_i': [0, 0, 0], 'Cell_ID': [1, 2, 3],
        'left': [0, 1, -1], 'right': [0, -1, 1],
    })
    with pytest.warns(UserWarning):
        result = tp.TimepointPredictor.from_equation(
            df, '(left_other > 0) & (right_other > 0)', 'first_true',
            SizeT=1, IDs=[1], other_cells=tp.CellFilter.default_other()
        )
    assert result == {1: []}


@pytest.mark.parametrize('function', [
    'MIN', 'MAX', 'MEAN', 'AVERAGE', 'MEDIAN', 'SUM', 'STD', 'FIRST', 'LAST',
    'COUNT', 'NUNIQUE',
])
def test_other_aggregate_returns_zero_without_candidates(function):
    df = pd.DataFrame({'frame_i': [0], 'Cell_ID': [1], 'area': [10.]})
    prepared, equation = tp.OtherCellMeasurements.prepare_inputs(
        df, f'{function}(area_other)', tp.CellFilter.default_other(), 1
    )
    assert tp.CellEquation.evaluate(prepared, equation).tolist() == [0]


def test_direct_daughter_input_in_reported_equation():
    df = pd.DataFrame({
        'frame_i': [0, 1, 2, 3, 4], 'Cell_ID': [12, 12, 12, 13, 14],
        'parent_ID_tree': [-1, -1, -1, 12, 12],
        'cell_vol_fl': [1., 1., 10., 4., 4.],
    }).set_index(['frame_i', 'Cell_ID'])
    condition, start, end = tp.CellFilter.other_presets(df)['daughter']
    settings = dict(
        tp.CellFilter.default_other(), condition=condition,
        range_start=start, range_end=end
    )
    assert tp.TimepointPredictor.from_equation(
        df, '(cell_vol_fl_other>0) & cell_vol_fl>AVERAGE(cell_vol_fl)*1.5',
        'first_true', SizeT=5, other_cells=settings
    ) == {12: [2], 13: [], 14: []}


@pytest.mark.parametrize('column', ['area', 'cell area'])
@pytest.mark.parametrize('condition,expected', [
    ('', [5., 5., 5.]), ('Cell_ID < 0', [0., 0., 0.]),
])
def test_direct_input_and_aggregate_share_filtered_window(column, condition, expected):
    df = pd.DataFrame({
        'frame_i': [0, 1, 2, 0, 1, 2],
        'Cell_ID': [1, 1, 1, 2, 2, 2],
        column: [10., 10., 10., 5., 5., np.nan],
    })
    name = f'{column}_other'
    equation = f'(`{name}` + MEAN(`{name}`)) / 2'
    prepared, rewritten = tp.OtherCellMeasurements.prepare_inputs(
        df, equation,
        dict(tp.CellFilter.default_other(), condition=condition,
             range_start='0', range_end='2'),
        3, IDs=[1]
    )
    np.testing.assert_allclose(tp.CellEquation.evaluate(prepared, rewritten), expected)


def test_direct_other_unknown_column_is_not_zero():
    df = pd.DataFrame({'frame_i': [0], 'Cell_ID': [1], 'area': [10.]})
    with pytest.raises(ValueError, match='Unknown other-cell input column'):
        tp.OtherCellMeasurements.prepare_inputs(
            df, 'unknown_other', tp.CellFilter.default_other(), 1
        )


@pytest.mark.parametrize('mode', ['where_true', 'any', 'all'])
@pytest.mark.parametrize('condition', [
    'parent_ID_tree == this_Cell_ID',
    'Cell_ID == this_parent_ID_tree',
    '(parent_ID_tree > 0) & (parent_ID_tree == this_parent_ID_tree)',
    '(relative_ID == this_Cell_ID) & (relationship == "bud")',
    '(Cell_ID == this_relative_ID) & (this_relationship == "bud")',
    '(area > this_area) & (area > SizeT)',
    '(area > this_area) | (area < 5)',
    'True', 'False',
    'area > MEAN(area)',
    'DIFF(area) > 0',
    '`area` > this_area',
])
def test_batched_filters_match_per_candidate_evaluation(
        measurements, mode, condition
    ):
    flat = tp.CellTable.flatten(measurements)
    flat['parent_ID_tree'] = np.where(flat['frame_i'] < 2, 1, 2)
    flat['relative_ID'] = np.where(flat['Cell_ID'] == 1, 2, 1)
    flat['relationship'] = np.where(flat['Cell_ID'] == 1, 'mother', 'bud')
    settings = dict(
        tp.CellFilter.default_other(), condition=condition,
        condition_mode=mode, range_start='0', range_end='4'
    )
    window = flat.sort_values(['frame_i', 'Cell_ID'])
    columns = tp.CellFilter._this_columns(flat, condition)
    rowwise = tp.OtherCellMeasurements._is_rowwise_condition(condition)
    for _, target in flat.iterrows():
        expected = full_scan_rows(flat, target, settings, 5)
        actual = tp.OtherCellMeasurements._select_candidates(
            window, target, condition, mode, columns, {'SizeT': 5},
            rowwise=rowwise
        )
        pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize('condition', [
    'area > MEAN(area)', 'DIFF(area) > 0', 'any(area > this_area)',
    'index > 0', '`area` > this_area',
])
def test_non_rowwise_filters_keep_general_evaluation(condition):
    assert not tp.OtherCellMeasurements._is_rowwise_condition(condition)


def test_daughter_equality_avoids_per_candidate_pandas_evaluation(
        measurements, monkeypatch
    ):
    flat = tp.CellTable.flatten(measurements)
    flat['parent_ID_tree'] = 1
    condition, start, end = tp.CellFilter.other_presets(flat)['daughter']
    evaluate = Mock(wraps=tp.CellEquation.evaluate)
    monkeypatch.setattr(tp.CellEquation, 'evaluate', evaluate)
    settings = dict(
        tp.CellFilter.default_other(), condition=condition,
        range_start=start, range_end=end
    )
    tp.TimepointPredictor.from_equation(
        flat, '(MAX(area_other) > 0) & (area > AVERAGE(area) * 2)',
        'first_true', SizeT=5, other_cells=settings
    )
    assert evaluate.call_count == flat['Cell_ID'].nunique()


def test_daughter_equality_index_limits_candidate_rows(measurements, monkeypatch):
    flat = pd.DataFrame({
        'frame_i': [0] + [1] * 100,
        'Cell_ID': [1] + list(range(2, 102)),
        'parent_ID_tree': [-1, 1, 1] + [-1] * 98,
        'area': [1.] * 101,
    })
    windowSizes = []
    select = tp.OtherCellMeasurements._select_candidates

    def record_window_size(window, *args, **kwargs):
        windowSizes.append(len(window))
        return select(window, *args, **kwargs)

    monkeypatch.setattr(
        tp.OtherCellMeasurements, '_select_candidates', record_window_size
    )
    settings = dict(
        tp.CellFilter.default_other(), source=tp.SOURCE_DAUGHTER,
        range_start='t + 1', range_end='t + 1'
    )

    prepared, _ = tp.OtherCellMeasurements.prepare_inputs(
        flat, 'COUNT(area_other)', settings, SizeT=2, IDs=[1]
    )

    assert windowSizes == [2]
    assert prepared['_acdc_other_count_0'].tolist() == [2]


@pytest.mark.parametrize('mode', ['where_true', 'any', 'all'])
@pytest.mark.parametrize('start,end', [
    ('t - 1', 't + 1'), ('0', '6'), ('t + 20', 't + 21'), ('-5', '-1'),
])
def test_equality_windows_match_full_scan_without_between(monkeypatch, mode, start, end):
    df = pd.DataFrame({
        'frame_i': [0, 1, 3, 6] * 3,
        'Cell_ID': np.repeat([1, 2, 2**32 - 1], 4),
        'stage': ['S', 'G1', np.nan, 'S', 'G1', 'S', 'S', 'G1', 'S', 'S', 'G1', 'S'],
        'area': np.arange(12, dtype=float),
    }).sample(frac=1, random_state=42)
    original = df.copy(deep=True)
    settings = dict(
        tp.CellFilter.default_other(), condition='stage == this_stage',
        condition_mode=mode, range_start=start, range_end=end
    )
    targets = df[df['Cell_ID'].isin([1, 2**32 - 1])].iloc[:5]
    expected_count, expected_first = [], []
    for _, target in targets.iterrows():
        rows = full_scan_rows(df, target, settings, 7)
        expected_count.append(len(rows))
        expected_first.append(rows['area'].iloc[0] if len(rows) else 0)
    between = Mock(wraps=pd.Series.between)
    monkeypatch.setattr(pd.Series, 'between', between)
    prepared, _ = tp.OtherCellMeasurements.prepare_inputs(
        df, 'COUNT(area_other) + FIRST(area_other)', settings, 7,
        IDs=[1, 2**32 - 1], max_targets=5
    )
    assert prepared['_acdc_other_count_0'].tolist() == expected_count
    assert prepared['_acdc_other_first_1'].tolist() == expected_first
    pd.testing.assert_frame_equal(prepared[df.columns], targets)
    pd.testing.assert_frame_equal(df, original)
    between.assert_not_called()


def test_budding_filter_is_evaluated_once_per_nonempty_frame_window(
        measurements, monkeypatch
    ):
    flat = tp.CellTable.flatten(measurements)
    flat['relative_ID'] = 1
    flat['relationship'] = 'bud'
    condition, start, end = tp.CellFilter.other_presets(flat)['daughter']
    evaluate = Mock(wraps=tp.CellEquation.evaluate)
    monkeypatch.setattr(tp.CellEquation, 'evaluate', evaluate)
    tp.OtherCellMeasurements.prepare_inputs(
        flat, 'MAX(area_other)',
        dict(tp.CellFilter.default_other(), condition=condition,
             range_start=start, range_end=end),
        5
    )
    frames = set(flat['frame_i'])
    nonemptyWindows = 0
    for t in frames:
        lower, upper = tp.FrameRange.bounds(start, end, t, 5)
        nonemptyWindows += any(lower <= frame <= upper for frame in frames)
    assert evaluate.call_count == nonemptyWindows < len(flat)


def test_batched_filter_errors_identify_frame_and_condition(measurements):
    with pytest.raises(ValueError, match="frame \\d+ with condition 'missing"):
        tp.OtherCellMeasurements.prepare_inputs(
            measurements, 'MAX(area_other)',
            dict(tp.CellFilter.default_other(), condition='missing > this_area'),
            5
        )


@pytest.mark.parametrize('condition', [
    'COUNT(area_other) > 0', 'MAX(area_other) > 0', 'area_other > 0',
    'COUNT_OTHER(area) > 0',
])
@pytest.mark.parametrize('mode,expected', [('any', {1: [1]}), ('all', {})])
def test_main_filter_with_daughters_retains_complete_cell_history(
        condition, mode, expected
    ):
    df = pd.DataFrame({
        'frame_i': [0, 1, 2, 3, 0, 1, 2, 3, 4, 4],
        'Cell_ID': [1, 1, 1, 1, 2, 2, 2, 2, 3, 4],
        'area': [1., 10., 1., 1., 1., 10., 1., 1., 5., 5.],
        'parent_ID_tree': [-1] * 8 + [1, 1],
    }).set_index(['frame_i', 'Cell_ID'])
    source_condition, start, end = tp.CellFilter.other_presets(df)['daughter']
    original = df.copy(deep=True)
    result = tp.TimepointPredictor.from_equation(
        df, 'area > AVERAGE(area) * 2', 'first_true', SizeT=5,
        IDs=iter([1, 2]),
        other_cells=dict(
            tp.CellFilter.default_other(), condition=source_condition,
            range_start=start, range_end=end
        ),
        main_cell_filter={'condition': condition, 'condition_mode': mode}
    )
    assert result == expected
    pd.testing.assert_frame_equal(df, original)


def test_other_inputs_in_main_filter_require_configuration(measurements):
    with pytest.raises(ValueError, match='Configure Other-cell measurements'):
        tp.TimepointPredictor.from_equation(
            measurements, 'area > 0', 'first_true',
            main_cell_filter={'condition': 'COUNT(area_other) > 0'}
        )


@pytest.mark.parametrize('candidate_mode', ['where_true', 'any', 'all'])
def test_main_any_parent_comparison_with_blank_next_frame_filter(candidate_mode):
    df = pd.DataFrame({
        'frame_i': [0, 0, 1, 1],
        'Cell_ID': [12, 13, 14, 15],
        'parent_ID_tree': [-1, -1, 12, -1],
    }).set_index(['frame_i', 'Cell_ID'])
    with pytest.warns(UserWarning):
        result = tp.TimepointPredictor.from_equation(
            df, 'Cell_ID > 0', 'first_true', SizeT=2,
            other_cells=dict(
                tp.CellFilter.default_other(), condition='',
                condition_mode=candidate_mode,
                range_start='t + 1', range_end='t + 1'
            ),
            main_cell_filter={
                'condition': 'any(Cell_ID == parent_ID_tree_other)',
                'condition_mode': 'any',
            }
        )
    assert result == {12: [0]}


@pytest.mark.parametrize('equation', [
    'any(Cell_ID == parent_ID_tree_other)',
    'Cell_ID == parent_ID_tree_other',
    'ANY((Cell_ID == parent_ID_tree_other) & (stage_other == "S"))',
    'any(parent_ID_tree_other == 0)',
])
@pytest.mark.parametrize('condition', ['', 'area > this_area'])
@pytest.mark.parametrize('source,start,end', [
    ('same_frame', 't + 1', 't + 1'), ('same_frame', 't - 1', 't + 1'),
    ('same_frame', 't + 20', 't + 21'),
    ('same_frame', '4 - t', '5 - t'),
    ('same_frame', 't % 2', 't % 2 + 1'),
    ('same_frame', '0', '4'), ('same_frame', '-5', '-1'),
    ('daughter', 't + 1', 't + 1'), ('mother', 't - 1', 't - 1'),
    ('sister', 't', 't'),
])
def test_constant_main_any_matches_general_path(
        monkeypatch, equation, condition, source, start, end
    ):
    df = pd.DataFrame({
        'frame_i': [0, 1, 3, 0, 2, 4, 1, 2, 3],
        'Cell_ID': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'parent_ID_tree': [-1, -1, -1, 1, 1, 1, 2, 2, 2],
        'stage': ['G1', 'G1', 'G1', 'S', 'G1', 'S', 'S', 'S', 'G1'],
        'area': [1., 1., 1., 2., 2., 2., 3., 3., 3.],
    }).set_index(['frame_i', 'Cell_ID'])
    kwargs = dict(
        SizeT=5, IDs=[1, 2],
        other_cells=dict(
            tp.CellFilter.default_other(), condition=condition,
            source=source, range_start=start, range_end=end
        ),
        main_cell_filter={'condition': equation, 'condition_mode': 'any'},
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        actual = tp.TimepointPredictor.from_equation(
            df, 'area > 0', 'first_true', **kwargs
        )
        monkeypatch.setattr(
            tp.OtherCellMeasurements, 'constant_main_any', lambda *args: None
        )
        expected = tp.TimepointPredictor.from_equation(
            df, 'area > 0', 'first_true', **kwargs
        )
    assert actual == expected


def test_constant_main_any_selects_and_evaluates_once_per_cell(monkeypatch):
    df = pd.DataFrame({
        'frame_i': np.tile(np.arange(20), 3),
        'Cell_ID': np.repeat([1, 2, 3], 20),
        'parent_ID_tree': np.repeat([-1, 1, 2], 20),
    })
    select = Mock(wraps=tp.OtherCellMeasurements._select_candidates)
    evaluate = Mock(wraps=tp.CellEquation.evaluate)
    bounds = Mock(wraps=tp.FrameRange.bounds)
    array_bounds = Mock(wraps=tp.FrameRange.array_bounds)
    monkeypatch.setattr(tp.OtherCellMeasurements, '_select_candidates', select)
    monkeypatch.setattr(tp.CellEquation, 'evaluate', evaluate)
    monkeypatch.setattr(tp.FrameRange, 'bounds', bounds)
    monkeypatch.setattr(tp.FrameRange, 'array_bounds', array_bounds)
    with pytest.warns(UserWarning):
        result = tp.OtherCellMeasurements.constant_main_any(
            df, 'any(Cell_ID == parent_ID_tree_other)',
            dict(tp.CellFilter.default_other(),
                 range_start='t + 1', range_end='t + 1'),
            20, [1, 2, 3]
        )
    assert result == [1, 2]
    assert select.call_count == evaluate.call_count == 3
    bounds.assert_called_once()  # Settings validation, not per-frame evaluation.
    assert array_bounds.call_count == 3


def test_constant_main_any_falls_back_for_changing_own_values(measurements):
    assert tp.OtherCellMeasurements.constant_main_any(
        measurements, 'any(area > area_other)',
        tp.CellFilter.default_other(), 5, None
    ) is None
