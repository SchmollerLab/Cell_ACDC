import ast
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from cellacdc import timepoint_predictors as tp
from cellacdc._equations import equation_variable_spans


@pytest.fixture
def measurements():
    df = pd.DataFrame({
        'frame_i': np.repeat([0, 1, 3, 4], 3),
        'Cell_ID': np.tile([1, 2, 3], 4),
        'area': [10., 4., 7., 8., np.nan, 12., 15., 6., 3., 9., 20., 11.],
        'stage': ['G1', 'S', 'G1'] * 4,
    })
    return df.sample(frac=1, random_state=42).set_index(['frame_i', 'Cell_ID'])


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
    select = Mock(wraps=tp.OtherCellMeasurements._select_candidates)
    monkeypatch.setattr(tp.OtherCellMeasurements, '_select_candidates', select)
    result = tp.TimepointPredictor.from_equation(
        measurements, 'MEAN(area_other)', 'max',
        SizeT=5, IDs=iter([1, 2]),
        main_cell_filter={
            'condition': f'Cell_ID == {eligible_ID}', 'condition_mode': 'all'
        },
        other_cells=tp.CellFilter.default_other()
    )
    assert result == expected
    assert select.call_count == count
    assert all(call.args[1]['Cell_ID'] == 1 for call in select.call_args_list)


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


def test_direct_other_input_requires_aggregation_for_different_values():
    df = pd.DataFrame({
        'frame_i': [0, 0, 0], 'Cell_ID': [1, 2, 3], 'area': [10., 4., 7.]
    })
    with pytest.raises(ValueError, match='area_other.*multiple.*aggregate'):
        tp.OtherCellMeasurements.prepare_inputs(
            df, 'area_other > 0', tp.CellFilter.default_other(), 1, IDs=[1]
        )


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


def test_budding_filter_is_evaluated_once_per_nonempty_target_window(
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
    assert evaluate.call_count == len(flat)


def test_batched_filter_errors_identify_target(measurements):
    with pytest.raises(ValueError, match='frame .*enquiring cell'):
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
