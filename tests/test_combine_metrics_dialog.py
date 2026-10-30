import pandas as pd
import pytest
from types import SimpleNamespace
from qtpy.QtCore import QPoint, Qt
from qtpy.QtWidgets import QApplication, QWidget

from cellacdc import apps, cli, config, measurements, timepoint_predictors, widgets


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def tables():
    return {
        'first': pd.DataFrame({
            'GFP_mean': [2., 4., 8.],
            'cell_cycle_stage': pd.Categorical(['G1', 'S', 'G1']),
            'relationship': ['mother', 'bud', 'mother'],
            'is_history_known': [True, False, True],
            'count': pd.Series([1, 2, 3], dtype='Int64'),
            'numeric_category': pd.Categorical([1, 2, 1]),
            'cell area': [10., 20., 30.],
        }),
        'second': pd.DataFrame({'GFP_mean': [1., 2., 4.]}),
    }


@pytest.fixture
def dialog(app, tables):
    dialog = apps.CombineMetricsMultiDfsDialog(tables, ['GFP'])
    yield dialog
    dialog.close()
    dialog.deleteLater()


def leaf_names(item):
    if item.childCount() == 0:
        return [item.text(0)]
    return [
        name for i in range(item.childCount())
        for name in leaf_names(item.child(i))
    ]


def tree_items(parent):
    for row in range(parent.childCount()):
        item = parent.child(row)
        yield item
        yield from tree_items(item)


@pytest.mark.parametrize('multi_table', [False, True])
def test_measurement_search_matches_recipe_fuzzy_logic(app, tables, multi_table):
    if multi_table:
        dialog = apps.CombineMetricsMultiDfsDialog(tables, ['GFP'])
    else:
        dialog = apps.CombineMetricsEquationDialog(
            ['GFP'], False, False, acdc_df=tables['first']
        )
    for tree in dialog.metricsTreeWidgets:
        tree.collapseAll()
    initial = [
        (item, item.isHidden(), item.isExpanded())
        for tree in dialog.metricsTreeWidgets
        for item in tree_items(tree.invisibleRootItem())
    ]
    for query in ('GFP', 'cell_cycle', 'relationship', 'cell areaa', 'zzzzzzzzzz'):
        dialog.measurementsSearchLineEdit.setText(query)
        for tree, delegate in zip(
                dialog.metricsTreeWidgets, dialog.metricsHighlightDelegates
            ):
            items = list(tree_items(tree.invisibleRootItem()))
            flat = widgets.QStandardItemModel()
            for item in items:
                if not hasattr(item, 'isCol'):
                    continue
                reference_item = widgets.QStandardItem(item.text(0))
                reference_item.setData(
                    item.text(0), widgets.ButtonSearchCompleter.SEARCH_NAME_ROLE
                )
                flat.appendRow(reference_item)
            reference = widgets.ButtonSearchCompleter(
                searchRoles=(widgets.ButtonSearchCompleter.SEARCH_NAME_ROLE,),
                tooltipRole=None,
            )
            reference.setSourceModel(flat)
            reference.setFilterText(query)
            expected = {
                reference.index(row, 0).data()
                for row in range(reference.rowCount())
            }
            actual = {
                item.text(0) for item in items
                if hasattr(item, 'isCol') and not item.isHidden()
            }
            assert actual == expected
            assert delegate._filterText == query.strip()
            for item in items:
                if item.isHidden():
                    continue
                parent = item.parent()
                while parent is not None:
                    assert not parent.isHidden()
                    assert parent.isExpanded()
                    parent = parent.parent()
            if not expected:
                assert all(
                    tree.topLevelItem(row).isHidden()
                    for row in range(tree.topLevelItemCount())
                )
    dialog.measurementsSearchLineEdit.clear()
    for item, hidden, expanded in initial:
        assert item.isHidden() == hidden
        assert item.isExpanded() == expanded
    dialog.deleteLater()


def test_search_preserves_formula_and_inserts_suffixed_match(dialog):
    dialog.formulaEditWidget.setText('GFP_mean_table1 / 2')
    dialog.test_cb()
    assert dialog.okButton.isEnabled()
    dialog.measurementsSearchLineEdit.setText('cell area')
    assert dialog.formulaEditWidget.text() == 'GFP_mean_table1 / 2'
    assert dialog.okButton.isEnabled()
    tree = dialog.metricsTreeWidgets[0]
    item, = [
        item for item in tree_items(tree.invisibleRootItem())
        if item.text(0) == 'cell area' and not item.isHidden()
    ]
    dialog.clearEquation()
    tree.itemDoubleClicked.emit(item, 0)
    assert dialog.formulaEditWidget.text() == '`cell area_table1`'
    assert not dialog.okButton.isEnabled()


def test_both_dialogs_share_editor_and_controls(app, tables):
    single = apps.CombineMetricsEquationDialog(
        ['GFP'], False, False, acdc_df=tables['first']
    )
    multi = apps.CombineMetricsMultiDfsDialog(tables, ['GFP'])
    assert type(single).__bases__ == type(multi).__bases__
    for dialog in (single, multi):
        assert isinstance(dialog.equationDisplay, apps.FormulaEditWidget)
        assert not dialog.okButton.isEnabled()
        assert not dialog.formulaEditWidget.checkValidity()[0]
        dialog.deleteLater()


@pytest.mark.parametrize('multi_table', [False, True])
def test_name_and_equation_inputs_stay_aligned(app, tables, multi_table):
    if multi_table:
        dialog = apps.CombineMetricsMultiDfsDialog(tables, ['GFP'])
    else:
        dialog = apps.CombineMetricsEquationDialog(
            ['GFP'], False, False, acdc_df=tables['first']
        )
    suffix = '_table1' if multi_table else ''
    dialog.show()
    try:
        for width in (900, 1400):
            dialog.resize(width, 850)
            for formula in (
                '', f'GFP_mean{suffix} / 2',
                f'GFP_mean{suffix} + missing_variable',
                f'GFP_mean{suffix} + ' + 'very_long_missing_variable_' * 8,
            ):
                dialog.formulaEditWidget.setText(formula)
                app.processEvents()
                name = dialog.newColNameLineEdit
                equation = dialog.formulaEditWidget._edit
                assert name.mapTo(dialog, QPoint(0, 0)).y() == (
                    equation.mapTo(dialog, QPoint(0, 0)).y()
                )
                assert name.height() == equation.height()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_categorical_and_numerical_groups_follow_dtype(dialog):
    tree = dialog.metricsTreeWidgets[0]
    numerical = set(leaf_names(tree.topLevelItem(0)))
    categorical = set(leaf_names(tree.topLevelItem(1)))
    assert numerical == {'GFP_mean', 'count', 'cell area'}
    assert categorical == {
        'cell_cycle_stage', 'relationship', 'is_history_known',
        'numeric_category',
    }


@pytest.mark.parametrize('loaded', [False, True])
def test_catalog_uses_loaded_columns_and_preserves_groups(app, monkeypatch, loaded):
    monkeypatch.setattr(
        measurements, 'standard_metrics_desc',
        lambda isZstack, channel, **kwargs: (
            {f'{channel}_mean': '', f'{channel}_missing': ''},
            {f'{channel}_background': ''}
        )
    )
    monkeypatch.setattr(
        measurements, 'custom_metrics_desc',
        lambda isZstack, channel, **kwargs: {f'{channel}_custom': ''}
    )
    monkeypatch.setattr(
        measurements, 'get_size_metrics_desc',
        lambda *args: {'cell_area_pxl': '', 'missing_size': ''}
    )
    monkeypatch.setattr(
        measurements, 'get_props_names', lambda: ['solidity', 'missing_prop']
    )
    df = pd.DataFrame({
        'GFP_mean': [2.],
        'GFP_background': [1.],
        'GFP_custom': [3.],
        '_mean': [4.],
        'cell_area_pxl': [10.],
        'solidity': [0.9],
        'unlisted measurement': [7.],
        'stage': ['G1'],
        'known': [True],
    })
    dialog = apps.CombineMetricsEquationDialog(
        ['GFP'], False, False, acdc_df=df if loaded else None
    )
    items = list(tree_items(dialog.metricsTreeWidgets[0].invisibleRootItem()))
    leaves = {
        item.variableName: item for item in items
        if getattr(item, 'isCol', False)
    }
    if loaded:
        assert set(leaves) == set(df.columns)
        assert set(dialog.variableNames) == set(df.columns)
        assert len(dialog.variableNames) == len(df.columns)
        for name in ('unlisted measurement', 'stage', 'known'):
            assert leaves[name].parent().text(0) == 'Other'
        assert leaves['unlisted measurement'].parent().parent().text(0) == (
            'Numerical inputs'
        )
        for name in ('stage', 'known'):
            assert leaves[name].parent().parent().text(0) == 'Categorical inputs'
    else:
        assert {'GFP_missing', 'missing_size', 'missing_prop'} <= set(leaves)
        assert 'unlisted measurement' not in leaves
    for name, group in (
            ('GFP_mean', 'Cell measurements'),
            ('GFP_background', 'Background values'),
            ('GFP_custom', 'Custom measurements'),
            ('_mean', 'Cell signal measurements'),
            ('cell_area_pxl', 'Size measurements'),
            ('solidity', 'Region properties')
        ):
        assert leaves[name].parent().text(0) == group
    dialog.deleteLater()


def test_catalog_with_empty_dataframe_has_no_measurements(app):
    dialog = apps.CombineMetricsEquationDialog(
        ['GFP'], False, False, acdc_df=pd.DataFrame()
    )
    assert dialog.variableNames == []
    assert not any(
        getattr(item, 'isCol', False)
        for item in tree_items(dialog.metricsTreeWidgets[0].invisibleRootItem())
    )
    dialog.deleteLater()


@pytest.mark.parametrize('multi_table', [False, True])
@pytest.mark.parametrize('label, before, after, expected', [
    ('==', 'cell_cycle_stage{suffix}', '"G1"', [True, False, True]),
    ('!=', 'cell_cycle_stage{suffix}', '"G1"', [False, True, False]),
    ('<', 'GFP_mean{suffix}', '4', [True, False, False]),
    ('<=', 'GFP_mean{suffix}', '4', [True, True, False]),
    ('>', 'GFP_mean{suffix}', '4', [False, False, True]),
    ('>=', 'GFP_mean{suffix}', '4', [False, True, True]),
    ('AND', '(cell_cycle_stage{suffix} == "G1")',
     '(GFP_mean{suffix} > 3)', [False, False, True]),
    ('OR', '(cell_cycle_stage{suffix} == "G1")',
     '(GFP_mean{suffix} > 3)', [True, True, True]),
    ('NOT', '', '(cell_cycle_stage{suffix} == "G1")', [False, True, False]),
    ('True', 'is_history_known{suffix} == ', '', [True, False, True]),
    ('False', 'is_history_known{suffix} == ', '', [False, True, False]),
])
def test_categorical_buttons_build_valid_formulas(
        app, tables, multi_table, label, before, after, expected
    ):
    if multi_table:
        dialog = apps.CombineMetricsMultiDfsDialog(tables, ['GFP'])
    else:
        dialog = apps.CombineMetricsEquationDialog(
            ['GFP'], False, False, acdc_df=tables['first']
        )
    suffix = '_table1' if multi_table else ''
    before = before.format(suffix=suffix)
    after = after.format(suffix=suffix)
    editor = dialog.formulaEditWidget
    editor.setText(before + after)
    editor._edit.setCursorPosition(len(before))
    button, = [
        button for button in dialog.categoricalOperatorButtons
        if button.text() == label
    ]
    assert button.toolTip()
    button.click()
    assert editor.checkValidity()[0]
    dialog.test_cb()
    assert dialog.okButton.isEnabled()
    assert dialog.getTestData().eval(editor.text()).tolist() == expected
    button.click()
    assert not dialog.okButton.isEnabled()
    assert not dialog.statusLabel.text()
    dialog.close()
    dialog.deleteLater()


@pytest.mark.parametrize('formula, expected', [
    ('GFP_mean_table1 / GFP_mean_table2', [2., 2., 2.]),
    ('log10(GFP_mean_table1)', [0.30103, 0.60206, 0.90309]),
    ('cell_cycle_stage_table1 == "G1"', [True, False, True]),
    ('relationship_table1 != "mother"', [False, True, False]),
    ('numeric_category_table1 == 1', [True, False, True]),
    ('(cell_cycle_stage_table1 == "G1") & (GFP_mean_table1 > 3)',
     [False, False, True]),
    ('is_history_known_table1 & (GFP_mean_table1 > 3)',
     [False, False, True]),
    ('`cell area_table1` / 10', [1., 2., 3.]),
])
def test_pandas_formulas_validate_and_evaluate(dialog, formula, expected):
    dialog.formulaEditWidget.setText(formula)
    assert dialog.formulaEditWidget.checkValidity()[0]
    dialog.test_cb()
    assert dialog.okButton.isEnabled()
    actual = dialog.getCombinedDf().eval(formula)
    assert actual.tolist() == pytest.approx(expected, abs=1e-5)
    assert 'successful' in dialog.statusLabel.text()


@pytest.mark.parametrize('function, expected', [
    ('any', [True, True, True, False, False, True, True]),
    ('all', [False, False, False, False, False, True, True]),
])
def test_any_all_reduce_per_cell_and_repeat_on_each_frame(
        app, function, expected
    ):
    df = pd.DataFrame(
        {'cell area': [1, 4, 2, 1, 2, 4, 5]},
        index=pd.MultiIndex.from_tuples(
            [(0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (0, 3), (1, 3)],
            names=['frame_i', 'Cell_ID']
        )
    )
    dialog = apps.CombineMetricsEquationDialog(
        ['GFP'], False, False, acdc_df=df, saveToConfig=False,
        allowCellAggregations=True
    )
    formula = f'{function}(`cell area` > 3)'
    dialog.formulaEditWidget.setText(formula)
    assert dialog.formulaEditWidget.checkValidity()[0]
    assert dialog.evaluateEquation(df, formula).tolist() == expected
    labels = {
        button.text() for button in dialog.categoricalOperatorButtons
    }
    assert {'ANY', 'ALL'} <= labels
    dialog.clearEquation()
    next(
        button for button in dialog.categoricalOperatorButtons
        if button.text() == function.upper()
    ).click()
    assert dialog.formulaEditWidget.text() == f'{function}('
    dialog.deleteLater()


def test_any_all_are_unavailable_outside_measurements_viewer(dialog):
    dialog.formulaEditWidget.setText('any(GFP_mean_table1 > 2)')
    valid, message = dialog.formulaEditWidget.checkValidity()
    assert not valid
    assert 'Measurements viewer' in message


@pytest.mark.parametrize('formula', [
    'any(`cell area` > MEAN(`cell area`))',
    'ALL(`cell area` > 0)',
    'DIFF(`cell area`)',
    '`cell area` / MAX(`cell area`)',
])
@pytest.mark.parametrize('indexed', [True, False])
def test_evaluate_equation_matches_per_cell_runtime(app, formula, indexed):
    df = pd.DataFrame({
        'frame_i': [2, 0, 1, 0, 1],
        'Cell_ID': [1, 2, 1, 1, 2],
        'cell area': [6., 10., 4., 2., 20.],
    })
    if indexed:
        df = df.set_index(['frame_i', 'Cell_ID'])
    dialog = apps.CombineMetricsEquationDialog(
        [], False, False, acdc_df=df, saveToConfig=False,
        allowCellAggregations=True
    )
    actual = dialog.evaluateEquation(df, formula)
    expected = []
    flat = timepoint_predictors._flat_df(df)
    for _, cell in flat.groupby('Cell_ID', sort=False):
        cell = cell.sort_values('frame_i')
        values = timepoint_predictors.eval_cell_equation(
            cell.reset_index(drop=True), formula
        )
        expected.append(pd.Series(
            values if not isinstance(values, pd.Series) else values.to_numpy(),
            index=cell.index
        ))
    expected = pd.concat(expected).reindex(flat.index)
    expected.index = df.index
    pd.testing.assert_series_equal(actual, expected)
    dialog.formulaEditWidget.setText(formula)
    dialog.test_cb()
    assert dialog.okButton.isEnabled()
    dialog.deleteLater()


@pytest.mark.parametrize('cell_functions', [False, True])
@pytest.mark.parametrize('formula,dtype,valid', [
    ('value > 0', bool, True),
    ('value * 0', bool, True),
    ('value', bool, False),
    ('value / 0', bool, False),
    ('value', 'float64', True),
    ('value', 'int64', False),
    ('value > 0', 'float64', False),
])
def test_evaluate_equation_enforces_dtype(
        app, cell_functions, formula, dtype, valid
    ):
    df = pd.DataFrame({'value': [1., 2., 3.]})
    dialog = apps.CombineMetricsEquationDialog(
        [], False, False, acdc_df=df, saveToConfig=False,
        allowCellAggregations=cell_functions
    )
    if valid:
        result = dialog.evaluateEquation(
            df, formula, enforce_dtype_output=dtype
        )
        assert pd.api.types.is_dtype_equal(result.dtype, dtype)
    else:
        with pytest.raises(ValueError, match='dtype|bool|0/1'):
            dialog.evaluateEquation(df, formula, enforce_dtype_output=dtype)
    dialog.deleteLater()


@pytest.mark.parametrize('formula', [
    '', '2 + 2', 'missing_table1 + GFP_mean_table1',
    'cell_cycle_stage_table1 + GFP_mean_table1',
    'GFP_mean_table1 + (', 'GFP_mean_table1 = 2',
])
def test_invalid_equations_show_error_and_disable_ok(dialog, formula):
    dialog.formulaEditWidget.setText('GFP_mean_table1 / 2')
    dialog.test_cb()
    assert dialog.okButton.isEnabled()
    dialog.formulaEditWidget.setText(formula)
    assert not dialog.formulaEditWidget.checkValidity()[0]
    dialog.test_cb()
    assert not dialog.okButton.isEnabled()
    assert dialog.statusLabel.text()
    assert dialog.testOutputDisplay.toPlainText()
    assert 'red' in dialog.testOutputDisplay.styleSheet()


def test_changes_invalidate_previous_success(dialog):
    dialog.newColNameLineEdit.setText('ratio')
    dialog.formulaEditWidget.setText('GFP_mean_table1 / 2')
    dialog.test_cb()
    dialog.newColNameLineEdit.setText('new_ratio')
    assert not dialog.okButton.isEnabled()
    assert not dialog.statusLabel.text()
    dialog.test_cb()
    dialog.formulaEditWidget.setText('GFP_mean_table1 / 4')
    assert not dialog.okButton.isEnabled()
    assert not dialog.testOutputDisplay.toPlainText()
    dialog.clearEquation()
    assert dialog.equationColNames == []


def test_insertion_uses_cursor_and_backticks(dialog):
    tree = dialog.metricsTreeWidgets[0]
    item, = tree.findItems(
        'cell area', Qt.MatchExactly | Qt.MatchRecursive
    )
    dialog.formulaEditWidget.setText(' / 2')
    dialog.formulaEditWidget._edit.setCursorPosition(0)
    dialog.addColname(item, 0)
    assert dialog.formulaEditWidget.text() == '`cell area_table1` / 2'
    assert dialog.equationColNames == ['cell area_table1']
    dialog.formulaEditWidget._edit.setCursorPosition(
        len(dialog.formulaEditWidget.text())
    )
    dialog.clearEntryEquation()
    assert dialog.formulaEditWidget.text().endswith(' / ')


def test_manual_input_tracks_columns_not_string_literals(dialog):
    dialog.formulaEditWidget.setText(
        'relationship_table1 == "GFP_mean_table1"'
    )
    assert dialog.equationColNames == ['relationship_table1']
    dialog.formulaEditWidget.setText('GFP_mean_table1 + GFP_mean_table1')
    assert dialog.equationColNames == ['GFP_mean_table1']


def test_multi_table_signals_and_close_contract(dialog):
    accepted = []
    closed = []
    dialog.sigOk.connect(lambda name, expression: accepted.append((name, expression)))
    dialog.sigClose.connect(closed.append)
    dialog.newColNameLineEdit.setText('g1')
    expression = 'cell_cycle_stage_table1 == "G1"'
    dialog.equationDisplay.setPlainText(expression)
    dialog.test_cb()
    dialog.ok_cb()
    assert not dialog.cancel
    assert dialog.newColname == 'g1'
    assert dialog.expression == expression
    assert accepted == [('g1', expression)]
    assert closed == [False]


def test_single_table_categorical_inputs_and_persistence(app, tables, monkeypatch):
    dialog = apps.CombineMetricsEquationDialog(
        ['GFP'], False, False, closeOnOk=False, acdc_df=tables['first']
    )
    tree = dialog.metricsTreeWidget
    assert 'cell_cycle_stage' in leaf_names(tree.topLevelItem(1))
    dialog.newColNameLineEdit.setText('g1')
    dialog.formulaEditWidget.setText('cell_cycle_stage == "G1"')
    assert dialog.formulaEditWidget.checkValidity()[0]
    dialog.test_cb()
    assert dialog.getEquationsDict() == (
        {'g1': 'cell_cycle_stage == "G1"'}, True
    )
    monkeypatch.setattr(measurements, 'read_saved_user_combine_config', dict)
    saved = []
    monkeypatch.setattr(
        measurements, 'add_user_combine_metrics',
        lambda config, equation, name, mixed: config | {name: equation}
    )
    monkeypatch.setattr(measurements, 'save_common_combine_metrics', saved.append)
    accepted = []
    dialog.sigOk.connect(accepted.append)
    dialog.ok_cb()
    assert saved == [{'g1': 'cell_cycle_stage == "G1"'}]
    assert accepted == [dialog]
    assert not dialog.cancel
    dialog.deleteLater()


def test_single_channel_and_channel_less_equations_use_exact_tokens(app):
    dialog = apps.CombineMetricsEquationDialog(['GFP', 'RFP'], False, False)
    dialog.newColNameLineEdit.setText('ratio')
    dialog.formulaEditWidget.setText('GFP_mean / GFP_sum')
    assert dialog.getEquationsDict() == (
        {'GFP_ratio': 'GFP_mean / GFP_sum'}, False
    )
    dialog.formulaEditWidget.setText('_mean / GFP_mean')
    assert dialog.channelLessColnames == ['_mean']
    assert dialog.getEquationsDict() == ({
        'GFP_ratio': 'GFP_mean / GFP_mean',
        'RFP_ratio': 'RFP_mean / GFP_mean',
    }, False)
    dialog.formulaEditWidget.setText('GFP_mean / RFP_mean')
    assert dialog.channelLessColnames == []
    assert dialog.getEquationsDict() == (
        {'ratio': 'GFP_mean / RFP_mean'}, True
    )
    dialog.test_cb()
    assert dialog.okButton.isEnabled()
    dialog.deleteLater()


def test_formula_editor_no_longer_requires_channel_parent(app):
    parent = QWidget()
    editor = apps.FormulaEditWidget(['img1', 'img2'], parent=parent)
    assert editor.checkValidity()[0]
    editor.setText('img1 + img2 * 0.5')
    assert editor.checkValidity()[0]
    editor.setText('img1 + missing')
    valid, error = editor.checkValidity()
    assert not valid
    assert 'Missing variables' in error
    parent.deleteLater()


def test_channel_formula_preserves_forbidden_inputs_and_empty_formula(app, monkeypatch):
    monkeypatch.setattr(apps.CombineChannelsSetupDialog, 'setLoadLastRecipe', lambda _: None)
    dialog = apps.CombineChannelsSetupDialog(['GFP', 'RFP'])
    assert dialog.formulaEditWidget.checkValidity()[0]
    dialog.formulaEditWidget.setText('img1 * 2')
    assert dialog.formulaEditWidget.checkValidity()[0]
    dialog.forbiddenChannels = {'GFP'}
    valid, error = dialog.formulaEditWidget.checkValidity()
    assert not valid
    assert 'forbidden' in error
    dialog.formulaEditWidget.setText('')
    assert not dialog.formulaEditWidget.checkValidity()[0]
    dialog.forbiddenChannels.clear()
    assert dialog.formulaEditWidget.checkValidity()[0]
    dialog.deleteLater()


def test_variable_replacement_preserves_literals_and_quoted_names(app):
    editor = apps.FormulaEditWidget(['_mean', 'GFP_mean', 'cell area'])
    editor.setText('(_mean / GFP_mean) + (`cell area` == "_mean")')
    assert editor.referencedVariableNames() == ['_mean', 'GFP_mean', 'cell area']
    assert editor.replaceVariableNames({'_mean': 'RFP_mean'}) == (
        '(RFP_mean / GFP_mean) + (`cell area` == "_mean")'
    )
    editor.deleteLater()


def test_incomplete_multiline_equation_does_not_break_live_editor(dialog):
    dialog.formulaEditWidget.setText(
        'GFP_mean_table1 + (\n    GFP_mean_table2\n  GFP_mean_table1'
    )
    assert not dialog.formulaEditWidget.checkValidity()[0]
    assert not dialog.okButton.isEnabled()


def test_full_table_test_catches_errors_outside_preview(app):
    dialog = apps.CombineMetricsMultiDfsDialog({
        'first': pd.DataFrame({'value': ['1'] * 10 + ['bad']})
    }, [])
    dialog.formulaEditWidget.setText('value_table1.astype("float")')
    assert dialog.formulaEditWidget.checkValidity()[0]
    dialog.test_cb()
    assert not dialog.okButton.isEnabled()
    assert 'bad' in dialog.statusLabel.text()
    dialog.deleteLater()


def test_saved_categorical_formulas_remain_available(tmp_path, monkeypatch):
    monkeypatch.setattr(measurements, 'acdc_metrics_path', str(tmp_path))
    monkeypatch.setattr(
        measurements, 'combine_metrics_ini_path',
        str(tmp_path / 'combine_metrics.ini')
    )
    saved = config.ConfigParser()
    saved['mixed_channels_equations'] = {
        'selected': 'stage == "G1_phase"',
        'missing': 'unknown == "G1_phase"',
        'with_function': 'log(area) > 1',
        'quoted': '`cell area` > 2',
    }
    measurements.save_common_combine_metrics(saved)
    equations = measurements.get_user_combine_mixed_channels_equations()
    assert equations['selected'] == 'stage == "G1_phase"'
    _, available = measurements.combine_mixed_channels_desc(
        available_cols=['stage', 'area', 'cell area']
    )
    assert set(available) == {'selected', 'with_function', 'quoted'}


def test_saved_channel_less_formulas_expand_exact_tokens(monkeypatch):
    saved = config.ConfigParser()
    saved['equations'] = {}
    saved['channelLess_equations'] = {
        'ratio': '_mean / GFP_mean;_mean'
    }
    monkeypatch.setattr(
        measurements, 'read_saved_user_combine_config', lambda: saved
    )
    _, equations = measurements.channel_combine_metrics_desc('RFP')
    assert equations['RFP_ratio'] == 'RFP_mean / GFP_mean'


def test_saved_channel_specific_categorical_formulas_remain_available(monkeypatch):
    saved = config.ConfigParser()
    saved['equations'] = {'GFP_selected': 'GFP_state == "G1_phase"'}
    saved['channelLess_equations'] = {}
    monkeypatch.setattr(
        measurements, 'read_saved_user_combine_config', lambda: saved
    )
    equations = measurements.get_user_combine_metrics_equations('GFP')
    assert equations['GFP_selected'] == 'GFP_state == "G1_phase"'


def test_measurement_execution_handles_categorical_literals_with_underscores():
    kernel = object.__new__(cli.ComputeMeasurementsKernel)
    kernel.metricsToSkip = {}
    kernel.mixedChCombineMetricsToSkip = []
    position = SimpleNamespace(
        loadedChNames=['GFP'],
        combineMetricsConfig={
            'equations': {},
            'user_path_equations': {},
            'mixed_channels_equations': {
                'selected': 'stage == "G1_phase"',
                'large': '`cell area` > 2',
                'missing': 'unknown_column == "G1_phase"',
            },
        },
    )
    df = pd.DataFrame({
        'stage': ['G1_phase', 'S_phase'],
        'cell area': [1., 3.],
    })
    kernel._add_combined_metrics(position, df)
    assert df['selected'].tolist() == [True, False]
    assert df['large'].tolist() == [False, True]
    assert 'missing' not in df
