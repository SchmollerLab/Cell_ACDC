import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from qtpy.QtWidgets import QApplication, QPushButton

from cellacdc import apps, apps_measurements_viewer, measurements_viewer, phase_translator, widgets
from cellacdc.timepoint_predictors import TIMEPOINT_PREFIX
from cellacdc.measurements_viewer import MeasurementsViewerWin
from cellacdc.timepoint_predictors import TimepointIdentifier

load_configuration = MeasurementsViewerWin.loadConfiguration


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def table():
    return pd.DataFrame({
        'frame_i': [0, 1, 2, 3] * 2,
        'Cell_ID': [1] * 4 + [2] * 4,
        'area': [0., 10., 20., 30., 3., 3., 3., 3.],
        f'{TIMEPOINT_PREFIX}Start': [False, True, False, False] * 2,
        f'{TIMEPOINT_PREFIX}End': [False, False, False, True] * 2,
        'Flag': [1] * 8,
    }).set_index(['frame_i', 'Cell_ID'])


def rule(**kwargs):
    return {
        'name': 'Cycle', 'predictor': {'type': 'none'},
        'timepoints': [f'{TIMEPOINT_PREFIX}Start', f'{TIMEPOINT_PREFIX}End'],
        'annotations': ['Flag'], 'phases': ['G1', 'S', 'G2'],
        **kwargs
    }


def fallback():
    return rule(
        name='Fallback', timepoints=[], annotations=[], phases=['Other']
    )


def test_intervals_and_boolean_columns(table):
    original = table.copy()
    result = phase_translator.evaluate(table, [rule()])
    assert result.table['phase'].tolist() == ['G1', 'S', 'S', 'G2'] * 2
    assert result.columns == {'phase', 'phase_G1', 'phase_S', 'phase_G2'}
    for phase in ('G1', 'S', 'G2'):
        values = result.table[f'phase_{phase}']
        assert values.dtype == bool
        assert values.equals(result.table['phase'].eq(phase).rename(values.name))
    assert not result.warnings
    pd.testing.assert_frame_equal(table, original)


@pytest.mark.parametrize('kind', ['formula', 'annotation', 'missing', 'order', 'equal'])
def test_ineligible_rule_falls_through_per_cell(table, kind):
    first = rule()
    if kind == 'formula':
        first['predictor'] = {'type': 'equation', 'equation': 'MAX(area) > 20'}
    elif kind == 'annotation':
        table.loc[(slice(None), 2), 'Flag'] = 0
    elif kind == 'missing':
        table.loc[(3, 2), f'{TIMEPOINT_PREFIX}End'] = False
    elif kind == 'order':
        table.loc[(1, 2), f'{TIMEPOINT_PREFIX}Start'] = False
        table.loc[(3, 2), f'{TIMEPOINT_PREFIX}Start'] = True
        table.loc[(3, 2), f'{TIMEPOINT_PREFIX}End'] = False
        table.loc[(1, 2), f'{TIMEPOINT_PREFIX}End'] = True
    else:
        table.loc[(3, 2), f'{TIMEPOINT_PREFIX}End'] = False
        table.loc[(1, 2), f'{TIMEPOINT_PREFIX}End'] = True
    result = phase_translator.evaluate(table, [first, fallback()]).table
    assert result.xs(1, level='Cell_ID')['phase'].tolist() == ['G1', 'S', 'S', 'G2']
    assert result.xs(2, level='Cell_ID')['phase'].tolist() == ['Other'] * 4


@pytest.mark.parametrize('mode, expected', [
    ('all', ['', '', '', '', '', '', '', '']),
    ('none', ['', '', '', '', 'G1', 'S', 'S', 'G2']),
    ('most', ['G1', 'S', 'S', 'G2', '', '', '', '']),
])
def test_formula_consolidation(table, mode, expected):
    result = phase_translator.evaluate(table, [rule(predictor={
        'type': 'equation', 'equation': 'area > 5', 'consolidation': mode
    })])
    assert result.table['phase'].tolist() == expected


def test_formula_main_filter(table):
    result = phase_translator.evaluate(table, [rule(predictor={
        'type': 'equation', 'equation': 'area >= 0',
        'main_cell_filter': {'condition': 'area > 20', 'condition_mode': 'any'}
    })])
    assert result.table.xs(2, level='Cell_ID')['phase'].eq('').all()
    assert result.table.xs(1, level='Cell_ID')['phase'].ne('').all()


def test_file_predictor_uses_entire_table_and_cell_scope(table, tmp_path):
    path = tmp_path / 'predictor.py'
    path.write_text(
        'def predictor(df, entire_df=None):\n'
        '    assert len(entire_df) == 8\n'
        '    return df["area"].max() > 20\n'
    )
    result = phase_translator.evaluate(table, [rule(predictor={
        'type': 'file', 'filepath': str(path)
    })], IDs=[1])
    assert result.table.xs(1, level='Cell_ID')['phase'].ne('').all()
    assert result.table.xs(2, level='Cell_ID')['phase'].eq('').all()


def test_annotations_are_requirements_anywhere_not_boundaries(table):
    table['Flag'] = 0
    table.loc[(2, 1), 'Flag'] = 1
    result = phase_translator.evaluate(table, [rule()])
    assert result.table.xs(1, level='Cell_ID')['phase'].tolist() == ['G1', 'S', 'S', 'G2']
    assert result.table.xs(2, level='Cell_ID')['phase'].eq('').all()


def test_only_first_matching_rule_supplies_all_intervals(table):
    result = phase_translator.evaluate(
        table, [rule(phases=['', 'S', '']), fallback()]
    )
    assert result.table['phase'].tolist() == ['', 'S', 'S', ''] * 2
    assert not result.table['phase_Other'].any()


def test_duplicates_warn_only_for_unmatched_cells(table):
    table.loc[(2, 2), f'{TIMEPOINT_PREFIX}Start'] = True
    result = phase_translator.evaluate(table, [rule()])
    assert result.table.xs(2, level='Cell_ID')['phase'].eq('').all()
    assert result.warnings == [
        'No phase rule could be applied to cells with duplicate time points: 2.'
    ]
    assert not phase_translator.evaluate(table, [rule(), fallback()]).warnings


def test_missing_input_is_reported_and_next_rule_tried(table):
    result = phase_translator.evaluate(
        table.drop(columns='Flag'), [rule(), fallback()]
    )
    assert result.table['phase'].eq('Other').all()
    assert result.warnings == ['Rule "Cycle" skipped: missing inputs Flag.']


def test_per_cell_apply_keeps_other_cells_and_cleans_obsolete_outputs(table):
    initial = phase_translator.evaluate(table, [rule()])
    changed = phase_translator.evaluate(
        initial.table, [fallback()], IDs=[1], owned_columns=initial.columns
    )
    assert changed.table.xs(1, level='Cell_ID')['phase'].eq('Other').all()
    assert changed.table.xs(2, level='Cell_ID')['phase'].tolist() == ['G1', 'S', 'S', 'G2']
    assert not changed.table.xs(1, level='Cell_ID')['phase_S'].any()
    final = phase_translator.evaluate(
        changed.table, [fallback()], owned_columns=changed.columns
    )
    assert final.table.columns.tolist() == [*table.columns, 'phase', 'phase_Other']


def test_reapply_exported_table_including_never_used_phases(table, tmp_path):
    first = phase_translator.evaluate(table, [rule(), fallback()])
    path = tmp_path / 'phases.csv'
    first.table.to_csv(path)
    loaded = pd.read_csv(path, index_col=['frame_i', 'Cell_ID'])
    result = phase_translator.evaluate(loaded, [rule(), fallback()])
    pd.testing.assert_frame_equal(result.table, first.table)


@pytest.mark.parametrize('column, value', [('phase', 'measurement'), ('phase_S', 7)])
def test_outputs_do_not_overwrite_unrelated_measurements(table, column, value):
    table[column] = value
    original = table.copy()
    with pytest.raises(ValueError, match='overwrite existing measurements'):
        phase_translator.evaluate(table, [rule()])
    pd.testing.assert_frame_equal(table, original)


def test_unsorted_rows_keep_their_own_assignments(table):
    ordered = phase_translator.evaluate(table, [rule()]).table
    shuffled = table.sample(frac=1, random_state=4)
    result = phase_translator.evaluate(shuffled, [rule()]).table
    assert result.index.equals(shuffled.index)
    pd.testing.assert_frame_equal(result.sort_index(), ordered.sort_index())


def test_flat_tables_and_uint32_cell_ids(table):
    table = table.reset_index()
    table.loc[table['Cell_ID'].eq(1), 'Cell_ID'] = 2 ** 32 - 1
    result = phase_translator.evaluate(table, [rule()], IDs=[2 ** 32 - 1])
    assert result.table['phase'].tolist() == ['G1', 'S', 'S', 'G2', '', '', '', '']


def test_empty_translator_cannot_be_applied(table):
    with pytest.raises(ValueError, match='at least one phase rule'):
        phase_translator.evaluate(table, [])


@pytest.mark.parametrize('invalid', [
    {}, [None], [rule(name='')], [rule(), rule()],
    [rule(phases=['S'])], [rule(phases=['', '', ''])],
    [rule(timepoints=['area'])], [rule(annotations=['Flag', 'Flag'])],
    [rule(predictor={'type': 'equation', 'equation': ''})],
    [rule(annotations=['phase_S'])],
])
def test_invalid_rules(invalid):
    with pytest.raises(ValueError):
        phase_translator.validate_rules(invalid)


def test_formula_failure_does_not_change_source(table):
    original = table.copy()
    with pytest.raises(ValueError):
        phase_translator.evaluate(table, [rule(predictor={
            'type': 'equation', 'equation': 'area'
        })])
    pd.testing.assert_frame_equal(table, original)


def test_rule_editor_roundtrip_and_reordering(app, table):
    dialog = apps_measurements_viewer.PhaseTranslatorRuleDialog(
        [f'{TIMEPOINT_PREFIX}Start', f'{TIMEPOINT_PREFIX}End'], ['Flag'],
        {'acdc_df': table}, rule=rule()
    )
    assert dialog.predictorGroupbox.autoApplyCheckBox.isHidden()
    assert [edit.text() for edit in dialog.phaseEdits.values()] == ['G1', 'S', 'G2']
    dialog.okButton.click()
    assert not dialog.cancel
    assert dialog.rule == rule()
    dialog.deleteLater()

    dialog = apps_measurements_viewer.PhaseTranslatorRuleDialog(
        [f'{TIMEPOINT_PREFIX}Start', f'{TIMEPOINT_PREFIX}End'], ['Flag'],
        {'acdc_df': table}, rule=rule()
    )
    down = next(
        button for button in dialog.findChildren(QPushButton)
        if button.text() == 'Down' and button.isEnabled()
    )
    down.click()
    assert dialog.boundaries == [f'{TIMEPOINT_PREFIX}End', f'{TIMEPOINT_PREFIX}Start']
    # Named intervals stay associated with their two boundaries.
    dialog.moveBoundary(f'{TIMEPOINT_PREFIX}Start', -1)
    assert [edit.text() for edit in dialog.phaseEdits.values()] == ['G1', 'S', 'G2']
    checkbox, _ = dialog.requirementChecks[f'{TIMEPOINT_PREFIX}End']
    checkbox.setChecked(False)
    assert dialog.boundaries == [f'{TIMEPOINT_PREFIX}Start']
    assert len(dialog.phaseEdits) == 2
    assert dialog.requirementChecks['Flag'][0].isChecked()
    dialog.deleteLater()


def test_rule_editor_reuses_annotation_formula_editor(app, table, monkeypatch):
    dialog = apps_measurements_viewer.PhaseTranslatorRuleDialog(
        [f'{TIMEPOINT_PREFIX}Start', f'{TIMEPOINT_PREFIX}End'], ['Flag'],
        {'acdc_df': table, 'allChNames': ['GFP'], 'SizeT': 4}, rule=rule()
    )
    equation = Mock(
        cancel=False, equation='MAX(area) > 20',
        otherCells=None, mainCellFilter={
            'condition': 'area > 5', 'condition_mode': 'any'
        }
    )
    factory = Mock(return_value=equation)
    monkeypatch.setattr(apps_measurements_viewer, 'CustomAnnotationPredictorEquationDialog', factory)
    group = dialog.predictorGroupbox
    group.predictorCombobox.setCurrentIndex(
        group.predictorCombobox.findData('equation')
    )
    group.consolidationCombobox.setCurrentIndex(
        group.consolidationCombobox.findData('most')
    )
    group.editEquationButton.click()
    equation.exec_.assert_called_once()
    assert factory.call_args.kwargs['acdc_df'] is table
    assert factory.call_args.kwargs['consolidation'] == 'most'
    dialog.okButton.click()
    assert not dialog.cancel
    assert dialog.rule['predictor'] == {
        'type': 'equation', 'equation': 'MAX(area) > 20',
        'consolidation': 'most', 'main_cell_filter': equation.mainCellFilter
    }
    dialog.deleteLater()


def test_rule_editor_rejects_missing_inputs_and_duplicate_names(app, monkeypatch):
    warning = Mock()
    monkeypatch.setattr(widgets.myMessageBox, 'warning', warning)
    dialog = apps_measurements_viewer.PhaseTranslatorRuleDialog([], [], {}, rule=rule())
    dialog.okButton.click()
    assert dialog.cancel
    assert 'missing inputs' in warning.call_args.args[-1]
    dialog.deleteLater()
    dialog = apps_measurements_viewer.PhaseTranslatorRuleDialog(
        [], [], {}, rule=fallback(), existingNames=['Fallback']
    )
    dialog.okButton.click()
    assert dialog.cancel
    assert 'already exists' in warning.call_args.args[-1]
    dialog.deleteLater()


def test_manager_priority_apply_and_cancel_are_isolated(app):
    rules = [rule(), fallback()]
    dialog = apps_measurements_viewer.PhaseTranslatorDialog(
        rules, [f'{TIMEPOINT_PREFIX}Start', f'{TIMEPOINT_PREFIX}End'], ['Flag'], {}
    )
    applied = Mock()
    dialog.sigApply.connect(applied)
    dialog.moveRule(1)
    assert dialog.rules[0]['name'] == 'Fallback'
    assert rules[0]['name'] == 'Cycle'
    dialog.applyCellButton.click()
    assert applied.call_args.args == (dialog.rules, False)
    dialog.applyAllButton.click()
    assert applied.call_args.args == (dialog.rules, True)
    dialog.removeRule()
    assert len(dialog.rules) == 1
    dialog.reject()
    assert dialog.cancel
    dialog.deleteLater()
    empty = apps_measurements_viewer.PhaseTranslatorDialog([], [], [], {})
    assert not empty.applyCellButton.isEnabled()
    assert not empty.applyAllButton.isEnabled()
    empty.deleteLater()


@pytest.fixture
def viewer(app, table, monkeypatch):
    monkeypatch.setattr(MeasurementsViewerWin, 'loadConfiguration', Mock())
    window = MeasurementsViewerWin()
    window.acdc_df = table
    window.selectedID = 1
    window.data = [SimpleNamespace(
        acdc_df=table, SizeT=4, SizeZ=1, chNames=['GFP'],
        pos_path=r'C:\experiment\Position_1'
    )]
    monkeypatch.setattr(window, 'plotMeasurements', Mock())
    monkeypatch.setattr(window, 'measurementProcessingWarning', Mock())
    window.centralWidget().setEnabled(True)
    yield window
    window.deleteLater()
    app.processEvents()


def test_viewer_apply_persists_table_rules_and_warns(viewer):
    viewer.acdc_df.loc[(2, 1), f'{TIMEPOINT_PREFIX}Start'] = True
    assert viewer.applyPhaseTranslator([rule()], allCells=False)
    assert viewer.posData().acdc_df is viewer.acdc_df
    assert viewer.phaseTranslatorRules == [rule()]
    assert viewer.acdc_df['phase'].eq('').all()
    viewer.measurementProcessingWarning.assert_called_once()
    viewer.plotMeasurements.assert_called_once()
    assert viewer.applyPhaseTranslator([fallback()])
    assert viewer.acdc_df['phase'].eq('Other').all()


def test_viewer_apply_displayed_cell_leaves_other_cell_and_position_untouched(viewer):
    other = SimpleNamespace(acdc_df=viewer.acdc_df.copy())
    viewer.data.append(other)
    original = other.acdc_df.copy()
    assert viewer.applyPhaseTranslator([rule()], allCells=False)
    assert viewer.acdc_df.xs(1, level='Cell_ID')['phase'].ne('').all()
    assert viewer.acdc_df.xs(2, level='Cell_ID')['phase'].eq('').all()
    pd.testing.assert_frame_equal(other.acdc_df, original)


def test_viewer_failure_is_atomic_and_processing_blocks_apply(viewer):
    original = viewer.acdc_df
    assert not viewer.applyPhaseTranslator([rule(predictor={
        'type': 'equation', 'equation': 'area'
    })])
    assert viewer.acdc_df is original
    assert viewer.phaseTranslatorRules == []
    viewer.measurementProcessingThread = object()
    assert not viewer.applyPhaseTranslator([rule()])
    viewer.measurementProcessingThread = None
    assert viewer.measurementProcessingWarning.call_count == 2


def test_missing_selected_cell_reports_error_without_changing_table(viewer):
    viewer.selectedID = 999
    original = viewer.acdc_df
    assert not viewer.applyPhaseTranslator([rule()], allCells=False)
    assert viewer.acdc_df is original
    assert viewer.phaseTranslatorRules == []
    assert 'No measurement rows' in viewer.measurementProcessingWarning.call_args.args[1]


@pytest.mark.parametrize('choice, expected', [
    ('Evaluate and export', True), ('Export existing values', False), ('Cancel', None)
])
def test_export_prompts_and_evaluates(viewer, tmp_path, monkeypatch, choice, expected):
    viewer.phaseTranslatorRules = [rule()]
    path = tmp_path / 'export.csv'
    monkeypatch.setattr(
        measurements_viewer.QFileDialog, 'getSaveFileName',
        Mock(return_value=(str(path), 'CSV (*.csv)'))
    )

    def question(msg, *args, **kwargs):
        buttons = kwargs['buttonsTexts']
        msg.cancel = choice == 'Cancel'
        msg.clickedButton = choice
        return buttons

    monkeypatch.setattr(widgets.myMessageBox, 'question', question)
    viewer.exportMeasurements()
    if expected is None:
        assert not path.exists()
    else:
        result = pd.read_csv(path)
        assert ('phase' in result) is expected
        if expected:
            assert result['phase'].tolist() == ['G1', 'S', 'S', 'G2'] * 2
            assert result['phase_S'].dtype == bool


def test_export_aborts_if_phase_evaluation_fails(viewer, tmp_path, monkeypatch):
    viewer.phaseTranslatorRules = [rule(predictor={
        'type': 'equation', 'equation': 'area'
    })]
    path = tmp_path / 'export.csv'
    monkeypatch.setattr(
        measurements_viewer.QFileDialog, 'getSaveFileName',
        Mock(return_value=(str(path), 'CSV (*.csv)'))
    )

    def question(msg, *args, **kwargs):
        msg.cancel = False
        msg.clickedButton = 'Evaluate and export'
        return kwargs['buttonsTexts']

    monkeypatch.setattr(widgets.myMessageBox, 'question', question)
    viewer.exportMeasurements()
    assert not path.exists()
    viewer.measurementProcessingWarning.assert_called_once()


def test_configuration_save_load_and_legacy(viewer, tmp_path, monkeypatch):
    pos = viewer.posData()
    pos.images_path = str(tmp_path)
    pos.basename = 'test_'
    pos.segm_npz_path = str(tmp_path / 'test_segm.npz')
    viewer.user_ch_name = 'GFP'
    viewer.allOverlayToolbuttons = {
        'GFP': SimpleNamespace(isChecked=lambda: True, setChecked=Mock())
    }
    viewer.phaseTranslatorRules = [rule(), fallback()]
    viewer.phaseColors = {'S': '#123456'}
    viewer.autoApplyPhases = True
    viewer.isDataLoaded = True
    viewer.configurationFilepath = str(tmp_path / 'configuration.json')
    assert viewer.saveConfiguration()
    configuration = json.loads((tmp_path / 'configuration.json').read_text())
    assert configuration['phase_translator_rules'] == [rule(), fallback()]
    assert configuration['phase_colors'] == {'S': '#123456'}
    assert configuration['auto_apply_phases'] is True
    assert 'phase' not in viewer.acdc_df
    legacy = copy.deepcopy(configuration)
    del legacy['phase_translator_rules']
    del legacy['phase_colors']
    del legacy['auto_apply_phases']
    MeasurementsViewerWin.validateConfiguration(legacy)
    with pytest.raises(ValueError):
        MeasurementsViewerWin.validateConfiguration(
            dict(configuration, phase_translator_rules=[rule(phases=[])])
        )
    with pytest.raises(ValueError):
        MeasurementsViewerWin.validateConfiguration(
            dict(configuration, phase_colors={'S': 'red'})
        )
    with pytest.raises(ValueError):
        MeasurementsViewerWin.validateConfiguration(
            dict(configuration, auto_apply_phases='yes')
        )
    monkeypatch.setattr(
        measurements_viewer, 'get_channels_and_segm_endnames',
        Mock(return_value=(['GFP'], ['segm']))
    )
    monkeypatch.setattr(measurements_viewer, 'load_position', Mock(return_value=pos))
    monkeypatch.setattr(
        viewer, 'selectConfigurationInput',
        lambda config: viewer.configurationInputOptions(config)[0]
    )
    monkeypatch.setattr(viewer, 'setData', Mock())
    monkeypatch.setattr(viewer, 'setOverlayItemsOpacities', Mock())
    viewer.phaseTranslatorRules = []
    viewer.phaseColors = {}
    viewer.autoApplyPhases = False
    assert load_configuration(
        viewer, filepath=viewer.configurationFilepath
    )
    assert viewer.phaseTranslatorRules == [rule(), fallback()]
    assert viewer.phaseColors == {'S': '#123456'}
    assert viewer.autoApplyPhases is True
    viewer.phaseTranslatorRules[0]['phases'][0] = 'Changed'
    assert configuration['phase_translator_rules'][0]['phases'][0] == 'G1'


def test_timepoint_and_annotation_renames_keep_rule_requirements(viewer, monkeypatch):
    identifier = TimepointIdentifier(name='Start', color=(255, 0, 0), shortcut='Q')
    viewer.addTimepointIdentifierItem(identifier)
    viewer.phaseTranslatorRules = [rule()]
    values = identifier.to_dict()
    values['name'] = 'Renamed'
    monkeypatch.setattr(viewer, 'execTimepointIdentifierDialog', Mock(return_value=values))
    monkeypatch.setattr(viewer, 'updateTimepointLines', Mock())
    viewer.editTimepointIdentifier(identifier.uid)
    assert viewer.phaseTranslatorRules[0]['timepoints'][0] == f'{TIMEPOINT_PREFIX}Renamed'
    viewer.renameCustomAnnotColumn('Flag', 'NewFlag')
    assert viewer.phaseTranslatorRules[0]['annotations'] == ['NewFlag']


def test_phase_names_and_colors():
    rules = [rule(phases=['G1', '', 'S']), fallback(), rule(name='B', phases=['S', 'G1', 'M'])]
    names = phase_translator.phase_names(rules)
    assert names == ['G1', 'S', 'Other', 'M']
    colors = phase_translator.phase_colors(names, {'S': '#ABCDEF', 'M': 'red'})
    assert list(colors) == names
    assert colors['S'] == '#abcdef'
    assert colors['M'] == phase_translator.default_phase_color(3, 4)
    assert len(set(colors.values())) == 4
    phase_translator.validate_phase_colors({'S': '#000000'})
    for invalid in ({'S': 'red'}, {'': '#000000'}, ['S']):
        with pytest.raises(ValueError):
            phase_translator.validate_phase_colors(invalid)


def test_manager_phase_color_buttons(app):
    dialog = apps_measurements_viewer.PhaseTranslatorDialog(
        [rule(), fallback()], [], [], {}, phaseColors={'S': '#ff0000', 'Gone': '#00ff00'}
    )
    assert list(dialog.phaseColorButtons) == ['G1', 'S', 'G2', 'Other']
    assert dialog.phaseColorButtons['S'].color().name() == '#ff0000'
    dialog.phaseColorButtons['G1'].setColor('#0000ff')
    colors = dialog.currentPhaseColors()
    assert colors['G1'] == '#0000ff'
    assert 'Gone' not in colors
    dialog.removeRule()
    assert list(dialog.phaseColorButtons) == ['Other']
    dialog.deleteLater()


def test_phase_runs():
    runs = MeasurementsViewerWin.phaseRuns(
        [1, 2, 3, 5, 6, 7], ['A', 'A', 'B', 'B', '', 'B']
    )
    assert runs == [(1, 2, 'A'), (3, 3, 'B'), (5, 5, 'B'), (7, 7, 'B')]


def test_phase_band_and_comparison_colors(viewer, tmp_path):
    viewer.settings_csv_path = str(tmp_path / 'settings.csv')
    viewer.showPhasesCheckbox.setChecked(True)
    viewer.phaseBandPlot.setVisible(True)
    viewer.updatePhaseBand()
    assert viewer.phaseBandPlot.isHidden()
    assert not viewer.showPhasesCheckbox.isEnabled()
    assert viewer.applyPhaseTranslator([rule()])
    viewer.phaseColors = {'S': '#123456'}
    viewer.updatePhaseBand()
    assert not viewer.phaseBandPlot.isHidden()
    assert viewer.showPhasesCheckbox.isEnabled()
    bars = viewer.phaseBandItems[0]
    assert list(bars.opts['x0']) == [0.5, 1.5, 3.5]
    assert list(bars.opts['x1']) == [1.5, 3.5, 4.5]
    assert bars.opts['brushes'][1].color().name() == '#123456'
    assert 'S: frames 2-3' in viewer.phaseBandPlot.toolTip()

    brushes = viewer.comparisonPhaseBrushes(viewer.phaseValues(viewer.acdc_df))
    assert brushes[1].color().name() == '#123456'

    viewer.showPhasesCheckbox.setChecked(False)
    assert not viewer.isShowingPhases()
    viewer.updatePhaseBand()
    assert viewer.phaseBandPlot.isHidden()
    assert viewer._get_setting_value(
        'measurements_viewer_show_phases', True, cast=bool
    ) is False
    settings = pd.read_csv(viewer.settings_csv_path, index_col='setting')
    assert settings.at['measurements_viewer_show_phases', 'value'] == 'No'
    viewer.showPhasesCheckbox.setChecked(True)
    viewer.measurementsModeCombobox.setCurrentIndex(1)
    viewer.updatePhaseBand()
    assert viewer.phaseBandPlot.isHidden()


def test_auto_apply_phases_on_timepoint_change(viewer, app, monkeypatch):
    identifier = TimepointIdentifier(name='Start', color=(255, 0, 0), shortcut='Q')
    viewer.addTimepointIdentifierItem(identifier)
    viewer.phaseTranslatorRules = [rule()]
    viewer.setAutoApplyPhases(False)
    viewer.replaceTimepoints(identifier.uid, {1: [2]}, [1])
    app.processEvents()
    assert 'phase' not in viewer.acdc_df

    apply = Mock(wraps=viewer.applyPhaseTranslator)
    monkeypatch.setattr(viewer, 'applyPhaseTranslator', apply)
    monkeypatch.setattr(viewer, 'ensurePhasePrediction', Mock())
    viewer.setAutoApplyPhases(True)
    viewer.replaceTimepoints(identifier.uid, {1: [2]}, [1])
    viewer.replaceTimepoints(identifier.uid, {}, [999])
    app.processEvents()
    apply.assert_called_once_with(IDs={1}, quiet=True)
    assert viewer.acdc_df.xs(1, level='Cell_ID')['phase'].tolist() == [
        'G1', 'G1', 'S', 'G2'
    ]
    # Other cells are not evaluated
    assert viewer.acdc_df.xs(2, level='Cell_ID')['phase'].eq('').all()
    # Errors do not open message boxes
    viewer.phaseTranslatorRules = [rule(predictor={'type': 'equation', 'equation': 'area'})]
    viewer.replaceTimepoints(identifier.uid, {1: [1]}, [1])
    app.processEvents()
    viewer.measurementProcessingWarning.assert_not_called()


def test_auto_apply_phases_when_missing(viewer, app, monkeypatch):
    viewer.phaseTranslatorRules = [rule()]
    apply = Mock(wraps=viewer.applyPhaseTranslator)
    monkeypatch.setattr(viewer, 'applyPhaseTranslator', apply)
    viewer.ensurePhasePrediction()
    app.processEvents()
    apply.assert_not_called()
    # Enabling evaluates the displayed cell without phases
    viewer.setAutoApplyPhases(True)
    app.processEvents()
    apply.assert_called_once_with(IDs={1}, quiet=True)
    assert viewer.acdc_df.xs(1, level='Cell_ID')['phase'].ne('').all()
    # Cells are tried once; cells that already have phases are skipped
    viewer.posData().phaseAutoEvaluatedIDs.clear()
    viewer.ensurePhasePrediction()
    viewer.ensurePhasePrediction(ID=2)
    viewer.ensurePhasePrediction(ID=2)
    app.processEvents()
    assert apply.call_count == 2
    assert apply.call_args.kwargs['IDs'] == {2}
    # Viewing a cell (predictor hook) triggers the check
    viewer.acdc_df.loc[(slice(None), 1), 'phase'] = ''
    viewer.posData().phaseAutoEvaluatedIDs.clear()
    viewer.ensurePredictorPredictions()
    app.processEvents()
    assert apply.call_args.kwargs['IDs'] == {1}
