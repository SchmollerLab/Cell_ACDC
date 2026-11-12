"""Measurements viewer: custom annotations with the "Overrides phases" option
(state key ``overridesPhases``) give annotated cells the annotation's name as
phase, re-evaluated automatically when annotations change."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pandas as pd
import pytest
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QApplication

from cellacdc import gui_custom_annotations, measurements_viewer, phase_translator
from cellacdc import measurements_viewer_utils
from cellacdc.measurements_viewer import MeasurementsViewerWin
from cellacdc.timepoint_predictors import TIMEPOINT_PREFIX


load_configuration = MeasurementsViewerWin.loadConfiguration

START = f'{TIMEPOINT_PREFIX}Start'
END = f'{TIMEPOINT_PREFIX}End'
CYCLE = {
    'name': 'Cycle', 'predictor': {'type': 'none'},
    'timepoints': [START, END], 'annotations': [],
    'phases': ['G1', 'S', 'G2'],
}
RULE_PHASES = ['G1', 'S', 'S', 'G2']


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def acdc_df():
    return pd.DataFrame({
        'frame_i': [0, 1, 2, 3] * 3,
        'Cell_ID': [1] * 4 + [2] * 4 + [3] * 4,
        'area': [10., 20., 15., 30., 5., 6., 7., 4., 200., 210., 220., 230.],
        START: [False, True, False, False] * 3,
        END: [False, False, False, True] * 3,
    }).set_index(['frame_i', 'Cell_ID'])


@pytest.fixture
def viewer(app, monkeypatch, acdc_df, tmp_path):
    monkeypatch.setattr(
        gui_custom_annotations, 'custom_annot_path',
        str(tmp_path / 'custom_annotations.json')
    )
    monkeypatch.setattr(MeasurementsViewerWin, 'loadConfiguration', Mock())
    window = MeasurementsViewerWin()
    labels = np.zeros((4, 30, 30), dtype=np.uint16)
    labels[:, 2:5, 3:7] = 1
    labels[:, 15:19, 20:25] = 2
    labels[:, 22:27, 5:9] = 3
    window.data = [SimpleNamespace(
        SizeT=4, SizeZ=1, frame_i=0, segm_data=labels, chNames=['GFP'],
        pos_path=r'C:\experiment\Position_1', acdc_df=acdc_df
    )]
    window.isSegm3D = False
    window.acdc_df = acdc_df
    monkeypatch.setattr(window, 'updateImages', Mock())
    monkeypatch.setattr(window, 'setViewRangeToObjBbox', Mock())
    monkeypatch.setattr(window, 'measurementProcessingWarning', Mock())
    monkeypatch.setattr(window, 'drawCustomAnnotationIDs', Mock())
    window.centralWidget().setEnabled(True)
    window.initFramesNavigation()
    window.initIDsNavigation(ID=1)
    window.updateColumnsList()
    monkeypatch.setattr(window, 'plotMeasurements', Mock())
    yield window
    app.processEvents()
    window.reinitCustomAnnot()
    window.deleteLater()
    app.processEvents()


def annot_state(
        name='Arrested', shortcut='A', overrides=None, predictor=None,
        autoApply=False
    ):
    state = {
        'type': 'Single time-point', 'name': name, 'symbol': "'o' circle",
        'shortcut': shortcut, 'description': '', 'keepActive': True,
        'isHideChecked': True, 'symbolColor': [255, 0, 0, 255],
    }
    if predictor is not None:
        state['predictor'] = predictor
        state['autoApply'] = autoApply
    if overrides is not None:
        state['overridesPhases'] = overrides
    return state


def equation(text):
    return {'type': 'equation', 'equation': text}


def configure(
        viewer, app, states, rules=(CYCLE,), autoApply=True, evaluate=True
    ):
    """Set the rules and annotations, evaluate all cells, then set the
    auto apply option."""
    viewer.phaseTranslatorRules = copy.deepcopy(list(rules))
    viewer.setCustomAnnotations(states)
    if evaluate:
        assert viewer.applyPhaseTranslator()
    viewer.autoApplyPhases = autoApply
    app.processEvents()


def phases(viewer, ID):
    return viewer.acdc_df.xs(ID, level='Cell_ID')['phase'].tolist()


def spy(viewer, monkeypatch, method):
    mock = Mock(wraps=getattr(viewer, method))
    monkeypatch.setattr(viewer, method, mock)
    return mock


def toggle(viewer, name, ID):
    assert viewer.toggleCustomAnnotation(
        viewer.customAnnotButtonByName(name), ID
    )


def modify_annotation(viewer, monkeypatch, name, state):
    state = dict(state)
    state['symbolColor'] = QColor(*state['symbolColor'])
    dialog = SimpleNamespace(
        cancel=False, state=state, symbol='o', exec_=Mock(),
        sigDeleteSelecAnnot=SimpleNamespace(connect=Mock())
    )
    monkeypatch.setattr(
        gui_custom_annotations.apps, 'customAnnotationDialog',
        Mock(return_value=dialog)
    )
    viewer.customAnnotModify(viewer.customAnnotButtonByName(name))


def test_toggling_overriding_annotation_replaces_rule_phases(
        viewer, app, monkeypatch
    ):
    configure(viewer, app, [annot_state(overrides=True)])
    assert viewer.phaseOverrideAnnotations() == ['Arrested']
    assert phases(viewer, 1) == RULE_PHASES
    apply = spy(viewer, monkeypatch, 'applyPhaseTranslator')
    toggle(viewer, 'Arrested', 1)
    app.processEvents()
    apply.assert_called_once_with(IDs={1}, quiet=True)
    assert phases(viewer, 1) == ['Arrested'] * 4
    assert viewer.acdc_df.xs(1, level='Cell_ID')['phase_Arrested'].all()
    for ID in (2, 3):
        assert phases(viewer, ID) == RULE_PHASES
    assert viewer.acdc_df['phase_Arrested'].sum() == 4
    assert viewer.posData().acdc_df is viewer.acdc_df
    # Toggling it off applies the rules again
    toggle(viewer, 'Arrested', 1)
    app.processEvents()
    assert apply.call_args == call(IDs={1}, quiet=True)
    assert phases(viewer, 1) == RULE_PHASES
    assert not viewer.acdc_df['phase_Arrested'].any()


def test_annotation_that_does_not_affect_phases_evaluates_nothing(
        viewer, app, monkeypatch
    ):
    configure(viewer, app, [annot_state('Big')])
    schedule = spy(viewer, monkeypatch, 'schedulePhaseAutoApply')
    toggle(viewer, 'Big', 1)
    assert viewer.replaceCustomAnnotations('Big', {2: [0]}, {2})
    app.processEvents()
    schedule.assert_not_called()
    assert not viewer.annotationAffectsPhases('Big')
    # Required annotation of a rule, or read by a rule's formula
    viewer.phaseTranslatorRules[0]['annotations'] = ['Big']
    assert viewer.annotationAffectsPhases('Big')
    viewer.phaseTranslatorRules[0]['annotations'] = []
    viewer.phaseTranslatorRules[0]['predictor'] = equation('Big == 0')
    assert viewer.annotationAffectsPhases('Big')
    toggle(viewer, 'Big', 1)
    schedule.assert_called_once_with([1], None)
    app.processEvents()
    assert phases(viewer, 1) == RULE_PHASES


def test_predictions_evaluate_only_cells_whose_annotation_changed(
        viewer, app, monkeypatch
    ):
    # Displayed cell 1 is predicted (not arrested) when the annotation is set
    configure(viewer, app, [annot_state(
        overrides=True, predictor=equation('area > 100'), autoApply=True
    )])
    schedule = spy(viewer, monkeypatch, 'schedulePhaseAutoApply')
    # First view of cell 2: predicted, but its annotation does not change
    viewer.setSelectedID(2)
    assert viewer.replaceCustomAnnotations('Arrested', {}, {1, 2})
    app.processEvents()
    schedule.assert_not_called()
    assert phases(viewer, 2) == RULE_PHASES
    # First view of cell 3: annotated by its predictor
    viewer.setSelectedID(3)
    schedule.assert_called_once_with({3}, viewer.posData())
    app.processEvents()
    assert phases(viewer, 3) == ['Arrested'] * 4
    for ID in (1, 2):
        assert phases(viewer, ID) == RULE_PHASES
    # Same values again: nothing to evaluate
    assert viewer.replaceCustomAnnotations('Arrested', {3: [0]}, {1, 2, 3})
    assert schedule.call_count == 1


def test_clearing_overriding_annotation_restores_rule_phases(
        viewer, app, monkeypatch
    ):
    configure(viewer, app, [annot_state(overrides=True)])
    toggle(viewer, 'Arrested', 1)
    toggle(viewer, 'Arrested', 2)
    app.processEvents()
    assert phases(viewer, 1) == phases(viewer, 2) == ['Arrested'] * 4
    apply = spy(viewer, monkeypatch, 'applyPhaseTranslator')
    viewer.clearCustomAnnotations(viewer.customAnnotButtonByName('Arrested'))
    app.processEvents()
    apply.assert_called_once_with(IDs={1, 2}, quiet=True)
    for ID in (1, 2, 3):
        assert phases(viewer, ID) == RULE_PHASES


def test_switching_overrides_option_reevaluates_annotated_cells(
        viewer, app, monkeypatch, capsys
    ):
    configure(viewer, app, [annot_state()])
    toggle(viewer, 'Arrested', 1)
    app.processEvents()
    assert phases(viewer, 1) == RULE_PHASES
    button = viewer.customAnnotButtonByName('Arrested')
    assert 'Overrides phases' not in button.toolTip()
    apply = spy(viewer, monkeypatch, 'applyPhaseTranslator')
    capsys.readouterr()
    modify_annotation(viewer, monkeypatch, 'Arrested', annot_state(overrides=True))
    assert 'Custom annotation "Arrested" overrides phases' in capsys.readouterr().out
    assert 'Overrides phases' in button.toolTip()
    app.processEvents()
    apply.assert_called_once_with(IDs={1}, quiet=True)
    assert phases(viewer, 1) == ['Arrested'] * 4
    assert phases(viewer, 2) == RULE_PHASES

    modify_annotation(viewer, monkeypatch, 'Arrested', annot_state(overrides=False))
    assert 'no longer overrides phases' in capsys.readouterr().out
    app.processEvents()
    assert apply.call_args == call(IDs={1}, quiet=True)
    assert phases(viewer, 1) == RULE_PHASES
    assert 'phase_Arrested' not in viewer.acdc_df


def test_renaming_overriding_annotation_renames_phase_and_color(
        viewer, app, monkeypatch
    ):
    configure(viewer, app, [annot_state(overrides=True)])
    toggle(viewer, 'Arrested', 1)
    app.processEvents()
    viewer.phaseColors = {'Arrested': '#123456', 'G1': '#654321'}
    modify_annotation(
        viewer, monkeypatch, 'Arrested', annot_state('Stopped', overrides=True)
    )
    app.processEvents()
    assert phases(viewer, 1) == ['Stopped'] * 4
    assert viewer.acdc_df.xs(1, level='Cell_ID')['phase_Stopped'].all()
    assert 'phase_Arrested' not in viewer.acdc_df
    assert phases(viewer, 2) == RULE_PHASES
    assert viewer.phaseColors == {'Stopped': '#123456', 'G1': '#654321'}


@pytest.mark.parametrize('remove_column', [False, True])
def test_removing_overriding_annotation_restores_rule_phases(
        viewer, app, monkeypatch, remove_column
    ):
    configure(viewer, app, [annot_state(overrides=True)])
    toggle(viewer, 'Arrested', 1)
    app.processEvents()
    assert phases(viewer, 1) == ['Arrested'] * 4
    texts = []

    class FakeMessageBox:
        def __init__(self, *args, **kwargs):
            self.cancel = False

        def question(self, parent, title, text, buttonsTexts=()):
            texts.append(text)
            buttons = tuple(object() for _ in buttonsTexts)
            # 'Remove only button' or 'Remove also column with annotations'
            self.clickedButton = buttons[2 if remove_column else 1]
            return buttons

    monkeypatch.setattr(
        gui_custom_annotations.widgets, 'myMessageBox', FakeMessageBox
    )
    apply = spy(viewer, monkeypatch, 'applyPhaseTranslator')
    viewer.removeCustomAnnotButton(viewer.customAnnotButtonByName('Arrested'))
    assert 'overrides phases' in texts[0]
    assert 'phase rules again' in texts[0]
    assert ('Arrested' in viewer.acdc_df) is not remove_column
    app.processEvents()
    apply.assert_called_once_with(IDs={1}, quiet=True)
    for ID in (1, 2, 3):
        assert phases(viewer, ID) == RULE_PHASES
    assert 'phase_Arrested' not in viewer.acdc_df


@pytest.mark.parametrize('order', [
    ('Excluded', 'Arrested'), ('Arrested', 'Excluded')
])
def test_first_overriding_annotation_in_toolbar_order_wins(viewer, app, order):
    configure(viewer, app, [
        annot_state(order[0], 'A', overrides=True),
        annot_state('Big', 'F'),
        annot_state(order[1], 'D', overrides=True),
    ])
    assert viewer.phaseOverrideAnnotations() == list(order)
    toggle(viewer, order[1], 1)
    toggle(viewer, order[0], 1)
    toggle(viewer, order[1], 2)
    app.processEvents()
    assert phases(viewer, 1) == [order[0]] * 4
    assert phases(viewer, 2) == [order[1]] * 4
    assert phases(viewer, 3) == RULE_PHASES


def test_without_auto_apply_annotation_changes_evaluate_nothing(
        viewer, app, monkeypatch
    ):
    configure(viewer, app, [annot_state(overrides=True)], autoApply=False)
    apply = spy(viewer, monkeypatch, 'applyPhaseTranslator')
    toggle(viewer, 'Arrested', 1)
    viewer.replaceCustomAnnotations('Arrested', {2: [0]}, {2})
    app.processEvents()
    apply.assert_not_called()
    assert viewer._pendingPhaseIDs == {}
    assert phases(viewer, 1) == RULE_PHASES
    # Explicit evaluation applies the override
    assert viewer.selectedID == 1
    assert viewer.applyPhaseTranslator(allCells=False)
    assert phases(viewer, 1) == ['Arrested'] * 4
    assert phases(viewer, 2) == RULE_PHASES


def test_overriding_annotations_without_rules(viewer, app, monkeypatch):
    viewer.isDataLoaded = True
    configure(
        viewer, app, [annot_state(overrides=True)], rules=(),
        autoApply=False, evaluate=False
    )
    assert viewer.hasPhaseTranslation()
    toggle(viewer, 'Arrested', 1)
    assert viewer.applyPhasesOnDisplayedCell()
    assert phases(viewer, 1) == ['Arrested'] * 4
    assert phases(viewer, 2) == [''] * 4
    assert viewer.phaseTranslatorRules == []
    # Automatic evaluation works with overrides only
    viewer.autoApplyPhases = True
    toggle(viewer, 'Arrested', 2)
    app.processEvents()
    assert phases(viewer, 2) == ['Arrested'] * 4
    assert viewer.applyPhaseTranslator()
    assert phases(viewer, 3) == [''] * 4
    # Neither rules nor overriding annotations
    viewer.setCustomAnnotations([annot_state()])
    assert not viewer.hasPhaseTranslation()
    apply = spy(viewer, monkeypatch, 'applyPhaseTranslator')
    assert not viewer.applyPhasesOnDisplayedCell()
    apply.assert_not_called()
    assert 'No phase rules' in viewer.statusBar().currentMessage()
    viewer.schedulePhaseAutoApply([1])
    assert viewer._pendingPhaseIDs == {}


def second_position(viewer, annotated, rules=(CYCLE,)):
    """A loaded, not displayed position: the displayed table with the
    'Arrested' annotation on the cells `annotated`, phases evaluated."""
    table = viewer.acdc_df.copy()
    cellIDs = table.index.get_level_values('Cell_ID')
    table['Arrested'] = cellIDs.isin(annotated).astype(int)
    evaluation = phase_translator.evaluate(
        table, list(rules), annotation_overrides=['Arrested']
    )
    second = SimpleNamespace(
        acdc_df=evaluation.table, SizeT=4, SizeZ=1, chNames=['GFP'],
        pos_path=r'C:\experiment\Position_2',
        phaseTranslatorColumns=evaluation.columns
    )
    viewer.data.append(second)
    return second


def position_phases(posData, ID):
    return posData.acdc_df.xs(ID, level='Cell_ID')['phase'].tolist()


@pytest.mark.parametrize('action, autoApply', [
    ('switch_off', False), ('remove_button', True), ('remove_column', True),
])
def test_without_rules_last_override_gone_clears_its_phase(
        viewer, app, monkeypatch, action, autoApply
    ):
    """Nothing can evaluate the cells again: their phase is cleared, on every
    loaded position, also with automatic evaluation off."""
    viewer.isDataLoaded = True
    configure(
        viewer, app, [annot_state(overrides=True)], rules=(),
        autoApply=autoApply, evaluate=False
    )
    toggle(viewer, 'Arrested', 1)
    assert viewer.applyPhaseTranslator()
    second = second_position(viewer, [3], rules=())
    assert phases(viewer, 1) == ['Arrested'] * 4
    assert position_phases(second, 3) == ['Arrested'] * 4
    texts = []
    if action == 'switch_off':
        modify_annotation(
            viewer, monkeypatch, 'Arrested', annot_state(overrides=False)
        )
    else:
        class FakeMessageBox:
            def __init__(self, *args, **kwargs):
                self.cancel = False

            def question(self, parent, title, text, buttonsTexts=()):
                texts.append(text)
                buttons = tuple(object() for _ in buttonsTexts)
                self.clickedButton = buttons[
                    2 if action == 'remove_column' else 1
                ]
                return buttons

        monkeypatch.setattr(
            gui_custom_annotations.widgets, 'myMessageBox', FakeMessageBox
        )
        viewer.removeCustomAnnotButton(
            viewer.customAnnotButtonByName('Arrested')
        )
        assert 'is cleared' in texts[0]
    assert not viewer.hasPhaseTranslation()
    app.processEvents()
    for ID in (1, 2, 3):
        assert phases(viewer, ID) == [''] * 4
        assert position_phases(second, ID) == [''] * 4
    for table in (viewer.acdc_df, viewer.posData().acdc_df, second.acdc_df):
        assert 'phase_Arrested' not in table
    assert viewer._pendingPhaseClears == {}


def test_saving_tables_evaluates_pending_cells_of_other_positions(
        viewer, app, monkeypatch
    ):
    """Renaming an overriding annotation re-evaluates the other positions
    when they are displayed, or before their tables are saved."""
    configure(viewer, app, [annot_state(overrides=True)])
    second = second_position(viewer, [3])
    assert position_phases(second, 3) == ['Arrested'] * 4
    modify_annotation(
        viewer, monkeypatch, 'Arrested', annot_state('Stopped', overrides=True)
    )
    app.processEvents()
    # Still pending: Position_2 is not displayed
    assert position_phases(second, 3) == ['Arrested'] * 4
    assert id(second) in viewer._pendingPhaseIDs
    # Saving fails (no file path) but evaluates the pending cells first
    viewer.saveAcdcDfs([second])
    assert position_phases(second, 3) == ['Stopped'] * 4
    assert position_phases(second, 1) == RULE_PHASES
    assert 'phase_Arrested' not in second.acdc_df
    assert id(second) not in viewer._pendingPhaseIDs


def test_translator_dialog_gets_and_applies_overriding_annotations(
        viewer, app, monkeypatch, capsys
    ):
    configure(
        viewer, app, [annot_state('Big', 'D'), annot_state(overrides=True)],
        rules=(), autoApply=False, evaluate=False
    )
    toggle(viewer, 'Arrested', 1)
    received = {}

    class FakeDialog:
        # The viewer passes the overrides as keyword (trailing parameter)
        def __init__(self, rules, timepoints, annotations, options, **kwargs):
            received.update(kwargs, annotations=annotations)
            self.cancel = False
            self.rules = []
            self.autoApplyCheckbox = Mock(isChecked=Mock(return_value=False))
            self.sigApply = Mock()
            self.currentPhaseColors = Mock(return_value={})

        def exec_(self):
            # "Apply phase evaluation" on all cells without rules
            apply = self.sigApply.connect.call_args.args[0]
            apply([], True)

        def deleteLater(self):
            pass

    monkeypatch.setattr(
        measurements_viewer.apps_measurements_viewer,
        'PhaseTranslatorDialog', FakeDialog
    )
    capsys.readouterr()
    viewer.configurePhaseTranslator()
    assert received['annotationOverrides'] == ['Arrested']
    assert received['annotations'] == ['Big', 'Arrested']
    assert phases(viewer, 1) == ['Arrested'] * 4
    assert phases(viewer, 2) == [''] * 4
    out = capsys.readouterr().out
    assert 'annotations overriding phases: "Arrested"' in out
    viewer.measurementProcessingWarning.assert_not_called()


def test_override_respects_last_analyzed_frame(viewer, app):
    viewer.posData().lastAnalyzedFrame = 2
    configure(viewer, app, [annot_state(overrides=True)])
    toggle(viewer, 'Arrested', 1)
    app.processEvents()
    assert phases(viewer, 1) == ['Arrested'] * 3 + ['']
    assert viewer.acdc_df.xs(1, level='Cell_ID')['phase_Arrested'].tolist() == [
        True, True, True, False
    ]


def test_setting_annotations_in_bulk_evaluates_nothing(
        viewer, app, monkeypatch
    ):
    # Saved table: cell 1 annotated, phases evaluated in an earlier session
    viewer.acdc_df['Arrested'] = [1] * 4 + [0] * 8
    viewer.acdc_df['phase'] = 'G1'
    viewer.phaseTranslatorRules = [copy.deepcopy(CYCLE)]
    viewer.autoApplyPhases = True
    schedule = spy(viewer, monkeypatch, 'schedulePhaseAutoApply')
    viewer.setCustomAnnotations([annot_state(overrides=True)])
    viewer.setCustomAnnotations([
        annot_state('Excluded', 'D', overrides=True),
        annot_state(overrides=True),
    ])
    viewer.setCustomAnnotations([])
    app.processEvents()
    schedule.assert_not_called()
    assert viewer.acdc_df['phase'].eq('G1').all()


def test_adding_overriding_annotation_with_annotated_cells(
        viewer, app, monkeypatch
    ):
    viewer.acdc_df['Arrested'] = [1] * 4 + [0] * 8
    configure(viewer, app, [])
    assert phases(viewer, 1) == RULE_PHASES
    state = annot_state(overrides=True)
    state['symbolColor'] = QColor(*state['symbolColor'])
    dialog = SimpleNamespace(
        cancel=False, state=state, symbol='o', exec_=Mock(),
        sigDeleteSelecAnnot=SimpleNamespace(connect=Mock())
    )
    monkeypatch.setattr(
        gui_custom_annotations.apps, 'customAnnotationDialog',
        Mock(return_value=dialog)
    )
    monkeypatch.setattr(viewer, 'checkNameExists', lambda name: True)
    apply = spy(viewer, monkeypatch, 'applyPhaseTranslator')
    assert viewer.addCustomAnnotation() is not None
    app.processEvents()
    apply.assert_called_once_with(IDs={1}, quiet=True)
    assert phases(viewer, 1) == ['Arrested'] * 4
    assert phases(viewer, 2) == RULE_PHASES


def test_phase_color_map_lists_overrides_after_rule_phases(viewer):
    viewer.phaseTranslatorRules = [copy.deepcopy(CYCLE)]
    before = viewer.phaseColorMap()
    viewer.setCustomAnnotations([
        annot_state('Excluded', overrides=True),
        annot_state('Big', 'D'),
        annot_state('G1', 'F', overrides=True),
        annot_state('Arrested', 'G', overrides=True),
    ])
    colors = viewer.phaseColorMap()
    assert list(colors) == ['G1', 'S', 'G2', 'Excluded', 'Arrested']
    assert all(colors[name] == before[name] for name in before)


def prepare_configuration(viewer, tmp_path):
    posData = viewer.posData()
    posData.images_path = str(tmp_path)
    posData.basename = 'test_'
    posData.segm_npz_path = str(tmp_path / 'test_segm.npz')
    viewer.user_ch_name = 'GFP'
    viewer.allOverlayToolbuttons = {
        'GFP': SimpleNamespace(isChecked=lambda: True, setChecked=Mock())
    }
    return viewer.configuration()


def test_configuration_keeps_and_validates_overrides_phases(viewer, tmp_path):
    viewer.phaseTranslatorRules = [copy.deepcopy(CYCLE)]
    viewer.setCustomAnnotations([
        annot_state(overrides=True), annot_state('Big', 'D'),
        annot_state('Excluded', 'F', overrides=False),
    ])
    configuration = prepare_configuration(viewer, tmp_path)
    MeasurementsViewerWin.validateConfiguration(configuration)
    saved = configuration['custom_annotations']
    assert saved[0]['overridesPhases'] is True
    # No default key is added to annotations without the option
    assert 'overridesPhases' not in saved[1]
    assert saved[2]['overridesPhases'] is False
    for invalid in ('yes', 1, None):
        states = copy.deepcopy(saved)
        states[0]['overridesPhases'] = invalid
        with pytest.raises(ValueError, match='Overrides phases'):
            MeasurementsViewerWin.validateConfiguration(
                dict(configuration, custom_annotations=states)
            )
    # Overriding annotations are part of the rules' input/output clash check
    for name in ('phase', 'phase_G1'):
        states = copy.deepcopy(saved)
        states[0]['name'] = name
        with pytest.raises(ValueError, match='Phase output columns'):
            MeasurementsViewerWin.validateConfiguration(
                dict(configuration, custom_annotations=states)
            )
        # Fine without the option
        states[0]['overridesPhases'] = False
        MeasurementsViewerWin.validateConfiguration(
            dict(configuration, custom_annotations=states)
        )


def test_loading_configuration_keeps_saved_phases(
        viewer, app, tmp_path, monkeypatch
    ):
    configure(viewer, app, [
        annot_state(overrides=True), annot_state('Big', 'D')
    ])
    toggle(viewer, 'Arrested', 1)
    app.processEvents()
    assert phases(viewer, 1) == ['Arrested'] * 4
    configuration = prepare_configuration(viewer, tmp_path)
    assert configuration['auto_apply_phases'] is True
    filepath = tmp_path / 'state.json'
    filepath.write_text(json.dumps(configuration))
    posData = viewer.posData()
    monkeypatch.setattr(
        measurements_viewer, 'get_channels_and_segm_endnames',
        Mock(return_value=(['GFP'], ['segm']))
    )
    monkeypatch.setattr(
        measurements_viewer_utils, 'load_position', Mock(return_value=posData)
    )
    monkeypatch.setattr(viewer, 'setData', Mock())
    monkeypatch.setattr(viewer, 'setOverlayItemsOpacities', Mock())
    monkeypatch.setattr(viewer, 'askSaveBeforeLoading', lambda: True)
    monkeypatch.setattr(
        viewer, 'selectConfigurationInput',
        lambda config: viewer.configurationInputOptions(config)[0]
    )
    schedule = spy(viewer, monkeypatch, 'schedulePhaseAutoApply')
    assert load_configuration(viewer, filepath=str(filepath))
    app.processEvents()
    # Removing the previous annotations and adding the loaded ones does not
    # re-evaluate the phases of the annotated cells
    schedule.assert_not_called()
    assert viewer.phaseOverrideAnnotations() == ['Arrested']
    states = viewer.customAnnotConfiguration()
    assert states[0]['overridesPhases'] is True
    assert 'overridesPhases' not in states[1]
    assert phases(viewer, 1) == ['Arrested'] * 4
    assert phases(viewer, 2) == RULE_PHASES
