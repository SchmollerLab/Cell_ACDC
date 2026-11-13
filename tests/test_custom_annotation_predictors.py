import json
import os
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from qtpy.QtCore import QPoint
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import QApplication, QLabel

from cellacdc import apps, apps_measurements_viewer, gui_custom_annotations, measurements_viewer
from cellacdc import measurements_viewer_utils
from cellacdc.custom_annotation_predictors import (
    CustomAnnotationPredictor, DEFAULT_SHORTCUTS
)
from cellacdc.measurements_viewer import MeasurementsViewerWin
from cellacdc.timepoint_predictors import (
    CellFilter, TIMEPOINT_PREFIX, TimepointPredictor
)


load_configuration = MeasurementsViewerWin.loadConfiguration


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def acdc_df():
    return pd.DataFrame({
        'frame_i': [0, 1, 2, 3, 0, 1, 2, 3],
        'Cell_ID': [1, 1, 1, 1, 2, 2, 2, 2],
        'area': [10., 20., 15., 30., 5., 6., 7., 4.],
        'stage': ['G1', 'S', 'S', 'G1', 'G1', 'G1', 'G1', 'G1'],
    }).set_index(['frame_i', 'Cell_ID'])


def equation(text, **kwargs):
    return {'type': 'equation', 'equation': text, **kwargs}


# --- Strict boolean results -------------------------------------------------
@pytest.mark.parametrize('values, expected', [
    (np.array([True, False, True]), [True, False, True]),
    (pd.Series([True, False, True]), [True, False, True]),
    (pd.array([True, pd.NA, True], dtype='boolean'), [True, False, True]),
    (np.array([True, None, False], dtype=object), [True, False, False]),
    (True, [True, True, True]),
    (np.bool_(False), [False, False, False]),
])
def test_strict_boolean_values_accepts_booleans(values, expected):
    result = CellFilter.strict_boolean_values(values, 3)
    assert result.dtype == bool
    assert result.tolist() == expected


@pytest.mark.parametrize('values', [
    np.array([1, 0, 1]), np.array([1., 0., 1.]), 1, 0.0, 'True',
    np.array(['a', 'b', 'c'], dtype=object), np.array([True, False]),
    pd.DataFrame({'a': [True, False, True]}),
])
def test_strict_boolean_values_rejects_non_booleans(values):
    with pytest.raises(ValueError):
        CellFilter.strict_boolean_values(values, 3)


# --- Predictors -------------------------------------------------------------
def test_equation_defaults_to_all_frames_true(acdc_df):
    predicted = CustomAnnotationPredictor.predict(
        acdc_df, equation('area > 12'), SizeT=4
    )
    assert predicted == {}


@pytest.mark.parametrize('mode, expected', [
    ('all', {}), ('none', {2: [0, 1, 2, 3]}),
    ('most', {1: [0, 1, 2, 3]}),
])
def test_equation_consolidation(acdc_df, mode, expected):
    assert CustomAnnotationPredictor.predict(
        acdc_df, equation('area > 12', consolidation=mode), SizeT=4
    ) == expected


@pytest.mark.parametrize('values, mode, expected', [
    ([True, True], 'all', True),
    ([True, False], 'all', False),
    ([False, False], 'none', True),
    ([True, False], 'none', False),
    ([True, False], 'most', False),
    ([True, True, False], 'most', True),
    ([True, False, False], 'most', False),
    (True, 'all', True), (False, 'all', False),
    (True, 'most', True), (False, 'none', True),
])
def test_boolean_consolidation_thresholds(values, mode, expected):
    length = len(values) if isinstance(values, list) else 4
    assert CustomAnnotationPredictor.consolidate(values, length, mode) is expected


def test_invalid_consolidation_is_rejected():
    with pytest.raises(ValueError, match='consolidation'):
        CustomAnnotationPredictor.validate(
            equation('area > 12', consolidation='any')
        )


def test_equation_with_cell_scalar_is_broadcast(acdc_df):
    predicted = CustomAnnotationPredictor.predict(
        acdc_df, equation('MAX(area) > 25'), SizeT=4
    )
    assert predicted == {1: [0, 1, 2, 3]}


@pytest.mark.parametrize('text', ['area', 'area * 0 + 1', 'stage'])
def test_equation_rejects_non_boolean_results(acdc_df, text):
    with pytest.raises(ValueError):
        CustomAnnotationPredictor.predict(acdc_df, equation(text), SizeT=4)


def test_equation_restricted_to_IDs_and_clipped_to_SizeT(acdc_df):
    predicted = CustomAnnotationPredictor.predict(
        acdc_df, equation('area > 0'), SizeT=2, IDs=[2]
    )
    assert predicted == {2: [0, 1]}


def test_main_cell_filter(acdc_df):
    mainFilter = {'condition': 'area > 25', 'condition_mode': 'any'}
    predicted = CustomAnnotationPredictor.predict(
        acdc_df, equation('area > 0', main_cell_filter=mainFilter), SizeT=4
    )
    assert predicted == {1: [0, 1, 2, 3]}
    IDs = TimepointPredictor.main_cell_filter_IDs(
        acdc_df, mainFilter, SizeT=4
    )
    assert set(IDs) == {1}


def test_file_predictor(tmp_path, acdc_df):
    path = tmp_path / 'predictor.py'
    path.write_text(
        'def predictor(df, entire_df=None):\n'
        '    assert entire_df is not None\n'
        '    return df["stage"] == "S"\n'
    )
    predicted = CustomAnnotationPredictor.predict(
        acdc_df, {'type': 'file', 'filepath': str(path)}, SizeT=4
    )
    assert predicted == {}
    assert CustomAnnotationPredictor.predict(
        acdc_df, {'type': 'file', 'filepath': str(path), 'consolidation': 'none'},
        SizeT=4
    ) == {2: [0, 1, 2, 3]}
    assert CustomAnnotationPredictor.predict(
        acdc_df, {'type': 'file', 'filepath': str(path), 'consolidation': 'most'},
        SizeT=4
    ) == {}
    path.write_text('def predictor(df):\n    return 1\n')
    with pytest.raises(ValueError, match='boolean'):
        CustomAnnotationPredictor.predict(
            acdc_df, {'type': 'file', 'filepath': str(path)}, SizeT=4
        )


def test_validate_and_describe():
    CustomAnnotationPredictor.validate({'type': 'none'})
    with pytest.raises(ValueError):
        CustomAnnotationPredictor.validate(equation(' '))
    with pytest.raises(ValueError):
        CustomAnnotationPredictor.validate({'type': 'file', 'filepath': ''})
    assert 'area > 1' in CustomAnnotationPredictor.describe(equation('area > 1'))
    assert CustomAnnotationPredictor.describe(None).startswith('None')
    assert DEFAULT_SHORTCUTS[:4] == ('A', 'S', 'D', 'F')


# --- Measurements viewer ----------------------------------------------------
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
    window.data = [SimpleNamespace(
        SizeT=4, SizeZ=1, frame_i=0, segm_data=labels, chNames=['GFP'],
        pos_path=r'C:\experiment\Position_1'
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
    window.plotMeasurements()
    yield window
    window.reinitCustomAnnot()
    window.deleteLater()
    app.processEvents()


def annot_state(name='Big', shortcut='A', predictor=None, autoApply=False):
    state = {
        'type': 'Single time-point', 'name': name, 'symbol': "'o' circle",
        'shortcut': shortcut, 'description': '', 'keepActive': True,
        'isHideChecked': True, 'symbolColor': [255, 0, 0, 255],
    }
    if predictor is not None:
        state['predictor'] = predictor
        state['autoApply'] = autoApply
    return state


@pytest.mark.parametrize('editing', [False, True])
@pytest.mark.parametrize('predictor_options', [None, {}])
def test_annotation_dialog_has_no_type_dropdown(app, editing, predictor_options):
    dialog = apps.customAnnotationDialog(
        {}, state=annot_state() if editing else None,
        predictorOptions=predictor_options
    )
    try:
        assert not hasattr(dialog, 'typeCombobox')
        assert not any(
            label.text().strip() == 'Type:'
            for label in dialog.findChildren(QLabel)
        )
        dialog.nameWidget.widget.setText('Big')
        dialog.cancel = False
        dialog.clickedButton = dialog.okButton
        event = Mock()
        dialog.closeEvent(event)
        event.ignore.assert_not_called()
        assert not dialog.cancel
        assert dialog.state['type'] == 'Single time-point'
        assert dialog.state['name'] == 'Big'
        assert 'Type:' not in dialog.toolTip
    finally:
        dialog.deleteLater()
        app.processEvents()


def test_annotation_dialog_loads_legacy_saved_annotation(app, monkeypatch):
    selection = Mock(
        cancel=False, selectedItemsText=['Big'], _additionalButtons=[]
    )
    selection.listBox.count.return_value = 1
    monkeypatch.setattr(apps.widgets, 'QDialogListbox', Mock(return_value=selection))
    dialog = apps.customAnnotationDialog({'Big': annot_state()})
    try:
        dialog.loadSavedAnnot()
        assert dialog.nameWidget.widget.text() == 'Big'
        assert dialog.shortcutWidget.widget.text() == 'A'
        dialog.cancel = False
        dialog.clickedButton = dialog.okButton
        event = Mock()
        dialog.closeEvent(event)
        event.ignore.assert_not_called()
        assert not dialog.cancel
        assert dialog.state['type'] == 'Single time-point'
    finally:
        dialog.deleteLater()
        app.processEvents()


def test_load_saved_annotation_restores_predictor_and_options(
        app, monkeypatch
    ):
    """'Load annotation...' restores the predictor, auto-apply and toggle
    options, like editing a saved state does (regression)."""
    saved = annot_state(
        predictor=equation('area > 12', consolidation='most'), autoApply=True
    )
    saved['keepActive'] = False
    saved['isHideChecked'] = False
    saved['overridesPhases'] = True
    selection = Mock(
        cancel=False, selectedItemsText=['Big'], _additionalButtons=[]
    )
    selection.listBox.count.return_value = 1
    monkeypatch.setattr(
        apps.widgets, 'QDialogListbox', Mock(return_value=selection)
    )
    dialog = apps.customAnnotationDialog({'Big': saved}, predictorOptions={})
    try:
        group = dialog.predictorGroupbox
        assert group.predictorType() != 'equation'
        assert not group.autoApplyCheckBox.isChecked()
        assert not dialog.overridesPhasesToggle.isChecked()
        dialog.loadSavedAnnot()
        assert group.predictorType() == 'equation'
        assert group.equationLineEdit.text() == 'area > 12'
        assert group.autoApplyCheckBox.isChecked()
        assert not dialog.keepActiveToggle.isChecked()
        assert not dialog.hideAnnotTooggle.isChecked()
        assert dialog.overridesPhasesToggle.isChecked()
        dialog.nameWidget.widget.setText('Big2')
        dialog.cancel = False
        dialog.clickedButton = dialog.okButton
        event = Mock()
        dialog.closeEvent(event)
        event.ignore.assert_not_called()
        assert not dialog.cancel
        state = dialog.state
        assert state['name'] == 'Big2'
        assert state['predictor'] == equation(
            'area > 12', consolidation='most'
        )
        assert state['autoApply'] is True
        assert state['keepActive'] is False
        assert state['isHideChecked'] is False
        assert state['overridesPhases'] is True
    finally:
        dialog.deleteLater()
        app.processEvents()


def accept_annotation_dialog(dialog):
    """Run the OK path of `customAnnotationDialog.closeEvent`; returns
    the resulting state."""
    dialog.cancel = False
    dialog.clickedButton = dialog.okButton
    event = Mock()
    dialog.closeEvent(event)
    event.ignore.assert_not_called()
    assert not dialog.cancel
    return dialog.state


@pytest.mark.parametrize('overrides', [False, True])
def test_main_gui_annotation_dialog_has_no_overrides_phases_option(
        app, monkeypatch, overrides
    ):
    """The "Overrides phases" option is viewer-only: the main GUI dialog
    neither shows it nor adds the key, even when the state has it."""
    state = annot_state()
    if overrides:
        state['overridesPhases'] = True
    dialog = apps.customAnnotationDialog({}, state=state)
    try:
        assert dialog.overridesPhasesToggle is None
        assert not any(
            'Overrides phases' in label.text()
            for label in dialog.findChildren(QLabel)
        )
        info = Mock()
        monkeypatch.setattr(apps.widgets, 'myMessageBox', info)
        dialog.showOptionsInfo()
        assert 'Overrides phases' not in info.return_value.addText.call_args.args[0]
        assert set(accept_annotation_dialog(dialog)) == {
            'type', 'name', 'symbol', 'shortcut', 'description',
            'keepActive', 'isHideChecked', 'symbolColor'
        }
    finally:
        dialog.deleteLater()
        app.processEvents()


@pytest.mark.parametrize('saved', [None, False, True])
def test_viewer_annotation_dialog_overrides_phases_option(
        app, monkeypatch, saved
    ):
    """In the viewer the toggle is prefilled from the state (absent means
    False) and the state always gets a strict bool."""
    state = annot_state(predictor=equation('area > 12'))
    if saved is not None:
        state['overridesPhases'] = saved
    dialog = apps.customAnnotationDialog(
        {}, state=state, predictorOptions={}
    )
    try:
        toggle = dialog.overridesPhasesToggle
        assert toggle is not None
        assert toggle.isChecked() is bool(saved)
        assert any(
            label.text() == 'Overrides phases: '
            and 'first one' in label.toolTip()
            for label in dialog.findChildren(QLabel)
        )
        info = Mock()
        monkeypatch.setattr(apps.widgets, 'myMessageBox', info)
        dialog.showOptionsInfo()
        assert 'Overrides phases' in info.return_value.addText.call_args.args[0]
        assert accept_annotation_dialog(dialog)['overridesPhases'] is bool(saved)
        toggle.setChecked(not saved)
        assert accept_annotation_dialog(dialog)['overridesPhases'] is (not saved)
    finally:
        dialog.deleteLater()
        app.processEvents()


def test_new_viewer_annotation_does_not_override_phases(app):
    dialog = apps.customAnnotationDialog({}, predictorOptions={})
    try:
        assert not dialog.overridesPhasesToggle.isChecked()
        dialog.nameWidget.widget.setText('Big')
        assert accept_annotation_dialog(dialog)['overridesPhases'] is False
    finally:
        dialog.deleteLater()
        app.processEvents()


def phase_rule(name='Cycle', phases=('G1', 'S'), timepoints=('Start',)):
    return {
        'name': name, 'predictor': {'type': 'none'},
        'timepoints': [f'{TIMEPOINT_PREFIX}{tp}' for tp in timepoints],
        'annotations': [], 'phases': list(phases),
    }


def test_phase_translator_dialog_lists_annotation_overrides(app):
    rules = [phase_rule(), phase_rule('Fallback', ['Other'], ())]
    without = apps_measurements_viewer.PhaseTranslatorDialog(rules, [], [], {})
    dialog = apps_measurements_viewer.PhaseTranslatorDialog(
        rules, [], [], {}, phaseColors={'Excluded': '#123456'},
        annotationOverrides=('Excluded', 'Arrested', 'G1')
    )
    try:
        assert dialog.autoApplyCheckbox.text() == (
            'Automatically evaluate phases when time points or annotations '
            'change or phases are missing'
        )
        assert 'annotations' in dialog.autoApplyCheckbox.toolTip()
        assert dialog.annotationOverrides == ['Excluded', 'Arrested', 'G1']
        # Rule phases first (unchanged defaults), then the new override names
        assert list(dialog.phaseColorButtons) == [
            'G1', 'S', 'Other', 'Excluded', 'Arrested'
        ]
        rule_colors = without.currentPhaseColors()
        colors = dialog.currentPhaseColors()
        assert {name: colors[name] for name in rule_colors} == rule_colors
        assert colors['Excluded'] == '#123456'
        assert dialog.phaseColorButtons['Excluded'].color().name() == '#123456'
        assert 'overriding' in dialog.phaseColorButtons['Arrested'].toolTip()
        dialog.phaseColorButtons['Arrested'].setColor('#0000ff')
        assert dialog.currentPhaseColors()['Arrested'] == '#0000ff'
        assert without.overridesLabel.isHidden()
        assert not dialog.overridesLabel.isHidden()
        text = dialog.overridesLabel.text()
        for name in dialog.annotationOverrides:
            assert f'<b>{name}</b>' in text
        assert 'first one' in text
        assert 'Overrides phases' in apps_measurements_viewer.PHASE_TRANSLATOR_HELP
        # Removing every rule keeps the override colors and Apply enabled
        dialog.removeRule()
        dialog.removeRule()
        assert dialog.rules == []
        assert list(dialog.phaseColorButtons) == ['Excluded', 'Arrested', 'G1']
        assert dialog.currentPhaseColors()['Arrested'] == '#0000ff'
        assert dialog.applyCellButton.isEnabled()
        assert dialog.applyAllButton.isEnabled()
        applied = Mock()
        dialog.sigApply.connect(applied)
        dialog.applyAllButton.click()
        assert applied.call_args.args == ([], True)
    finally:
        without.deleteLater()
        dialog.deleteLater()
        app.processEvents()


def test_phase_translator_dialog_without_rules_or_overrides(app):
    dialog = apps_measurements_viewer.PhaseTranslatorDialog(
        [], [], [], {}, annotationOverrides=[]
    )
    single = apps_measurements_viewer.PhaseTranslatorDialog(
        [], [], [], {}, annotationOverrides=['Arrested']
    )
    try:
        assert not dialog.applyCellButton.isEnabled()
        assert not dialog.applyAllButton.isEnabled()
        assert dialog.overridesLabel.isHidden()
        assert dialog.phaseColorButtons == {}
        assert single.applyCellButton.isEnabled()
        assert list(single.phaseColorButtons) == ['Arrested']
        # "first one wins" only matters with several overrides
        assert 'first one' not in single.overridesLabel.text()
    finally:
        dialog.deleteLater()
        single.deleteLater()
        app.processEvents()


def test_predictor_dialog_consolidation_roundtrip(app):
    dialog = apps.customAnnotationDialog(
        {}, state=annot_state(predictor=equation('area > 12', consolidation='most')),
        predictorOptions={}
    )
    try:
        group = dialog.predictorGroupbox
        assert group.consolidationCombobox.currentData() == 'most'
        assert group.predictor()['consolidation'] == 'most'
        group.consolidationCombobox.setCurrentIndex(0)
        assert group.predictor()['consolidation'] == 'all'
    finally:
        dialog.deleteLater()
        app.processEvents()


@pytest.mark.parametrize('mode, cells', [('all', 0), ('none', 1), ('most', 1)])
def test_equation_editor_preview_uses_consolidation(app, acdc_df, mode, cells):
    dialog = apps_measurements_viewer.CustomAnnotationPredictorEquationDialog(
        [], False, False, acdc_df=acdc_df, equation='area > 12',
        SizeT=4, consolidation=mode
    )
    try:
        dialog.test_cb()
        assert dialog.statusLabel.text() == (
            f'Annotated {cells * 4} frame(s) of {cells} of 2 cells.'
        )
        assert dialog.okButton.isEnabled()
    finally:
        dialog.deleteLater()
        app.processEvents()


def annotated(viewer, name):
    values = viewer.acdc_df[name]
    return sorted(values.index[values.to_numpy() != 0].tolist())


def test_toolbar_and_column(viewer):
    assert viewer.annotateToolbar.isVisibleTo(viewer)
    viewer.setCustomAnnotations([annot_state()])
    button = viewer.customAnnotButtonByName('Big')
    assert button is not None
    assert viewer.acdc_df['Big'].tolist() == [0] * 8
    assert 'Big' in [
        item.text() for item in viewer.columnsModel.measurementItems()
    ]
    # The shortcut is assigned to window actions, not to the button
    assert button.shortcut().isEmpty()
    toggleAction, autoAction = viewer.customAnnotShortcutActions[button]
    assert toggleAction.shortcut() == QKeySequence('A')
    assert autoAction.shortcut() == QKeySequence('Shift+A')
    reserved = viewer.reservedShortcuts()
    assert 'A' in reserved and 'Shift+A' in reserved
    assert 'A' not in viewer.reservedShortcuts(excludeCustomAnnot=button)


def test_hide_annotations_menu_toggles_and_keeps_state(viewer, monkeypatch):
    viewer.setCustomAnnotations([annot_state()])
    button = viewer.customAnnotButtonByName('Big')
    expected = iter([True, False])

    def select_hide(menu, position):
        action = next(
            action for action in menu.actions()
            if action.text() == 'Hide annotations'
        )
        assert action.isChecked() == next(expected)
        action.trigger()

    monkeypatch.setattr(apps.widgets.QMenu, 'exec', select_hide)
    event = Mock()
    event.globalPos.return_value = QPoint()
    button.showContextMenu(event)
    assert not button.isHideChecked
    assert not viewer.customAnnotDict[button]['state']['isHideChecked']
    assert not viewer.savedCustomAnnot['Big']['isHideChecked']
    assert not viewer.posData().customAnnot['Big']['isHideChecked']
    assert not os.path.exists(gui_custom_annotations.custom_annot_path)
    button.showContextMenu(event)
    assert button.isHideChecked
    assert viewer.savedCustomAnnot['Big']['isHideChecked']


def test_hide_annotations_option_is_a_user_setting_per_annotation(
        viewer, monkeypatch
    ):
    states = [annot_state(), annot_state('Small', 'S')]
    viewer.setCustomAnnotations(states)
    big = viewer.customAnnotButtonByName('Big')
    # The button does not shadow `QObject.parent()`
    assert big.parent() is not None
    big.hideAction(False)
    saved = pd.read_csv(viewer.settings_csv_path, index_col='setting')
    assert saved.at['Big_annotation_hidden', 'value'] == 'False'
    assert 'Small_annotation_hidden' not in saved.index

    # Reloading annotations whose saved state hides them (e.g., from a
    # configuration) keeps the user's choice of each annotation
    viewer.setCustomAnnotations(states)
    big = viewer.customAnnotButtonByName('Big')
    small = viewer.customAnnotButtonByName('Small')
    assert not big.isHideChecked and small.isHideChecked
    assert not viewer.customAnnotDict[big]['state']['isHideChecked']
    assert viewer.customAnnotDict[small]['state']['isHideChecked']
    viewer.viewAllCustomAnnotAction.setChecked(False)
    assert viewer.customAnnotButtonsToDraw() == [big]
    checked = {}

    def record_hide(menu, position):
        checked[menu.parent().name] = next(
            action.isChecked() for action in menu.actions()
            if action.text() == 'Hide annotations'
        )

    monkeypatch.setattr(apps.widgets.QMenu, 'exec', record_hide)
    event = Mock()
    event.globalPos.return_value = QPoint()
    big.showContextMenu(event)
    small.showContextMenu(event)
    assert checked == {'Big': False, 'Small': True}


@pytest.mark.parametrize('hide', [False, True])
@pytest.mark.parametrize('show_all', [False, True])
def test_annotation_visibility_after_deactivation(viewer, monkeypatch, hide, show_all):
    state = annot_state()
    state['isHideChecked'] = hide
    viewer.setCustomAnnotations([state])
    viewer.viewAllCustomAnnotAction.setChecked(show_all)
    button = viewer.customAnnotButtonByName('Big')
    button.setChecked(True)
    clear = Mock()
    monkeypatch.setattr(viewer, 'clearScatterPlotCustomAnnotButton', clear)
    button.setChecked(False)
    if hide and not show_all:
        clear.assert_called_once_with(button)
    else:
        clear.assert_not_called()
        assert button in viewer.customAnnotButtonsToDraw()


def test_switching_annotation_tools_keeps_nonhidden_annotations(viewer, monkeypatch):
    state = annot_state()
    state['isHideChecked'] = False
    viewer.setCustomAnnotations([state, annot_state('Small', 'S')])
    first = viewer.customAnnotButtonByName('Big')
    second = viewer.customAnnotButtonByName('Small')
    first.setChecked(True)
    clear = Mock()
    monkeypatch.setattr(viewer, 'clearScatterPlotCustomAnnotButton', clear)
    second.setChecked(True)
    assert not first.isChecked()
    assert second.isChecked()
    assert first in viewer.customAnnotButtonsToDraw()
    assert not any(call.args == (first,) for call in clear.call_args_list)


def test_default_shortcut_skips_reserved(viewer):
    viewer.setCustomAnnotations([annot_state()])
    kwargs = viewer.customAnnotationDialogKwargs()
    # 'S' is skipped: its auto shortcut Shift+S shuffles the colormap
    assert 'S' in kwargs['reservedShortcuts']
    assert kwargs['state']['shortcut'] == 'D'
    assert kwargs['existingNames'] == ['Big']
    assert kwargs['predictorOptions']['SizeT'] == 4
    button = viewer.customAnnotButtonByName('Big')
    kwargs = viewer.customAnnotationDialogKwargs(
        state=viewer.customAnnotDict[button]['state']
    )
    assert 'state' not in kwargs
    assert kwargs['existingNames'] == []
    assert 'A' not in kwargs['reservedShortcuts']


def test_auto_shortcut_never_duplicates_another_shortcut(viewer):
    # 'S' (e.g., from an older configuration): Shift+S shuffles the colormap
    # --> no auto shortcut instead of an ambiguous one
    viewer.setCustomAnnotations([
        annot_state(), annot_state('Small', 'S'),
        annot_state('Tiny', 'Shift+Q')
    ])
    button = viewer.customAnnotButtonByName('Small')
    toggleAction, autoAction = viewer.customAnnotShortcutActions[button]
    assert toggleAction.shortcut() == QKeySequence('S')
    assert autoAction.shortcut().isEmpty()
    assert 'Randomly shuffle colormap' in button.toolTip()
    # A Shift shortcut gets no auto shortcut (it would be the same)
    button = viewer.customAnnotButtonByName('Tiny')
    toggleAction, autoAction = viewer.customAnnotShortcutActions[button]
    assert toggleAction.shortcut() == QKeySequence('Shift+Q')
    assert autoAction.shortcut().isEmpty()
    assert 'Auto: none' in button.toolTip()
    shortcuts = [
        action.shortcut().toString() for action in viewer.actions()
        if not action.shortcut().isEmpty()
    ]
    assert len(shortcuts) == len(set(shortcuts))
    # Q is reserved because Shift+Q is used by "Tiny"
    assert 'Q' in viewer.reservedShortcuts()
    assert 'Q' not in viewer.reservedShortcuts(excludeCustomAnnot=button)


def test_annotation_shortcut_already_used_is_not_set(viewer):
    # Configurations and 'Load annotation...' skip the dialog checks: two
    # actions sharing a key would make Qt trigger neither of them
    identifier = viewer.defaultTimepointIdentifier()
    identifier.shortcut = 'Q'
    timepointAction = viewer.addTimepointIdentifierItem(identifier)
    states = [annot_state(), annot_state('Small', 'Q')]
    viewer.setCustomAnnotations(states)
    button = viewer.customAnnotButtonByName('Small')
    toggleAction, autoAction = viewer.customAnnotShortcutActions[button]
    assert toggleAction.shortcut().isEmpty()
    assert autoAction.shortcut().isEmpty()
    # The note is wrapped to the tool tip width (`qutils.wrap_tooltip`)
    toolTip = ' '.join(button.toolTip().split())
    assert 'the shortcut "Q" (and its auto shortcut) is not set' in toolTip
    assert f'used by "{identifier.name}"' in toolTip
    assert timepointAction.shortcut() == QKeySequence('Q')
    # Reloading frees the keys of the removed annotations first
    viewer.setCustomAnnotations(states)
    button = viewer.customAnnotButtonByName('Big')
    toggleAction, autoAction = viewer.customAnnotShortcutActions[button]
    assert toggleAction.shortcut() == QKeySequence('A')
    assert autoAction.shortcut() == QKeySequence('Shift+A')
    assert 'is not set' not in button.toolTip()


@pytest.mark.parametrize('filename', sorted(
    name for name in os.listdir(measurements_viewer.PREMADE_CONFIGURATIONS_PATH)
    if name.endswith('.json')
))
def test_premade_shortcuts_do_not_collide(viewer, filename):
    path = os.path.join(measurements_viewer.PREMADE_CONFIGURATIONS_PATH, filename)
    with open(path) as file:
        template = json.load(file)
    items = (
        template.get('timepoint_identifiers', [])
        + template.get('custom_annotations', [])
    )
    shortcuts = []
    for item in items:
        shortcut = item.get('shortcut') or ''
        if not shortcut:
            continue
        shortcuts.append(QKeySequence(shortcut).toString())
        autoText = measurements_viewer.auto_shortcut_text(shortcut)
        if autoText:
            shortcuts.append(autoText)
    reserved = viewer.reservedShortcuts()
    assert not [shortcut for shortcut in shortcuts if shortcut in reserved]
    assert len(shortcuts) == len(set(shortcuts))


def test_annotation_symbols_are_above_image_layers(viewer):
    viewer.setCustomAnnotations([annot_state()])
    button = viewer.customAnnotButtonByName('Big')
    item = viewer.customAnnotDict[button]['scatterPlotItem']
    assert 0.57 < item.zValue() < 1


def test_shortcut_toggles_selected_cell_for_entire_video(viewer):
    viewer.setCustomAnnotations([annot_state()])
    button = viewer.customAnnotButtonByName('Big')
    toggleAction, _ = viewer.customAnnotShortcutActions[button]
    viewer.frame_i = 2
    toggleAction.trigger()
    assert annotated(viewer, 'Big') == [(frame, 1) for frame in range(4)]
    assert viewer.editedCustomAnnotCells()['Big'] == {1}
    assert viewer.customAnnotFrameIDs('Big') == [1]
    toggleAction.trigger()
    assert annotated(viewer, 'Big') == []


def test_right_click_annotation_with_active_button(viewer):
    viewer.setCustomAnnotations([annot_state()])
    button = viewer.customAnnotButtonByName('Big')
    button.setChecked(True)
    assert viewer.customAnnotButton is button
    viewer.posData().lab = viewer.data[0].segm_data[0]
    viewer.annotateClickedObject(21, 16)
    assert annotated(viewer, 'Big') == [(frame, 2) for frame in range(4)]
    viewer.setUncheckedAllCustomAnnotButtons()
    assert viewer.customAnnotButton is None


def test_shortcut_annotates_cell_absent_from_current_frame(viewer):
    # Cell 1 has no row at frame 3, but stays selected there
    viewer.acdc_df = viewer.acdc_df.drop(index=(3, 1))
    viewer.setCustomAnnotations([annot_state()])
    button = viewer.customAnnotButtonByName('Big')
    viewer.setSelectedID(1)
    viewer.frame_i = 3
    viewer.customAnnotShortcutTriggered(button)
    assert annotated(viewer, 'Big') == [(frame, 1) for frame in range(3)]
    assert viewer.editedCustomAnnotCells()['Big'] == {1}
    viewer.customAnnotShortcutTriggered(button)
    assert annotated(viewer, 'Big') == []


def test_predictor_reevaluation_and_overwrite_warning(viewer, monkeypatch):
    predictor = equation('area > 12', consolidation='most')
    viewer.setCustomAnnotations([annot_state(predictor=predictor)])
    button = viewer.customAnnotButtonByName('Big')
    # autoApply is False --> nothing predicted automatically
    assert annotated(viewer, 'Big') == []
    assert viewer.reevaluateAllCellsCustomAnnot(button)
    assert annotated(viewer, 'Big') == [(frame, 1) for frame in range(4)]
    
    viewer.frame_i = 0
    viewer.customAnnotShortcutTriggered(button)
    ask = Mock(return_value='skip')
    monkeypatch.setattr(viewer, 'askOverwriteManualTimepoints', ask)
    assert viewer.reevaluateAllCellsCustomAnnot(button)
    assert ask.call_args.kwargs['what'] == 'Custom annotations'
    assert ask.call_args.args[0] == {'Big': [1]}
    assert annotated(viewer, 'Big') == []
    
    ask.return_value = 'overwrite'
    assert viewer.reevaluateDisplayedCellCustomAnnot(button)
    assert annotated(viewer, 'Big') == [(frame, 1) for frame in range(4)]
    assert viewer.editedCustomAnnotCells()['Big'] == set()
    
    viewer.clearCustomAnnotations(button)
    assert annotated(viewer, 'Big') == []


def test_global_reevaluation_actions_use_expected_scope(viewer, monkeypatch):
    assert not viewer.reevaluateAllCustomAnnotationsAction.isEnabled()
    assert not viewer.reevaluateCellCustomAnnotationsAction.isEnabled()
    viewer.setCustomAnnotations([
        annot_state(predictor=equation('area > 12')),
        annot_state(
            name='Small', shortcut='S', predictor=equation('area < 8')
        ),
    ])
    assert viewer.reevaluateAllCustomAnnotationsAction.isEnabled()
    assert viewer.reevaluateCellCustomAnnotationsAction.isEnabled()

    reevaluate = Mock()
    monkeypatch.setattr(viewer, 'reevaluateCustomAnnotations', reevaluate)
    viewer.reevaluateAllCustomAnnotationsAction.trigger()
    reevaluate.assert_called_once_with(buttons=None)

    reevaluate.reset_mock()
    viewer.reevaluateCellCustomAnnotationsAction.trigger()
    reevaluate.assert_called_once_with(buttons=None, IDs=[1])

    viewer.setCustomAnnotations([annot_state()])
    assert not viewer.reevaluateAllCustomAnnotationsAction.isEnabled()
    assert not viewer.reevaluateCellCustomAnnotationsAction.isEnabled()


def test_auto_apply_runs_once_per_displayed_cell(viewer, monkeypatch):
    predict = Mock(wraps=CustomAnnotationPredictor.predict)
    monkeypatch.setattr(CustomAnnotationPredictor, 'predict', predict)
    viewer.setCustomAnnotations([annot_state(
        predictor=equation('area < 8'), autoApply=True
    )])
    # Selected cell 1 is predicted when the annotation is added
    assert predict.call_count == 1
    assert annotated(viewer, 'Big') == []
    viewer.setSelectedID(2)
    assert annotated(viewer, 'Big') == [(frame, 2) for frame in range(4)]
    viewer.setSelectedID(1)
    viewer.setSelectedID(2)
    assert predict.call_count == 2


def test_failed_auto_prediction_is_not_retried(viewer, monkeypatch):
    viewer.setCustomAnnotations([annot_state(
        predictor=equation('area'), autoApply=True
    )])
    assert viewer.measurementProcessingWarning.call_count == 1
    viewer.setSelectedID(2)
    viewer.setSelectedID(1)
    assert viewer.measurementProcessingWarning.call_count == 2


def test_modify_rename_and_remove(viewer, monkeypatch):
    viewer.setCustomAnnotations([annot_state()])
    button = viewer.customAnnotButtonByName('Big')
    viewer.customAnnotShortcutTriggered(button)
    newState = annot_state(name='Large', shortcut='D')
    newState['symbolColor'] = measurements_viewer.QColor(0, 255, 0)
    dialog = SimpleNamespace(
        cancel=False, state=newState, symbol='o', exec_=Mock(),
        sigDeleteSelecAnnot=SimpleNamespace(connect=Mock())
    )
    monkeypatch.setattr(
        gui_custom_annotations.apps, 'customAnnotationDialog', 
        Mock(return_value=dialog)
    )
    viewer.customAnnotModify(button)
    assert 'Big' not in viewer.acdc_df.columns
    assert annotated(viewer, 'Large') == [(frame, 1) for frame in range(4)]
    assert viewer.editedCustomAnnotCells()['Large'] == {1}
    toggleAction, autoAction = viewer.customAnnotShortcutActions[button]
    assert toggleAction.shortcut() == QKeySequence('D')
    
    viewer.removeCustomAnnotButton(button, askHow=False)
    assert button not in viewer.customAnnotShortcutActions
    assert viewer.customAnnotDict == {}
    # Only the button was removed, the column is kept
    assert 'Large' in viewer.acdc_df.columns


def test_removing_annotation_column_warns_about_dependents(viewer, monkeypatch):
    viewer.setCustomAnnotations([annot_state()])
    button = viewer.customAnnotButtonByName('Big')
    viewer.customAnnotShortcutTriggered(button)
    assert 'Big' in viewer.acdc_df.columns
    viewer.conditionalFormatting = {
        'area': [{'condition': 'Big > 0', 'color': '#ff0000'}],
        'Big': [{'condition': 'Big > 0', 'color': '#00ff00'}],
    }
    viewer.measurementsColors['Big'] = '#0000ff'
    texts = []

    class FakeMessageBox:
        def __init__(self, *args, **kwargs):
            self.cancel = False

        def question(self, parent, title, text, buttonsTexts=()):
            texts.append(text)
            buttons = tuple(object() for _ in buttonsTexts)
            # 'Remove also column with annotations'
            self.clickedButton = buttons[-1]
            return buttons

    monkeypatch.setattr(
        gui_custom_annotations.widgets, 'myMessageBox', FakeMessageBox
    )
    viewer.removeCustomAnnotButton(button)
    assert len(texts) == 1
    assert 'conditional formatting of "area"' in texts[0]
    assert 'will fail until edited' in texts[0]
    # The column's own formatting is removed with it, not reported
    assert 'conditional formatting of "Big"' not in texts[0]
    assert 'Big' not in viewer.acdc_df.columns
    assert 'Big' not in viewer.conditionalFormatting
    assert 'Big' not in viewer.measurementsColors
    assert 'area' in viewer.conditionalFormatting


def modify_annotation(viewer, monkeypatch, button, state):
    state = dict(state)
    state['symbolColor'] = measurements_viewer.QColor(*state['symbolColor'])
    dialog = SimpleNamespace(
        cancel=False, state=state, symbol='o', exec_=Mock(),
        sigDeleteSelecAnnot=SimpleNamespace(connect=Mock())
    )
    monkeypatch.setattr(
        gui_custom_annotations.apps, 'customAnnotationDialog',
        Mock(return_value=dialog)
    )
    viewer.customAnnotModify(button)


def test_predictor_edit_keeps_assignments_of_visited_cells(viewer, monkeypatch):
    ask = Mock(return_value='overwrite')
    monkeypatch.setattr(viewer, 'askOverwriteManualTimepoints', ask)
    viewer.setCustomAnnotations([annot_state(
        predictor=equation('area < 8'), autoApply=True
    )])
    button = viewer.customAnnotButtonByName('Big')
    viewer.setSelectedID(2)
    viewer.setSelectedID(1)
    viewer.customAnnotShortcutTriggered(button)
    expected = [(frame, ID) for ID in (1, 2) for frame in range(4)]
    assert sorted(annotated(viewer, 'Big')) == sorted(expected)
    # Assignments saved in an earlier session: edits are not tracked
    viewer.editedCustomAnnotCells()['Big'].clear()

    modify_annotation(viewer, monkeypatch, button, annot_state(
        predictor=equation('area > 100'), autoApply=True
    ))
    viewer.setSelectedID(2)
    viewer.setSelectedID(1)
    assert sorted(annotated(viewer, 'Big')) == sorted(expected)
    ask.assert_not_called()
    # Explicit re-evaluation applies the new predictor to visited cells
    assert viewer.reevaluateAllCellsCustomAnnot(button)
    assert annotated(viewer, 'Big') == []


def test_enabling_auto_apply_keeps_assignments_of_visited_cells(
        viewer, monkeypatch
    ):
    viewer.setCustomAnnotations([annot_state(
        predictor=equation('area < 8'), autoApply=False
    )])
    button = viewer.customAnnotButtonByName('Big')
    viewer.setSelectedID(2)
    viewer.setSelectedID(1)
    modify_annotation(viewer, monkeypatch, button, annot_state(
        predictor=equation('area < 8'), autoApply=True
    ))
    viewer.setSelectedID(2)
    viewer.setSelectedID(1)
    assert annotated(viewer, 'Big') == []
    # Explicit re-evaluation still applies the predictor
    assert viewer.reevaluateAllCellsCustomAnnot(button)
    assert annotated(viewer, 'Big') == [(frame, 2) for frame in range(4)]


def test_removal_note_lists_phase_rules_using_column_as_input(viewer):
    viewer.setCustomAnnotations([annot_state()])
    viewer.phaseTranslatorRules = [
        {'name': 'Boundary', 'timepoints': ['timepoint_A'],
         'annotations': [], 'predictor': None},
        {'name': 'Required', 'timepoints': [], 'annotations': ['Big'],
         'predictor': None},
    ]
    assert 'phase rule "Required"' in viewer.customAnnotColumnRemovalNote('Big')
    assert viewer.columnDependents(
        'timepoint_A', includePhaseInputs=True
    ) == ['phase rule "Boundary"']
    # Renaming rewrites boundaries/required annotations: not a dependent
    assert viewer.columnDependents('timepoint_A') == []
    assert viewer.columnDependents('Big', excludeOutput='Big') == []


def test_loading_existing_column_keeps_assignments_of_visited_cells(viewer):
    viewer.acdc_df['Big'] = [0, 0, 0, 0, 1, 1, 1, 1]
    viewer.acdc_df[measurements_viewer.VISITED_COLUMN] = True
    viewer.setCustomAnnotations([annot_state(
        predictor=equation('area > 100'), autoApply=True
    )])
    viewer.setSelectedID(2)
    viewer.setSelectedID(1)
    assert annotated(viewer, 'Big') == [(frame, 2) for frame in range(4)]


def test_rename_is_refused_while_other_predictors_use_the_column(viewer):
    viewer.setCustomAnnotations([
        annot_state(),
        annot_state('NotBig', 'D', predictor=equation('Big == 0')),
    ])
    viewer.conditionalFormatting['area'] = [
        {'condition': 'Big == 1', 'color': '#ff0000'}
    ]
    validate = viewer.customAnnotationDialogKwargs(
        state=viewer.customAnnotDict[viewer.customAnnotButtonByName('Big')]['state']
    )['predictorOptions']['validateDependencies']
    with pytest.raises(ValueError) as error:
        validate('Large', {'type': 'none'})
    assert 'custom annotation "NotBig"' in str(error.value)
    assert 'conditional formatting of "area"' in str(error.value)
    # Keeping the name (or editing the dependent itself) is allowed
    validate('Big', {'type': 'none'})
    viewer.validatePredictorDependencies('NotBig', 'NotLarge', equation('Big == 0'))


def test_rename_onto_existing_table_column_is_refused(viewer):
    # e.g., a column kept by 'Remove only button' or a measurement
    viewer.acdc_df['mitotic'] = 0
    viewer.setCustomAnnotations([annot_state()])
    validate = viewer.customAnnotationDialogKwargs(
        state=viewer.customAnnotDict[viewer.customAnnotButtonByName('Big')]['state']
    )['predictorOptions']['validateDependencies']
    for name in ('mitotic', 'area'):
        with pytest.raises(ValueError, match='already a column of the table'):
            validate(name, {'type': 'none'})
    validate('Big', {'type': 'none'})
    # A new annotation may reuse an existing (e.g., exported) column
    validate = viewer.customAnnotationDialogKwargs()['predictorOptions'][
        'validateDependencies'
    ]
    validate('mitotic', {'type': 'none'})
    assert list(viewer.acdc_df.columns).count('mitotic') == 1


def test_rename_moves_selection_and_formatting(viewer, monkeypatch):
    viewer.setCustomAnnotations([annot_state()])
    button = viewer.customAnnotButtonByName('Big')
    for item in viewer.columnsModel.measurementItems():
        if item.text() == 'Big':
            item.setCheckState(measurements_viewer.Qt.Checked)
    rules = [{'condition': 'area > 12', 'color': '#ff0000'}]
    viewer.conditionalFormatting['Big'] = rules
    color = measurements_viewer.QColor(0, 255, 0)
    viewer.measurementsColors['Big'] = color
    modify_annotation(viewer, monkeypatch, button, annot_state(name='Large'))
    assert 'Large' in viewer.checkedColumns()
    assert 'Big' not in viewer.checkedColumns()
    assert viewer.conditionalFormatting == {'Large': rules}
    assert viewer.measurementsColors == {'Large': color}


def test_reload_keeps_annotations(viewer, monkeypatch):
    viewer.setCustomAnnotations([annot_state()])
    button = viewer.customAnnotButtonByName('Big')
    viewer.customAnnotShortcutTriggered(button)
    posData = viewer.posData()
    previous = viewer.acdc_df
    posData.acdc_df = previous.drop(columns='Big')
    viewer.restoreTimepointColumns(posData, previous)
    assert posData.acdc_df.loc[(0, 1), 'Big'] == 1
    assert posData.acdc_df.xs(1, level='Cell_ID')['Big'].eq(1).all()


def test_legacy_frame_annotations_expand_to_entire_cell(viewer):
    viewer.acdc_df['Big'] = [0, 1, 0, 0, 0, 0, 0, 0]
    viewer.setCustomAnnotations([annot_state()])
    assert annotated(viewer, 'Big') == [(frame, 1) for frame in range(4)]


def test_load_legacy_definitions_does_not_overwrite_global_library(viewer, monkeypatch):
    path = gui_custom_annotations.custom_annot_path
    legacy = json.dumps({'Big': annot_state()})
    with open(path, 'w') as file:
        file.write(legacy)
    selection = Mock(
        cancel=False, selectedItemsText=['Big'], _additionalButtons=[]
    )
    monkeypatch.setattr(apps.widgets, 'QDialogListbox', Mock(return_value=selection))
    viewer.loadCustomAnnotations()
    button = viewer.customAnnotButtonByName('Big')
    assert button is not None
    viewer.customAnnotShortcutTriggered(button)
    button.hideAction(False)
    viewer.saveCustomAnnot()
    with open(path) as file:
        assert file.read() == legacy
    saved = viewer.customAnnotConfiguration()
    assert not saved[0]['isHideChecked']
    assert saved[0]['name'] == 'Big'


def test_export_contains_cellwide_annotation_column(viewer, tmp_path, monkeypatch):
    viewer.setCustomAnnotations([annot_state()])
    button = viewer.customAnnotButtonByName('Big')
    viewer.customAnnotShortcutTriggered(button)
    identifier = viewer.defaultTimepointIdentifier()
    viewer.addTimepointIdentifierItem(identifier)
    viewer.replaceTimepoints(identifier.uid, {1: [2]}, {1, 2})
    path = tmp_path / 'measurements.csv'
    monkeypatch.setattr(
        measurements_viewer.QFileDialog, 'getSaveFileName',
        Mock(return_value=(str(path), 'CSV (*.csv)'))
    )
    viewer.exportMeasurements()
    exported = pd.read_csv(path).set_index(['frame_i', 'Cell_ID'])
    assert exported.xs(1, level='Cell_ID')['Big'].eq(1).all()
    assert exported.xs(2, level='Cell_ID')['Big'].eq(0).all()
    column = viewer.timepointColumn(identifier)
    assert exported.index[exported[column]].tolist() == [(2, 1)]


def test_predictor_replacement_marks_entire_cell_not_only_result_frames(viewer):
    viewer.setCustomAnnotations([annot_state()])
    assert viewer.replaceCustomAnnotations('Big', {1: [2]}, {1})
    assert annotated(viewer, 'Big') == [(frame, 1) for frame in range(4)]


def test_configuration_roundtrip(viewer, tmp_path, monkeypatch):
    viewer.setCustomAnnotations([
        annot_state(predictor=equation('area > 12', consolidation='most')),
        annot_state(name='Small', shortcut='S'),
    ])
    posData = viewer.posData()
    posData.images_path = str(tmp_path)
    posData.basename = 'test_'
    posData.segm_npz_path = str(tmp_path / 'test_segm.npz')
    viewer.user_ch_name = 'GFP'
    viewer.allOverlayToolbuttons = {
        'GFP': SimpleNamespace(isChecked=lambda: True, setChecked=Mock())
    }
    configuration = viewer.configuration()
    MeasurementsViewerWin.validateConfiguration(configuration)
    saved = configuration['custom_annotations']
    assert [state['name'] for state in saved] == ['Big', 'Small']
    assert saved[0]['predictor'] == equation('area > 12', consolidation='most')
    filepath = tmp_path / 'state.json'
    filepath.write_text(json.dumps(configuration))
    
    viewer.setCustomAnnotations([])
    assert viewer.customAnnotDict == {}
    monkeypatch.setattr(
        measurements_viewer, 'get_channels_and_segm_endnames',
        Mock(return_value=(['GFP'], ['segm']))
    )
    monkeypatch.setattr(
        measurements_viewer_utils, 'load_position', Mock(return_value=posData)
    )
    monkeypatch.setattr(viewer, 'setData', Mock())
    monkeypatch.setattr(viewer, 'setOverlayItemsOpacities', Mock())
    monkeypatch.setattr(
        viewer, 'selectConfigurationInput',
        lambda config: viewer.configurationInputOptions(config)[0]
    )
    assert load_configuration(viewer, filepath=str(filepath))
    assert viewer.customAnnotNames() == ['Big', 'Small']
    button = viewer.customAnnotButtonByName('Small')
    toggleAction, _ = viewer.customAnnotShortcutActions[button]
    assert toggleAction.shortcut() == QKeySequence('S')
    
    legacy = dict(configuration)
    legacy.pop('custom_annotations')
    MeasurementsViewerWin.validateConfiguration(legacy)
    invalid_states = (
        {}, [{'name': ''}], [annot_state(), annot_state()],
        [dict(annot_state(), symbolColor='red')],
        [dict(annot_state(), predictor={'type': 'equation', 'equation': ''})],
    )
    for invalid in invalid_states:
        with pytest.raises(ValueError):
            MeasurementsViewerWin.validateConfiguration(
                dict(configuration, custom_annotations=invalid)
            )
