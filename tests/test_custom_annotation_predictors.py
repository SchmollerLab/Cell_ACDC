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

from cellacdc import apps, gui_custom_annotations, measurements_viewer
from cellacdc.custom_annotation_predictors import (
    CustomAnnotationPredictor, DEFAULT_SHORTCUTS
)
from cellacdc.measurements_viewer import MeasurementsViewerWin
from cellacdc.timepoint_predictors import CellFilter, TimepointPredictor


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
    dialog = apps.CustomAnnotationPredictorEquationDialog(
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
    assert kwargs['state']['shortcut'] == 'S'
    assert kwargs['existingNames'] == ['Big']
    assert kwargs['predictorOptions']['SizeT'] == 4
    button = viewer.customAnnotButtonByName('Big')
    kwargs = viewer.customAnnotationDialogKwargs(
        state=viewer.customAnnotDict[button]['state']
    )
    assert 'state' not in kwargs
    assert kwargs['existingNames'] == []
    assert 'A' not in kwargs['reservedShortcuts']


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
        measurements_viewer, 'load_position', Mock(return_value=posData)
    )
    monkeypatch.setattr(viewer, 'setData', Mock())
    monkeypatch.setattr(viewer, 'setOverlayItemsOpacities', Mock())
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
