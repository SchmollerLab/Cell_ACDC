import json
import os
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QCloseEvent
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication

from cellacdc import apps, apps_measurements_viewer, measurements_viewer, timepoint_predictors
from cellacdc.custom_annotation_predictors import CustomAnnotationPredictor
from cellacdc.measurements_viewer import MeasurementsViewerWin, _TimepointLine, VISITED_COLUMN
from cellacdc.timepoint_predictors import TimepointIdentifier, TIMEPOINT_PREFIX

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


@pytest.fixture
def viewer(app, monkeypatch, acdc_df):
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
    window.centralWidget().setEnabled(True)
    window.initFramesNavigation()
    window.initIDsNavigation(ID=1)
    window.updateColumnsList()
    window.plotMeasurements()
    yield window
    window.reinitCustomAnnot()
    window.deleteLater()
    app.processEvents()


def lines(viewer):
    return [(line.uid, line.ID, line.frame_i) for line in viewer.timepointLines]


def timepoints(viewer, uid):
    return {
        ID: viewer.cellTimepoints(uid, ID) 
        for ID in sorted(viewer.timepointCellIDs(uid))
    }


def set_timepoints(viewer, uid, value):
    viewer.replaceTimepoints(uid, value, {1, 2})


def annotation_predictors(viewer, predictors, autoApply=False):
    viewer.setCustomAnnotations([
        {
            'name': name, 'symbol': "'o' circle",
            'symbolColor': [255, 0, 0, 255], 'shortcut': '',
            'predictor': {'type': 'equation', 'equation': equation},
            'autoApply': autoApply,
        }
        for name, equation in predictors
    ])


@pytest.mark.parametrize('predictor, expected', [
    ({'equation': f'`{TIMEPOINT_PREFIX}Bud emergence` & (Flag == 1)'},
     {f'{TIMEPOINT_PREFIX}Bud emergence': False, 'Flag': False}),
    ({'equation': 'stage == "Flag"'}, {'stage': False}),
    ({'equation': 'area > 0', 'main_cell_filter': {'condition': 'Flag == 1'}},
     {'area': False, 'Flag': False}),
    ({'equation': 'MAX(Flag_other) > 0',
      'other_cells': {'condition': 'Other == this_Target'}},
     {'Flag': True, 'Other': True, 'Target': False}),
    ({'equation': 'MAX_OTHER(Flag) > 0'}, {'Flag': True}),
    ({'equation': 'MAX_OTHER(`Flag name`) > 0'}, {'Flag name': True}),
    ({'equation': '(stage == "MAX_OTHER(Flag)") & (Flag == 1)'},
     {'stage': False, 'Flag': False}),
    ({'equation': 'area > 0', 'other_cells': {'condition': 'Flag == 1'}},
     {'area': False}),
])
def test_predictor_dependency_inputs(predictor, expected):
    assert timepoint_predictors.PredictorDependencies.inputs({
        'type': 'equation', **predictor
    }) == expected


def test_predictor_dependencies_ignore_files_and_order_target_closure():
    predictors = {
        'Last': {'type': 'equation', 'equation': 'First & Manual'},
        'First': {'type': 'equation', 'equation': 'area > 0'},
        'Manual': {'type': 'none'},
        'Unrelated': {'type': 'equation', 'equation': 'Unrelated'},
    }
    assert timepoint_predictors.PredictorDependencies.order(
        predictors, ['Last']
    ) == ['First', 'Manual', 'Last']
    assert timepoint_predictors.PredictorDependencies.inputs({
        'type': 'file', 'equation': 'Last'
    }) == {}


@pytest.mark.parametrize('predictors, cycle', [
    ({'A': 'A'}, 'A -> A'),
    ({'A': 'B', 'B': 'C', 'C': 'A'}, 'A -> B -> C -> A'),
])
def test_predictor_dependency_cycles(predictors, cycle):
    with pytest.raises(ValueError, match=cycle):
        timepoint_predictors.PredictorDependencies.order({
            name: {'type': 'equation', 'equation': equation}
            for name, equation in predictors.items()
        })


@pytest.mark.parametrize('automatic', [False, True])
def test_mixed_predictors_run_prerequisites_first(viewer, automatic):
    final = TimepointIdentifier(
        'Final', (0, 0, 255), '',
        {'type': 'equation', 'equation': '(Eligible == 1) & (frame_i > 1)',
         'mode': 'first_true'},
        autoApply=automatic
    )
    start = TimepointIdentifier(
        'Start', (255, 0, 0), '',
        {'type': 'equation', 'equation': 'area > 16', 'mode': 'rising_true'},
        autoApply=automatic
    )
    viewer.addTimepointIdentifierItem(final)
    viewer.addTimepointIdentifierItem(start)
    viewer.acdc_df[VISITED_COLUMN] = False
    annotation_predictors(
        viewer, [('Eligible', 'any(timepoint_Start)')], autoApply=automatic
    )
    if not automatic:
        assert viewer.reevaluateTimepoints([final.uid], IDs=[1])
    assert viewer.cellTimepoints(start.uid, 1) == [1, 3]
    assert viewer.cellTimepoints(final.uid, 1) == [2]
    assert viewer.acdc_df.xs(1, level='Cell_ID')['Eligible'].tolist() == [1]*4
    assert viewer.cellTimepoints(start.uid, 2) == []
    viewer.measurementProcessingWarning.assert_not_called()


@pytest.mark.parametrize('automatic', [False, True])
def test_other_cell_predictor_prerequisites_cover_all_cells(viewer, automatic):
    final = TimepointIdentifier(
        'Final', (0, 0, 255), '',
        {
            'type': 'equation', 'equation': 'MAX(Flag_other) > 0',
            'mode': 'first_true',
            'other_cells': timepoint_predictors.CellFilter.default_other(),
        },
        autoApply=automatic
    )
    viewer.addTimepointIdentifierItem(final)
    viewer.acdc_df[VISITED_COLUMN] = False
    annotation_predictors(
        viewer, [('Flag', 'area < 8')], autoApply=automatic
    )
    if not automatic:
        assert viewer.reevaluateTimepoints([final.uid], IDs=[1])
    assert viewer.cellTimepoints(final.uid, 1) == [0]
    assert viewer.acdc_df.xs(2, level='Cell_ID')['Flag'].tolist() == [1]*4
    assert viewer.cellTimepoints(final.uid, 2) == []
    viewer.measurementProcessingWarning.assert_not_called()


def test_automatic_prediction_keeps_edited_prerequisite_cells(viewer, monkeypatch):
    final = TimepointIdentifier(
        'Final', (0, 0, 255), '',
        {
            'type': 'equation', 'equation': 'MAX(Flag_other) > 0',
            'mode': 'first_true',
            'other_cells': timepoint_predictors.CellFilter.default_other(),
        },
        autoApply=True
    )
    viewer.addTimepointIdentifierItem(final)
    viewer.acdc_df[VISITED_COLUMN] = False
    annotation_predictors(viewer, [('Flag', 'area < 8')], autoApply=True)
    assert viewer.acdc_df.xs(2, level='Cell_ID')['Flag'].tolist() == [1]*4
    # Manual correction of the prerequisite on a cell not viewed yet
    viewer.acdc_df.loc[(slice(None), 2), 'Flag'] = 0
    viewer.editedCustomAnnotCells()['Flag'].add(2)
    ask = Mock(return_value='overwrite')
    monkeypatch.setattr(viewer, 'askOverwriteManualTimepoints', ask)
    annotPredict = Mock(wraps=CustomAnnotationPredictor.predict)
    monkeypatch.setattr(CustomAnnotationPredictor, 'predict', annotPredict)
    predict = Mock(wraps=timepoint_predictors.TimepointPredictor.predict)
    monkeypatch.setattr(timepoint_predictors.TimepointPredictor, 'predict', predict)
    viewer.setSelectedID(2)
    assert predict.call_args.kwargs['IDs'] == {2}
    viewer.acdc_df.loc[(slice(None), 1), VISITED_COLUMN] = False
    viewer.timepointAutoPredictions.clear()
    viewer.setSelectedID(1)
    assert predict.call_args.kwargs['IDs'] == {1}
    ask.assert_not_called()
    # Already predicted prerequisite cells are not recomputed
    annotPredict.assert_not_called()
    assert viewer.acdc_df.xs(2, level='Cell_ID')['Flag'].tolist() == [0]*4
    assert viewer.editedCustomAnnotCells()['Flag'] == {2}
    assert viewer.cellTimepoints(final.uid, 1) == []


def test_all_cells_auto_prerequisite_keeps_visited_cells_of_loaded_table(
        viewer, monkeypatch
    ):
    start = TimepointIdentifier(
        'A', (255, 0, 0), '',
        {'type': 'equation', 'equation': 'area > 16', 'mode': 'rising_true'},
        autoApply=True
    )
    final = TimepointIdentifier(
        'B', (0, 0, 255), '',
        {
            'type': 'equation', 'equation': 'MAX(timepoint_A_other) > 0',
            'mode': 'first_true',
            'other_cells': timepoint_predictors.CellFilter.default_other(),
        },
        autoApply=True
    )
    viewer.addTimepointIdentifierItem(start)
    viewer.addTimepointIdentifierItem(final)
    # Table saved in an earlier session: cell 1 visited with a manual A.
    viewer.acdc_df[VISITED_COLUMN] = False
    viewer.acdc_df.loc[(slice(None), 1), VISITED_COLUMN] = True
    viewer.replaceTimepoints(start.uid, {1: [0]}, {1, 2})
    # Session-only state is empty after (re)loading
    viewer.timepointAutoPredictions.clear()
    viewer.editedTimepointCells().clear()
    viewer.seedVisitedTimepointPredictions()
    predict = Mock(wraps=timepoint_predictors.TimepointPredictor.predict)
    monkeypatch.setattr(timepoint_predictors.TimepointPredictor, 'predict', predict)
    viewer.setSelectedID(2)
    # A is predicted on all cells not visited yet, B on the new cell only
    assert [call.kwargs['IDs'] for call in predict.call_args_list] == [{2}, {2}]
    assert viewer.cellTimepoints(start.uid, 1) == [0]
    viewer.measurementProcessingWarning.assert_not_called()


def test_new_identifier_with_existing_column_keeps_visited_cells(
        viewer, monkeypatch
    ):
    viewer.acdc_df['timepoint_A'] = False
    viewer.acdc_df.loc[(0, 1), 'timepoint_A'] = True
    viewer.acdc_df[VISITED_COLUMN] = False
    viewer.acdc_df.loc[(slice(None), 1), VISITED_COLUMN] = True
    final = TimepointIdentifier(
        'B', (0, 0, 255), '',
        {
            'type': 'equation', 'equation': 'MAX(timepoint_A_other) > 0',
            'mode': 'first_true',
            'other_cells': timepoint_predictors.CellFilter.default_other(),
        },
        autoApply=True
    )
    viewer.addTimepointIdentifierItem(final)
    monkeypatch.setattr(viewer, 'execTimepointIdentifierDialog', Mock(
        return_value={
            'name': 'A', 'color': (255, 0, 0), 'shortcut': '', 'autoApply': True,
            'predictor': {
                'type': 'equation', 'equation': 'area > 16',
                'mode': 'rising_true'
            }
        }
    ))
    viewer.addTimepointIdentifier()
    start = viewer.timepointIdentifiers[-1]
    viewer.setSelectedID(2)
    assert viewer.cellTimepoints(start.uid, 1) == [0]


def column_names(viewer):
    return [item.text() for item in viewer.columnsModel.measurementItems()]


def test_timepoint_column_list_and_axes_follow_add_rename_remove(viewer, monkeypatch):
    values = {
        'name': 'Bud', 'color': (255, 0, 0), 'shortcut': '', 'autoApply': False,
        'predictor': {'type': timepoint_predictors.PREDICTOR_NONE},
    }
    dialog = Mock(return_value=dict(values))
    monkeypatch.setattr(viewer, 'execTimepointIdentifierDialog', dialog)
    identifier = viewer.addTimepointIdentifier()
    assert 'timepoint_Bud' in column_names(viewer)

    viewer.measurementsModeCombobox.setCurrentIndex(1)
    viewer.measurementsAxes = {'X': 'area', 'Y': 'timepoint_Bud'}
    viewer.conditionalFormatting['timepoint_Bud'] = [
        {'condition': 'area > 1', 'color': '#ff0000'}
    ]
    dialog.return_value = dict(values, name='Bud2')
    viewer.editTimepointIdentifier(identifier.uid)
    names = column_names(viewer)
    assert 'timepoint_Bud2' in names and 'timepoint_Bud' not in names
    assert viewer.measurementsAxes == {'X': 'area', 'Y': 'timepoint_Bud2'}
    assert list(viewer.conditionalFormatting) == ['timepoint_Bud2']
    viewer.plotMeasurements()

    viewer.removeTimepointIdentifier(identifier.uid, ask=False)
    assert 'timepoint_Bud2' not in column_names(viewer)
    assert viewer.measurementsAxes['Y'] == ''
    viewer.plotMeasurements()


def test_removing_identifier_drops_its_color_and_formatting(
        viewer, monkeypatch
    ):
    point = TimepointIdentifier(
        'Bud', (255, 0, 0), '',
        {'type': 'equation', 'equation': 'area > 16', 'mode': 'first_true'}
    )
    viewer.addTimepointIdentifierItem(point)
    viewer.measurementsColors['timepoint_Bud'] = (255, 0, 0)
    viewer.conditionalFormatting['timepoint_Bud'] = [
        {'condition': 'area > 1', 'color': '#ff0000'}
    ]
    assert 'conditional formatting of "timepoint_Bud"' in (
        viewer.columnDependents('area')
    )
    # The removal note does not list the column's own formatting;
    # "Remove" is never clicked --> cancelled
    question = Mock(return_value=(None, object()))
    monkeypatch.setattr(
        measurements_viewer.widgets.myMessageBox, 'question', question
    )
    viewer.removeTimepointIdentifier(point.uid)
    assert 'conditional formatting' not in question.call_args.args[2]

    viewer.removeTimepointIdentifier(point.uid, ask=False)
    assert 'timepoint_Bud' not in viewer.conditionalFormatting
    assert 'timepoint_Bud' not in viewer.measurementsColors
    assert viewer.columnDependents('area') == []


def test_comparison_plot_ignores_axis_missing_from_table(viewer):
    viewer.measurementsModeCombobox.setCurrentIndex(1)
    viewer.measurementsAxes = {'X': 'area', 'Y': 'timepoint_Missing'}
    viewer.plotMeasurements()
    assert 'Select X and Y' in viewer.measurementsPlot.plotItem.titleLabel.text


def test_timepoint_rename_is_refused_while_referenced(viewer, monkeypatch):
    point = TimepointIdentifier(
        'Point', (255, 0, 0), '',
        {'type': 'equation', 'equation': 'area > 16', 'mode': 'first_true'}
    )
    viewer.addTimepointIdentifierItem(point)
    annotation_predictors(viewer, [('Flag', 'any(timepoint_Point)')])
    with pytest.raises(ValueError, match='custom annotation "Flag"'):
        viewer.validatePredictorDependencies(
            'timepoint_Point', 'timepoint_Moved', point.predictor
        )
    viewer.validatePredictorDependencies(
        'timepoint_Point', 'timepoint_Point', point.predictor
    )
    # Removal warns about the dependents
    # The "Remove" button is never clicked --> cancelled
    question = Mock(return_value=(None, object()))
    monkeypatch.setattr(measurements_viewer.widgets.myMessageBox, 'question', question)
    viewer.removeTimepointIdentifier(point.uid)
    assert 'custom annotation "Flag"' in question.call_args.args[2]
    assert viewer.timepointIdentifier(point.uid) is point


def test_timepoint_rename_is_refused_while_read_by_recorded_steps(
        viewer, monkeypatch
    ):
    # Recorded steps are replayed with the original column names on load
    point = TimepointIdentifier(
        'Bud', (255, 0, 0), '',
        {'type': 'equation', 'equation': 'area > 16', 'mode': 'first_true'}
    )
    viewer.addTimepointIdentifierItem(point)
    viewer.combinedMeasurementSteps.append(
        {'order': 0, 'equations': {'x': 'area * timepoint_Bud'}}
    )
    viewer.measurementProcessingSteps.append({
        'order': 1, 'columns': ['timepoint_Bud'], 'recipe': [],
        'output_columns': ['timepoint_Bud_smooth']
    })
    dependents = viewer.columnDependents(
        'timepoint_Bud', excludeOutput='timepoint_Bud'
    )
    assert dependents == [
        'combined measurement "x"',
        'measurement processing step "timepoint_Bud_smooth"'
    ]
    with pytest.raises(ValueError, match='combined measurement "x"'):
        viewer.validatePredictorDependencies(
            'timepoint_Bud', 'timepoint_Other', point.predictor
        )
    # Removal warns about them; "Remove" is never clicked --> cancelled
    question = Mock(return_value=(None, object()))
    monkeypatch.setattr(
        measurements_viewer.widgets.myMessageBox, 'question', question
    )
    viewer.removeTimepointIdentifier(point.uid)
    assert 'combined measurement "x"' in question.call_args.args[2]
    assert viewer.timepointIdentifier(point.uid) is point


def test_loading_annotation_predictors_defers_forward_dependencies(viewer):
    annotation_predictors(viewer, [
        ('Dependent', 'Source == 1'), ('Source', 'MAX(area) > 25')
    ], autoApply=True)
    assert viewer.acdc_df.xs(1, level='Cell_ID')['Dependent'].tolist() == [1]*4
    viewer.measurementProcessingWarning.assert_not_called()


def test_annotation_reevaluation_includes_timepoint_prerequisite(viewer):
    point = TimepointIdentifier(
        'Point', (255, 0, 0), '',
        {'type': 'equation', 'equation': 'area > 16', 'mode': 'first_true'}
    )
    viewer.addTimepointIdentifierItem(point)
    annotation_predictors(viewer, [('Flag', 'any(timepoint_Point)')])
    assert viewer.reevaluateDisplayedCellCustomAnnot(
        viewer.customAnnotButtonByName('Flag')
    )
    assert viewer.cellTimepoints(point.uid, 1) == [1]
    assert viewer.acdc_df.xs(1, level='Cell_ID')['Flag'].tolist() == [1]*4
    assert viewer.cellTimepoints(point.uid, 2) == []


@pytest.mark.parametrize('answer', ['cancel', 'skip', 'overwrite'])
def test_prerequisite_manual_annotations_are_protected(viewer, monkeypatch, answer):
    annotation_predictors(viewer, [('Flag', 'area < 0')])
    final = TimepointIdentifier(
        'Final', (0, 0, 255), '',
        {'type': 'equation', 'equation': 'Flag == 1', 'mode': 'first_true'}
    )
    viewer.addTimepointIdentifierItem(final)
    viewer.acdc_df.loc[(slice(None), 1), 'Flag'] = 1
    viewer.editedCustomAnnotCells()['Flag'].add(1)
    ask = Mock(return_value=answer)
    monkeypatch.setattr(viewer, 'askOverwriteManualTimepoints', ask)
    assert viewer.reevaluateTimepoints([final.uid], IDs=[1]) == (answer != 'cancel')
    assert ask.call_args.args[0] == {'Flag': [1]}
    assert viewer.cellTimepoints(final.uid, 1) == ([0] if answer == 'skip' else [])
    assert viewer.acdc_df.xs(1, level='Cell_ID')['Flag'].tolist() == (
        [0]*4 if answer == 'overwrite' else [1]*4
    )


def test_prerequisite_failure_stops_dependent_prediction(viewer):
    annotation_predictors(viewer, [('Flag', 'missing_column > 0')])
    final = TimepointIdentifier(
        'Final', (0, 0, 255), '',
        {'type': 'equation', 'equation': 'Flag == 0', 'mode': 'first_true'}
    )
    viewer.addTimepointIdentifierItem(final)
    assert not viewer.reevaluateTimepoints([final.uid], IDs=[1])
    assert viewer.cellTimepoints(final.uid, 1) == []
    viewer.measurementProcessingWarning.assert_called_once()


def test_failed_auto_prerequisites_are_not_retried_on_revisit(viewer):
    point = TimepointIdentifier(
        'Point', (255, 0, 0), '',
        {'type': 'equation', 'equation': 'missing_column > 0',
         'mode': 'first_true'},
        autoApply=True
    )
    viewer.addTimepointIdentifierItem(point)
    viewer.acdc_df[VISITED_COLUMN] = False
    annotation_predictors(
        viewer, [('Flag', 'any(timepoint_Point)')], autoApply=True
    )
    viewer.ensureTimepointPredictions()
    viewer.ensureCustomAnnotPredictions()
    viewer.measurementProcessingWarning.assert_called_once()
    assert viewer.acdc_df['Flag'].tolist() == [0]*8


def test_timepoint_customization_warns_about_mixed_cycle(viewer, monkeypatch):
    point = TimepointIdentifier(
        'Point', (255, 0, 0), '',
        {'type': 'equation', 'equation': 'area > 0', 'mode': 'first_true'}
    )
    viewer.addTimepointIdentifierItem(point)
    annotation_predictors(viewer, [('Flag', 'any(timepoint_Point)')])
    warn = Mock()
    monkeypatch.setattr(apps_measurements_viewer.TimepointIdentifierDialog, 'warn', warn)

    def accept(dialog):
        assert dialog.acdc_df is viewer.acdc_df
        dialog.equation = 'Flag == 1'
        dialog.ok_cb()
        assert dialog.cancel

    monkeypatch.setattr(apps_measurements_viewer.TimepointIdentifierDialog, 'exec_', accept)
    assert viewer.execTimepointIdentifierDialog(point, 'Edit') is None
    assert 'Circular predictor dependency' in warn.call_args.args[1]


def test_annotation_customization_warns_about_mixed_cycle(viewer, monkeypatch):
    annotation_predictors(viewer, [('Flag', 'area > 0')])
    point = TimepointIdentifier(
        'Point', (255, 0, 0), '',
        {'type': 'equation', 'equation': 'Flag == 1', 'mode': 'first_true'}
    )
    viewer.addTimepointIdentifierItem(point)
    state = viewer.customAnnotDict[viewer.customAnnotButtonByName('Flag')]['state']
    dialog = apps.customAnnotationDialog(
        {}, state=state, **viewer.customAnnotationDialogKwargs(state)
    )
    warning = Mock()
    monkeypatch.setattr(apps.widgets, 'myMessageBox', Mock(return_value=warning))
    try:
        dialog.predictorGroupbox.equation = 'any(timepoint_Point)'
        dialog.clickedButton = dialog.okButton
        event = QCloseEvent()
        dialog.closeEvent(event)
        assert not event.isAccepted()
        assert dialog.cancel
        assert 'Circular predictor dependency' in warning.warning.call_args.args[2]
    finally:
        dialog.clickedButton = dialog.cancelButton
        dialog.deleteLater()


@pytest.mark.parametrize('Dialog', [
    apps_measurements_viewer.TimepointPredictorEquationDialog,
    apps_measurements_viewer.CustomAnnotationPredictorEquationDialog,
])
def test_equation_customization_warns_before_accepting_cycle(
        viewer, monkeypatch, Dialog
    ):
    point = TimepointIdentifier('Point', (255, 0, 0), '')
    viewer.addTimepointIdentifierItem(point)
    dialog = Dialog([], False, False, acdc_df=viewer.acdc_df,
                    equation=f'{TIMEPOINT_PREFIX}Point')
    warning = Mock()
    monkeypatch.setattr(apps.widgets, 'myMessageBox', Mock(return_value=warning))
    dialog.validatePredictorDependencies = (
        lambda predictor: viewer.validatePredictorDependencies(
            f'{TIMEPOINT_PREFIX}Point', f'{TIMEPOINT_PREFIX}Point', predictor
        )
    )
    try:
        assert f'{TIMEPOINT_PREFIX}Point' in dialog.variableNames
        dialog.okButton.setEnabled(True)
        dialog.ok_cb()
        assert dialog.cancel
        assert 'Circular predictor dependency' in warning.warning.call_args.args[2]
    finally:
        dialog.deleteLater()


@pytest.mark.parametrize('mode, expected', [
    ('first_true', {1: [1], 2: []}),
    ('last_true', {1: [2], 2: []}),
    ('rising_true', {1: [1], 2: []}),
])
def test_equation_condition_modes(acdc_df, mode, expected):
    timepoints = timepoint_predictors.TimepointPredictor.from_equation(
        acdc_df, 'stage == "S"', mode
    )
    assert timepoints == expected


def test_equation_rising_and_extrema(acdc_df):
    assert timepoint_predictors.TimepointPredictor.from_equation(
        acdc_df, 'area > 16', 'rising_true'
    ) == {1: [1, 3], 2: []}
    assert timepoint_predictors.TimepointPredictor.from_equation(
        acdc_df, 'area', 'max'
    ) == {1: [3], 2: [2]}
    assert timepoint_predictors.TimepointPredictor.from_equation(
        acdc_df, 'area * 2', 'min'
    ) == {1: [0], 2: [3]}


@pytest.mark.parametrize('equation, mode, expected', [
    # cell 1 area: 10, 20, 15, 30 | cell 2 area: 5, 6, 7, 4
    ('area >= 0.5 * MAX(area)', 'first_true', {1: [1], 2: [0]}),
    ('area > MEAN(area)', 'first_true', {1: [1], 2: [1]}),
    ('area > AVERAGE(area)', 'first_true', {1: [1], 2: [1]}),
    ('area == MIN(area)', 'first_true', {1: [0], 2: [3]}),
    ('area > MEDIAN(area)', 'last_true', {1: [3], 2: [2]}),
    ('area / SUM(area) > 0.3', 'first_true', {1: [3], 2: [2]}),
    ('area - MEAN(area) > STD(area)', 'first_true', {1: [3], 2: [2]}),
    ('area > FIRST(area) * 1.4', 'first_true', {1: [1], 2: []}),
    ('area == LAST(area)', 'first_true', {1: [3], 2: [3]}),
    ('DIFF(area)', 'max', {1: [3], 2: [1]}),
    ('DIFF(area) < 0', 'rising_true', {1: [2], 2: [3]}),
    ('area > SHIFT(area, 2)', 'first_true', {1: [2], 2: [2]}),
    ('area < SHIFT(area, -1)', 'last_true', {1: [2], 2: [1]}),
    ('log(area) == MAX(log(area))', 'first_true', {1: [3], 2: [2]}),
    ('(area > MEAN(area)) & (stage == "S")', 'first_true', {1: [1], 2: []}),
])
def test_equation_cell_functions(acdc_df, equation, mode, expected):
    assert timepoint_predictors.TimepointPredictor.from_equation(
        acdc_df, equation, mode
    ) == expected


@pytest.mark.parametrize('function, expected', [
    ('any(area > 16)', {1: [0], 2: []}),
    ('all(area > 0)', {1: [0], 2: [0]}),
])
def test_equation_any_all_reduce_over_cell_frames(acdc_df, function, expected):
    assert timepoint_predictors.TimepointPredictor.from_equation(
        acdc_df, function, 'first_true'
    ) == expected


def test_other_cell_equation_uses_filtered_frame_window(acdc_df):
    settings = {
        'source': 'same_frame',
        'condition': 'area > 5',
        'condition_mode': 'where_true',
        'range_start': 't',
        'range_end': 't + 1',
    }
    predictor = {
        'type': timepoint_predictors.PREDICTOR_EQUATION,
        'equation': 'area > 2 * MAX(area_other)',
        'mode': 'rising_true',
        'other_cells': settings,
    }
    assert timepoint_predictors.TimepointPredictor.predict(
        acdc_df, predictor
    ) == {1: [1]}


def test_main_cell_filter_editor_supports_other_inputs(app, acdc_df):
    dialog = apps_measurements_viewer._TimepointCellFilterEquationDialog(
        [], False, False, acdc_df=acdc_df, SizeT=4,
        condition='COUNT(area_other) > 0',
        otherCells=timepoint_predictors.CellFilter.default_other()
    )
    try:
        assert 'area_other' in dialog.otherMeasurementColumns
        assert dialog.validateEquation('COUNT(area_other) > 0')[0]
        dialog.test_cb()
        assert dialog.okButton.isEnabled()
        dialog.otherCells = None
        valid, message = dialog.validateEquation('COUNT(area_other) > 0')
        assert not valid
        assert 'Configure Other-cell measurements' in message
    finally:
        dialog.deleteLater()


def test_direct_other_dialog_validation(app, acdc_df):
    dialog = apps_measurements_viewer.TimepointPredictorEquationDialog(
        [], False, False, acdc_df=acdc_df, equation='area_other > 0'
    )
    try:
        dialog.otherCellsGroup.setChecked(False)
        valid, message = dialog.validateEquation('area_other > 0')
        assert not valid
        assert 'Enable Other-cell measurements' in message
        dialog.otherCellsGroup.setChecked(True)
        assert dialog.validateEquation('area_other > 0')[0]
    finally:
        dialog.deleteLater()


@pytest.mark.parametrize('Dialog', [
    apps_measurements_viewer.TimepointPredictorEquationDialog,
    apps_measurements_viewer.CustomAnnotationPredictorEquationDialog,
])
@pytest.mark.parametrize('equation', [
    'frame_i == FIRST(frame_i) + 1', 'LAST(frame_i) == SizeT - 1',
    'Cell_ID == 1',
])
def test_equation_dialog_accepts_index_only_inputs(
        app, acdc_df, Dialog, equation
    ):
    # frame_i and Cell_ID are index levels, not tree inputs, but valid
    dialog = Dialog([], False, False, acdc_df=acdc_df, SizeT=4)
    try:
        assert dialog.validateEquation(equation)[0]
        dialog.formulaEditWidget.setText(equation)
        dialog.test_cb()
        assert dialog.okButton.isEnabled(), dialog.statusLabel.text()
        # SizeT alone is a constant, not a per-cell input
        valid, message = dialog.validateEquation('SizeT - 1 > 0')
        assert not valid
        assert message == 'No measurement inputs used'
    finally:
        dialog.deleteLater()


def test_other_cell_inputs_use_equation_aggregations(acdc_df):
    settings = {
        'source': 'same_frame',
        'condition': '',
        'condition_mode': 'where_true',
        'range_start': 't',
        'range_end': 't + 1',
    }
    assert timepoint_predictors.TimepointPredictor.from_equation(
        acdc_df, 'area < MEAN(area_other)', 'first_true',
        other_cells=settings
    ) == {1: [], 2: [0]}


def test_main_cell_filter_selects_eligible_cells(acdc_df):
    predictor = {
        'type': timepoint_predictors.PREDICTOR_EQUATION,
        'equation': 'stage == "S"',
        'mode': 'first_true',
        'main_cell_filter': {
            'condition': 'area >= 10',
            'condition_mode': 'any',
        },
    }
    assert timepoint_predictors.TimepointPredictor.predict(
        acdc_df, predictor
    ) == {1: [1]}


def test_main_cell_filter_all_requires_every_frame(acdc_df):
    assert timepoint_predictors.TimepointPredictor.from_equation(
        acdc_df, 'area > 0', 'first_true',
        main_cell_filter={
            'condition': 'stage == "G1"',
            'condition_mode': 'all',
        }
    ) == {2: [0]}


def test_main_cell_filter_validation():
    with pytest.raises(ValueError, match='main-cell filter mode'):
        timepoint_predictors.CellFilter.validate_main({
            'condition': 'area > 0',
            'condition_mode': 'where_true',
        })


def test_other_cell_equation_supports_relationship_presets(lineage_df):
    predictor = {
        'type': timepoint_predictors.PREDICTOR_EQUATION,
        'equation': 'MEAN_OTHER(area) > 10',
        'mode': 'first_true',
        'other_cells': {
            'source': 'mother',
            'condition': '',
            'condition_mode': 'where_true',
            'range_start': 't - 2',
            'range_end': 't',
        },
    }
    assert timepoint_predictors.TimepointPredictor.predict(
        lineage_df, predictor
    ) == {2: [2], 3: [2]}


def test_cell_functions_ignore_nan():
    df = pd.DataFrame({
        'frame_i': [0, 1, 2], 'Cell_ID': [1, 1, 1], 'area': [np.nan, 4., 2.]
    }).set_index(['frame_i', 'Cell_ID'])
    assert timepoint_predictors.TimepointPredictor.from_equation(
        df, 'area == MAX(area)', 'first_true'
    ) == {1: [1]}
    assert timepoint_predictors.TimepointPredictor.from_equation(
        df, 'area == FIRST(area)', 'first_true'
    ) == {1: [1]}


def test_file_predictor_receives_single_cell(tmp_path, acdc_df):
    filepath = tmp_path / 'pred.py'
    filepath.write_text(
        'def predictor(df):\n'
        '    assert df["Cell_ID"].nunique() == 1\n'
        '    if df["Cell_ID"].iloc[0] == 1:\n'
        '        return [3, 1, 9]\n'
        '    return int(df["frame_i"].iloc[-1])\n'
    )
    predicted = timepoint_predictors.TimepointPredictor.predict(
        acdc_df, {'type': 'file', 'filepath': str(filepath)}, SizeT=4
    )
    assert predicted == {1: [1, 3], 2: [3]}


def test_file_predictor_errors(tmp_path, acdc_df):
    filepath = tmp_path / 'pred.py'
    filepath.write_text('def other(df):\n    return 1\n')
    with pytest.raises(AttributeError):
        timepoint_predictors.TimepointPredictor.load_function(str(filepath))
    filepath.write_text('def predictor(df):\n    return "a"\n')
    with pytest.raises(ValueError, match='Cell_ID 1'):
        timepoint_predictors.TimepointPredictor.predict(
            acdc_df, {'type': 'file', 'filepath': str(filepath)}
        )


def test_identifier_dict_roundtrip():
    identifier = TimepointIdentifier(
        'Bud', (1, 2, 3), 'Q', 
        {'type': 'equation', 'equation': 'area > 1', 'mode': 'max'}
    )
    restored = TimepointIdentifier.from_dict(identifier.to_dict())
    assert restored.to_dict() == identifier.to_dict()
    assert restored.uid != identifier.uid
    with pytest.raises(ValueError):
        TimepointIdentifier.from_dict({**identifier.to_dict(), 'color': [1]})


def test_toolbar_and_default_identifier(viewer):
    assert viewer.addTimepointIdentifierAction in viewer.timepointsToolBar.actions()
    identifier = viewer.defaultTimepointIdentifier()
    assert identifier.name == 'Time point 1'
    assert identifier.shortcut == 'Q'
    assert identifier.predictor == {'type': 'none'}
    viewer.addTimepointIdentifierItem(identifier)
    second = viewer.defaultTimepointIdentifier()
    assert (second.name, second.shortcut) == ('Time point 2', 'W')
    assert second.color != identifier.color
    reserved = viewer.reservedShortcuts()
    assert {'1', 'H', 'Q', 'Ctrl+O'} <= set(reserved)
    assert 'Q' not in viewer.reservedShortcuts(excludeUid=identifier.uid)


def test_add_identifier_creates_action_with_shortcut(viewer, monkeypatch):
    def accept(dialog):
        assert dialog.nameLineEdit.text() == 'Time point 1'
        assert dialog.shortcutLineEdit.text() == 'Q'
        dialog.nameLineEdit.setText('Bud')
        dialog.ok_cb()

    monkeypatch.setattr(apps_measurements_viewer.TimepointIdentifierDialog, 'exec_', accept)
    identifier = viewer.addTimepointIdentifier()
    action = viewer.timepointActions[identifier.uid]
    assert action in viewer.timepointsToolBar.actions()
    assert action.text() == 'Bud'
    assert action.shortcut().toString() == 'Q'


def test_shortcut_adds_timepoint_at_hovered_frame(viewer, monkeypatch):
    identifier = TimepointIdentifier('Bud', (255, 0, 0), 'Q')
    viewer.addTimepointIdentifierItem(identifier)
    monkeypatch.setattr(viewer, 'hoveredMeasurementsPlotFrame', lambda: 2)
    viewer.timepointActions[identifier.uid].trigger()
    assert timepoints(viewer, identifier.uid) == {1: [2]}
    assert lines(viewer) == [(identifier.uid, 1, 2)]
    line = viewer.timepointLines[0]
    assert isinstance(line, _TimepointLine)
    assert line.value() == 3
    
    # Not hovering the plot nor clicking the toolbar --> nothing added
    monkeypatch.setattr(viewer, 'hoveredMeasurementsPlotFrame', lambda: None)
    viewer.timepointActions[identifier.uid].trigger()
    assert timepoints(viewer, identifier.uid) == {1: [2]}


def test_auto_shortcut_predicts_selected_cell_and_replots(viewer, app):
    identifier = TimepointIdentifier(
        'Big', (0, 255, 0), 'Q',
        {'type': 'equation', 'equation': 'area', 'mode': 'max'}
    )
    viewer.addTimepointIdentifierItem(identifier)
    viewer.show()
    viewer.activateWindow()
    viewer.setFocus()
    app.processEvents()

    QTest.keyClick(viewer, Qt.Key_Q, Qt.ShiftModifier)
    app.processEvents()

    assert timepoints(viewer, identifier.uid) == {1: [3]}
    assert lines(viewer) == [(identifier.uid, 1, 3)]
    viewer.hide()


def test_auto_shortcut_is_never_ambiguous(viewer, app, monkeypatch):
    shifted = TimepointIdentifier('Bud', (255, 0, 0), 'Shift+Q')
    empty = TimepointIdentifier('Septum', (0, 255, 0), '')
    # Shift+S shuffles the colormap (e.g., S from an older configuration)
    clashing = TimepointIdentifier('Death', (0, 0, 255), 'S')
    for identifier in (shifted, empty, clashing):
        viewer.addTimepointIdentifierItem(identifier)
        assert viewer.timepointAutoActions[identifier.uid].shortcut().isEmpty()
    assert 'Randomly shuffle colormap' in (
        viewer.timepointActions[clashing.uid].toolTip()
    )
    shortcuts = [
        action.shortcut().toString() for action in viewer.actions()
        if not action.shortcut().isEmpty()
    ]
    assert len(shortcuts) == len(set(shortcuts))
    # Q would get Shift+Q (used by "Bud") as auto shortcut
    assert 'Q' in viewer.reservedShortcuts()
    reserved = viewer.reservedShortcuts(excludeUid=shifted.uid)
    assert 'Q' not in reserved and 'Shift+Q' not in reserved

    monkeypatch.setattr(viewer, 'hoveredMeasurementsPlotFrame', lambda: 2)
    viewer.show()
    viewer.activateWindow()
    viewer.setFocus()
    app.processEvents()
    QTest.keyClick(viewer, Qt.Key_Q, Qt.ShiftModifier)
    app.processEvents()
    viewer.hide()
    assert timepoints(viewer, shifted.uid) == {1: [2]}


def test_identifier_dialog_rejects_key_with_used_auto_shortcut(
        viewer, monkeypatch
    ):
    warnings = []
    monkeypatch.setattr(
        apps_measurements_viewer.TimepointIdentifierDialog, 'warn',
        lambda self, title, text: warnings.append((title, text))
    )
    reserved = viewer.reservedShortcuts()
    assert 'S' in reserved
    dialog = apps_measurements_viewer.TimepointIdentifierDialog(
        'Bud', (255, 0, 0), 'S', reservedShortcuts=reserved,
        acdc_df=viewer.acdc_df, SizeT=4
    )
    dialog.ok_cb()
    assert dialog.cancel
    assert warnings[-1][0] == 'Shortcut already used'
    assert 'Shift+S' in warnings[-1][1]
    dialog.deleteLater()


def test_move_and_remove_timepoint_lines(viewer, app):
    identifier = TimepointIdentifier('Bud', (255, 0, 0), 'Q')
    viewer.addTimepointIdentifierItem(identifier)
    assert viewer.addTimepoint(identifier.uid, 1)
    assert not viewer.addTimepoint(identifier.uid, 1)
    assert viewer.addTimepoint(identifier.uid, 3)
    line = viewer.timepointLines[0]
    line.setValue(2.8)
    viewer.timepointLineMoved(line)
    app.processEvents()
    assert timepoints(viewer, identifier.uid) == {1: [2, 3]}
    # Moving onto another time point merges them
    line = viewer.timepointLines[0]
    line.setValue(10)
    viewer.timepointLineMoved(line)
    app.processEvents()
    assert timepoints(viewer, identifier.uid) == {1: [3]}
    viewer.timepointLineRemoveRequested(viewer.timepointLines[0])
    assert timepoints(viewer, identifier.uid) == {}
    assert viewer.timepointLines == []


def test_lines_follow_selected_cell_and_mode(viewer):
    identifier = TimepointIdentifier('Bud', (255, 0, 0), 'Q')
    viewer.addTimepointIdentifierItem(identifier)
    viewer.addTimepoint(identifier.uid, 1, ID=2)
    assert viewer.timepointLines == []
    viewer.setSelectedID(2)
    assert lines(viewer) == [(identifier.uid, 2, 1)]
    viewer.measurementsModeCombobox.setCurrentIndex(1)
    assert viewer.timepointLines == []
    assert not viewer.addTimepoint(identifier.uid, 2)


def test_predictor_runs_and_export_columns(viewer):
    identifier = TimepointIdentifier(
        'Big', (0, 255, 0), 'W', 
        {'type': 'equation', 'equation': 'area > 16', 'mode': 'rising_true'},
        autoApply=True
    )
    viewer.addTimepointIdentifierItem(identifier)
    viewer.acdc_df[VISITED_COLUMN] = False
    viewer.ensureTimepointPredictions()
    assert timepoints(viewer, identifier.uid) == {1: [1, 3]}
    viewer.updateTimepointLines()
    assert lines(viewer) == [(identifier.uid, 1, 1), (identifier.uid, 1, 3)]
    flagged = viewer.acdc_df.index[viewer.acdc_df[f'{TIMEPOINT_PREFIX}Big']].tolist()
    assert flagged == [(1, 1), (3, 1)]
    # Already predicted --> manual edits are not overwritten
    viewer.removeTimepoint(identifier.uid, 1, 1)
    viewer.ensureTimepointPredictions()
    assert timepoints(viewer, identifier.uid) == {1: [3]}


def test_combine_measurements_uses_timepoint_columns_and_cell_aggregations(
        viewer, monkeypatch
    ):
    identifier = TimepointIdentifier(
        'Bud', (0, 255, 0), 'W'
    )
    viewer.addTimepointIdentifierItem(identifier)
    viewer.addTimepoint(identifier.uid, 1, ID=1)

    originalDialog = apps.CombineMetricsEquationDialog

    class TestDialog(originalDialog):
        def exec_(self):
            assert f'{TIMEPOINT_PREFIX}Bud' in self.acdcDf
            assert f'{TIMEPOINT_PREFIX}Bud' in self.variableNames
            self.newColNameLineEdit.setText('ever_bud')
            self.formulaEditWidget.setText('any(timepoint_Bud)')
            self.test_cb()
            self.ok_cb()

    monkeypatch.setattr(apps, 'CombineMetricsEquationDialog', TestDialog)
    viewer.combineMeasurements()

    assert viewer.acdc_df['ever_bud'].tolist() == (
        [True, True, True, True, False, False, False, False]
    )


def big_identifier(viewer):
    identifier = TimepointIdentifier(
        'Big', (0, 255, 0), 'W', 
        {'type': 'equation', 'equation': 'area > 16', 'mode': 'rising_true'},
        autoApply=True
    )
    viewer.addTimepointIdentifierItem(identifier)
    viewer.acdc_df[VISITED_COLUMN] = False
    viewer.ensureTimepointPredictions()
    return identifier


def test_automatic_predictions_only_run_on_newly_viewed_cells(viewer, monkeypatch):
    identifier = big_identifier(viewer)
    predict = Mock(wraps=timepoint_predictors.TimepointPredictor.predict)
    monkeypatch.setattr(timepoint_predictors.TimepointPredictor, 'predict', predict)
    assert set(viewer.acdc_df.index.get_level_values('Cell_ID')[
        viewer.acdc_df[VISITED_COLUMN]
    ]) == {1}
    viewer.setSelectedID(2)
    assert predict.call_args.kwargs['IDs'] == {2}
    assert timepoints(viewer, identifier.uid) == {1: [1, 3]}
    viewer.setSelectedID(1)
    viewer.setSelectedID(2)
    viewer.initIDsNavigation(ID=1)
    assert predict.call_count == 1
    assert viewer.acdc_df[VISITED_COLUMN].all()


def test_failed_automatic_prediction_is_not_retried_on_revisit(viewer, monkeypatch):
    identifier = big_identifier(viewer)
    predict = Mock(side_effect=ValueError('Failed'))
    monkeypatch.setattr(timepoint_predictors.TimepointPredictor, 'predict', predict)
    viewer.setSelectedID(2)
    viewer.measurementProcessingWarning.assert_called_once()
    viewer.setSelectedID(1)
    viewer.setSelectedID(2)
    assert predict.call_count == 1
    predict.side_effect = None
    predict.return_value = {2: [2]}
    assert viewer.reevaluateDisplayedCellTimepoints()
    assert predict.call_count == 2
    assert timepoints(viewer, identifier.uid)[2] == [2]


def test_viewed_cells_are_scoped_to_position_table(viewer, monkeypatch):
    big_identifier(viewer)
    predict = Mock(wraps=timepoint_predictors.TimepointPredictor.predict)
    monkeypatch.setattr(timepoint_predictors.TimepointPredictor, 'predict', predict)
    posData = viewer.posData()
    originalPath = posData.pos_path
    posData.pos_path = r'C:\experiment\Position_2'
    original = viewer.acdc_df
    other = original.copy()
    other[VISITED_COLUMN] = False
    viewer.acdc_df = other
    viewer.ensureTimepointPredictions()
    assert predict.call_args.kwargs['IDs'] == {1}
    viewer.acdc_df = original
    posData.pos_path = originalPath
    posData.acdc_df = original
    viewer.ensureTimepointPredictions()
    assert predict.call_count == 1
    assert original.xs(1, level='Cell_ID')[VISITED_COLUMN].all()
    assert other.xs(1, level='Cell_ID')[VISITED_COLUMN].all()


def test_adding_identifier_does_not_repredict_viewed_cells(viewer, monkeypatch):
    big_identifier(viewer)
    other = TimepointIdentifier(
        'Other', (0, 0, 255), 'E',
        {'type': 'equation', 'equation': 'area > 0', 'mode': 'first_true'},
        autoApply=True
    )
    viewer.addTimepointIdentifierItem(other)
    predict = Mock(wraps=timepoint_predictors.TimepointPredictor.predict)
    monkeypatch.setattr(timepoint_predictors.TimepointPredictor, 'predict', predict)
    viewer.ensureTimepointPredictions()
    predict.assert_not_called()
    viewer.setSelectedID(2)
    assert predict.call_count == 2
    assert all(call.kwargs['IDs'] == {2} for call in predict.call_args_list)


def test_predictor_edit_does_not_repredict_viewed_cells(viewer, monkeypatch):
    identifier = big_identifier(viewer)
    monkeypatch.setattr(viewer, 'execTimepointIdentifierDialog', Mock(return_value={
        'name': identifier.name, 'color': identifier.color,
        'shortcut': identifier.shortcut, 'autoApply': True,
        'predictor': {
            'type': 'equation', 'equation': 'area > 0', 'mode': 'first_true'
        }
    }))
    predict = Mock(wraps=timepoint_predictors.TimepointPredictor.predict)
    monkeypatch.setattr(timepoint_predictors.TimepointPredictor, 'predict', predict)
    viewer.editTimepointIdentifier(identifier.uid)
    viewer.setSelectedID(1)
    predict.assert_not_called()
    assert timepoints(viewer, identifier.uid) == {1: [1, 3]}
    viewer.setSelectedID(2)
    assert predict.call_args.kwargs['IDs'] == {2}
    assert timepoints(viewer, identifier.uid) == {1: [1, 3], 2: [0]}


def test_explicit_shortcut_preserves_manual_edit_warning(viewer, monkeypatch):
    identifier = big_identifier(viewer)
    viewer.addTimepoint(identifier.uid, 0)
    ask = Mock(return_value='cancel')
    monkeypatch.setattr(viewer, 'askOverwriteManualTimepoints', ask)
    predict = Mock(wraps=timepoint_predictors.TimepointPredictor.predict)
    monkeypatch.setattr(timepoint_predictors.TimepointPredictor, 'predict', predict)
    viewer.timepointAutoActions[identifier.uid].trigger()
    ask.assert_called_once_with({'Big': [1]}, 'the displayed cell (ID 1)')
    predict.assert_not_called()
    assert timepoints(viewer, identifier.uid) == {1: [0, 1, 3]}


def test_viewed_cells_configuration_roundtrip(viewer, tmp_path, monkeypatch):
    big_identifier(viewer)
    viewer.setSelectedID(2)
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
    path = os.path.normcase(os.path.abspath(posData.pos_path))
    assert 'viewed_cells' not in configuration
    assert viewer.acdc_df[VISITED_COLUMN].all()
    filepath = tmp_path / 'state.json'
    filepath.write_text(json.dumps(configuration))
    viewer.timepointAutoPredictions.clear()
    monkeypatch.setattr(
        measurements_viewer, 'get_channels_and_segm_endnames',
        Mock(return_value=(['GFP'], ['segm']))
    )
    monkeypatch.setattr(measurements_viewer, 'load_position', Mock(return_value=posData))
    monkeypatch.setattr(
        viewer, 'selectConfigurationInput',
        lambda config: viewer.configurationInputOptions(config)[0]
    )
    predict = Mock(wraps=timepoint_predictors.TimepointPredictor.predict)
    monkeypatch.setattr(timepoint_predictors.TimepointPredictor, 'predict', predict)

    def set_data(data):
        assert data == [posData]
        assert data[0].acdc_df[VISITED_COLUMN].all()
        viewer.ensureTimepointPredictions()

    monkeypatch.setattr(viewer, 'setData', set_data)
    monkeypatch.setattr(viewer, 'setOverlayItemsOpacities', Mock())
    assert load_configuration(viewer, filepath=str(filepath))
    predict.assert_not_called()
    assert viewer.acdc_df[VISITED_COLUMN].all()
    # Visited cells count as predicted for the loaded identifiers
    loaded_uid = viewer.timepointIdentifiers[0].uid
    assert viewer.timepointAutoPredictions[(path, loaded_uid)] == {1, 2}
    assert 'viewed_cells' not in viewer.configuration()
    legacy = dict(configuration)
    legacy['viewed_cells'] = {path: [1, 2]}
    MeasurementsViewerWin.validateConfiguration(legacy)
    for invalid in ([], {'position': [True]}, {'position': [-1]}, {'position': ['1']}):
        with pytest.raises(ValueError, match='Invalid viewed cells'):
            MeasurementsViewerWin.validateConfiguration(
                dict(configuration, viewed_cells=invalid)
            )


def test_predictor_requires_auto_apply_unless_forced(viewer, monkeypatch):
    identifier = TimepointIdentifier(
        'Big', (0, 255, 0), 'W',
        {'type': 'equation', 'equation': 'area > 16', 'mode': 'rising_true'}
    )
    viewer.addTimepointIdentifierItem(identifier)
    predict = Mock(wraps=timepoint_predictors.TimepointPredictor.predict)
    monkeypatch.setattr(timepoint_predictors.TimepointPredictor, 'predict', predict)
    assert viewer.runTimepointPredictor(identifier.uid)
    predict.assert_not_called()
    assert not viewer.timepointAutoPredictions
    assert viewer.runTimepointPredictor(identifier.uid, force=True)
    assert predict.call_count == 1
    assert timepoints(viewer, identifier.uid) == {1: [1, 3]}


def test_auto_predictions_track_each_cell_including_empty_results(viewer, monkeypatch):
    identifier = big_identifier(viewer)
    viewer.timepointAutoPredictions.clear()
    predict = Mock(wraps=timepoint_predictors.TimepointPredictor.predict)
    monkeypatch.setattr(timepoint_predictors.TimepointPredictor, 'predict', predict)
    assert viewer.runTimepointPredictor(identifier.uid, IDs=[1], skipIDs=[1])
    predict.assert_not_called()
    assert viewer.runTimepointPredictor(identifier.uid, IDs=[1])
    assert viewer.runTimepointPredictor(identifier.uid, IDs=[1, 2])
    assert [call.kwargs['IDs'] for call in predict.call_args_list] == [{1}, {2}]
    set_timepoints(viewer, identifier.uid, {1: [0]})
    assert viewer.runTimepointPredictor(identifier.uid)
    assert predict.call_count == 2
    assert timepoints(viewer, identifier.uid) == {1: [0]}
    assert viewer.runTimepointPredictor(identifier.uid, IDs=[1], force=True)
    assert predict.call_count == 3
    assert timepoints(viewer, identifier.uid) == {1: [1, 3]}


def test_auto_predictions_are_scoped_to_position_path_and_identifier(viewer, monkeypatch):
    identifier = big_identifier(viewer)
    predict = Mock(wraps=timepoint_predictors.TimepointPredictor.predict)
    monkeypatch.setattr(timepoint_predictors.TimepointPredictor, 'predict', predict)
    original = viewer.posData()
    other = SimpleNamespace(SizeT=4, pos_path=r'C:\experiment\Position_2')
    monkeypatch.setattr(viewer, 'getAcdcDf', lambda pos: viewer.acdc_df)
    assert viewer.runTimepointPredictor(identifier.uid, posData=other, IDs=[1])
    assert predict.call_count == 1
    same = SimpleNamespace(SizeT=4, pos_path=original.pos_path)
    assert viewer.runTimepointPredictor(identifier.uid, posData=same, IDs=[1])
    assert predict.call_count == 1
    second = TimepointIdentifier(
        'Other', (0, 0, 255), 'E', dict(identifier.predictor), autoApply=True
    )
    viewer.addTimepointIdentifierItem(second)
    assert viewer.runTimepointPredictor(second.uid)
    assert predict.call_count == 2


def test_auto_predictions_retry_failures(viewer, monkeypatch):
    identifier = big_identifier(viewer)
    viewer.timepointAutoPredictions.clear()
    predict = Mock(side_effect=[ValueError('Failed'), {1: [2]}])
    monkeypatch.setattr(timepoint_predictors.TimepointPredictor, 'predict', predict)
    assert not viewer.runTimepointPredictor(identifier.uid, IDs=[1])
    assert not any(viewer.timepointAutoPredictions.values())
    assert viewer.runTimepointPredictor(identifier.uid, IDs=[1])
    assert predict.call_count == 2
    assert timepoints(viewer, identifier.uid)[1] == [2]


def test_auto_predictions_skip_cells_without_marking_them_completed(viewer, monkeypatch):
    identifier = big_identifier(viewer)
    viewer.timepointAutoPredictions.clear()
    predict = Mock(wraps=timepoint_predictors.TimepointPredictor.predict)
    monkeypatch.setattr(timepoint_predictors.TimepointPredictor, 'predict', predict)
    assert viewer.runTimepointPredictor(identifier.uid, skipIDs=[1])
    assert predict.call_args.kwargs['IDs'] == {2}
    viewer.acdc_df[VISITED_COLUMN] = False
    viewer.ensureTimepointPredictions()
    assert predict.call_args.kwargs['IDs'] == {1}
    assert predict.call_count == 2


def test_reevaluation_bypasses_auto_apply_guard(viewer):
    identifier = TimepointIdentifier(
        'Big', (0, 255, 0), 'W',
        {'type': 'equation', 'equation': 'area > 16', 'mode': 'rising_true'}
    )
    viewer.addTimepointIdentifierItem(identifier)
    assert viewer.reevaluateAllTimepoints()
    assert timepoints(viewer, identifier.uid) == {1: [1, 3]}


def test_enabling_auto_apply_after_position_load(viewer):
    identifier = TimepointIdentifier(
        'Big', (0, 255, 0), 'W',
        {'type': 'equation', 'equation': 'area > 16', 'mode': 'rising_true'}
    )
    viewer.addTimepointIdentifierItem(identifier)
    viewer.ensureTimepointPredictions()
    assert timepoints(viewer, identifier.uid) == {}
    identifier.autoApply = True
    viewer.ensureTimepointPredictions()
    assert timepoints(viewer, identifier.uid) == {}
    assert viewer.reevaluateDisplayedCellTimepoints()
    assert timepoints(viewer, identifier.uid) == {1: [1, 3]}


@pytest.mark.parametrize('auto_apply', [False, True])
def test_edit_identifier_preserves_and_updates_auto_apply(viewer, monkeypatch, auto_apply):
    identifier = TimepointIdentifier(
        'Bud', (255, 0, 0), 'Q', autoApply=auto_apply
    )
    viewer.addTimepointIdentifierItem(identifier)

    def accept(dialog):
        assert dialog.autoApplyCheckBox.isChecked() == auto_apply
        dialog.autoApplyCheckBox.setChecked(not auto_apply)
        dialog.ok_cb()

    monkeypatch.setattr(apps_measurements_viewer.TimepointIdentifierDialog, 'exec_', accept)
    viewer.editTimepointIdentifier(identifier.uid)
    assert identifier.autoApply == (not auto_apply)


def ask(monkeypatch, viewer, answer):
    calls = []
    def askOverwrite(editedCells, scope):
        calls.append((editedCells, scope))
        return answer
    monkeypatch.setattr(viewer, 'askOverwriteManualTimepoints', askOverwrite)
    return calls


def test_reevaluate_actions_enabled_only_with_predictors(viewer):
    toolbarActions = viewer.timepointsToolBar.actions()
    assert viewer.reevaluateAllTimepointsAction in toolbarActions
    assert viewer.reevaluateCellTimepointsAction in toolbarActions
    assert not viewer.reevaluateAllTimepointsAction.isEnabled()
    manual = TimepointIdentifier('Manual', (255, 0, 0), 'Q')
    viewer.addTimepointIdentifierItem(manual)
    assert not viewer.reevaluateCellTimepointsAction.isEnabled()
    identifier = big_identifier(viewer)
    assert viewer.reevaluateAllTimepointsAction.isEnabled()
    assert viewer.reevaluateCellTimepointsAction.isEnabled()
    viewer.removeTimepointIdentifier(identifier.uid, ask=False)
    assert not viewer.reevaluateAllTimepointsAction.isEnabled()


def test_reevaluate_without_edits_does_not_warn(viewer, monkeypatch):
    identifier = big_identifier(viewer)
    calls = ask(monkeypatch, viewer, 'cancel')
    set_timepoints(viewer, identifier.uid, {2: [0]})
    assert viewer.reevaluateAllTimepointsAction.trigger() is None
    assert calls == []
    assert timepoints(viewer, identifier.uid) == {1: [1, 3]}


def test_manual_edits_are_tracked(viewer, app):
    identifier = big_identifier(viewer)
    edited = viewer.editedTimepointCells()
    assert not edited[identifier.uid]
    viewer.addTimepoint(identifier.uid, 0, ID=2)
    assert edited[identifier.uid] == {2}
    viewer.updateTimepointLines()
    line = viewer.timepointLines[0]
    line.setValue(line.value())
    viewer.timepointLineMoved(line)
    assert edited[identifier.uid] == {2}
    line.setValue(3)
    viewer.timepointLineMoved(line)
    app.processEvents()
    assert edited[identifier.uid] == {1, 2}
    viewer.clearTimepoints(identifier.uid)
    assert not edited[identifier.uid]
    viewer.timepointLineRemoveRequested(
        SimpleNamespace(uid=identifier.uid, ID=1, frame_i=3)
    )
    assert edited[identifier.uid] == {1}


@pytest.mark.parametrize('answer, expected', [
    ('cancel', {1: [3], 2: [0]}),
    ('skip', {1: [3], 2: [0]}),
    ('overwrite', {1: [1, 3]}),
])
def test_reevaluate_all_cells_warns_on_edits(viewer, monkeypatch, answer, expected):
    identifier = big_identifier(viewer)
    viewer.removeTimepoint(identifier.uid, 1, 1)
    viewer.markTimepointEdited(identifier.uid, 1)
    viewer.addTimepoint(identifier.uid, 0, ID=2)
    calls = ask(monkeypatch, viewer, answer)
    assert viewer.reevaluateAllTimepoints() is (answer != 'cancel')
    assert calls == [({'Big': [1, 2]}, '2 cell(s)')]
    assert timepoints(viewer, identifier.uid) == expected
    edited = viewer.editedTimepointCells()[identifier.uid]
    assert edited == (set() if answer == 'overwrite' else {1, 2})


def test_reevaluate_displayed_cell_only(viewer, monkeypatch):
    identifier = big_identifier(viewer)
    viewer.addTimepoint(identifier.uid, 0, ID=1)
    viewer.addTimepoint(identifier.uid, 2, ID=2)
    calls = ask(monkeypatch, viewer, 'overwrite')
    assert viewer.selectedID == 1
    viewer.reevaluateCellTimepointsAction.trigger()
    assert calls == [({'Big': [1]}, 'the displayed cell (ID 1)')]
    assert timepoints(viewer, identifier.uid) == {1: [1, 3], 2: [2]}
    assert viewer.editedTimepointCells()[identifier.uid] == {2}
    # Displayed cell not edited anymore --> no warning
    calls.clear()
    viewer.reevaluateDisplayedCellTimepoints()
    assert calls == []


def test_reevaluate_single_identifier(viewer, monkeypatch):
    identifier = big_identifier(viewer)
    other = TimepointIdentifier(
        'Small', (0, 0, 255), 'E', 
        {'type': 'equation', 'equation': 'area < 6', 'mode': 'first_true'},
        autoApply=True
    )
    viewer.addTimepointIdentifierItem(other)
    viewer.reevaluateTimepoints([other.uid])
    assert timepoints(viewer, other.uid) == {2: [0]}
    set_timepoints(viewer, identifier.uid, {})
    set_timepoints(viewer, other.uid, {})
    viewer.reevaluateTimepoints([other.uid])
    assert timepoints(viewer, identifier.uid) == {}
    assert timepoints(viewer, other.uid) == {2: [0]}


def test_remove_identifier(viewer):
    identifier = TimepointIdentifier('Bud', (255, 0, 0), 'Q')
    action = viewer.addTimepointIdentifierItem(identifier)
    autoAction = viewer.timepointAutoActions[identifier.uid]
    assert autoAction in viewer.actions()
    viewer.addTimepoint(identifier.uid, 1)
    viewer.removeTimepointIdentifier(identifier.uid, ask=False)
    assert action not in viewer.timepointsToolBar.actions()
    assert autoAction not in viewer.actions()
    assert identifier.uid not in viewer.timepointAutoActions
    assert viewer.timepointIdentifiers == []
    assert f'{TIMEPOINT_PREFIX}Big' not in viewer.acdc_df.columns
    assert viewer.timepointLines == []


def test_configuration_contains_identifiers():
    identifier = TimepointIdentifier('Bud', (255, 0, 0), 'Q')
    configuration = {
        'version': 1, 'images_paths': ['a'], 'channel': 'GFP',
        'segm_endname': 'segm', 'overlay_channels': [],
        'visible_channels': ['GFP'], 'measurements_mode': 0,
        'time_course_columns': [], 'normalize_measurements': False,
        'measurements_axes': {'X': '', 'Y': ''},
        'timepoint_identifiers': [identifier.to_dict()],
    }
    MeasurementsViewerWin.validateConfiguration(configuration)
    # Duplicate names would overwrite each other's timepoint_<name> columns
    configuration['timepoint_identifiers'].append(identifier.to_dict())
    with pytest.raises(ValueError, match='Duplicated time point identifier'):
        MeasurementsViewerWin.validateConfiguration(configuration)


def test_identifier_dialog_validation(app, monkeypatch, acdc_df):
    warnings = []
    monkeypatch.setattr(
        apps_measurements_viewer.TimepointIdentifierDialog, 'warn', 
        lambda self, title, text: warnings.append(title)
    )
    dialog = apps_measurements_viewer.TimepointIdentifierDialog(
        'Bud', (255, 0, 0), 'Q', existingNames={'Other'},
        reservedShortcuts={'Q': 'Something'}, acdc_df=acdc_df, SizeT=4
    )
    dialog.ok_cb()
    assert dialog.cancel and warnings == ['Shortcut already used']
    dialog.shortcutLineEdit.setText('E')
    dialog.nameLineEdit.setText('Other')
    dialog.ok_cb()
    assert dialog.cancel and warnings[-1] == 'Name already used'
    dialog.nameLineEdit.setText('Bud')
    dialog.predictorCombobox.setCurrentIndex(1)
    dialog.ok_cb()
    assert dialog.cancel and warnings[-1] == 'Invalid predictor'
    dialog.equation = 'area > 12'
    dialog.equationMode = 'first_true'
    assert dialog.testPredictor()
    assert '1 of 2 cells' in dialog.statusLabel.text()
    dialog.ok_cb()
    assert not dialog.cancel
    assert dialog.identifier == {
        'name': 'Bud', 'color': (255, 0, 0), 'shortcut': 'E',
        'autoApply': False,
        'predictor': {
            'type': 'equation', 'equation': 'area > 12', 'mode': 'first_true'
        },
    }
    dialog.deleteLater()


def test_predictor_equation_dialog(app, acdc_df):
    dialog = apps_measurements_viewer.TimepointPredictorEquationDialog(
        ['GFP'], False, False, acdc_df=acdc_df, name='Bud'
    )
    assert {'ANY', 'ALL'} <= {
        button.text() for button in dialog.categoricalOperatorButtons
    }
    assert dialog.otherMeasurementColumns['area_other'] == 'area'
    assert dialog.otherMeasurementColumns['stage_other'] == 'stage'
    assert not hasattr(dialog, 'otherCellFunctionButtons')
    assert not dialog.otherMeasurementsTreeWidget.isEnabled()
    assert dialog.currentOtherCells() is None
    dialog.otherCellsGroup.setChecked(True)
    assert dialog.otherMeasurementsTreeWidget.isEnabled()
    areaOther = [
        dialog.otherMeasurementsTreeWidget.topLevelItem(0).child(i)
        for i in range(
            dialog.otherMeasurementsTreeWidget.topLevelItem(0).childCount()
        )
    ]
    # assert any(item.text(0) == 'Other numerical inputs' for item in areaOther)
    dialog.otherCellsGroup.setChecked(False)
    assert not dialog.otherMeasurementsTreeWidget.isEnabled()
    dialog.mainCellGroup.setChecked(True)
    dialog.mainCellFilterCombobox.setCurrentIndex(
        dialog.mainCellFilterCombobox.findData('custom')
    )
    dialog.mainCellConditionEdit.setText('area > 5')
    dialog.mainCellMatchCombobox.setCurrentIndex(
        dialog.mainCellMatchCombobox.findData('all')
    )
    assert dialog.currentMainCellFilter() == {
        'condition': 'area > 5',
        'condition_mode': 'all',
    }
    dialog.otherCellsGroup.setChecked(True)
    dialog.otherCellsRangeStartEdit.setText('t - 2')
    dialog.otherCellsRangeEndEdit.setText('t + 1')
    assert dialog.currentOtherCells()['range_start'] == 't - 2'
    assert dialog.currentOtherCells()['range_end'] == 't + 1'
    dialog.otherCellsGroup.setChecked(False)
    dialog.formulaEditWidget.setText('area > 16')
    dialog.modeCombobox.setCurrentIndex(
        dialog.modeCombobox.findData('rising_true')
    )
    dialog.ok_cb()
    assert dialog.cancel
    assert dialog.okButton.isEnabled()
    assert '1 of 1 cells' in dialog.statusLabel.text()
    dialog.ok_cb()
    assert not dialog.cancel
    assert (dialog.equation, dialog.mode) == ('area > 16', 'rising_true')
    assert dialog.mainCellFilter == {
        'condition': 'area > 5',
        'condition_mode': 'all',
    }
    dialog.deleteLater()


def test_other_cell_predictor_equation_dialog_saves_filter_settings(app, acdc_df):
    settings = {
        'source': 'same_frame',
        'condition': 'area > 5',
        'condition_mode': 'any',
        'range_start': 't - 1',
        'range_end': 't + 2',
    }
    dialog = apps_measurements_viewer.TimepointPredictorEquationDialog(
        ['GFP'], False, False, acdc_df=acdc_df, name='Bud',
        equation='MEAN(area_other) > 10', otherCells=settings
    )
    assert dialog.otherCellsGroup.isChecked()
    assert dialog.currentOtherCells() == settings
    dialog.test_cb()
    dialog.ok_cb()
    assert not dialog.cancel
    assert dialog.otherCells == settings
    dialog.deleteLater()


def test_unified_cell_filter_selectors(app, acdc_df):
    dialog = apps_measurements_viewer.TimepointPredictorEquationDialog(
        [], False, False, acdc_df=acdc_df, SizeT=4
    )
    for group, selector, edit, button, controls in (
            (dialog.mainCellGroup, dialog.mainCellFilterCombobox,
             dialog.mainCellConditionEdit, dialog.mainCellConditionButton,
             (dialog.mainCellMatchCombobox,)),
            (dialog.otherCellsGroup, dialog.otherCellsSourceCombobox,
             dialog.otherCellsConditionEdit, dialog.otherCellsConditionButton,
             (dialog.otherCellsMatchCombobox,
              dialog.otherCellsRangeStartEdit, dialog.otherCellsRangeEndEdit))
        ):
        assert not selector.isEnabled()
        assert all(not control.isEnabled() for control in controls)
        group.setChecked(True)
        assert selector.isEnabled()
        assert not edit.isEnabled()
        assert not button.isEnabled()
        assert all(not control.isEnabled() for control in controls)
        selector.setCurrentIndex(selector.findData('custom'))
        assert edit.isEnabled()
        assert not edit.isReadOnly()
        assert button.isEnabled()
        assert all(control.isEnabled() for control in controls)
        edit.setText('area > 5')
        selector.setCurrentIndex(0)
        assert not edit.isEnabled()
        assert edit.text() == ''
        assert all(not control.isEnabled() for control in controls)
        selector.setCurrentIndex(selector.findData('custom'))
        assert edit.text() == 'area > 5'
        assert all(control.isEnabled() for control in controls)
        group.setChecked(False)
        assert not edit.isEnabled()
        assert all(not control.isEnabled() for control in controls)
        group.setChecked(True)
        assert edit.isEnabled()
        assert all(control.isEnabled() for control in controls)
    dialog.deleteLater()


@pytest.mark.parametrize('preset,expected', [
    ('All cells', {1: [0], 2: [1]}),
    ('Present at the first frame', {1: [0]}),
    ('Not present at the first frame', {2: [1]}),
    ('Present at the last frame', {1: [0], 2: [1]}),
    ('Not present at the last frame', {}),
    ('Present throughout the video', {1: [0]}),
])
def test_main_cell_filter_presets(app, acdc_df, preset, expected):
    df = acdc_df.drop(index=(0, 2))
    dialog = apps_measurements_viewer.TimepointPredictorEquationDialog(
        [], False, False, acdc_df=df, SizeT=4
    )
    dialog.mainCellGroup.setChecked(True)
    dialog.mainCellFilterCombobox.setCurrentIndex(
        dialog.mainCellFilterCombobox.findData(preset)
    )
    settings = dialog.currentMainCellFilter()
    assert not dialog.mainCellConditionEdit.isEnabled()
    assert not dialog.mainCellMatchCombobox.isEnabled()
    assert dialog.mainCellConditionEdit.text() == settings['condition']
    assert timepoint_predictors.TimepointPredictor.predict(df, {
        'type': 'equation',
        'equation': 'area > 0',
        'mode': 'first_true',
        'main_cell_filter': settings,
    }, SizeT=4) == expected
    reopened = apps_measurements_viewer.TimepointPredictorEquationDialog(
        [], False, False, acdc_df=df, SizeT=4, mainCellFilter=settings
    )
    assert reopened.mainCellFilterCombobox.currentData() == preset
    assert reopened.currentMainCellFilter() == settings
    reopened.deleteLater()
    dialog.deleteLater()


def test_main_cell_annotation_presets(app, acdc_df):
    df = acdc_df.copy()
    df['relationship'] = np.where(
        df.index.get_level_values('Cell_ID') == 1, 'mother', 'bud'
    )
    df['cell_cycle_stage'] = df['stage']
    dialog = apps_measurements_viewer.TimepointPredictorEquationDialog(
        [], False, False, acdc_df=df
    )
    dialog.mainCellGroup.setChecked(True)
    assert 'Present at the last frame' not in dialog.mainCellFilterPresets
    for preset, expected in (
            ('Mother cells (budding)', {1: [0]}),
            ('Bud cells (budding)', {2: [0]}),
            ('Cells in S (budding)', {1: [0]}),
            ('Cells in G1 (budding)', {1: [0], 2: [0]})
        ):
        dialog.mainCellFilterCombobox.setCurrentIndex(
            dialog.mainCellFilterCombobox.findData(preset)
        )
        assert timepoint_predictors.TimepointPredictor.predict(df, {
            'type': 'equation', 'equation': 'area > 0',
            'mode': 'first_true',
            'main_cell_filter': dialog.currentMainCellFilter(),
        }) == expected
    dialog.deleteLater()


def test_unified_other_cell_filter_materializes_legacy_source(app, lineage_df):
    settings = dict(
        timepoint_predictors.CellFilter.default_other(),
        source='mother', condition='area > 5', condition_mode='all'
    )
    dialog = apps_measurements_viewer.TimepointPredictorEquationDialog(
        [], False, False, acdc_df=lineage_df, otherCells=settings
    )
    expected = dict(
        settings, source='same_frame',
        condition='(Cell_ID == this_parent_ID_tree) & (area > 5)'
    )
    assert dialog.otherCellsSourceCombobox.currentData() == 'custom'
    assert dialog.otherCellsConditionEdit.isEnabled()
    assert dialog.currentOtherCells() == expected
    dialog.otherCellsSourceCombobox.setCurrentIndex(
        dialog.otherCellsSourceCombobox.findData('sister')
    )
    assert dialog.currentOtherCells()['source'] == 'same_frame'
    assert dialog.currentOtherCells()['condition'] == (
        '(parent_ID_tree > 0) & (parent_ID_tree == this_parent_ID_tree)'
    )
    dialog.otherCellsSourceCombobox.setCurrentIndex(
        dialog.otherCellsSourceCombobox.findData('custom')
    )
    assert dialog.currentOtherCells() == expected
    dialog.deleteLater()


@pytest.mark.parametrize('dialogClass', [
    apps_measurements_viewer.TimepointPredictorEquationDialog,
    apps_measurements_viewer.CustomAnnotationPredictorEquationDialog,
])
@pytest.mark.parametrize('withData', [True, False])
def test_equation_dialog_keeps_unavailable_relation_source(
        app, acdc_df, dialogClass, withData
    ):
    # acdc_df has no lineage tree or budding columns
    settings = dict(
        timepoint_predictors.CellFilter.default_other(),
        source='daughter', range_start='t + 1', range_end='t + 1'
    )
    dialog = dialogClass(
        ['GFP'], False, False, acdc_df=acdc_df if withData else None,
        otherCells=settings, SizeT=4
    )
    combobox = dialog.otherCellsSourceCombobox
    assert combobox.currentData() == 'daughter'
    assert 'annotations missing' in combobox.currentText()
    assert not dialog.otherCellsMissingSourceLabel.isHidden()
    assert not dialog.otherCellsConditionEdit.isEnabled()
    assert dialog.currentOtherCells() == settings
    assert dialog.otherCells == settings
    combobox.setCurrentIndex(combobox.findData('custom'))
    assert dialog.otherCellsMissingSourceLabel.isHidden()
    assert dialog.currentOtherCells()['source'] == 'same_frame'
    combobox.setCurrentIndex(combobox.findData('daughter'))
    assert dialog.currentOtherCells() == settings
    dialog.deleteLater()


def test_identifier_edit_equation_opens_with_unavailable_relation_source(
        app, acdc_df, monkeypatch
    ):
    settings = dict(
        timepoint_predictors.CellFilter.default_other(),
        source='mother', range_start='t', range_end='t'
    )
    predictor = {
        'type': timepoint_predictors.PREDICTOR_EQUATION,
        'equation': 'COUNT(area_other) > 0',
        'mode': 'first_true',
        'other_cells': settings,
        'main_cell_filter': None,
    }
    opened = []

    def exec_(win):
        opened.append(win.currentOtherCells())
        win.cancel = True

    monkeypatch.setattr(
        apps_measurements_viewer.TimepointPredictorEquationDialog, 'exec_',
        exec_
    )
    dialog = apps_measurements_viewer.TimepointIdentifierDialog(
        'M', (255, 0, 0), 'Q', predictor=predictor,
        acdc_df=acdc_df, SizeT=4, allChNames=['GFP']
    )
    dialog.editEquation()
    assert opened == [settings]
    assert dialog.otherCells == settings
    dialog.deleteLater()


@pytest.mark.parametrize('start, end', [('', 't'), ('t -', 't'), ('t + 1', 't')])
def test_edit_filter_with_invalid_frame_window(app, acdc_df, monkeypatch, start, end):
    opened = []

    def exec_(win):
        opened.append(win.formulaEditWidget.text())
        win.cancel = True

    monkeypatch.setattr(
        apps_measurements_viewer._TimepointCellFilterEquationDialog, 'exec_',
        exec_
    )
    dialog = apps_measurements_viewer.TimepointPredictorEquationDialog(
        ['GFP'], False, False, acdc_df=acdc_df, SizeT=4,
        otherCells=timepoint_predictors.CellFilter.default_other(),
        mainCellFilter=timepoint_predictors.CellFilter.default_main()
    )
    for combobox in (
            dialog.mainCellFilterCombobox, dialog.otherCellsSourceCombobox
        ):
        combobox.setCurrentIndex(combobox.findData('custom'))
    dialog.otherCellsConditionEdit.setText('area > this_area')
    dialog.otherCellsRangeStartEdit.setText(start)
    dialog.otherCellsRangeEndEdit.setText(end)
    # The candidate filter editor does not need a valid frame window
    dialog.editOtherCellsCondition()
    assert opened == ['area > this_area']
    # The main-cell filter editor needs it: report the error instead
    dialog.editMainCellCondition()
    assert len(opened) == 1
    assert 'Invalid other-cell settings' in dialog.statusLabel.text()
    dialog.deleteLater()


def test_predictor_equation_dialog_functions(app, acdc_df):
    dialog = apps_measurements_viewer.TimepointPredictorEquationDialog(
        ['GFP'], False, False, acdc_df=acdc_df, name='Bud'
    )
    names = [button.text() for button in dialog.cellFunctionButtons]
    assert names == [
        name for name in timepoint_predictors.CELL_FUNCTIONS
        if name not in ('any', 'all', 'ANY', 'ALL')
    ]
    assert {'ANY', 'ALL'} <= {
        button.text() for button in dialog.categoricalOperatorButtons
    }
    dialog.formulaEditWidget.setText('area > 0.5 * ')
    dialog.cellFunctionButtons[names.index('MAX')].click()
    dialog.insertEquationText('area)')
    assert dialog.formulaEditWidget.text() == 'area > 0.5 * MAX(area)'
    assert dialog.formulaEditWidget.checkValidity()[0]
    dialog.ok_cb()
    assert dialog.okButton.isEnabled()
    assert '2 of 2 cells' in dialog.statusLabel.text()
    dialog.formulaEditWidget.setText('area > UNKNOWN(area)')
    assert not dialog.formulaEditWidget.checkValidity()[0]
    dialog.deleteLater()


def test_timepoint_filter_equation_dialogs(app, acdc_df):
    mainFilter = apps_measurements_viewer._TimepointCellFilterEquationDialog(
        ['GFP'], False, False, acdc_df=acdc_df, SizeT=4,
        condition='area > 5'
    )
    assert mainFilter.validateEquation('area > 5')[0]
    assert not mainFilter.validateEquation('area')[0]
    mainFilter.ok_cb()
    assert not mainFilter.cancel
    assert mainFilter.equation == 'area > 5'
    mainFilter.deleteLater()

    otherFilter = apps_measurements_viewer._TimepointCellFilterEquationDialog(
        ['GFP'], False, False, acdc_df=acdc_df, SizeT=4,
        condition='area < this_area', includeEnquiringCell=True
    )
    assert 'this_area' in otherFilter.variableNames
    assert otherFilter.validateEquation('area < this_area')[0]
    otherFilter.deleteLater()


def test_filter_editor_checks_all_rows_and_matches_runtime(app):
    df = pd.DataFrame({
        'frame_i': list(range(12)),
        'Cell_ID': [1] * 12,
        'value': [1.] * 11 + [2.],
    }).set_index(['frame_i', 'Cell_ID'])
    dialog = apps_measurements_viewer._TimepointCellFilterEquationDialog(
        [], False, False, acdc_df=df, condition='value'
    )
    assert dialog.validateEquation('value')[0]
    dialog.test_cb()
    assert not dialog.okButton.isEnabled()
    assert '0/1' in dialog.statusLabel.text()
    with pytest.raises(ValueError, match='0/1'):
        timepoint_predictors.TimepointPredictor.from_equation(
            df, 'value > 0', 'first_true',
            main_cell_filter={'condition': 'value', 'condition_mode': 'any'}
        )
    dialog.deleteLater()


def test_filter_editor_ok_skips_retest_after_passing_test(app, monkeypatch):
    df = pd.DataFrame({
        'frame_i': list(range(12)),
        'Cell_ID': [1] * 12,
        'value': [1.] * 11 + [2.],
    }).set_index(['frame_i', 'Cell_ID'])
    Dialog = apps_measurements_viewer._TimepointCellFilterEquationDialog
    # OK without a prior Test still validates the whole table
    dialog = Dialog([], False, False, acdc_df=df, condition='value')
    dialog.ok_cb()
    assert dialog.cancel and not dialog.okButton.isEnabled()
    assert '0/1' in dialog.statusLabel.text()
    dialog.deleteLater()

    dialog = Dialog([], False, False, acdc_df=df, condition='value > 1')
    fullEvaluations = []
    evaluate = dialog.evaluateEquation

    def countingEvaluate(*args, preview=False, **kwargs):
        if not preview:
            fullEvaluations.append(args)
        return evaluate(*args, preview=preview, **kwargs)

    monkeypatch.setattr(dialog, 'evaluateEquation', countingEvaluate)
    dialog.test_cb()
    assert dialog.okButton.isEnabled() and len(fullEvaluations) == 1
    dialog.ok_cb()
    assert len(fullEvaluations) == 1
    assert not dialog.cancel and dialog.equation == 'value > 1'
    dialog.deleteLater()


def test_other_cell_filter_editor_adds_enquiring_inputs(app, acdc_df):
    dialog = apps_measurements_viewer._TimepointCellFilterEquationDialog(
        [], False, False, acdc_df=acdc_df, condition='area > this_area',
        includeEnquiringCell=True
    )
    sample = dialog.getTestData()
    assert len(sample) == len(
        timepoint_predictors.CellTable.flatten(acdc_df)
    )
    assert 'this_area' in sample.columns and 'area' in sample.columns
    dialog.test_cb()
    assert dialog.okButton.isEnabled()
    dialog.deleteLater()


def test_filter_editor_preserves_frame_index_inputs(app, acdc_df):
    dialog = apps_measurements_viewer._TimepointCellFilterEquationDialog(
        [], False, False, acdc_df=acdc_df,
        condition='FIRST(frame_i) == 0'
    )
    sample = dialog.getTestData()
    assert 'frame_i' not in sample.columns
    result = dialog.evaluateEquation(
        sample, 'FIRST(frame_i) == 0', enforce_dtype_output=bool
    )
    assert result.all()
    dialog.test_cb()
    assert dialog.okButton.isEnabled()
    dialog.deleteLater()


def test_predictor_identifier_retains_other_cell_options(app, acdc_df):
    settings = {
        'source': 'same_frame',
        'condition': 'area > this_area',
        'condition_mode': 'any',
        'range_start': 't - 1',
        'range_end': 't + 2',
    }
    predictor = {
        'type': timepoint_predictors.PREDICTOR_EQUATION,
        'equation': 'MAX_OTHER(area) > 10',
        'mode': 'first_true',
        'other_cells': settings,
        'main_cell_filter': {
            'condition': 'area > 5',
            'condition_mode': 'any',
        },
    }
    dialog = apps_measurements_viewer.TimepointIdentifierDialog(
        'Bud', (255, 0, 0), 'Q', predictor=predictor,
        acdc_df=acdc_df, SizeT=4, allChNames=['GFP']
    )
    assert dialog.predictor() == predictor
    dialog.deleteLater()

# def test_cell_constants_presets_and_values(acdc_df):
#     constants = [
#         timepoint_predictors.constant_from_preset('Maximum value', 'area'),
#         timepoint_predictors.constant_from_preset('Was at beginning'),
#         timepoint_predictors.constant_from_preset('Was at end'),
#         timepoint_predictors.constant_from_preset('Number of frames'),
#         {'name': 'half_max', 'equation': 'max_area / 2'},
#     ]
#     assert constants[0] == timepoint_predictors.default_constant(
#         'max_area', 'MAX(area)'
#     )
#     df = acdc_df.drop(index=(0, 2))
#     values = timepoint_predictors.compute_cell_constants(df, constants, SizeT=5)
#     assert values == {
#         'max_area': {1: 30.0, 2: 7.0},
#         'was_at_beginning': {1: True, 2: False},
#         'was_at_end': {1: False, 2: False},
#         'num_frames': {1: 4, 2: 3},
#         'half_max': {1: 15.0, 2: 3.5},
#     }
#     withConstants = timepoint_predictors.add_constants_columns(df, values)
#     assert 'max_area' not in df
#     assert withConstants.loc[(3, 2), 'max_area'] == 7.0
#     predicted = timepoint_predictors.predict_timepoints(
#         withConstants, 
#         {'type': 'equation', 'equation': 'area >= half_max', 'mode': 'first_true'}
#     )
#     assert predicted == {1: [1], 2: [1]}


# @pytest.mark.parametrize('constants, match', [
#     ([{'name': 'x', 'equation': 'area'}], 'more than one value'),
#     ([{'name': '1x', 'equation': 'MAX(area)'}], 'Invalid constant name'),
#     ([{'name': 'MAX', 'equation': 'MAX(area)'}], 'built-in'),
#     ([{'name': 'area', 'equation': 'MAX(area)'}], 'already a column'),
#     ([{'name': 'a', 'equation': '1'}, {'name': 'a', 'equation': '2'}], 'more than once'),
#     ([{'name': 'a', 'equation': ' '}], 'cannot be empty'),
# ])
# def test_cell_constants_validation(acdc_df, constants, match):
#     with pytest.raises(ValueError, match=match):
#         timepoint_predictors.validate_constants(
#             constants, columns=['area', 'frame_i']
#         )
#         timepoint_predictors.compute_cell_constants(acdc_df, constants)


# def test_cell_constants_dialog(app, acdc_df, monkeypatch):
#     monkeypatch.setattr(
#         apps.CellConstantDefinitionDialog, 'exec_',
#         lambda editor: editor.ok_cb()
#     )
#     dialog = apps.CellConstantsDialog(
#         constants=[{'name': 'first_area', 'equation': 'FIRST(area)'}],
#         acdc_df=acdc_df, SizeT=4
#     )
#     assert 'area' in [
#         dialog.columnCombobox.itemText(i) 
#         for i in range(dialog.columnCombobox.count())
#     ]
#     dialog.presetCombobox.setCurrentText('Was at beginning')
#     assert not dialog.columnCombobox.isEnabled()
#     dialog.addPreset()
#     dialog.presetCombobox.setCurrentText('Maximum value')
#     assert dialog.columnCombobox.isEnabled()
#     dialog.columnCombobox.setCurrentText('area')
#     dialog.addPreset()
#     dialog.addPreset()
#     assert [c['name'] for c in dialog.currentConstants()] == [
#         'first_area', 'was_at_beginning', 'max_area', 'max_area_2'
#     ]
#     assert dialog.preview()
#     assert 'max_area_2' in dialog.previewDisplay.toHtml()
#     dialog.table.selectRow(3)
#     dialog.removeSelected()
#     dialog.addConstant(timepoint_predictors.default_constant())
#     warnings = []
#     monkeypatch.setattr(dialog, 'warn', lambda *args: warnings.append(args))
#     dialog.ok_cb()
#     assert dialog.cancel and warnings
#     dialog.updateConstantRow(3, timepoint_predictors.default_constant(
#         'constant', 'MAX(area) > 20'
#     ))
#     dialog.ok_cb()
#     assert not dialog.cancel
#     assert dialog.constants[-1] == timepoint_predictors.default_constant(
#         'constant', 'MAX(area) > 20'
#     )
#     dialog.deleteLater()


# def test_cell_constants_dialog_source_cells(app, acdc_df, monkeypatch):
#     df = acdc_df.copy()
#     df['relative_ID'] = [2, 2, 2, 2, 1, 1, 1, 1]
#     df['relationship'] = ['mother'] * 4 + ['bud'] * 4
#     dialog = apps.CellConstantsDialog(
#         constants=[{'name': 'first_area', 'equation': 'FIRST(area)'}],
#         acdc_df=df, SizeT=4
#     )
#     assert dialog.currentConstants()[0]['source'] == 'this'
#     def accept_definition(editor):
#         if editor.constant['name'] == 'max_area':
#             editor.sourceCombobox.setCurrentIndex(
#                 editor.sourceCombobox.findData('daughter')
#             )
#             editor.nameEdit.setText('max_area_daughter')
#             editor.ok_cb()
#             return
#         editor.sourceCombobox.setCurrentIndex(
#             editor.sourceCombobox.findData('custom')
#         )
#         editor.scopeCombobox.setCurrentIndex(1)
#         editor.equationEdit.setText('area')
#         # Custom source requires a condition.
#         editor.ok_cb()
#         assert editor.cancel
#         editor.conditionEdit.setText('Cell_ID == this_relative_ID')
#         editor.ok_cb()
#         assert not editor.cancel
#     monkeypatch.setattr(apps.CellConstantDefinitionDialog, 'exec_', accept_definition)
#     dialog.presetCombobox.setCurrentText('Maximum value')
#     dialog.columnCombobox.setCurrentText('area')
#     dialog.addPreset()
#     dialog.addCustomConstant()
#     assert dialog.preview()
#     assert 'frame_i' in dialog.previewDisplay.toHtml()
#     constants = dialog.currentConstants()
#     assert constants[1] == timepoint_predictors.default_constant(
#         'max_area_daughter', 'MAX(area)', 'daughter'
#     )
#     assert constants[2] == dict(
#         timepoint_predictors.default_constant('constant', 'area', 'custom'),
#         condition='Cell_ID == this_relative_ID', per_frame=True, time_scope='frame'
#     )
#     dialog.ok_cb()
#     assert not dialog.cancel and dialog.constants == constants
#     dialog.deleteLater()


# @pytest.mark.parametrize('mode,expected', [
#     ('any', 7.0), ('all', np.nan), ('where_true', 7.0),
# ])
# @pytest.mark.parametrize('per_frame', [False, True])
# def test_constant_filter_frame_rules(acdc_df, mode, expected, per_frame):
#     constant = dict(
#         timepoint_predictors.default_constant('filtered', 'MAX(area)', 'custom'),
#         condition='area > 5', condition_mode=mode, per_frame=per_frame
#     )
#     values = timepoint_predictors.compute_cell_constants(acdc_df, [constant])['filtered']
#     # Enquirer 1 sees candidate 2 with areas [5, 6, 7, 4].
#     if per_frame:
#         actual = values.xs(1, level='Cell_ID').to_numpy()
#         expected = {
#             'any': [5, 6, 7, 4],
#             'all': [np.nan] * 4,
#             'where_true': [np.nan, 6, 7, np.nan],
#         }[mode]
#         np.testing.assert_allclose(actual, expected, equal_nan=True)
#     elif np.isnan(expected):
#         assert np.isnan(values[1])
#     else:
#         # Only true frames are [6, 7]; test MAX and FIRST independently.
#         assert values[1] == expected
#         constant['equation'] = 'FIRST(area)'
#         first = timepoint_predictors.compute_cell_constants(acdc_df, [constant])
#         assert first['filtered'][1] == (6 if mode == 'where_true' else 5)


# @pytest.mark.parametrize('offset', [-1, 1])
# def test_constant_range_and_enquiring_inputs(acdc_df, offset):
#     constant = dict(
#         timepoint_predictors.default_constant('windowed', 'area + this_area', 'custom'),
#         condition='Cell_ID != this_Cell_ID', time_scope='custom',
#         range_start=f't + {offset}', range_end=f't + {offset}'
#     )
#     result = timepoint_predictors.compute_cell_constants(acdc_df, [constant])['windowed']
#     candidate = acdc_df.xs(2, level='Cell_ID')['area']
#     enquiring = acdc_df.xs(1, level='Cell_ID')['area']
#     for t in range(4):
#         actual = result.loc[(t, 1)]
#         if 0 <= t + offset < 4:
#             assert actual == candidate.loc[t + offset] + enquiring.loc[t + offset]
#         else:
#             assert np.isnan(actual)


# @pytest.mark.parametrize('output', ['2', '-1', '0.5', 'float("nan")', '"yes"', '[True]', '[[True]]'])
# def test_custom_file_filter_rejects_invalid_output(tmp_path, acdc_df, output):
#     path = tmp_path / 'cell_filter.py'
#     path.write_text(f'def filter_cells(df, enquiring_df):\n    return {output}\n')
#     constant = dict(
#         timepoint_predictors.default_constant('filtered', 'MAX(area)', 'custom'),
#         condition_type='file', condition_filepath=str(path)
#     )
#     with pytest.raises(ValueError, match='filter|Filter'):
#         timepoint_predictors.compute_cell_constants(acdc_df, [constant])


# def test_custom_file_filter_range_and_alignment(tmp_path, acdc_df):
#     path = tmp_path / 'cell_filter.py'
#     path.write_text(
#         'def filter_cells(df, enquiring_df):'
#         '\n    assert (df["frame_i"] == enquiring_df["frame_i"]).all()'
#         '\n    return (df["area"].to_numpy() < enquiring_df["area"].to_numpy()).astype(int)\n'
#     )
#     constant = dict(
#         timepoint_predictors.default_constant('filtered', 'MAX(area)', 'custom'),
#         condition_type='file', condition_filepath=str(path), time_scope='custom',
#         range_start='t', range_end='t + 1'
#     )
#     result = timepoint_predictors.compute_cell_constants(acdc_df, [constant])['filtered']
#     np.testing.assert_allclose(
#         result.xs(1, level='Cell_ID'), [6, 7, 7, 4], equal_nan=True
#     )


# @pytest.mark.parametrize('formula', ['area', '2', 'area / this_area'])
# def test_custom_equation_filter_requires_boolean(acdc_df, formula):
#     constant = dict(
#         timepoint_predictors.default_constant('filtered', 'MAX(area)', 'custom'),
#         condition=formula
#     )
#     with pytest.raises(ValueError, match='bool|0/1'):
#         timepoint_predictors.compute_cell_constants(acdc_df, [constant])


# def test_constant_formula_editor_two_input_groups(app, acdc_df):
#     constant = dict(
#         timepoint_predictors.default_constant('filtered', 'MAX(area)', 'custom'),
#         condition='area < this_area'
#     )
#     dialog = apps.CellConstantEquationDialog(
#         constant, acdc_df=acdc_df, SizeT=4, condition=True
#     )
#     assert len(dialog.metricsTreeWidgets) == 2
#     assert 'area' in dialog.variableNames and 'this_area' in dialog.variableNames
#     assert 'area_table1' not in dialog.variableNames
#     dialog.formulaEditWidget.setText('area < this_area')
#     dialog.ok_cb()
#     assert not dialog.cancel
#     dialog.formulaEditWidget.setText('2')
#     dialog.test_cb()
#     assert not dialog.okButton.isEnabled()
#     dialog.deleteLater()


# def test_constant_value_formula_editor(app, acdc_df):
#     dialog = apps.CellConstantEquationDialog(
#         timepoint_predictors.default_constant('max_area', 'MAX(area)'),
#         acdc_df=acdc_df, SizeT=4
#     )
#     dialog.ok_cb()
#     assert not dialog.cancel and dialog.equation == 'MAX(area)'
#     dialog.deleteLater()


# @pytest.mark.parametrize('condition,expected', [
#     ('True', 1), ('1', 1), ('False', 0), ('0', 0),
#     ('area > 0', 1),
# ])
# def test_constant_all_accepts_bool_and_binary_values(acdc_df, condition, expected):
#     constant = dict(
#         timepoint_predictors.default_constant('selected', 'NUNIQUE(Cell_ID)', 'custom'),
#         condition=condition, condition_mode='all'
#     )
#     result = timepoint_predictors.compute_cell_constants(acdc_df, [constant])
#     assert result['selected'] == {1: expected, 2: expected}


# def test_constant_range_uses_source_frame_in_formula(acdc_df):
#     constant = dict(
#         timepoint_predictors.default_constant('next_frame', 'FIRST(frame_i)', 'custom'),
#         condition='True', time_scope='custom', range_start='t+1', range_end='t+1'
#     )
#     result = timepoint_predictors.compute_cell_constants(acdc_df, [constant])
#     np.testing.assert_allclose(
#         result['next_frame'].xs(1, level='Cell_ID'), [1, 2, 3, np.nan],
#         equal_nan=True
#     )


# def test_constant_definition_preserves_options(app, acdc_df):
#     editor = apps.CellConstantDefinitionDialog(
#         timepoint_predictors.default_constant('filtered', 'MEAN(area)', 'custom'),
#         acdc_df=acdc_df, SizeT=4
#     )
#     editor.scopeCombobox.setCurrentIndex(editor.scopeCombobox.findData('custom'))
#     editor.conditionEdit.setText('area < this_area')
#     editor.matchCombobox.setCurrentIndex(editor.matchCombobox.findData('any'))
#     editor.rangeStartEdit.setText('t - 1')
#     editor.rangeEndEdit.setText('t + 1')
#     editor.ok_cb()
#     assert not editor.cancel
#     assert editor.constant == dict(
#         timepoint_predictors.default_constant('filtered', 'MEAN(area)', 'custom'),
#         condition='area < this_area', condition_mode='any', time_scope='custom',
#         range_start='t - 1', range_end='t + 1', per_frame=True
#     )
#     editor.deleteLater()


# @pytest.mark.parametrize('source', ['this', 'same_frame'])
# def test_constant_custom_range_inclusive_and_clipped(acdc_df, source):
#     constant = dict(
#         timepoint_predictors.default_constant('window_max', 'MAX(area)', source),
#         time_scope='custom', range_start='t - 1', range_end='t + 1'
#     )
#     result = timepoint_predictors.compute_cell_constants(acdc_df, [constant])
#     expected = [20, 20, 30, 30] if source == 'this' else [6, 7, 7, 7]
#     np.testing.assert_allclose(
#         result['window_max'].xs(1, level='Cell_ID'), expected
#     )
#     assert timepoint_predictors.normalize_constant(constant)['per_frame']


# @pytest.mark.parametrize('mode,expected', [
#     ('any', [6, 7, 7, 7]),
#     ('all', [np.nan, np.nan, np.nan, np.nan]),
#     ('where_true', [6, 7, 7, 7]),
# ])
# def test_constant_custom_range_filters_each_window(acdc_df, mode, expected):
#     constant = dict(
#         timepoint_predictors.default_constant('window_max', 'MAX(area)', 'custom'),
#         time_scope='custom', range_start='t - 1', range_end='t + 1',
#         condition='area > 5', condition_mode=mode
#     )
#     result = timepoint_predictors.compute_cell_constants(acdc_df, [constant])
#     # At t=1 the window is [5,6,7]: All fails, Any retains all, Where retains [6,7].
#     np.testing.assert_allclose(
#         result['window_max'].xs(1, level='Cell_ID'), expected, equal_nan=True
#     )
#     constant['equation'] = 'FIRST(area)'
#     first = timepoint_predictors.compute_cell_constants(acdc_df, [constant])
#     if mode == 'any':
#         assert first['window_max'].loc[(1, 1)] == 5
#     elif mode == 'where_true':
#         assert first['window_max'].loc[(1, 1)] == 6
#     # A narrower [t,t+1] window at t=1 has both values True.
#     constant['range_start'] = 't'
#     constant['equation'] = 'MAX(area)'
#     narrow = timepoint_predictors.compute_cell_constants(acdc_df, [constant])
#     assert narrow['window_max'].loc[(1, 1)] == 7


# def test_constant_custom_range_gaps_empty_counts_and_dependencies(acdc_df):
#     df = acdc_df.drop(index=[(1, 1), (1, 2)])
#     constants = [
#         dict(
#             timepoint_predictors.default_constant('window_max', 'MAX(area)'),
#             time_scope='custom', range_start='t', range_end='t + 1'
#         ),
#         timepoint_predictors.default_constant('max_window', 'MAX(window_max)'),
#         dict(
#             timepoint_predictors.default_constant('future_count', 'NUNIQUE(frame_i)'),
#             time_scope='custom', range_start='t + 10', range_end='t + 20'
#         ),
#     ]
#     result = timepoint_predictors.compute_cell_constants(df, constants)
#     np.testing.assert_allclose(
#         result['window_max'].xs(1, level='Cell_ID'), [10, 30, 30]
#     )
#     assert result['max_window'] == {1: 30, 2: 7}
#     assert (result['future_count'] == 0).all()
#     added = timepoint_predictors.add_constants_columns(df, result)
#     assert added.loc[(2, 1), 'window_max'] == 30
#     assert added.index.equals(df.index)


# @pytest.mark.parametrize('start,end', [
#     ('t + 1', 't - 1'), ('', 't'), ('t', 't/2'),
#     ('t', '1.5'), ('t', 'True'), ('x', 't'),
#     ('t', '__import__("os")'), ('t', 't // 0'),
# ])
# def test_constant_custom_range_rejects_invalid_bounds(start, end):
#     constant = dict(
#         timepoint_predictors.default_constant('c', 'MAX(area)'),
#         time_scope='custom', range_start=start, range_end=end
#     )
#     with pytest.raises(ValueError, match='range|bounds|integers'):
#         timepoint_predictors.validate_constants([constant])


# def test_constant_frame_bound_arithmetic():
#     tp = timepoint_predictors
#     assert tp.frame_range_bounds('(t // 2) * 2', 't + (3 % 2)', 5) == (4, 6)
#     assert tp.normalize_constant({'name': 'x', 'equation': 'MAX(area)'})[
#         'time_scope'
#     ] == 'video'
#     assert tp.normalize_constant({
#         'name': 'x', 'equation': 'MAX(area)', 'per_frame': True
#     })['time_scope'] == 'frame'
#     assert 'frame_shift' not in tp.normalize_constant({
#         'name': 'x', 'equation': 'MAX(area)', 'frame_shift': 1
#     })


# def test_constant_custom_range_ui_visibility(app, acdc_df):
#     editor = apps.CellConstantDefinitionDialog(
#         timepoint_predictors.default_constant('window_max', 'MAX(area)'),
#         acdc_df=acdc_df, SizeT=4
#     )
#     assert editor.rangeStartEdit.isHidden() and editor.rangeEndEdit.isHidden()
#     assert not hasattr(editor, 'shiftSpinbox')
#     editor.scopeCombobox.setCurrentIndex(editor.scopeCombobox.findData('custom'))
#     assert not editor.rangeStartEdit.isHidden()
#     assert not editor.rangeEndEdit.isHidden()
#     editor.rangeStartEdit.setText('t + 2')
#     editor.rangeEndEdit.setText('t')
#     editor.ok_cb()
#     assert editor.cancel and 'exceeds' in editor.statusLabel.text()
#     editor.rangeStartEdit.setText('t - 2')
#     editor.ok_cb()
#     assert not editor.cancel
#     editor.deleteLater()


# def test_constant_manager_edits_with_previous_constants(app, acdc_df, monkeypatch):
#     tp = timepoint_predictors
#     dialog = apps.CellConstantsDialog(
#         constants=[
#             tp.default_constant('max_area', 'MAX(area)'),
#             tp.default_constant('half_max', 'max_area / 2'),
#         ], acdc_df=acdc_df, SizeT=4
#     )
#     def edit_definition(editor):
#         assert 'max_area' in editor.acdc_df
#         assert 'half_max' not in editor.acdc_df
#         editor.equationEdit.setText('max_area / 4')
#         editor.ok_cb()
#     monkeypatch.setattr(apps.CellConstantDefinitionDialog, 'exec_', edit_definition)
#     dialog.table.selectRow(1)
#     dialog.editSelected()
#     assert dialog.currentConstants()[1]['equation'] == 'max_area / 4'
#     assert dialog.preview()
#     dialog.deleteLater()


@pytest.fixture
def lineage_df():
    # Cell 1 divides into 2 and 3 at frame 2, cell 4 is unrelated
    rows = [
        (0, 1, 10., -1, -1), (1, 1, 12., -1, -1),
        (2, 2, 6., 1, 3), (2, 3, 7., 1, 2),
        (3, 2, 8., 1, 3), (3, 3, 9., 1, 2), (3, 4, 3., -1, -1),
    ]
    return pd.DataFrame(rows, columns=[
        'frame_i', 'Cell_ID', 'area', 'parent_ID_tree', 'sister_ID_tree'
    ]).set_index(['frame_i', 'Cell_ID'])


@pytest.mark.parametrize('source,start,end,expected', [
    ('mother', 't - 1', 't - 1', {2: [2], 3: [2]}),
    ('sister', 't', 't', {2: [2], 3: [2]}),
    ('daughter', 't + 1', 't + 1', {1: [1]}),
])
def test_symmetric_presets_show_and_save_equations(
        app, lineage_df, source, start, end, expected
    ):
    dialog = apps_measurements_viewer.TimepointPredictorEquationDialog(
        [], False, False, acdc_df=lineage_df,
        equation='MEAN(area_other) > 0'
    )
    dialog.otherCellsGroup.setChecked(True)
    dialog.otherCellsSourceCombobox.setCurrentIndex(
        dialog.otherCellsSourceCombobox.findData(source)
    )
    settings = dialog.currentOtherCells()
    assert settings['source'] == 'same_frame'
    assert not dialog.otherCellsMatchCombobox.isEnabled()
    assert not dialog.otherCellsRangeStartEdit.isEnabled()
    assert not dialog.otherCellsRangeEndEdit.isEnabled()
    assert settings['condition'] == dialog.otherCellsConditionEdit.text()
    assert settings['condition']
    assert (settings['range_start'], settings['range_end']) == (start, end)
    assert timepoint_predictors.TimepointPredictor.predict(lineage_df, {
        'type': 'equation', 'equation': 'MEAN(area_other) > 0',
        'mode': 'first_true', 'other_cells': settings
    }) == expected
    dialog.test_cb()
    assert dialog.okButton.isEnabled()
    reopened = apps_measurements_viewer.TimepointPredictorEquationDialog(
        [], False, False, acdc_df=lineage_df, otherCells=settings
    )
    assert reopened.currentOtherCells() == settings
    assert reopened.otherCellsSourceCombobox.currentData() == source
    reopened.deleteLater()
    dialog.deleteLater()


@pytest.mark.parametrize('source,expected', [
    ('mother', {2: [0]}), ('daughter', {1: [0]})
])
def test_budding_presets_use_same_frame(app, source, expected):
    df = pd.DataFrame({
        'frame_i': [0, 0, 1, 1], 'Cell_ID': [1, 2, 1, 2],
        'relationship': ['mother', 'bud', 'mother', 'mother'],
        'relative_ID': [2, 1, -1, -1], 'area': [10., 2., 10., 5.],
        'parent_ID_tree': [-1, -1, -1, 1],
    }).set_index(['frame_i', 'Cell_ID'])
    dialog = apps_measurements_viewer.TimepointPredictorEquationDialog(
        [], False, False, acdc_df=df
    )
    dialog.otherCellsGroup.setChecked(True)
    dialog.otherCellsSourceCombobox.setCurrentIndex(
        dialog.otherCellsSourceCombobox.findData(source)
    )
    settings = dialog.currentOtherCells()
    assert (settings['range_start'], settings['range_end']) == ('t', 't')
    assert 'relative_ID' in dialog.otherCellsConditionEdit.text()
    assert timepoint_predictors.TimepointPredictor.predict(df, {
        'type': 'equation', 'equation': 'MEAN(area_other) > 0',
        'mode': 'first_true', 'other_cells': settings
    }) == expected
    assert dialog.otherCellsSourceCombobox.findData('sister') == -1
    dialog.deleteLater()


@pytest.mark.parametrize('with_tree_IDs', [False, True])
@pytest.mark.parametrize('source,equation,expected', [
    ('mother', 'MEAN(area_other) == 20', {1: [], 2: [1], 3: [1]}),
    ('sister', 'MEAN(area_other) > 0', {1: [], 2: [1], 3: [1]}),
    ('daughter', 'MEAN(area_other) == 8.5', {1: [0], 2: [], 3: []}),
])
def test_symmetric_presets_use_segmentation_ID(
        source, equation, expected, with_tree_IDs
    ):
    df = pd.DataFrame({
        'frame_i': [0, 1, 1], 'Cell_ID': [1, 2, 3],
        'parent_ID_tree': [-1, 1, 1],
        'area': [20., 8., 9.],
    }).set_index(['frame_i', 'Cell_ID'])
    if with_tree_IDs:
        df['Cell_ID_tree'] = 10
    presets = timepoint_predictors.CellFilter.other_presets(df)
    assert presets['mother'][0] == 'Cell_ID == this_parent_ID_tree'
    assert presets['daughter'][0] == 'parent_ID_tree == this_Cell_ID'
    condition, start, end = presets[source]
    assert timepoint_predictors.TimepointPredictor.from_equation(
        df, equation, 'first_true',
        other_cells=dict(
            timepoint_predictors.CellFilter.default_other(),
            condition=condition, range_start=start, range_end=end
        )
    ) == expected


def test_if_condition_and_shift_tooltips(app, acdc_df):
    dialog = apps_measurements_viewer.TimepointPredictorEquationDialog(
        [], False, False, acdc_df=acdc_df
    )
    for edit in (
            dialog.mainCellConditionEdit, dialog.otherCellsConditionEdit,
            dialog.formulaEditWidget._edit
        ):
        assert '"if"' in edit.toolTip()
        assert 'SHIFT(area, 1)' in edit.toolTip()
    assert 't - 1' in dialog.otherCellsRangeStartEdit.toolTip()
    assert timepoint_predictors.TimepointPredictor.from_equation(
        acdc_df, '(area > SHIFT(area, 1)) & (stage == "S")', 'first_true'
    ) == {1: [1], 2: []}
    dialog.deleteLater()


def test_constants_are_disabled(viewer):
    assert not hasattr(viewer, 'editCellConstantsAction')
    assert not hasattr(viewer, 'cellConstants')
    assert not hasattr(apps, 'CellConstantsDialog')
    assert not hasattr(timepoint_predictors, 'compute_cell_constants')
    assert not hasattr(timepoint_predictors, 'default_constant')


def test_class_api_has_no_module_level_compatibility_functions():
    for name in (
            'predict_timepoints', 'predict_from_equation', 'predict_from_function',
            'validate_predictor', 'eval_cell_equation', 'validate_other_cells',
            'frame_range_bounds', '_flat_df'
        ):
        assert not hasattr(timepoint_predictors, name)


@pytest.mark.parametrize('start,end,t,expected', [
    ('t - 1', 't', 5, (4, 5)),
    ('0', 't + 2', 3, (0, 5)),
    ('(t // 2) * 2', 't + 1', 3, (2, 4)),
])
def test_frame_range_class_bounds(start, end, t, expected):
    assert timepoint_predictors.FrameRange.bounds(start, end, t) == expected


@pytest.mark.parametrize('start,end,t,sizeT,expected', [
    ('t - 5', 't + 3', 0, 3, (0, 2)),
    ('t + 5', 't + 8', 0, 3, (5, 2)),
    ('t - 8', 't - 5', 0, 3, (0, -5)),
    ('t - 5', 't + 3', 0, 0, (0, -1)),
])
def test_frame_range_class_clips_to_video(
        start, end, t, sizeT, expected
    ):
    assert timepoint_predictors.FrameRange.bounds(
        start, end, t, SizeT=sizeT
    ) == expected


@pytest.mark.parametrize('start,end', [
    ('t + 1', 't'), ('t / 2', 't'), ('unknown', 't'),
])
def test_frame_range_class_rejects_invalid_bounds(start, end):
    with pytest.raises(ValueError):
        timepoint_predictors.FrameRange.bounds(start, end, 3)


def test_class_table_and_aggregate_helpers_preserve_inputs(acdc_df):
    original = acdc_df.copy(deep=True)
    flat = timepoint_predictors.CellTable.flatten(acdc_df)
    assert {'Cell_ID', 'frame_i'}.issubset(flat.columns)
    assert [
        ID for ID, _ in timepoint_predictors.CellTable.iter_cells(
            timepoint_predictors.CellTable.subset(flat, [2])
        )
    ] == [2]
    settings = timepoint_predictors.CellFilter.default_other()
    prepared, equation = (
        timepoint_predictors.OtherCellMeasurements.prepare_inputs(
            acdc_df, 'area > MEAN(area_other)', settings, 4
        )
    )
    assert 'MEAN(area_other)' not in equation
    assert prepared.index.equals(acdc_df.index)
    assert len(prepared.columns) == len(acdc_df.columns) + 1
    pd.testing.assert_frame_equal(acdc_df, original)
    assert settings == timepoint_predictors.CellFilter.default_other()


# def test_cell_constants_source_cells(lineage_df):
#     tp = timepoint_predictors
#     constants = [
#         tp.constant_from_preset('Maximum value', 'area', 'mother'),
#         tp.constant_from_preset('Number of source cells', source='daughter'),
#         dict(tp.default_constant('sis_area', 'area', 'sister'), per_frame=True),
#         dict(
#             tp.default_constant('mean_others', 'MEAN(area)', 'same_frame'), 
#             per_frame=True
#         ),
#         dict(
#             tp.default_constant('n_bigger', 'NUNIQUE(Cell_ID)', 'custom'), 
#             condition='area > this_area', per_frame=True
#         ),
#         dict(
#             tp.default_constant('max_big_daughter', 'MAX(area)', 'daughter'), 
#             condition='area > 8'
#         ),
#         dict(
#             tp.default_constant('rel_area', 'area / max_area_mother'), 
#             per_frame=True
#         ),
#     ]
#     assert [c['name'] for c in constants[:2]] == [
#         'max_area_mother', 'num_cells_daughter'
#     ]
#     values = tp.compute_cell_constants(lineage_df, constants, SizeT=4)
#     df = tp.add_constants_columns(lineage_df, values)
#     assert values['max_area_mother'][2] == 12.0
#     assert np.isnan(values['max_area_mother'][1])
#     assert values['num_cells_daughter'] == {1: 2, 2: 0, 3: 0, 4: 0}
#     assert values['max_big_daughter'][1] == 9.0
#     assert df.loc[(3, 2), 'sis_area'] == 9.0
#     assert np.isnan(df.loc[(3, 4), 'sis_area'])
#     assert df.loc[(3, 4), 'mean_others'] == 8.5
#     assert np.isnan(df.loc[(0, 1), 'mean_others'])
#     assert df['n_bigger'].tolist() == [0, 0, 1, 0, 1, 0, 2]
#     assert df.loc[(2, 2), 'rel_area'] == 0.5
    
#     predicted = tp.predict_timepoints(df, {
#         'type': 'equation', 'equation': 'area > sis_area', 
#         'mode': 'first_true'
#     })
#     assert predicted == {3: [2]}


# @pytest.mark.parametrize('constant, match', [
#     ({'name': 'c', 'equation': 'MAX(area)', 'source': 'custom'}, 'condition'),
#     ({'name': 'c', 'equation': 'MAX(area)', 'source': 'cousin'}, 'source'),
#     ({'name': 'this_c', 'equation': 'MAX(area)'}, 'this_'),
# ])
# def test_cell_constants_source_validation(constant, match):
#     with pytest.raises(ValueError, match=match):
#         timepoint_predictors.validate_constants([constant])


# def test_file_predictor_entire_df(tmp_path, acdc_df):
#     path = tmp_path / 'pred.py'
#     path.write_text(
#         'def predictor(df, entire_df=None):\n'
#         '    ID = df["Cell_ID"].iloc[0]\n'
#         '    others = entire_df[entire_df["Cell_ID"] != ID]\n'
#         '    larger = df["area"].values > others["area"].values\n'
#         '    return [int(i) for i in df["frame_i"][larger]]\n'
#     )
#     predictor = {'type': 'file', 'filepath': str(path)}
#     assert timepoint_predictors.predict_timepoints(acdc_df, predictor) == {
#         1: [0, 1, 2, 3]
#     }
#     # Only cell 1 is predicted, but entire_df still contains all cells
#     assert timepoint_predictors.predict_timepoints(
#         acdc_df, predictor, IDs=[1]
#     ) == {1: [0, 1, 2, 3]}


# def test_viewer_cell_constants_used_by_predictors(viewer, monkeypatch):
#     assert viewer.editCellConstantsAction in viewer.timepointsToolBar.actions()
#     identifier = TimepointIdentifier(
#         'Half', (0, 255, 0), 'W',
#         {'type': 'equation', 'equation': 'area >= half_max', 'mode': 'first_true'},
#         autoApply=True
#     )
#     viewer.addTimepointIdentifierItem(identifier)
#     viewer.ensureTimepointPredictions()
#     # Unknown constant --> prediction fails and warns
#     assert viewer.measurementProcessingWarning.called
#     assert timepoints(viewer, identifier.uid) == {}
    
#     constants = [
#         {'name': 'max_area', 'equation': 'MAX(area)'},
#         {'name': 'half_max', 'equation': 'max_area / 2'},
#     ]
#     class FakeDialog:
#         def __init__(self, constants=(), acdc_df=None, SizeT=None, parent=None):
#             assert SizeT == 4 and 'max_area' not in acdc_df
#             self.cancel = False
#             self.constants = newConstants
#         def exec_(self):
#             pass
#     newConstants = constants
#     monkeypatch.setattr(apps, 'CellConstantsDialog', FakeDialog)
#     asked = []
#     monkeypatch.setattr(
#         viewer, 'askReevaluateAfterConstantsChanged', 
#         lambda: asked.append(True) or True
#     )
#     assert viewer.editCellConstants()
#     assert asked == [True]
#     posData = viewer.posData()
#     assert posData.cellConstants == {
#         'max_area': {1: 30.0, 2: 7.0}, 'half_max': {1: 15.0, 2: 3.5}
#     }
#     assert timepoints(viewer, identifier.uid) == {1: [1], 2: [0]}
#     assert 'half_max' in viewer.acdcDfWithConstants()
#     assert 'half_max' not in viewer.acdc_df
    
#     # Cached until the table or the constants change
#     source = posData.cellConstantsSource
#     viewer.positionCellConstants()
#     assert posData.cellConstantsSource is source
#     viewer.acdc_df = viewer.acdc_df.copy()
#     viewer.acdc_df['area'] *= 2
#     assert viewer.positionCellConstants()['max_area'] == {1: 60.0, 2: 14.0}
    
#     # Unchanged constants --> no re-evaluation prompt
#     asked.clear()
#     assert viewer.editCellConstants()
#     assert asked == []
    
#     assert viewer.cellConstants == [
#         timepoint_predictors.normalize_constant(c) for c in constants
#     ]
    
#     # Changing only the source cells invalidates the cached values
#     viewer.setCellConstants([
#         {'name': 'max_area', 'equation': 'MAX(area)', 'source': 'same_frame'},
#     ])
#     assert viewer.positionCellConstants()['max_area'] == {1: 14.0, 2: 60.0}
#     viewer.setCellConstants([])
#     assert viewer.positionCellConstants() == {}
#     assert 'max_area' not in viewer.acdcDfWithConstants()


# def test_configuration_rejects_invalid_constants():
#     with pytest.raises(ValueError, match='Invalid constant name'):
#         timepoint_predictors.validate_constants([{'name': 'a b', 'equation': '1'}])
