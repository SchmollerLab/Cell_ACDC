import time
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from qtpy.QtCore import QItemSelectionModel, QMutex, QWaitCondition, Qt
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QLabel, QWidget

from cellacdc.measurements_viewer import (
    annotation_data as viewer_annotation_data,
    workers as viewer_workers,
)
from cellacdc.measurements_viewer.dialogs import processing_recipe as viewer_dialogs_processing_recipe
from cellacdc.shared.dialogs import preprocessing as dialogs_preprocessing
from cellacdc.shared.widgets import (
    combo_boxes as widgets_combo_boxes,
    message_box as widgets_message_box,
)
from cellacdc.shared.workers import base as workers_base
from qtpy.QtWidgets import QFileDialog
from cellacdc.measurements_viewer.main_window.window import MeasurementsViewerWin
from tests.conftest import flushDeferredDeletes

pytestmark = pytest.mark.viewer


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def viewer(app, monkeypatch, dummy_processor):
    monkeypatch.setattr(MeasurementsViewerWin, 'loadConfiguration', Mock())
    window = MeasurementsViewerWin()
    window.acdc_df = pd.DataFrame({
        'frame_i': [0, 1, 2, 0, 1, 2],
        'Cell_ID': [1, 1, 1, 2, 2, 2],
        'area': [0, 0, 9, 100, 100, 100],
    }).set_index(['frame_i', 'Cell_ID'])
    window.data = [SimpleNamespace(acdc_df=window.acdc_df, chNames=[])]
    window.centralWidget().setEnabled(True)
    monkeypatch.setattr(window, 'plotMeasurements', Mock())
    monkeypatch.setattr(window, 'measurementProcessingWarning', Mock())
    window.updateColumnsList()
    yield window
    wait_for_worker(window)
    window.deleteLater()
    app.processEvents()


def wait_for_worker(window):
    deadline = time.monotonic() + 10
    while window.measurementProcessingThread is not None and time.monotonic() < deadline:
        QTest.qWait(10)
    assert window.measurementProcessingThread is None


def measurement_item(model, name):
    """Return the measurement leaf called ``name`` of a measurement tree."""
    item, = [item for item in model.measurementItems() if item.text() == name]
    return item


def select_measurements(dialog, names):
    """Select exactly the measurement leaves ``names`` in a recipe dialog."""
    selection = dialog.listSelector.selectionModel()
    selection.clearSelection()
    for name in names:
        item = measurement_item(dialog.measurementsModel, name)
        selection.select(
            dialog.measurementsSearchProxy.mapFromSource(item.index()),
            QItemSelectionModel.Select
        )


def test_measurement_recipe_ui_uses_separate_registry_and_defaults(app):
    dialog = viewer_dialogs_processing_recipe.MeasurementProcessingRecipeDialog(['area', 'intensity'])
    params = dialog.preProcessParamsWidget
    assert params.recipe() == [
        {'method': 'Smooth measurement', 'kwargs': {'window': 3}}
    ]
    selector = params.stepsWidgets[1]['selector']
    assert [selector.itemText(i) for i in range(selector.count())] == [
        'Smooth measurement', 'Min-max normalization', 'Ramp detection'
    ]
    imageSelector = widgets_combo_boxes.PreProcessingSelector()
    assert imageSelector.preprocessMapper is not selector.preprocessMapper
    assert dialog.keepInputDataTypeToggle.isHidden()
    assert any(
        label.text() == 'Select measurement(s) to process:'
        for label in dialog.findChildren(QLabel)
    )
    # Without a suffix the output would overwrite the input measurement.
    dialog.append_new_colname.setText('smooth')
    dialog.ok_cb()
    assert not dialog.cancel
    assert dialog.selectedChannels == ['area']
    assert dialog.saveNames == ['area_smooth']
    step, = dialog.selectedRecipe
    assert step['method'] == 'Smooth measurement'
    assert step['kwargs'] == {'window': 3}
    assert not step['keep_input_data_type']
    imageSelector.deleteLater()
    dialog.deleteLater()


def test_recipe_parameters_roundtrip(app, dummy_processor):
    original = viewer_dialogs_processing_recipe.MeasurementProcessingParamsWidget()
    original.stepsWidgets[1]['step_kwargs'] = {'window': 7}
    original.addStep()
    original.stepsWidgets[2]['selector'].setCurrentText('Dummy')
    saved = original.recipeConfigPars('acdc')
    loaded = viewer_dialogs_processing_recipe.MeasurementProcessingParamsWidget()
    loaded.stepsWidgets[1]['step_kwargs'] = {'window': 99}
    loaded.loadRecipe({section: saved[section] for section in saved.sections()})
    assert loaded.recipe() == original.recipe()
    original.deleteLater()
    loaded.deleteLater()


def test_empty_measurement_selection_is_rejected(app, monkeypatch):
    dialog = viewer_dialogs_processing_recipe.MeasurementProcessingRecipeDialog(['area'])
    dialog.listSelector.clearSelection()
    warning = Mock()
    monkeypatch.setattr(widgets_message_box.myMessageBox, 'warning', warning)
    dialog.ok_cb()
    assert dialog.cancel
    warning.assert_called_once()
    dialog.deleteLater()


def test_unsupported_recipe_does_not_change_steps(app, monkeypatch):
    params = viewer_dialogs_processing_recipe.MeasurementProcessingParamsWidget()
    original = params.recipe()
    warning = Mock()
    monkeypatch.setattr(widgets_message_box.myMessageBox, 'warning', warning)
    params.loadRecipe({'acdc.preprocess.step1': {'method': 'Gaussian filter'}})
    assert params.recipe() == original
    warning.assert_called_once()
    params.deleteLater()


def test_image_recipe_ui_still_uses_image_registry(app):
    params = dialogs_preprocessing.PreProcessParamsWidget()
    selector = params.stepsWidgets[1]['selector']
    assert type(selector) is widgets_combo_boxes.PreProcessingSelector
    assert 'Dummy' not in selector.preprocessMapper
    params.deleteLater()


def run_worker_job(df, recipe, cols, new_colnames):
    """Run a measurement job synchronously and collect its signals."""
    worker = viewer_workers.MeasurementProcessingWorkerGUI(QMutex(), QWaitCondition())
    done, failed, finished = [], [], []
    worker.sigDone.connect(done.append)
    worker.signals.critical.connect(failed.append)
    worker.signals.finished.connect(finished.append)
    worker.setupJob(
        df=df, cols=cols, recipe=recipe, new_colnames=new_colnames
    )
    # The result is emitted through sigDone, not returned.
    assert worker.runJob() is None
    assert finished == [worker]
    return worker, done, failed


def test_worker_run_is_a_one_shot_job(app):
    df = pd.DataFrame(
        {'area': [0., 9.]},
        index=pd.MultiIndex.from_tuples([(0, 1), (1, 1)], names=['frame_i', 'Cell_ID'])
    )
    worker = viewer_workers.MeasurementProcessingWorkerGUI(QMutex(), QWaitCondition())
    done, finished = [], []
    worker.sigDone.connect(done.append)
    worker.signals.finished.connect(finished.append)
    worker.setupJob(
        df=df, cols=['area'],
        recipe=[{'method': 'Smooth measurement', 'kwargs': {'window': 3}}],
        new_colnames=['area_smooth']
    )
    worker.run()
    result, = done
    assert finished == [worker]
    np.testing.assert_allclose(result['area_smooth'], [4.5, 4.5])
    with pytest.raises(NotImplementedError):
        worker.enqueue(df, ['area'], [], ['area_smooth'])
    worker.deleteLater()


def test_worker_subclasses_preprocessing_and_keeps_fractional_values(app):
    assert issubclass(
        viewer_workers.MeasurementProcessingWorkerGUI, workers_base.CustomPreprocessWorkerGUI
    )
    df = pd.DataFrame(
        {'area': [0, 9]},
        index=pd.MultiIndex.from_tuples([(0, 1), (1, 1)], names=['frame_i', 'Cell_ID'])
    )
    worker, done, failed = run_worker_job(
        df, [{'method': 'Smooth measurement', 'kwargs': {'window': 3}}],
        ['area'], ['area_smooth']
    )
    assert not failed
    result, = done
    pd.testing.assert_series_equal(result['area'], df['area'])
    # The integer input is not cast back: fractional results are kept.
    np.testing.assert_allclose(result['area_smooth'], [4.5, 4.5])
    worker.deleteLater()


def test_worker_reports_recipe_errors_through_critical(app):
    df = pd.DataFrame(
        {'area': [0, 9]},
        index=pd.MultiIndex.from_tuples([(0, 1), (1, 1)], names=['frame_i', 'Cell_ID'])
    )
    worker, done, failed = run_worker_job(
        df, [{'method': 'Smooth measurement', 'kwargs': {'window': 0}}],
        ['area'], ['area_bad']
    )
    assert not done
    (source, error), = failed
    assert source is worker
    assert isinstance(error, ValueError)
    worker.deleteLater()


def test_processing_adds_columns_and_preserves_originals(viewer):
    original = viewer.acdc_df.copy()
    viewer.startMeasurementsProcessing(
        ['area'], [{'method': 'Smooth measurement', 'kwargs': {'window': 3}}], ['area_smooth']
    )
    assert not viewer.processMeasurementsButton.isEnabled()
    wait_for_worker(viewer)
    pd.testing.assert_frame_equal(viewer.acdc_df[['area']], original)
    np.testing.assert_allclose(
        viewer.acdc_df['area_smooth'], [0, 3, 4.5, 100, 100, 100]
    )
    assert viewer.posData().acdc_df is viewer.acdc_df
    assert [item.text() for item in viewer.columnsModel.measurementItems()] == [
        'area', 'area_smooth'
    ]
    assert viewer.processMeasurementsButton.isEnabled()
    viewer.plotMeasurements.assert_called_once()
    viewer.measurementProcessingWarning.assert_not_called()


def test_worker_error_is_reported_once_and_controls_reenabled(viewer):
    original = viewer.acdc_df.copy()
    viewer.startMeasurementsProcessing(
        ['area'], [{'method': 'Smooth measurement', 'kwargs': {'window': 0}}], ['area_bad']
    )
    wait_for_worker(viewer)
    viewer.measurementProcessingWarning.assert_called_once()
    assert viewer.measurementProcessingWarning.call_args.args[0] == 'Measurement processing failed'
    details = viewer.measurementProcessingWarning.call_args.kwargs['details']
    assert 'Traceback' in details
    assert 'SmoothMeasurement' in details
    assert 'ValueError' in details
    pd.testing.assert_frame_equal(viewer.acdc_df, original)
    assert viewer.processMeasurementsButton.isEnabled()
    assert viewer.refreshButton.isEnabled()


@pytest.mark.slow
@pytest.mark.parametrize('mode', [0, 1])
@pytest.mark.parametrize('normalize', [False, True])
def test_processed_columns_plot_in_both_modes(viewer, mode, normalize):
    viewer.startMeasurementsProcessing(
        ['area'], [{'method': 'Smooth measurement', 'kwargs': {'window': 3}}], ['area_smooth']
    )
    wait_for_worker(viewer)
    viewer.posData().SizeT = 3
    viewer.selectedID = 1
    viewer.frame_i = 0
    viewer.measurementsModeCombobox.setCurrentIndex(mode)
    for item in viewer.columnsModel.measurementItems():
        item.setCheckState(Qt.Unchecked)
    measurement_item(viewer.columnsModel, 'area_smooth').setCheckState(Qt.Checked)
    if mode == 1:
        measurement_item(viewer.columnsModel, 'area').setCheckState(Qt.Checked)
    viewer.normalizeCheckbox.setChecked(normalize)
    MeasurementsViewerWin.plotMeasurements(viewer)
    item, = viewer.measurementsPlot.listDataItems()
    smooth = np.array([0, 3, 4.5, 100, 100, 100])
    if mode == 0:
        expected = smooth[:3] / 4.5 if normalize else smooth[:3]
        np.testing.assert_allclose(item.yData, expected)
    else:
        expected = smooth / 100 if normalize else smooth
        np.testing.assert_allclose(item.xData, expected)


def test_result_applies_to_original_position_after_switch(viewer):
    target = viewer.posData()
    viewer.startMeasurementsProcessing(
        ['area'], [{'method': 'Dummy', 'kwargs': {}}], ['area_copy']
    )
    other = SimpleNamespace(acdc_df=viewer.acdc_df.copy() * 2)
    viewer.data.append(other)
    viewer.pos_i = 1
    viewer.acdc_df = other.acdc_df
    wait_for_worker(viewer)
    assert 'area_copy' in target.acdc_df
    assert 'area_copy' not in viewer.acdc_df
    viewer.plotMeasurements.assert_not_called()


def test_returning_to_position_before_completion_keeps_results(viewer):
    viewer.startMeasurementsProcessing(
        ['area'], [{'method': 'Dummy', 'kwargs': {}}], ['area_copy']
    )
    viewer.acdc_df = viewer.getAcdcDf()
    wait_for_worker(viewer)
    assert 'area_copy' in viewer.acdc_df
    viewer.measurementProcessingWarning.assert_not_called()


def test_closing_during_processing_is_blocked(viewer):
    viewer.startMeasurementsProcessing(
        ['area'], [{'method': 'Dummy', 'kwargs': {}}], ['area_copy']
    )
    event = Mock()
    viewer.closeEvent(event)
    event.ignore.assert_called_once()
    viewer.measurementProcessingWarning.assert_called_once()
    wait_for_worker(viewer)


def test_changed_source_does_not_receive_stale_results(viewer):
    viewer.startMeasurementsProcessing(
        ['area'], [{'method': 'Dummy', 'kwargs': {}}], ['area_copy']
    )
    # An input column changed while processing was running.
    viewer.acdc_df.loc[(0, 1), 'area'] = 55
    wait_for_worker(viewer)
    assert 'area_copy' not in viewer.posData().acdc_df
    assert 'area_copy' not in viewer.acdc_df
    viewer.measurementProcessingWarning.assert_called_once()


def test_edits_during_processing_are_kept(viewer):
    viewer.startMeasurementsProcessing(
        ['area'], [{'method': 'Dummy', 'kwargs': {}}], ['area_copy']
    )
    # In-place edits of other columns (time points, visited cells) ...
    viewer.acdc_df['timepoint_Bud'] = False
    viewer.acdc_df.loc[(2, 1), 'timepoint_Bud'] = True
    viewer.acdc_df[viewer_annotation_data.VISITED_COLUMN] = [
        True, True, True, False, False, False
    ]
    edited = viewer.acdc_df.copy()
    wait_for_worker(viewer)
    viewer.measurementProcessingWarning.assert_not_called()
    # ... survive, and the output column is added next to them.
    pd.testing.assert_frame_equal(viewer.acdc_df[edited.columns], edited)
    np.testing.assert_array_equal(
        viewer.acdc_df['area_copy'], viewer.acdc_df['area']
    )
    assert viewer.posData().acdc_df is viewer.acdc_df


def test_output_added_during_processing_is_not_overwritten(viewer):
    viewer.startMeasurementsProcessing(
        ['area'], [{'method': 'Dummy', 'kwargs': {}}], ['area_copy']
    )
    viewer.acdc_df['area_copy'] = -1.0
    wait_for_worker(viewer)
    assert (viewer.acdc_df['area_copy'] == -1.0).all()
    viewer.measurementProcessingWarning.assert_called_once()
    assert viewer.measurementProcessingWarning.call_args.args[0] == (
        'Invalid measurement name'
    )


def test_new_table_object_with_same_inputs_keeps_results(viewer):
    # Replacing the table with an equal copy (e.g. standardization on a
    # position switch) is not a change of the inputs.
    viewer.startMeasurementsProcessing(
        ['area'], [{'method': 'Dummy', 'kwargs': {}}], ['area_copy']
    )
    viewer.acdc_df = viewer.acdc_df.copy()
    viewer.posData().acdc_df = viewer.acdc_df
    wait_for_worker(viewer)
    viewer.measurementProcessingWarning.assert_not_called()
    assert 'area_copy' in viewer.acdc_df


def test_reload_during_processing_discards_results(viewer, tmp_path):
    posData = viewer.posData()
    path = tmp_path / 'acdc_output.csv'
    viewer.acdc_df.to_csv(path)
    posData.acdc_output_csv_path = str(path)

    def loadAcdcDf(filepath):
        posData.acdc_df = pd.read_csv(filepath).set_index(['frame_i', 'Cell_ID'])

    posData.loadAcdcDf = loadAcdcDf
    viewer.startMeasurementsProcessing(
        ['area'], [{'method': 'Dummy', 'kwargs': {}}], ['area_copy']
    )
    # Same values on disk: only the reload counter detects the reload.
    viewer.reloadAcdcDfFile()
    viewer.acdc_df = viewer.getAcdcDf()
    wait_for_worker(viewer)
    assert 'area_copy' not in viewer.acdc_df
    assert 'area_copy' not in posData.acdc_df
    viewer.measurementProcessingWarning.assert_called_once()
    assert viewer.measurementProcessingWarning.call_args.args[0] == (
        'Measurements changed'
    )


def test_processing_failure_shows_traceback_details(viewer, monkeypatch):
    # Use the real warning with a fake message box.
    monkeypatch.delattr(viewer, 'measurementProcessingWarning')
    boxes = []

    class FakeMessageBox:
        def __init__(self, *args, **kwargs):
            self.detailedText = None
            boxes.append(self)

        def setDetailedText(self, text, visible=False, wrap=True):
            self.detailedText = text

        def warning(self, parent, title, text, **kwargs):
            self.title = title

    monkeypatch.setattr(widgets_message_box, 'myMessageBox', FakeMessageBox)
    viewer.startMeasurementsProcessing(
        ['area'], [{'method': 'Smooth measurement', 'kwargs': {'window': 0}}], ['area_bad']
    )
    wait_for_worker(viewer)
    box, = boxes
    assert box.title == 'Measurement processing failed'
    assert 'Traceback' in box.detailedText
    assert 'SmoothMeasurement' in box.detailedText


def test_recipe_dialog_rejects_output_name_collisions(app, monkeypatch):
    # Output names are chosen in the recipe dialog: <measurement>_<suffix>.
    dialog = viewer_dialogs_processing_recipe.MeasurementProcessingRecipeDialog(
        ['area', 'area_taken']
    )
    select_measurements(dialog, ['area'])
    warning = Mock()
    monkeypatch.setattr(widgets_message_box.myMessageBox, 'warning', warning)
    tip = dialog.append_new_colname_tip
    expected = {
        # No suffix would overwrite the input: neutral hint, not an error
        '': ('Enter a suffix for the output column(s).', ''),
        # 'taken' overwrites area_taken
        'taken': (
            'Output column(s) already exist: area_taken. '
            'Choose another suffix.',
            'color: red;'
        ),
    }
    for suffix, (text, style) in expected.items():
        dialog.append_new_colname.setText(suffix)
        assert tip.text() == text
        assert tip.styleSheet() == style
        dialog.ok_cb()
        assert dialog.cancel
        assert not hasattr(dialog, 'saveNames')
        assert warning.call_args.args[2] == text
    assert warning.call_count == 2
    dialog.append_new_colname.setText('new')
    assert dialog.append_new_colname_tip.text() == 'area_new'
    dialog.ok_cb()
    assert not dialog.cancel
    assert dialog.selectedChannels == ['area']
    assert dialog.saveNames == ['area_new']
    dialog.deleteLater()


def test_recipe_dialog_opens_without_error_tip(app):
    # The first measurement is preselected; with no suffix typed yet the
    # dialog shows a neutral hint instead of a red "forbidden" error.
    dialog = viewer_dialogs_processing_recipe.MeasurementProcessingRecipeDialog(
        ['area', 'volume']
    )
    tip = dialog.append_new_colname_tip
    assert tip.text() == 'Enter a suffix for the output column(s).'
    assert tip.styleSheet() == ''
    dialog.deleteLater()


def test_recipe_dialog_names_one_output_per_selected_measurement(app):
    dialog = viewer_dialogs_processing_recipe.MeasurementProcessingRecipeDialog(
        ['area', 'volume']
    )
    select_measurements(dialog, ['area', 'volume'])
    dialog.append_new_colname.setText('smooth')
    dialog.ok_cb()
    assert not dialog.cancel
    assert dict(zip(dialog.selectedChannels, dialog.saveNames)) == {
        'area': 'area_smooth', 'volume': 'volume_smooth'
    }
    dialog.deleteLater()


def test_accept_passes_dialog_output_names_to_processing(viewer, monkeypatch):
    viewer.acdc_df['area_taken'] = 0
    dialog = SimpleNamespace(
        cancel=False, selectedChannels=['area'],
        selectedRecipe=[{'method': 'Dummy', 'kwargs': {}}],
        saveNames=['area_new'], exec_=Mock(), deleteLater=Mock()
    )
    dialogClass = Mock(return_value=dialog)
    monkeypatch.setattr(
        viewer_dialogs_processing_recipe, 'MeasurementProcessingRecipeDialog', dialogClass
    )
    start = Mock()
    monkeypatch.setattr(viewer, 'startMeasurementsProcessing', start)
    viewer.processMeasurements()
    # The dialog sees every existing measurement so it can reject collisions.
    columns, = dialogClass.call_args.args
    assert set(columns) == {'area', 'area_taken'}
    assert dialogClass.call_args.kwargs['acdc_df'] is viewer.acdc_df
    dialog.exec_.assert_called_once()
    start.assert_called_once_with(['area'], dialog.selectedRecipe, ['area_new'])
    viewer.measurementProcessingWarning.assert_not_called()


def test_cancel_does_not_process(viewer, monkeypatch):
    dialog = SimpleNamespace(
        cancel=True, selectedChannels=['area'], selectedRecipe=[],
        saveNames=['area_new'], exec_=Mock(), deleteLater=Mock()
    )
    monkeypatch.setattr(
        viewer_dialogs_processing_recipe, 'MeasurementProcessingRecipeDialog',
        Mock(return_value=dialog)
    )
    start = Mock()
    monkeypatch.setattr(viewer, 'startMeasurementsProcessing', start)
    viewer.processMeasurements()
    dialog.exec_.assert_called_once()
    start.assert_not_called()


@pytest.mark.slow
def test_recipe_dialog_is_deleted_after_use(viewer, monkeypatch):
    # A child of the viewer, it kept the table it was opened with alive for
    # the whole session (e.g. after a position switch or Reload table)
    def cancel(dialog):
        # Counted by name (see test_nested_editors_are_deleted_after_use)
        dialog.setObjectName('recipe_dialog')
        dialog.cancel = True

    monkeypatch.setattr(
        viewer_dialogs_processing_recipe.MeasurementProcessingRecipeDialog, 'exec_',
        cancel
    )
    table = weakref.ref(viewer.acdc_df)
    viewer.processMeasurements()
    viewer.acdc_df = viewer.acdc_df.sort_index().copy()
    viewer.posData().acdc_df = viewer.acdc_df
    flushDeferredDeletes()
    assert viewer.findChildren(QWidget, 'recipe_dialog') == []
    # PyQt releases the slots connected to the deleted widgets (they
    # reference the dialog) in the event loop
    QTest.qWait(10)
    flushDeferredDeletes()
    assert table() is None


def recipe_dialog_class(saveNames, onExec):
    """Fake recipe dialog accepted with the Dummy recipe on 'area' after
    calling `onExec` (what happened while the non-blocking dialog was
    open)."""
    class Dialog:
        def __init__(self, *args, **kwargs):
            self.cancel = False
            self.selectedChannels = ['area']
            self.selectedRecipe = [{'method': 'Dummy', 'kwargs': {}}]
            self.saveNames = saveNames

        def exec_(self):
            onExec()

        def deleteLater(self):
            pass

    return Dialog


def test_second_recipe_dialog_does_not_start_a_second_job(
        viewer, monkeypatch
    ):
    # A second recipe dialog, accepted first, started a job while this
    # one was open: one job at a time (the job state is kept on the window)
    def startOther():
        viewer.startMeasurementsProcessing(
            ['area'], [{'method': 'Dummy', 'kwargs': {}}], ['area_first']
        )

    monkeypatch.setattr(
        viewer_dialogs_processing_recipe, 'MeasurementProcessingRecipeDialog',
        recipe_dialog_class(['area_second'], startOther)
    )
    viewer.processMeasurements()
    viewer.measurementProcessingWarning.assert_called_once()
    assert viewer.measurementProcessingWarning.call_args.args[0] == (
        'Measurement processing running'
    )
    # Also refused when started directly
    viewer.startMeasurementsProcessing(
        ['area'], [{'method': 'Dummy', 'kwargs': {}}], ['area_third']
    )
    wait_for_worker(viewer)
    assert 'area_first' in viewer.acdc_df
    assert 'area_second' not in viewer.acdc_df
    assert 'area_third' not in viewer.acdc_df
    assert [step['output_columns'] for step in viewer.measurementProcessingSteps] == [
        ['area_first']
    ]
    viewer.measurementProcessingWarning.assert_called_once()


def test_recipe_inputs_dropped_meanwhile_are_refused(viewer, monkeypatch):
    # E.g. Reload table while the dialog was open dropped the input column
    def dropInput():
        viewer.acdc_df = viewer.acdc_df.drop(columns='area')
        viewer.posData().acdc_df = viewer.acdc_df

    monkeypatch.setattr(
        viewer_dialogs_processing_recipe, 'MeasurementProcessingRecipeDialog',
        recipe_dialog_class(['area_new'], dropInput)
    )
    start = Mock()
    monkeypatch.setattr(viewer, 'startMeasurementsProcessing', start)
    viewer.processMeasurements()
    start.assert_not_called()
    viewer.measurementProcessingWarning.assert_called_once()
    assert viewer.measurementProcessingWarning.call_args.args[0] == (
        'Measurements changed'
    )


@pytest.mark.slow
def test_export_includes_processed_columns(viewer, monkeypatch, tmp_path):
    viewer.startMeasurementsProcessing(
        ['area'], [{'method': 'Dummy', 'kwargs': {}}], ['area_copy']
    )
    wait_for_worker(viewer)
    path = tmp_path / 'measurements.csv'
    monkeypatch.setattr(
        QFileDialog, 'getSaveFileName',
        Mock(return_value=(str(path), 'CSV (*.csv)'))
    )
    viewer.exportMeasurements()
    saved = pd.read_csv(path, index_col=['frame_i', 'Cell_ID'])
    pd.testing.assert_frame_equal(saved, viewer.acdc_df, check_dtype=False)


def test_export_failure_is_reported(viewer, monkeypatch, tmp_path):
    monkeypatch.setattr(
        QFileDialog, 'getSaveFileName',
        Mock(return_value=(str(tmp_path / 'missing' / 'table.csv'), ''))
    )
    viewer.exportMeasurements()
    viewer.measurementProcessingWarning.assert_called_once()
    assert viewer.measurementProcessingWarning.call_args.args[0] == 'Export failed'
