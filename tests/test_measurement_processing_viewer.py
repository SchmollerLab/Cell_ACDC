import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from qtpy.QtCore import QItemSelectionModel, QMutex, QWaitCondition, Qt
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QLabel

from cellacdc import apps, apps_measurements_viewer, measurements_viewer, widgets, workers
from cellacdc.measurements_viewer import MeasurementsViewerWin


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def viewer(app, monkeypatch):
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
    dialog = apps_measurements_viewer.MeasurementProcessingRecipeDialog(['area', 'intensity'])
    params = dialog.preProcessParamsWidget
    assert params.recipe() == [
        {'method': 'Smooth measurement', 'kwargs': {'window': 3}}
    ]
    selector = params.stepsWidgets[1]['selector']
    assert [selector.itemText(i) for i in range(selector.count())] == [
        'Smooth measurement', 'Dummy', 'Min-max normalization', 'Ramp detection'
    ]
    imageSelector = widgets.PreProcessingSelector()
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


def test_recipe_parameters_roundtrip(app):
    original = apps_measurements_viewer.MeasurementProcessingParamsWidget()
    original.stepsWidgets[1]['step_kwargs'] = {'window': 7}
    original.addStep()
    original.stepsWidgets[2]['selector'].setCurrentText('Dummy')
    saved = original.recipeConfigPars('acdc')
    loaded = apps_measurements_viewer.MeasurementProcessingParamsWidget()
    loaded.stepsWidgets[1]['step_kwargs'] = {'window': 99}
    loaded.loadRecipe({section: saved[section] for section in saved.sections()})
    assert loaded.recipe() == original.recipe()
    original.deleteLater()
    loaded.deleteLater()


def test_empty_measurement_selection_is_rejected(app, monkeypatch):
    dialog = apps_measurements_viewer.MeasurementProcessingRecipeDialog(['area'])
    dialog.listSelector.clearSelection()
    warning = Mock()
    monkeypatch.setattr(widgets.myMessageBox, 'warning', warning)
    dialog.ok_cb()
    assert dialog.cancel
    warning.assert_called_once()
    dialog.deleteLater()


def test_unsupported_recipe_does_not_change_steps(app, monkeypatch):
    params = apps_measurements_viewer.MeasurementProcessingParamsWidget()
    original = params.recipe()
    warning = Mock()
    monkeypatch.setattr(widgets.myMessageBox, 'warning', warning)
    params.loadRecipe({'acdc.preprocess.step1': {'method': 'Gaussian filter'}})
    assert params.recipe() == original
    warning.assert_called_once()
    params.deleteLater()


def test_image_recipe_ui_still_uses_image_registry(app):
    params = apps.PreProcessParamsWidget()
    selector = params.stepsWidgets[1]['selector']
    assert type(selector) is widgets.PreProcessingSelector
    assert 'Dummy' not in selector.preprocessMapper
    params.deleteLater()


def run_worker_job(df, recipe, cols, new_colnames):
    """Run a measurement job synchronously and collect its signals."""
    worker = workers.MeasurementProcessingWorkerGUI(QMutex(), QWaitCondition())
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
    worker = workers.MeasurementProcessingWorkerGUI(QMutex(), QWaitCondition())
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
        workers.MeasurementProcessingWorkerGUI, workers.CustomPreprocessWorkerGUI
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
    pd.testing.assert_frame_equal(viewer.acdc_df, original)
    assert viewer.processMeasurementsButton.isEnabled()
    assert viewer.refreshButton.isEnabled()


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
    viewer.posData().acdc_df = viewer.acdc_df.copy()
    wait_for_worker(viewer)
    assert 'area_copy' not in viewer.posData().acdc_df
    viewer.measurementProcessingWarning.assert_called_once()


def test_recipe_dialog_rejects_output_name_collisions(app, monkeypatch):
    # Output names are chosen in the recipe dialog: <measurement>_<suffix>.
    dialog = apps_measurements_viewer.MeasurementProcessingRecipeDialog(
        ['area', 'area_taken']
    )
    select_measurements(dialog, ['area'])
    warning = Mock()
    monkeypatch.setattr(widgets.myMessageBox, 'warning', warning)
    for suffix in ('', 'taken'):
        # No suffix overwrites the input; 'taken' overwrites area_taken.
        dialog.append_new_colname.setText(suffix)
        assert dialog.append_new_colname_tip.text().startswith(
            'This name is forbidden'
        )
        dialog.ok_cb()
        assert dialog.cancel
        assert not hasattr(dialog, 'saveNames')
    assert warning.call_count == 2
    dialog.append_new_colname.setText('new')
    assert dialog.append_new_colname_tip.text() == 'area_new'
    dialog.ok_cb()
    assert not dialog.cancel
    assert dialog.selectedChannels == ['area']
    assert dialog.saveNames == ['area_new']
    dialog.deleteLater()


def test_recipe_dialog_names_one_output_per_selected_measurement(app):
    dialog = apps_measurements_viewer.MeasurementProcessingRecipeDialog(
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
        saveNames=['area_new'], exec_=Mock()
    )
    dialogClass = Mock(return_value=dialog)
    monkeypatch.setattr(
        apps_measurements_viewer, 'MeasurementProcessingRecipeDialog', dialogClass
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
        saveNames=['area_new'], exec_=Mock()
    )
    monkeypatch.setattr(
        apps_measurements_viewer, 'MeasurementProcessingRecipeDialog',
        Mock(return_value=dialog)
    )
    start = Mock()
    monkeypatch.setattr(viewer, 'startMeasurementsProcessing', start)
    viewer.processMeasurements()
    dialog.exec_.assert_called_once()
    start.assert_not_called()


def test_export_includes_processed_columns(viewer, monkeypatch, tmp_path):
    viewer.startMeasurementsProcessing(
        ['area'], [{'method': 'Dummy', 'kwargs': {}}], ['area_copy']
    )
    wait_for_worker(viewer)
    path = tmp_path / 'measurements.csv'
    monkeypatch.setattr(
        measurements_viewer.QFileDialog, 'getSaveFileName',
        Mock(return_value=(str(path), 'CSV (*.csv)'))
    )
    viewer.exportMeasurements()
    saved = pd.read_csv(path, index_col=['frame_i', 'Cell_ID'])
    pd.testing.assert_frame_equal(saved, viewer.acdc_df, check_dtype=False)


def test_export_failure_is_reported(viewer, monkeypatch, tmp_path):
    monkeypatch.setattr(
        measurements_viewer.QFileDialog, 'getSaveFileName',
        Mock(return_value=(str(tmp_path / 'missing' / 'table.csv'), ''))
    )
    viewer.exportMeasurements()
    viewer.measurementProcessingWarning.assert_called_once()
    assert viewer.measurementProcessingWarning.call_args.args[0] == 'Export failed'
