import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from qtpy.QtCore import QMutex, QWaitCondition, Qt
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QLabel

from cellacdc import apps, measurements_viewer, measurement_processing, widgets, workers
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
    window.data = [SimpleNamespace(acdc_df=window.acdc_df)]
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


def test_measurement_recipe_ui_uses_separate_registry_and_defaults(app):
    dialog = apps.MeasurementProcessingRecipeDialog(['area', 'intensity'])
    params = dialog.preProcessParamsWidget
    assert params.recipe() == [
        {'method': 'Smooth measurement', 'kwargs': {'window': 3}}
    ]
    selector = params.stepsWidgets[1]['selector']
    assert [selector.itemText(i) for i in range(selector.count())] == [
        'Smooth measurement', 'Dummy'
    ]
    imageSelector = widgets.PreProcessingSelector()
    assert imageSelector.preprocessMapper is not selector.preprocessMapper
    assert dialog.keepInputDataTypeToggle.isHidden()
    assert any(
        label.text() == 'Select measurement(s) to process:'
        for label in dialog.findChildren(QLabel)
    )
    dialog.ok_cb()
    assert not dialog.cancel
    assert dialog.selectedChannels == ['area']
    assert not dialog.selectedRecipe[0]['keep_input_data_type']
    imageSelector.deleteLater()
    dialog.deleteLater()


def test_recipe_parameters_roundtrip(app):
    original = apps.MeasurementProcessingParamsWidget()
    original.stepsWidgets[1]['step_kwargs'] = {'window': 7}
    original.addStep()
    original.stepsWidgets[2]['selector'].setCurrentText('Dummy')
    saved = original.recipeConfigPars('acdc')
    loaded = apps.MeasurementProcessingParamsWidget()
    loaded.stepsWidgets[1]['step_kwargs'] = {'window': 99}
    loaded.loadRecipe({section: saved[section] for section in saved.sections()})
    assert loaded.recipe() == original.recipe()
    original.deleteLater()
    loaded.deleteLater()


def test_empty_measurement_selection_is_rejected(app, monkeypatch):
    dialog = apps.MeasurementProcessingRecipeDialog(['area'])
    dialog.listSelector.clearSelection()
    warning = Mock()
    monkeypatch.setattr(widgets.myMessageBox, 'warning', warning)
    dialog.ok_cb()
    assert dialog.cancel
    warning.assert_called_once()
    dialog.deleteLater()


def test_unsupported_recipe_does_not_change_steps(app, monkeypatch):
    params = apps.MeasurementProcessingParamsWidget()
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


def test_worker_subclasses_preprocessing_and_keeps_fractional_values():
    worker = workers.MeasurementProcessingWorkerGUI(QMutex(), QWaitCondition())
    assert isinstance(worker, workers.CustomPreprocessWorkerGUI)
    df = pd.DataFrame(
        {'area': [0, 9]},
        index=pd.MultiIndex.from_tuples([(0, 1), (1, 1)], names=['frame_i', 'Cell_ID'])
    )
    worker.setupJob(
        measurement_processing.apply_recipe, df,
        [{'method': 'Smooth measurement', 'kwargs': {'window': 3},
          'keep_input_data_type': True}], 'smooth'
    )
    np.testing.assert_allclose(worker.runJob()['area'], [4.5, 4.5])


def test_processing_adds_columns_and_preserves_originals(viewer):
    original = viewer.acdc_df.copy()
    viewer.startMeasurementProcessing(
        ['area'], [{'method': 'Smooth measurement', 'kwargs': {'window': 3}}], 'smooth'
    )
    assert not viewer.processMeasurementsButton.isEnabled()
    wait_for_worker(viewer)
    pd.testing.assert_frame_equal(viewer.acdc_df[['area']], original)
    np.testing.assert_allclose(
        viewer.acdc_df['area_smooth'], [0, 3, 4.5, 100, 100, 100]
    )
    assert viewer.posData().acdc_df is viewer.acdc_df
    assert viewer.columnsModel.rowCount() == 2
    assert viewer.processMeasurementsButton.isEnabled()
    viewer.plotMeasurements.assert_called_once()
    viewer.measurementProcessingWarning.assert_not_called()


def test_worker_error_is_reported_once_and_controls_reenabled(viewer):
    original = viewer.acdc_df.copy()
    viewer.startMeasurementProcessing(
        ['area'], [{'method': 'Smooth measurement', 'kwargs': {'window': 0}}], 'bad'
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
    viewer.startMeasurementProcessing(
        ['area'], [{'method': 'Smooth measurement', 'kwargs': {'window': 3}}], 'smooth'
    )
    wait_for_worker(viewer)
    viewer.posData().SizeT = 3
    viewer.selectedID = 1
    viewer.frame_i = 0
    viewer.measurementsModeCombobox.setCurrentIndex(mode)
    for i in range(viewer.columnsModel.rowCount()):
        viewer.columnsModel.item(i).setCheckState(Qt.Unchecked)
    viewer.columnsModel.item(1).setCheckState(Qt.Checked)
    if mode == 1:
        viewer.columnsModel.item(0).setCheckState(Qt.Checked)
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
    viewer.startMeasurementProcessing(
        ['area'], [{'method': 'Dummy', 'kwargs': {}}], 'copy'
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
    viewer.startMeasurementProcessing(
        ['area'], [{'method': 'Dummy', 'kwargs': {}}], 'copy'
    )
    viewer.acdc_df = viewer.getAcdcDf()
    wait_for_worker(viewer)
    assert 'area_copy' in viewer.acdc_df
    viewer.measurementProcessingWarning.assert_not_called()


def test_closing_during_processing_is_blocked(viewer):
    viewer.startMeasurementProcessing(
        ['area'], [{'method': 'Dummy', 'kwargs': {}}], 'copy'
    )
    event = Mock()
    viewer.closeEvent(event)
    event.ignore.assert_called_once()
    viewer.measurementProcessingWarning.assert_called_once()
    wait_for_worker(viewer)


def test_changed_source_does_not_receive_stale_results(viewer):
    viewer.startMeasurementProcessing(
        ['area'], [{'method': 'Dummy', 'kwargs': {}}], 'copy'
    )
    viewer.posData().acdc_df = viewer.acdc_df.copy()
    wait_for_worker(viewer)
    assert 'area_copy' not in viewer.posData().acdc_df
    viewer.measurementProcessingWarning.assert_called_once()


def test_accept_prompts_for_name_and_rejects_collisions(viewer, monkeypatch):
    dialog = SimpleNamespace(
        cancel=False, selectedChannels=['area'],
        selectedRecipe=[{'method': 'Dummy', 'kwargs': {}}], exec_=Mock()
    )
    monkeypatch.setattr(apps, 'MeasurementProcessingRecipeDialog', Mock(return_value=dialog))
    viewer.acdc_df['area_taken'] = 0
    names = [
        SimpleNamespace(cancel=False, entryText=name, exec_=Mock())
        for name in ('taken', 'new')
    ]
    filename = Mock(side_effect=names)
    monkeypatch.setattr(apps, 'filenameDialog', filename)
    start = Mock()
    monkeypatch.setattr(viewer, 'startMeasurementProcessing', start)
    viewer.processMeasurements()
    assert filename.call_count == 2
    start.assert_called_once_with(['area'], dialog.selectedRecipe, 'new')
    viewer.measurementProcessingWarning.assert_called_once()


@pytest.mark.parametrize('cancel_recipe', [False, True])
def test_cancel_does_not_process(viewer, monkeypatch, cancel_recipe):
    dialog = SimpleNamespace(
        cancel=cancel_recipe, selectedChannels=['area'], selectedRecipe=[],
        exec_=Mock()
    )
    monkeypatch.setattr(apps, 'MeasurementProcessingRecipeDialog', Mock(return_value=dialog))
    monkeypatch.setattr(apps, 'filenameDialog', Mock(return_value=SimpleNamespace(
        cancel=True, exec_=Mock()
    )))
    start = Mock()
    monkeypatch.setattr(viewer, 'startMeasurementProcessing', start)
    viewer.processMeasurements()
    start.assert_not_called()


def test_export_includes_processed_columns(viewer, monkeypatch, tmp_path):
    viewer.startMeasurementProcessing(
        ['area'], [{'method': 'Dummy', 'kwargs': {}}], 'copy'
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
