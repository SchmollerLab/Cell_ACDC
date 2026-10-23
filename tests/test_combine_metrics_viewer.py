from types import SimpleNamespace
from unittest.mock import Mock
import time

import pandas as pd
import pytest
from qtpy.QtCore import Qt
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication

from cellacdc import apps, measurements, measurements_viewer
from cellacdc.measurements_viewer import MeasurementsViewerWin


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def viewer(app, monkeypatch):
    monkeypatch.setattr(MeasurementsViewerWin, 'loadConfiguration', Mock())
    window = MeasurementsViewerWin()
    window.acdc_df = pd.DataFrame({
        'frame_i': [0, 1, 2],
        'Cell_ID': [1, 1, 1],
        'GFP_mean': [2., 4., 8.],
        'RFP_mean': [1., 2., 4.],
        'stage': pd.Categorical(['G1', 'S', 'G1']),
    }).set_index(['frame_i', 'Cell_ID'])
    window.data = [SimpleNamespace(
        acdc_df=window.acdc_df, chNames=['GFP', 'RFP'], SizeZ=1, SizeT=3
    )]
    window.isSegm3D = False
    window.centralWidget().setEnabled(True)
    monkeypatch.setattr(window, 'plotMeasurements', Mock())
    monkeypatch.setattr(window, 'measurementProcessingWarning', Mock())
    window.updateColumnsList()
    yield window
    window.deleteLater()
    app.processEvents()


def accept_formula(monkeypatch, formula, name='combined', before_accept=None):
    dialogs = []

    def accept(dialog):
        dialogs.append(dialog)
        assert isinstance(dialog, apps.CombineMetricsEquationDialog)
        assert not dialog.saveToConfig
        dialog.newColNameLineEdit.setText(name)
        dialog.formulaEditWidget.setText(formula)
        dialog.test_cb()
        assert dialog.okButton.isEnabled()
        if before_accept is not None:
            before_accept()
        dialog.ok_cb()

    monkeypatch.setattr(apps.CombineMetricsEquationDialog, 'exec_', accept)
    return dialogs


def test_button_opens_dialog_and_adds_numeric_column(viewer, monkeypatch):
    original = viewer.acdc_df.copy()
    save = Mock()
    monkeypatch.setattr(measurements, 'save_common_combine_metrics', save)
    accept_formula(monkeypatch, 'GFP_mean / 2')
    viewer.combineMeasurementsButton.click()
    pd.testing.assert_frame_equal(viewer.acdc_df[original.columns], original)
    assert viewer.acdc_df['GFP_combined'].tolist() == [1., 2., 4.]
    assert viewer.posData().acdc_df is viewer.acdc_df
    assert len(list(viewer.columnsModel.measurementItems())) == 3
    viewer.plotMeasurements.assert_called_once()
    viewer.measurementProcessingWarning.assert_not_called()
    save.assert_not_called()


def test_channel_less_equation_adds_one_column_per_channel(viewer, monkeypatch):
    accept_formula(monkeypatch, '_mean / GFP_mean', 'ratio')
    viewer.combineMeasurements()
    assert viewer.acdc_df['GFP_ratio'].tolist() == [1., 1., 1.]
    assert viewer.acdc_df['RFP_ratio'].tolist() == [0.5, 0.5, 0.5]


def test_categorical_comparison_is_available_for_plotting(viewer, monkeypatch):
    accept_formula(monkeypatch, 'stage == "G1"', 'g1')
    viewer.combineMeasurements()
    assert viewer.acdc_df['g1'].tolist() == [True, False, True]
    names = [item.text() for item in viewer.columnsModel.measurementItems()]
    assert 'g1' in names


@pytest.mark.parametrize('formula,name', [
    ('GFP_mean / 2', 'mean'),
    ('_mean / GFP_mean', 'mean'),
])
def test_existing_columns_are_not_overwritten(viewer, monkeypatch, formula, name):
    original = viewer.acdc_df
    dialogs = accept_formula(monkeypatch, formula, name)
    viewer.combineMeasurements()
    assert viewer.acdc_df is original
    assert viewer.posData().acdc_df is original
    viewer.measurementProcessingWarning.assert_called_once()
    assert viewer.measurementProcessingWarning.call_args.args[0] == 'Invalid measurement name'
    assert dialogs[0].cancel
    viewer.plotMeasurements.assert_not_called()


def test_failed_channel_expansion_is_atomic(viewer, monkeypatch):
    viewer.acdc_df = viewer.acdc_df.drop(columns='RFP_mean')
    viewer.posData().acdc_df = viewer.acdc_df
    original = viewer.acdc_df
    accept_formula(monkeypatch, '_mean / GFP_mean', 'ratio')
    viewer.combineMeasurements()
    assert viewer.acdc_df is original
    assert 'GFP_ratio' not in original
    viewer.measurementProcessingWarning.assert_called_once()
    assert viewer.measurementProcessingWarning.call_args.args[0] == 'Combining measurements failed'


def test_cancel_does_not_change_table(viewer, monkeypatch):
    original = viewer.acdc_df
    monkeypatch.setattr(apps.CombineMetricsEquationDialog, 'exec_', lambda _: None)
    viewer.combineMeasurements()
    assert viewer.acdc_df is original
    viewer.plotMeasurements.assert_not_called()


def test_combining_requires_loaded_measurements(viewer):
    viewer.acdc_df = None
    viewer.combineMeasurements()
    viewer.measurementProcessingWarning.assert_called_once_with(
        'No measurements', 'Load measurements first.'
    )


def test_combining_is_blocked_during_processing(viewer):
    viewer.measurementProcessingThread = object()
    try:
        viewer.combineMeasurements()
        viewer.measurementProcessingWarning.assert_called_once()
        assert viewer.measurementProcessingWarning.call_args.args[0] == 'Measurement processing running'
    finally:
        viewer.measurementProcessingThread = None


def test_changed_table_is_not_overwritten(viewer, monkeypatch):
    original = viewer.acdc_df

    def replace_table():
        viewer.acdc_df = original.copy()
        viewer.posData().acdc_df = viewer.acdc_df

    accept_formula(monkeypatch, 'GFP_mean / 2', before_accept=replace_table)
    viewer.combineMeasurements()
    assert 'GFP_combined' not in viewer.acdc_df
    viewer.measurementProcessingWarning.assert_called_once()
    assert viewer.measurementProcessingWarning.call_args.args[0] == 'Measurements changed'


def test_combined_columns_are_exported(viewer, monkeypatch, tmp_path):
    accept_formula(monkeypatch, 'stage == "G1"', 'g1')
    viewer.combineMeasurements()
    filepath = tmp_path / 'combined.csv'
    monkeypatch.setattr(
        measurements_viewer.QFileDialog, 'getSaveFileName',
        lambda *args: (str(filepath), '')
    )
    viewer.exportMeasurements()
    exported = pd.read_csv(filepath).set_index(['frame_i', 'Cell_ID'])
    assert exported['g1'].tolist() == [True, False, True]
    assert exported['GFP_mean'].tolist() == [2., 4., 8.]


def test_category_result_is_retained_without_adding_non_numeric_plot(viewer, monkeypatch):
    accept_formula(monkeypatch, 'stage', 'stage_copy')
    viewer.combineMeasurements()
    assert viewer.acdc_df['stage_copy'].tolist() == ['G1', 'S', 'G1']
    assert len(list(viewer.columnsModel.measurementItems())) == 2


def test_combined_columns_stay_with_their_position(viewer, monkeypatch):
    target = viewer.posData()
    other = SimpleNamespace(acdc_df=viewer.acdc_df.copy())
    viewer.data.append(other)
    accept_formula(monkeypatch, 'GFP_mean / 2')
    viewer.combineMeasurements()
    viewer.pos_i = 1
    viewer.acdc_df = viewer.getAcdcDf()
    assert 'GFP_combined' not in viewer.acdc_df
    viewer.pos_i = 0
    viewer.acdc_df = viewer.getAcdcDf()
    assert 'GFP_combined' in target.acdc_df
    assert viewer.acdc_df['GFP_combined'].tolist() == [1., 2., 4.]


def test_failed_processing_restores_combine_button(viewer):
    original = viewer.acdc_df
    viewer.startSignalProcessing(
        ['GFP_mean'],
        [{'method': 'Smooth measurement', 'kwargs': {'window': 0}}], 'copy'
    )
    assert not viewer.combineMeasurementsButton.isEnabled()
    deadline = time.monotonic() + 10
    while viewer.measurementProcessingThread is not None and time.monotonic() < deadline:
        QTest.qWait(10)
    assert viewer.measurementProcessingThread is None
    assert viewer.combineMeasurementsButton.isEnabled()
    viewer.measurementProcessingWarning.assert_called_once()
    assert viewer.measurementProcessingWarning.call_args.args[0] == 'Measurement processing failed'
    assert viewer.acdc_df is original


@pytest.mark.parametrize('mode', [0, 1])
@pytest.mark.parametrize('normalize', [False, True])
def test_boolean_combined_column_plots(viewer, monkeypatch, mode, normalize):
    accept_formula(monkeypatch, 'stage == "G1"', 'g1')
    viewer.combineMeasurements()
    viewer.selectedID = 1
    viewer.frame_i = 0
    viewer.normalizeCheckbox.setChecked(normalize)
    viewer.measurementsModeCombobox.setCurrentIndex(mode)
    for item in viewer.columnsModel.measurementItems():
        selected = item.text() == 'g1' or (mode == 1 and item.text() == 'GFP_mean')
        item.setCheckState(Qt.Checked if selected else Qt.Unchecked)
    MeasurementsViewerWin.plotMeasurements(viewer)
    plot, = viewer.measurementsPlot.listDataItems()
    assert plot.yData.tolist() == [1., 0., 1.]
