import copy
import json
import os
import sys
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import tifffile
from qtpy.QtGui import QKeySequence
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication

from cellacdc import apps, measurements_viewer
from cellacdc.timepoint_predictors import TIMEPOINT_PREFIX
from cellacdc.measurements_viewer import MeasurementsViewerWin, VISITED_COLUMN
from cellacdc.timepoint_predictors import TimepointIdentifier

# The configurations below use the test-only identity processor 'Dummy'
pytestmark = pytest.mark.usefixtures('dummy_processor')


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def viewer(app, monkeypatch):
    window = MeasurementsViewerWin()
    monkeypatch.setattr(
        window, 'selectConfigurationInput',
        lambda config: window.configurationInputOptions(config)[0]
    )
    monkeypatch.setattr(window, 'configurationError', Mock())
    monkeypatch.setattr(window, 'measurementProcessingWarning', Mock())
    monkeypatch.setattr(
        window, 'askMeasurementReplay', Mock(return_value='recalculate')
    )
    yield window
    window.deleteLater()
    app.processEvents()


def input_option(images):
    return {
        'images_paths': [str(images)], 'channel': 'phase',
        'segm_endname': 'segm', 'overlay_channels': [],
        'visible_channels': ['phase']
    }


def config(images):
    option = input_option(images)
    return {
        'version': 2, **option, 'input_options': [option],
        'measurements_mode': 0, 'time_course_columns': ['final'],
        'measurements_axes': {'X': '', 'Y': ''},
        'normalize_measurements': False,
        'measurement_processing': [{
            'order': 1, 'columns': ['double'],
            'recipe': [{'method': 'Dummy', 'kwargs': {}}],
            'output_columns': ['copied']
        }],
        'combined_measurements': [
            {'order': 0, 'equations': {'double': 'area * 2'}},
            {'order': 2, 'equations': {'final': 'copied + 1'}}
        ]
    }


def make_experiment(root, positions=2):
    paths = []
    for i in range(positions):
        images = root / f'Position_{i + 1}' / 'Images'
        images.mkdir(parents=True)
        tifffile.imwrite(
            images / 'test_phase.tif',
            np.arange(900, dtype=np.uint16).reshape(30, 30)
        )
        labels = np.zeros((30, 30), dtype=np.uint16)
        labels[2:5, 3:7] = 1
        labels[15:18, 20:24] = 2
        np.savez_compressed(images / 'test_segm.npz', labels)
        pd.DataFrame({
            'frame_i': [0, 0], 'Cell_ID': [1, 2], 'area': [12., 24.]
        }).to_csv(images / 'test_acdc_output.csv', index=False)
        paths.append(str(images))
    return paths


def test_real_workflow_roundtrip_and_lazy_predictions(viewer, tmp_path):
    paths = make_experiment(tmp_path)
    configuration = config(paths[0])
    configuration['images_paths'] = paths
    configuration['input_options'][0]['images_paths'] = paths
    identifier = TimepointIdentifier(
        'Start', (255, 0, 0), 'Q',
        {'type': 'equation', 'equation': 'final > 0', 'mode': 'first_true'},
        autoApply=True
    )
    configuration['timepoint_identifiers'] = [identifier.to_dict()]
    configuration['custom_annotations'] = [{
        'type': 'Single time-point', 'description': '',
        'keepActive': True, 'isHideChecked': True,
        'name': 'Flag', 'symbol': "'o' circle",
        'symbolColor': [255, 0, 0, 255], 'shortcut': '',
        'predictor': {'type': 'equation', 'equation': 'final > 0'},
        'autoApply': True
    }]
    configuration['phase_translator_rules'] = [{
        'name': 'Cycle', 'predictor': {'type': 'none'},
        'timepoints': [f'{TIMEPOINT_PREFIX}Start'], 'annotations': ['Flag'],
        'phases': ['', 'Active']
    }]
    path = tmp_path / 'workflow.json'
    path.write_text(json.dumps(configuration))
    assert viewer.loadConfiguration(filepath=str(path))
    viewer.configurationError.assert_not_called()
    for pos in viewer.data:
        assert pos.acdc_df['final'].tolist() == [25., 49.]
    assert viewer.acdc_df[f'{TIMEPOINT_PREFIX}Start'].tolist() == [True, False]
    assert viewer.acdc_df[VISITED_COLUMN].tolist() == [True, False]
    assert not viewer.data[1].acdc_df[f'{TIMEPOINT_PREFIX}Start'].any()
    assert viewer.checkedColumns() == ['final']
    viewer.setSelectedID(2)
    assert viewer.acdc_df[f'{TIMEPOINT_PREFIX}Start'].all()
    assert viewer.acdc_df[VISITED_COLUMN].all()
    assert viewer.applyPhaseTranslator()
    assert viewer.acdc_df['phase'].eq('Active').all()
    assert viewer.saveConfiguration()
    saved = json.loads(path.read_text())
    assert 'viewed_cells' not in saved
    for key in ('measurement_processing', 'combined_measurements',
                'timepoint_identifiers', 'custom_annotations', 'phase_translator_rules'):
        assert saved[key] == configuration[key]
    viewer.loadPosition(1)
    assert viewer.acdc_df[f'{TIMEPOINT_PREFIX}Start'].tolist() == [True, False]
    assert viewer.acdc_df[VISITED_COLUMN].tolist() == [True, False]


def test_replay_is_atomic_and_refuses_collisions():
    table = pd.DataFrame(
        {'area': [2., 4.]},
        index=pd.MultiIndex.from_tuples([(0, 1), (1, 1)], names=['frame_i', 'Cell_ID'])
    )
    pos = SimpleNamespace(acdc_df=table, SizeT=2)
    configuration = config('unused')
    configuration['combined_measurements'][1]['equations']['final'] = 'missing + 1'
    with pytest.raises(ValueError, match='missing'):
        MeasurementsViewerWin.replayMeasurementSteps(pos, configuration)
    assert pos.acdc_df is table
    table['double'] = 7
    with pytest.raises(ValueError, match='overwrite'):
        MeasurementsViewerWin.replayMeasurementSteps(pos, config('unused'))
    assert pos.acdc_df is table


def write_stale_outputs(paths):
    for images in paths:
        tablepath = f'{images}/test_acdc_output.csv'
        table = pd.read_csv(tablepath)
        table['double'] = -1.
        table['copied'] = -1.
        table['final'] = -1.
        table.to_csv(tablepath, index=False)


def test_load_prompt_can_use_existing_values(viewer, tmp_path):
    paths = make_experiment(tmp_path)
    write_stale_outputs(paths)
    configuration = config(paths[0])
    configuration['images_paths'] = paths
    configuration['input_options'][0]['images_paths'] = paths
    path = tmp_path / 'workflow.json'
    path.write_text(json.dumps(configuration))
    viewer.askMeasurementReplay.return_value = 'existing'
    assert viewer.loadConfiguration(filepath=str(path))
    outputs, data = viewer.askMeasurementReplay.call_args.args
    assert outputs == ['double', 'copied', 'final']
    assert len(data) == 2
    for pos in viewer.data:
        assert pos.acdc_df['final'].tolist() == [-1., -1.]
    assert viewer.configuration()['combined_measurements'] == (
        configuration['combined_measurements']
    )


def test_load_prompt_recalculate_overwrites_outputs(viewer, tmp_path):
    paths = make_experiment(tmp_path)
    write_stale_outputs(paths)
    configuration = config(paths[0])
    configuration['images_paths'] = paths
    configuration['input_options'][0]['images_paths'] = paths
    path = tmp_path / 'workflow.json'
    path.write_text(json.dumps(configuration))
    assert viewer.loadConfiguration(filepath=str(path))
    viewer.configurationError.assert_not_called()
    for pos in viewer.data:
        assert pos.acdc_df['final'].tolist() == [25., 49.]


def test_load_prompt_cancel_preserves_window(viewer, tmp_path):
    paths = make_experiment(tmp_path, positions=1)
    path = tmp_path / 'workflow.json'
    path.write_text(json.dumps(config(paths[0])))
    before = viewer.data
    viewer.askMeasurementReplay.return_value = None
    assert not viewer.loadConfiguration(filepath=str(path))
    assert viewer.data is before
    viewer.configurationError.assert_not_called()


def test_load_without_steps_does_not_prompt(viewer, tmp_path):
    paths = make_experiment(tmp_path, positions=1)
    configuration = config(paths[0])
    configuration['measurement_processing'] = []
    configuration['combined_measurements'] = []
    configuration['time_course_columns'] = ['area']
    path = tmp_path / 'workflow.json'
    path.write_text(json.dumps(configuration))
    assert viewer.loadConfiguration(filepath=str(path))
    viewer.askMeasurementReplay.assert_not_called()


@pytest.mark.parametrize('clicked, complete, expected', [
    ('Calculate', False, 'recalculate'),
    ('Use existing values', True, 'existing'),
    ('Use existing values', False, ValueError),
    ('Cancel', True, None),
])
def test_ask_measurement_replay_choices(app, monkeypatch, clicked, complete, expected):
    class FakeMessageBox:
        def __init__(self, *args, **kwargs):
            self.cancel = False

        def question(self, parent, title, txt, detailsText='', buttonsTexts=()):
            FakeMessageBox.details = detailsText
            self.clickedButton = clicked
            return buttonsTexts

    monkeypatch.setattr(measurements_viewer.widgets, 'myMessageBox', FakeMessageBox)
    columns = {'area': [1.], 'final': [2.]} if complete else {'area': [1.]}
    data = [SimpleNamespace(
        acdc_df=pd.DataFrame(columns), pos_path='/exp/Position_1'
    )]
    viewer = SimpleNamespace()
    if expected is ValueError:
        with pytest.raises(ValueError, match='Position_1'):
            MeasurementsViewerWin.askMeasurementReplay(viewer, ['final'], data)
    else:
        assert MeasurementsViewerWin.askMeasurementReplay(
            viewer, ['final'], data
        ) == expected
    assert 'final' in FakeMessageBox.details


class FakeCloseBox:
    """Records the offered buttons and clicks the one named ``click``."""
    click = None
    offered = None

    def __init__(self, *args, **kwargs):
        self.cancel = False

    def warning(self, parent, title, txt, detailsText=None, buttonsTexts=()):
        FakeCloseBox.offered = list(buttonsTexts)
        self.clickedButton = FakeCloseBox.click
        return list(buttonsTexts)


def load_clean_workflow(viewer, tmp_path, steps=True):
    paths = make_experiment(tmp_path)
    configuration = config(paths[0])
    configuration['images_paths'] = paths
    configuration['input_options'][0]['images_paths'] = paths
    if not steps:
        configuration['measurement_processing'] = []
        configuration['combined_measurements'] = []
        configuration['time_course_columns'] = ['area']
    path = tmp_path / 'workflow.json'
    path.write_text(json.dumps(configuration))
    assert viewer.loadConfiguration(filepath=str(path))
    return paths, path


def test_configuration_changes_are_tracked(viewer, tmp_path):
    load_clean_workflow(viewer, tmp_path)
    assert not viewer.hasUnsavedConfiguration()
    viewer.normalizeCheckbox.setChecked(not viewer.normalizeCheckbox.isChecked())
    assert viewer.hasUnsavedConfiguration()
    assert viewer.saveConfiguration()
    assert not viewer.hasUnsavedConfiguration()


def test_table_changes_are_tracked_and_saved(viewer, tmp_path):
    paths, _ = load_clean_workflow(viewer, tmp_path)
    # Replayed workflow columns are not on disk yet.
    assert viewer.unsavedAcdcDfPositions() == viewer.data
    assert viewer.saveAcdcDfs(viewer.data)
    assert not viewer.unsavedAcdcDfPositions()
    # The saved column name is part of the file format
    assert VISITED_COLUMN == 'measurements_viewer_visited'
    for images in paths:
        saved = pd.read_csv(f'{images}/test_acdc_output.csv')
        assert saved['final'].tolist() == [25., 49.]
        assert VISITED_COLUMN in saved
    assert pd.read_csv(f'{paths[0]}/test_acdc_output.csv')[VISITED_COLUMN].tolist() == (
        viewer.data[0].acdc_df[VISITED_COLUMN].tolist()
    )
    viewer.reloadButtonClicked()
    assert not viewer.unsavedAcdcDfPositions()
    viewer.setSelectedID(2)
    assert viewer.unsavedAcdcDfPositions() == [viewer.posData()]


def test_save_tables_saves_only_changed_tables(viewer, tmp_path, monkeypatch):
    paths, _ = load_clean_workflow(viewer, tmp_path)
    saved = []
    saveAcdcDfs = viewer.saveAcdcDfs

    def record(positions):
        saved.append(list(positions))
        return saveAcdcDfs(positions)

    monkeypatch.setattr(viewer, 'saveAcdcDfs', record)
    # Replayed workflow columns are not on disk yet: all tables are saved
    viewer.saveTablesAction.trigger()
    assert saved == [viewer.data]
    assert not viewer.unsavedAcdcDfPositions()
    for images in paths:
        saved_df = pd.read_csv(f'{images}/test_acdc_output.csv')
        assert saved_df['final'].tolist() == [25., 49.]
    # Only the edited table is written
    viewer.setSelectedID(2)
    edited = viewer.posData()
    assert viewer.saveTables()
    assert saved[1:] == [[edited]]
    assert not viewer.unsavedAcdcDfPositions()
    # Nothing to save: no file is written (keeps files saved elsewhere)
    assert viewer.saveTables()
    assert len(saved) == 2


def test_empty_viewer_columns_are_not_changes(viewer, tmp_path):
    load_clean_workflow(viewer, tmp_path)
    viewer.saveAcdcDfs(viewer.data)
    pos = viewer.data[1]
    pos.acdc_df = pos.acdc_df.drop(columns=VISITED_COLUMN, errors='ignore')
    pos.savedAcdcDf = pos.acdc_df.copy()
    pos.acdc_df[VISITED_COLUMN] = False
    assert not viewer.hasUnsavedAcdcDf(pos)
    pos.acdc_df[VISITED_COLUMN] = True
    assert viewer.hasUnsavedAcdcDf(pos)


@pytest.mark.parametrize('tables, configChanged, expected', [
    (True, True, [
        'Cancel', "Don't save", 'Save tables only', 'Save configuration only',
        'Save both as...', 'Save both'
    ]),
    (True, False, ['Cancel', "Don't save", 'Save tables']),
    (False, True, [
        'Cancel', "Don't save", 'Save configuration as...', 'Save configuration'
    ]),
])
def test_close_prompt_options_are_dynamic(
        viewer, tmp_path, monkeypatch, tables, configChanged, expected
    ):
    load_clean_workflow(viewer, tmp_path)
    if not tables:
        viewer.saveAcdcDfs(viewer.data)
    if configChanged:
        viewer.normalizeCheckbox.setChecked(True)
    monkeypatch.setattr(measurements_viewer.widgets, 'myMessageBox', FakeCloseBox)
    FakeCloseBox.click = 'Cancel'
    assert not viewer.askSaveBeforeClosing()
    assert FakeCloseBox.offered == expected


@pytest.mark.parametrize('click, tablesSaved, configSaved, saveAs', [
    ("Don't save", False, False, None),
    ('Save tables only', True, False, None),
    ('Save configuration only', False, True, False),
    ('Save both as...', True, True, True),
    ('Save both', True, True, False),
])
def test_close_prompt_actions(
        viewer, tmp_path, monkeypatch, click, tablesSaved, configSaved, saveAs
    ):
    load_clean_workflow(viewer, tmp_path)
    viewer.normalizeCheckbox.setChecked(True)
    saveConfiguration = Mock(return_value=True)
    monkeypatch.setattr(viewer, 'saveConfiguration', saveConfiguration)
    monkeypatch.setattr(measurements_viewer.widgets, 'myMessageBox', FakeCloseBox)
    FakeCloseBox.click = click
    assert viewer.askSaveBeforeClosing()
    assert bool(viewer.unsavedAcdcDfPositions()) != tablesSaved
    if configSaved:
        saveConfiguration.assert_called_once_with(save_as=saveAs)
    else:
        saveConfiguration.assert_not_called()


def test_close_prompt_failed_save_cancels_close(viewer, tmp_path, monkeypatch):
    load_clean_workflow(viewer, tmp_path)
    monkeypatch.setattr(measurements_viewer.widgets, 'myMessageBox', FakeCloseBox)
    monkeypatch.setattr(viewer, 'saveAcdcDfs', Mock(return_value=False))
    FakeCloseBox.click = 'Save tables'
    assert not viewer.askSaveBeforeClosing()


def test_close_without_changes_does_not_prompt(viewer, tmp_path, monkeypatch):
    load_clean_workflow(viewer, tmp_path)
    viewer.saveAcdcDfs(viewer.data)
    box = Mock()
    monkeypatch.setattr(measurements_viewer.widgets, 'myMessageBox', box)
    assert viewer.askSaveBeforeClosing()
    box.assert_not_called()


@pytest.mark.parametrize('answer', ['reload', 'overwrite', None])
@pytest.mark.parametrize('reloaded', [False, True])
def test_saving_table_changed_on_disk_asks_first(
        viewer, tmp_path, monkeypatch, answer, reloaded
    ):
    load_clean_workflow(viewer, tmp_path, steps=False)
    if reloaded:
        # The stamp of the file is also updated by Reload table
        viewer.reloadButtonClicked()
    viewer.setSelectedID(2)
    pos = viewer.posData()
    csvPath = pos.acdc_output_csv_path
    # E.g., saved in the main GUI after the viewer loaded the table
    external = pd.read_csv(csvPath)
    external['external'] = [1., 2.]
    external.to_csv(csvPath, index=False)
    ask = Mock(return_value=answer)
    monkeypatch.setattr(viewer, 'askAcdcDfChangedOnDisk', ask)
    saved = viewer.saveAcdcDfs([pos])
    ask.assert_called_once_with(pos)
    table = pd.read_csv(csvPath)
    if answer is None:
        assert not saved
        assert 'external' in table and VISITED_COLUMN not in table
        return
    assert saved
    assert table[VISITED_COLUMN].all()
    assert ('external' in table) == (answer == 'reload')
    # The previous file is backed up like in the main GUI
    assert os.path.exists(pos.acdc_output_backup_zip_path)
    ask.reset_mock()
    assert viewer.saveAcdcDfs([pos])
    ask.assert_not_called()


def test_flag_columns_saved_by_main_gui_are_parsed_on_load(tmp_path):
    """The main GUI reindexes a frame's table with fill_value=0 when a cell
    is added (`guiWin.store_data`), turning the viewer's bool columns into
    object columns saved as True/False/0: only true values are marked."""
    images, = make_experiment(tmp_path, positions=1)
    csvPath = f'{images}/test_acdc_output.csv'
    table = pd.read_csv(csvPath).set_index('Cell_ID')
    table[f'{TIMEPOINT_PREFIX}Start'] = [True, False]
    table[VISITED_COLUMN] = [False, True]
    table['annot'] = [0, 1]
    table = table.reindex([1, 2, 3], fill_value=0)  # new cell 3 (like main GUI)
    table['frame_i'] = 0
    table.reset_index().to_csv(csvPath, index=False)
    assert pd.read_csv(csvPath)[VISITED_COLUMN].dtype == object

    pos = measurements_viewer.load_position(images, 'phase')
    acdc_df = MeasurementsViewerWin._standardizeAcdcDf(pos.acdc_df).sort_index()
    timepoints = acdc_df[f'{TIMEPOINT_PREFIX}Start']
    assert timepoints.dtype == bool and timepoints.tolist() == [True, False, False]
    visited = acdc_df[VISITED_COLUMN]
    assert visited.dtype == bool and visited.tolist() == [False, True, False]


def test_visited_ids_and_custom_annotations_parse_string_flags(viewer):
    table = pd.DataFrame(
        {VISITED_COLUMN: ['True', 'False', '0'], 'annot': ['True', '0', 'False']},
        index=pd.MultiIndex.from_tuples(
            [(0, 1), (0, 2), (0, 3)], names=['frame_i', 'Cell_ID']
        ), dtype=object
    )
    pos = SimpleNamespace(acdc_df=table)
    assert viewer.visitedCellIDs(pos) == {1}
    viewer.ensureCustomAnnotColumn(table, 'annot')
    assert table['annot'].tolist() == [1, 0, 0]


def phase_workflow(paths, autoApplyPhases=False):
    """Configuration with a 'Start' time point (predicted automatically on
    viewed cells) and a phase rule ('' before, 'Active' from 'Start')."""
    configuration = config(paths[0])
    configuration['images_paths'] = paths
    configuration['input_options'][0]['images_paths'] = paths
    configuration.update(
        measurement_processing=[], combined_measurements=[],
        time_course_columns=['area'], auto_apply_phases=autoApplyPhases
    )
    identifier = TimepointIdentifier(
        'Start', (255, 0, 0), 'Q',
        {'type': 'equation', 'equation': 'area > 0', 'mode': 'first_true'},
        autoApply=True
    )
    configuration['timepoint_identifiers'] = [identifier.to_dict()]
    configuration['phase_translator_rules'] = [{
        'name': 'Cycle', 'predictor': {'type': 'none'},
        'timepoints': [f'{TIMEPOINT_PREFIX}Start'], 'annotations': [],
        'phases': ['', 'Active']
    }]
    return configuration


def test_empty_phases_survive_save_and_reload_with_cca_columns(
        viewer, tmp_path, app
    ):
    from cellacdc import cca_df_colnames, base_cca_dict
    paths = make_experiment(tmp_path, positions=1)
    csvPath = f'{paths[0]}/test_acdc_output.csv'
    table = pd.read_csv(csvPath)
    for column in cca_df_colnames:
        table[column] = base_cca_dict[column]
    table.to_csv(csvPath, index=False)
    path = tmp_path / 'workflow.json'
    path.write_text(json.dumps(phase_workflow(paths)))
    assert viewer.loadConfiguration(filepath=str(path))
    # Only the displayed cell 1 gets the predicted time point and a phase
    assert viewer.applyPhaseTranslator(allCells=False)
    assert viewer.acdc_df['phase'].tolist() == ['Active', '']
    for _ in range(2):
        assert viewer.saveAcdcDfs(viewer.data)
        viewer.reloadButtonClicked()
        assert viewer.acdc_df['phase'].tolist() == ['Active', '']
        assert not viewer.unsavedAcdcDfPositions()
    pos = measurements_viewer.load_position(paths[0], 'phase')
    assert pos.acdc_df['phase'].tolist() == ['Active', '']
    # The reloaded phases are recognized as translator outputs
    assert viewer.applyPhaseTranslator()


def test_reload_and_save_keeps_phases_evaluated_in_memory(
        viewer, tmp_path, monkeypatch
    ):
    paths = make_experiment(tmp_path, positions=1)
    path = tmp_path / 'workflow.json'
    path.write_text(json.dumps(phase_workflow(paths)))
    assert viewer.loadConfiguration(filepath=str(path))
    assert viewer.applyPhaseTranslator(allCells=True)
    assert viewer.acdc_df['phase'].tolist() == ['Active', '']
    pos = viewer.posData()
    assert viewer.saveAcdcDfs([pos])
    # E.g., saved in the main GUI after the viewer saved the table
    csvPath = pos.acdc_output_csv_path
    external = pd.read_csv(csvPath)
    external['external'] = [1., 2.]
    external.to_csv(csvPath, index=False)
    # Time point predicted on the newly viewed cell, phases evaluated
    viewer.setSelectedID(2)
    assert viewer.applyPhaseTranslator(allCells=True)
    assert viewer.acdc_df['phase'].tolist() == ['Active', 'Active']
    monkeypatch.setattr(
        viewer, 'askAcdcDfChangedOnDisk', Mock(return_value='reload')
    )
    assert viewer.saveAcdcDfs([pos])
    table = pd.read_csv(csvPath)
    assert table['external'].tolist() == [1., 2.]
    assert table[f'{TIMEPOINT_PREFIX}Start'].tolist() == [True, True]
    assert table['phase'].fillna('').tolist() == ['Active', 'Active']
    assert table['phase_Active'].tolist() == [True, True]


def test_phase_evaluation_rebuilds_columns_only_when_they_change(
        viewer, tmp_path, monkeypatch
    ):
    paths = make_experiment(tmp_path, positions=1)
    path = tmp_path / 'workflow.json'
    path.write_text(json.dumps(phase_workflow(paths)))
    assert viewer.loadConfiguration(filepath=str(path))
    calls = []
    updateColumnsList = viewer.updateColumnsList
    def countingUpdateColumnsList(*args, **kwargs):
        calls.append(1)
        return updateColumnsList(*args, **kwargs)
    monkeypatch.setattr(viewer, 'updateColumnsList', countingUpdateColumnsList)
    assert 'phase' not in viewer.acdc_df
    # First evaluation adds the phase outputs
    assert viewer.applyPhaseTranslator(allCells=False)
    assert len(calls) == 1
    previous = viewer.acdc_df
    # Re-evaluation with the same outputs keeps the columns tree
    assert viewer.applyPhaseTranslator(allCells=False)
    assert len(calls) == 1
    assert viewer.acdc_df['phase'].tolist() == ['Active', '']
    # The evaluated table is a new object; the previous one is unchanged
    assert viewer.acdc_df is not previous
    assert previous['phase'].tolist() == ['Active', '']


def test_loading_configuration_auto_applies_phases_to_displayed_cell(
        viewer, tmp_path, app
    ):
    paths = make_experiment(tmp_path, positions=1)
    path = tmp_path / 'workflow.json'
    path.write_text(json.dumps(phase_workflow(paths, autoApplyPhases=True)))
    assert viewer.loadConfiguration(filepath=str(path))
    app.processEvents()
    assert viewer.selectedID == 1
    phases = viewer.acdc_df['phase']
    assert phases.xs(1, level='Cell_ID').tolist() == ['Active']


def test_navigation_toggles_are_persisted(viewer, app):
    for checkbox, name in (
            (viewer.rescalePlotCheckbox, 'rescalePlot'),
            (viewer.restrictFramesCheckbox, 'restrictFrames')):
        checkbox.setChecked(True)
        checkbox.setChecked(False)
        assert not viewer._get_setting_value(name, default=True, cast=bool)
    # Restored by a new window (from the isolated settings CSV)
    other = MeasurementsViewerWin()
    try:
        assert not other.rescalePlotCheckbox.isChecked()
        assert not other.restrictFramesCheckbox.isChecked()
    finally:
        other.deleteLater()
        app.processEvents()


def test_configuration_shows_base_channel_without_visible_layers(
        viewer, tmp_path
    ):
    load_clean_workflow(viewer, tmp_path, steps=False)
    for button in viewer.allOverlayToolbuttons.values():
        button.setChecked(False)
    assert viewer.configuration()['visible_channels'] == ['phase']
    assert viewer.saveConfiguration()
    viewer.configurationError.assert_not_called()


def test_new_experiment_without_visible_overlays_shows_base_channel(
        viewer, tmp_path, monkeypatch
    ):
    paths = make_experiment(tmp_path / 'old', positions=1)
    add_channel(paths[0], 'GFP')
    configuration = config(paths[0])
    configuration.update(
        measurement_processing=[], combined_measurements=[],
        time_course_columns=['area'], overlay_channels=['GFP'],
        visible_channels=['GFP']
    )
    configuration['input_options'] = []
    path = tmp_path / 'workflow.json'
    path.write_text(json.dumps(configuration))
    # The new experiment has no GFP channel: Continue without the overlay
    newPaths = make_experiment(tmp_path / 'new', positions=1)
    monkeypatch.setattr(
        viewer, 'selectConfigurationInput',
        lambda config: viewer.newExperimentInput(config, newPaths)
    )
    monkeypatch.setattr(measurements_viewer.widgets, 'myMessageBox', FakeCloseBox)
    FakeCloseBox.click = 'Continue'
    assert viewer.loadConfiguration(filepath=str(path))
    viewer.configurationError.assert_not_called()
    assert viewer.allOverlayToolbuttons['phase'].isChecked()
    assert viewer.configuration()['visible_channels'] == ['phase']
    assert viewer.saveConfiguration()
    viewer.configurationError.assert_not_called()


def test_invalid_selected_input_is_reported(viewer, tmp_path, monkeypatch):
    paths = make_experiment(tmp_path, positions=1)
    configuration = config(paths[0])
    path = tmp_path / 'workflow.json'
    path.write_text(json.dumps(configuration))
    monkeypatch.setattr(
        viewer, 'selectConfigurationInput',
        lambda config: dict(input_option(paths[0]), visible_channels=[])
    )
    before = viewer.data
    assert not viewer.loadConfiguration(filepath=str(path))
    viewer.configurationError.assert_called_once()
    assert viewer.data is before


def load_other_data(viewer, loader, root):
    """Load another experiment with a configuration or as a folder."""
    paths = make_experiment(root, positions=1)
    if loader == 'configuration':
        configuration = config(paths[0])
        configuration['measurement_processing'] = []
        configuration['combined_measurements'] = []
        configuration['time_course_columns'] = ['area']
        path = root / 'workflow.json'
        path.write_text(json.dumps(configuration))
        return viewer.loadConfiguration(filepath=str(path))
    # openFolder/openFile/openRecentFile all load through loadImagesPaths
    # (called directly so that no recent paths are written to the settings).
    return viewer.loadImagesPaths(paths)


@pytest.mark.parametrize('loader', ['configuration', 'folder'])
@pytest.mark.parametrize('click', ['Cancel', "Don't save", 'Save tables'])
def test_loading_other_data_asks_to_save_changes(
        viewer, tmp_path, monkeypatch, loader, click
    ):
    paths, _ = load_clean_workflow(viewer, tmp_path / 'first', steps=False)
    assert viewer.saveAcdcDfs(viewer.data)
    viewer.setSelectedID(2)
    edited = viewer.posData()
    assert viewer.unsavedAcdcDfPositions() == [edited]
    before = viewer.data
    texts = []

    class FakeLoadBox(FakeCloseBox):
        def warning(self, parent, title, txt, **kwargs):
            texts.append(txt)
            return super().warning(parent, title, txt, **kwargs)

    monkeypatch.setattr(measurements_viewer.widgets, 'myMessageBox', FakeLoadBox)
    FakeCloseBox.click = click
    loaded = load_other_data(viewer, loader, tmp_path / 'second')
    if loader == 'configuration':
        assert loaded == (click != 'Cancel')
    text, = texts
    assert 'before loading other data' in text
    saved = pd.read_csv(f'{paths[0]}/test_acdc_output.csv')
    if click == 'Cancel':
        assert viewer.data is before
        assert edited.acdc_df[VISITED_COLUMN].all()
        assert not saved[VISITED_COLUMN].all()
        return
    assert viewer.data is not before
    assert saved[VISITED_COLUMN].all() == (click == 'Save tables')


def test_loading_other_data_offers_unsaved_configuration(
        viewer, tmp_path, monkeypatch
    ):
    load_clean_workflow(viewer, tmp_path / 'first', steps=False)
    assert viewer.saveAcdcDfs(viewer.data)
    identifier = TimepointIdentifier('Added', (0, 255, 0), '', {'type': 'none'})
    viewer.addTimepointIdentifierItem(identifier)
    assert viewer.hasUnsavedConfiguration()
    before = viewer.data
    monkeypatch.setattr(measurements_viewer.widgets, 'myMessageBox', FakeCloseBox)
    FakeCloseBox.click = 'Cancel'
    viewer.loadImagesPaths(make_experiment(tmp_path / 'second', positions=1))
    assert any('configuration' in text for text in FakeCloseBox.offered)
    # Cancel keeps the data and the unsaved identifier.
    assert viewer.data is before
    assert identifier in viewer.timepointIdentifiers
    assert viewer.hasUnsavedConfiguration()


def test_loading_other_data_without_changes_does_not_prompt(
        viewer, tmp_path, monkeypatch
    ):
    load_clean_workflow(viewer, tmp_path / 'first', steps=False)
    assert viewer.saveAcdcDfs(viewer.data)
    before = viewer.data
    box = Mock()
    monkeypatch.setattr(measurements_viewer.widgets, 'myMessageBox', box)
    viewer.loadImagesPaths(make_experiment(tmp_path / 'second', positions=1))
    box.assert_not_called()
    assert viewer.data is not before


def test_reopening_unsaved_data_predicts_unvisited_cells_again(
        viewer, tmp_path, monkeypatch
    ):
    paths, _ = load_clean_workflow(viewer, tmp_path / 'first', steps=False)
    assert viewer.saveAcdcDfs(viewer.data)
    identifier = TimepointIdentifier(
        'Big', (0, 255, 0), '',
        {'type': 'equation', 'equation': 'area > 20', 'mode': 'first_true'},
        autoApply=True
    )
    viewer.addTimepointIdentifierItem(identifier)
    viewer.setCustomAnnotations([{
        'name': 'Large', 'symbol': "'o' circle",
        'symbolColor': [255, 0, 0, 255], 'shortcut': '',
        'predictor': {'type': 'equation', 'equation': 'area > 20'},
        'autoApply': True,
    }])

    def predicted():
        viewer.setSelectedID(2)
        return (
            viewer.cellTimepoints(identifier.uid, 2),
            viewer.acdc_df.xs(2, level='Cell_ID')['Large'].tolist()
        )

    assert predicted() == ([0], [1])
    # Reopen the same experiment discarding the session's table changes:
    # cell 2 is unvisited in the reloaded table and is predicted again.
    monkeypatch.setattr(measurements_viewer.widgets, 'myMessageBox', FakeCloseBox)
    FakeCloseBox.click = "Don't save"
    viewer.loadImagesPaths(paths)
    assert not viewer.acdc_df.xs(2, level='Cell_ID').get(
        VISITED_COLUMN, pd.Series([False])
    ).astype(bool).any()
    assert predicted() == ([0], [1])


def test_loading_failure_and_cancel_preserve_window(viewer, tmp_path, monkeypatch):
    path = tmp_path / 'workflow.json'
    configuration = config(tmp_path / 'missing')
    path.write_text(json.dumps(configuration))
    before = viewer.data
    assert not viewer.loadConfiguration(filepath=str(path))
    assert viewer.data is before
    viewer.configurationError.assert_called_once()
    viewer.configurationError.reset_mock()
    monkeypatch.setattr(viewer, 'selectConfigurationInput', lambda config: None)
    assert not viewer.loadConfiguration(filepath=str(path))
    viewer.configurationError.assert_not_called()


def add_channel(images, name, factor=2):
    tifffile.imwrite(
        f'{images}/test_{name}.tif',
        np.arange(900, dtype=np.uint16).reshape(30, 30) * factor
    )


def test_new_experiment_reuses_configuration_without_prompts(viewer, tmp_path):
    paths = make_experiment(tmp_path, positions=2)
    for images in paths:
        add_channel(images, 'GFP')
    configuration = config('unused')
    configuration.update(overlay_channels=['GFP'], visible_channels=['GFP'])
    viewer.selectItem = Mock(side_effect=AssertionError('unexpected prompt'))
    option = viewer.newExperimentInput(configuration, paths)
    assert option == {
        'images_paths': paths, 'channel': 'phase', 'segm_endname': 'segm',
        'overlay_channels': ['GFP'], 'visible_channels': ['GFP']
    }


def test_new_experiment_offers_only_unused_channels(viewer, tmp_path, monkeypatch):
    paths = make_experiment(tmp_path, positions=2)
    for images in paths:
        add_channel(images, 'GFP')
        add_channel(images, 'RFP', 3)
    add_channel(paths[0], 'onlyFirst', 4)
    configuration = config('unused')
    configuration.update(overlay_channels=['GFP'], visible_channels=['phase'])
    offered = []

    def select(channels, text=None):
        offered.append(channels)
        return ['RFP']

    monkeypatch.setattr(viewer, 'selectConfigurationOverlayChannels', select)
    option = viewer.newExperimentInput(configuration, paths)
    assert offered == [['RFP']]
    assert option['overlay_channels'] == ['GFP', 'RFP']
    assert option['visible_channels'] == ['phase', 'RFP']


def test_new_experiment_replaces_missing_channel_and_segm(viewer, tmp_path, monkeypatch):
    paths = make_experiment(tmp_path, positions=1)
    add_channel(paths[0], 'GFP')
    configuration = config('unused')
    configuration.update(channel='BF', segm_endname='other')
    prompts = []

    def select(title, text, items):
        prompts.append(title)
        return items[0]

    monkeypatch.setattr(viewer, 'selectItem', select)
    monkeypatch.setattr(
        viewer, 'selectConfigurationOverlayChannels', lambda c, text=None: []
    )
    option = viewer.newExperimentInput(configuration, paths)
    assert prompts == ['Select channel', 'Select segmentation']
    assert option['channel'] == 'GFP'
    assert option['segm_endname'] == 'segm'
    assert option['visible_channels'] == ['GFP']


@pytest.mark.parametrize('click, expected', [('Cancel', None), ('Continue', [])])
def test_new_experiment_missing_overlays_prompt(
        viewer, tmp_path, monkeypatch, click, expected
    ):
    paths = make_experiment(tmp_path, positions=1)
    configuration = config('unused')
    configuration.update(overlay_channels=['GFP'], visible_channels=['GFP'])

    class Box:
        def __init__(self, *args, **kwargs):
            self.cancel = False

        def warning(self, parent, title, txt, buttonsTexts=()):
            Box.txt = txt
            self.clickedButton = click
            return buttonsTexts

    monkeypatch.setattr(measurements_viewer.widgets, 'myMessageBox', Box)
    option = viewer.newExperimentInput(configuration, paths)
    assert 'GFP' in Box.txt
    if expected is None:
        assert option is None
    else:
        assert option['overlay_channels'] == expected
        # No configured visible channel remains: the base channel is shown
        assert option['visible_channels'] == ['phase']


def test_configuration_actions_in_menu_and_toolbar(viewer):
    actions = (
        viewer.saveTablesAction, viewer.loadConfigurationAction,
        viewer.saveConfigurationAction, viewer.saveConfigurationAsAction
    )
    for action in actions:
        assert action in viewer.fileToolBar.actions()
        assert action in viewer.fileMenu.actions()
        assert not action.icon().isNull()
    # Saving counterpart of loading: next to the load folder button
    toolbarActions = viewer.fileToolBar.actions()
    openIndex = toolbarActions.index(viewer.openFolderAction)
    assert toolbarActions[openIndex + 1] is viewer.saveTablesAction
    # The two save buttons are told apart
    assert (
        viewer.saveTablesAction.icon().cacheKey()
        != viewer.saveConfigurationAction.icon().cacheKey()
    )
    # Ctrl+S saves the tables (as in the main GUI)
    assert viewer.saveTablesAction.shortcut() == QKeySequence('Ctrl+S')
    assert (
        viewer.saveConfigurationAction.shortcut()
        == QKeySequence('Ctrl+Shift+S')
    )
    assert viewer.saveConfigurationAsAction.shortcut().isEmpty()
    hardcoded = viewer.getHardcodedShortcuts()
    assert hardcoded[QKeySequence('Ctrl+S').toString()][1] is (
        viewer.saveTablesAction
    )
    viewer.setEnabledFileToolbar(False)
    assert viewer.loadConfigurationAction.isEnabled()
    assert not viewer.saveTablesAction.isEnabled()
    assert not viewer.saveConfigurationAction.isEnabled()
    assert not viewer.saveConfigurationAsAction.isEnabled()
    # Qt5 triggers disabled actions: nothing is loaded, nothing is saved
    assert not viewer.saveTables()


def test_input_choices_include_new_experiment(viewer, tmp_path, monkeypatch):
    configuration = config(tmp_path / 'old')
    paths = make_experiment(tmp_path / 'new', positions=1)
    seen = []

    def select(title, text, items):
        seen.append(items)
        return items[-1]

    monkeypatch.setattr(viewer, 'selectItem', select)
    monkeypatch.setattr(
        measurements_viewer.QFileDialog, 'getExistingDirectory',
        Mock(return_value=str(tmp_path / 'new'))
    )
    option = MeasurementsViewerWin.selectConfigurationInput(viewer, configuration)
    assert seen[0][-1] == 'Load a new experiment...'
    assert option == input_option(paths[0])
    configuration['input_options'].append(option)
    monkeypatch.setattr(viewer, 'selectItem', lambda *args: args[-1][1])
    assert MeasurementsViewerWin.selectConfigurationInput(viewer, configuration) == option


@pytest.mark.parametrize('bad', [
    {'measurement_processing': {}},
    {'combined_measurements': [{'order': 0, 'equations': {}}]},
    {'measurement_processing': [{'order': 1, 'columns': ['area'],
                                'output_columns': [], 'recipe': []}]},
    {'input_options': [{'images_paths': []}]},
])
def test_invalid_workflow_is_rejected(bad):
    configuration = config('unused')
    configuration.update(bad)
    with pytest.raises(ValueError):
        MeasurementsViewerWin.validateConfiguration(configuration)


def test_duplicate_orders_are_rejected():
    configuration = config('unused')
    configuration['measurement_processing'][0]['order'] = 0
    with pytest.raises(ValueError, match='order'):
        MeasurementsViewerWin.validateConfiguration(configuration)


@pytest.mark.parametrize('second', [
    {'order': 3, 'columns': ['area'], 'output_columns': ['copied'],
     'recipe': [{'method': 'Dummy', 'kwargs': {'other': 1}}]},
    {'order': 3, 'equations': {'copied': 'area + 1'}},
])
def test_duplicate_outputs_across_steps_are_rejected(second):
    configuration = config('unused')
    section = (
        'combined_measurements' if 'equations' in second
        else 'measurement_processing'
    )
    configuration[section].append(second)
    with pytest.raises(ValueError, match='Duplicate measurement output.*copied'):
        MeasurementsViewerWin.validateConfiguration(configuration)


def smooth_step(repeats, outputs=('area_s',), columns=('area',)):
    """Processing step; different `repeats` give different recipes."""
    return {
        'columns': list(columns), 'output_columns': list(outputs),
        'recipe': [{'method': 'Dummy', 'kwargs': {}}] * repeats
    }


def test_reapplying_output_names_replaces_recorded_step(viewer):
    viewer.recordMeasurementStep(viewer.measurementProcessingSteps, smooth_step(3))
    viewer.recordMeasurementStep(
        viewer.combinedMeasurementSteps, {'equations': {'ratio': 'area_s / 2'}}
    )
    note = viewer.recordMeasurementStep(
        viewer.measurementProcessingSteps, smooth_step(5)
    )
    assert 'new parameters' in note and 'area_s' in note
    assert viewer.measurementProcessingSteps == [{'order': 0, **smooth_step(5)}]
    assert viewer.combinedMeasurementSteps == [
        {'order': 1, 'equations': {'ratio': 'area_s / 2'}}
    ]
    # Re-applying identical steps (e.g. at another position) changes nothing
    assert viewer.recordMeasurementStep(
        viewer.measurementProcessingSteps, smooth_step(5)
    ) == ''
    assert len(viewer.measurementProcessingSteps) == 1

    configuration = {
        'measurement_processing': viewer.measurementProcessingSteps,
        'combined_measurements': viewer.combinedMeasurementSteps,
    }
    MeasurementsViewerWin.validateMeasurementSteps(configuration)
    table = pd.DataFrame(
        {'area': [2., 4.]},
        index=pd.MultiIndex.from_tuples([(0, 1), (1, 1)], names=['frame_i', 'Cell_ID'])
    )
    pos = SimpleNamespace(acdc_df=table, SizeT=2)
    MeasurementsViewerWin.replayMeasurementSteps(pos, configuration)
    assert pos.acdc_df['ratio'].tolist() == [1., 2.]


def ratio_table():
    return pd.DataFrame(
        {'GFP': [2., 4.], 'mCherry': [1., 2.], 'area': [1., 2.]},
        index=pd.MultiIndex.from_tuples([(0, 1), (1, 1)], names=['frame_i', 'Cell_ID'])
    )


def test_replacement_reading_later_outputs_moves_after_them(viewer):
    combined = viewer.combinedMeasurementSteps
    viewer.recordMeasurementStep(combined, {'equations': {'norm': 'GFP / area'}})
    viewer.recordMeasurementStep(combined, {'equations': {'ratio': 'GFP / mCherry'}})
    # At another position: ratio again (identical), then norm from ratio
    viewer.recordMeasurementStep(combined, {'equations': {'ratio': 'GFP / mCherry'}})
    note = viewer.recordMeasurementStep(
        combined, {'equations': {'norm': 'ratio / area'}}
    )
    assert 'new parameters' in note
    assert combined == [
        {'order': 1, 'equations': {'ratio': 'GFP / mCherry'}},
        {'order': 2, 'equations': {'norm': 'ratio / area'}},
    ]
    configuration = {
        'measurement_processing': [], 'combined_measurements': combined
    }
    MeasurementsViewerWin.validateMeasurementSteps(configuration)
    pos = SimpleNamespace(acdc_df=ratio_table(), SizeT=2)
    MeasurementsViewerWin.replayMeasurementSteps(
        pos, configuration, overwrite=True
    )
    assert pos.acdc_df['norm'].tolist() == [2., 1.]


def test_replacement_reading_later_outputs_is_refused_when_read(
        viewer, monkeypatch
    ):
    combined = viewer.combinedMeasurementSteps
    viewer.recordMeasurementStep(combined, {'equations': {'norm': 'GFP / area'}})
    viewer.recordMeasurementStep(combined, {'equations': {'ratio': 'GFP / mCherry'}})
    viewer.recordMeasurementStep(combined, {'equations': {'twice': 'norm * 2'}})
    recorded = copy.deepcopy(combined)
    assert viewer.measurementStepReplacementOrder(
        {'equations': {'norm': 'ratio / area'}}
    ) == (0, ['ratio'])
    table = ratio_table()
    table['ratio'] = table['GFP'] / table['mCherry']
    pos = SimpleNamespace(acdc_df=table)
    viewer.data = [pos]
    viewer.acdc_df = table
    monkeypatch.setattr(viewer, 'posData', lambda: pos)
    dialog = SimpleNamespace(
        getEquationsDict=lambda: ({'norm': 'ratio / area'}, {}),
        evaluateEquation=Mock(), close=Mock()
    )
    viewer.measurementProcessingWarning.reset_mock()
    viewer.applyCombinedMeasurements(
        dialog, posData=pos, source=table, equationSource=table.copy()
    )
    viewer.measurementProcessingWarning.assert_called_once()
    dialog.evaluateEquation.assert_not_called()
    assert 'norm' not in viewer.acdc_df
    assert combined == recorded
    # Defensive: recording it anyway keeps the configuration replayable
    assert viewer.recordMeasurementStep(
        combined, {'equations': {'norm': 'ratio / area'}}
    ) == ''
    assert combined == recorded


@pytest.mark.parametrize('section, step', [
    ('combined_measurements', {'order': 0, 'equations': {'early': 'final + 1'}}),
    ('measurement_processing', {
        'order': 0, 'columns': ['final'], 'output_columns': ['early'],
        'recipe': [{'method': 'Dummy', 'kwargs': {}}]
    }),
])
def test_steps_reading_later_outputs_are_rejected(section, step):
    configuration = config('unused')
    for existing in (
            configuration['measurement_processing']
            + configuration['combined_measurements']):
        existing['order'] += 1
    configuration[section].append(step)
    with pytest.raises(ValueError, match='later step: final'):
        MeasurementsViewerWin.validateMeasurementSteps(configuration)


def test_partially_overlapping_outputs_are_refused(viewer, monkeypatch):
    viewer.recordMeasurementStep(
        viewer.measurementProcessingSteps,
        smooth_step(3, outputs=['a_s', 'b_s'], columns=['a', 'b'])
    )
    assert viewer.recordedOutputConflicts(['a_s']) == ['a_s']
    assert viewer.recordedOutputConflicts(['a_s', 'b_s']) == []
    assert viewer.recordedOutputConflicts(['c_s']) == []

    table = pd.DataFrame(
        {'area': [2., 4.]},
        index=pd.MultiIndex.from_tuples([(0, 1), (1, 1)], names=['frame_i', 'Cell_ID'])
    )
    pos = SimpleNamespace(acdc_df=table)
    viewer.data = [pos]
    viewer.acdc_df = table
    monkeypatch.setattr(viewer, 'posData', lambda: pos)
    dialog = SimpleNamespace(
        getEquationsDict=lambda: ({'a_s': 'area + 1'}, {}),
        evaluateEquation=Mock(), close=Mock()
    )
    viewer.measurementProcessingWarning.reset_mock()
    viewer.applyCombinedMeasurements(
        dialog, posData=pos, source=table, equationSource=table.copy()
    )
    viewer.measurementProcessingWarning.assert_called_once()
    dialog.evaluateEquation.assert_not_called()
    assert 'a_s' not in viewer.acdc_df
    assert viewer.combinedMeasurementSteps == []

    # Processing is refused before it starts
    class Dialog:
        def __init__(self, *args, **kwargs):
            self.cancel = False
            self.selectedChannels = ['area']
            self.selectedRecipe = [{'method': 'Dummy', 'kwargs': {}}]
            self.saveNames = ['b_s']
        def exec_(self):
            pass
    monkeypatch.setattr(
        measurements_viewer.apps_measurements_viewer,
        'MeasurementProcessingRecipeDialog', Dialog
    )
    pos.chNames = ['phase']
    start = Mock()
    monkeypatch.setattr(viewer, 'startMeasurementsProcessing', start)
    viewer.measurementProcessingWarning.reset_mock()
    viewer.processMeasurements()
    viewer.measurementProcessingWarning.assert_called_once()
    start.assert_not_called()


def test_successful_processing_and_combining_are_recorded(viewer, monkeypatch):
    table = pd.DataFrame(
        {'area': [2., 4.]},
        index=pd.MultiIndex.from_tuples([(0, 1), (1, 1)], names=['frame_i', 'Cell_ID'])
    )
    pos = SimpleNamespace(acdc_df=table)
    viewer.data = [pos]
    viewer.acdc_df = table
    monkeypatch.setattr(viewer, 'plotMeasurements', Mock())
    viewer.startMeasurementsProcessing(
        ['area'], [{'method': 'Dummy', 'kwargs': {}}], ['copy']
    )
    for _ in range(1000):
        if viewer.measurementProcessingThread is None:
            break
        QTest.qWait(10)
    assert viewer.measurementProcessingThread is None
    assert viewer.measurementProcessingSteps == [{
        'order': 0, 'columns': ['area'], 'output_columns': ['copy'],
        'recipe': [{'method': 'Dummy', 'kwargs': {}}]
    }]
    source = viewer.acdc_df
    dialog = SimpleNamespace(
        getEquationsDict=lambda: ({'combined': 'copy + 1'}, {}),
        evaluateEquation=lambda df, expression: apps.CombineMetricsEquationDialog.evaluateCellEquation(
            df, expression
        ),
        close=Mock()
    )
    viewer.applyCombinedMeasurements(
        dialog, posData=pos, source=source, equationSource=source.copy()
    )
    assert viewer.combinedMeasurementSteps == [
        {'order': 1, 'equations': {'combined': 'copy + 1'}}
    ]
    assert viewer.acdc_df['combined'].tolist() == [3., 5.]
    viewer.recordMeasurementStep(
        viewer.combinedMeasurementSteps, {'equations': {'combined': 'copy + 1'}}
    )
    assert len(viewer.combinedMeasurementSteps) == 1


def test_visited_flag_survives_export_and_reload(viewer, tmp_path, monkeypatch):
    paths = make_experiment(tmp_path, positions=1)
    configuration = config(paths[0])
    path = tmp_path / 'workflow.json'
    path.write_text(json.dumps(configuration))
    assert viewer.loadConfiguration(filepath=str(path))
    assert viewer.acdc_df[VISITED_COLUMN].tolist() == [True, False]
    export = tmp_path / 'export.csv'
    monkeypatch.setattr(
        measurements_viewer.QFileDialog, 'getSaveFileName',
        Mock(return_value=(str(export), 'CSV (*.csv)'))
    )
    viewer.exportMeasurements()
    table = pd.read_csv(export, index_col=['frame_i', 'Cell_ID'])
    assert table[VISITED_COLUMN].tolist() == [True, False]
    viewer.acdc_df = table
    viewer.posData().acdc_df = table
    viewer.ensurePredictorPredictions()
    assert table[VISITED_COLUMN].tolist() == [True, False]
    table[VISITED_COLUMN] = False
    viewer.restoreTimepointColumns(viewer.posData(), pd.read_csv(
        export, index_col=['frame_i', 'Cell_ID']
    ))
    assert viewer.posData().acdc_df[VISITED_COLUMN].tolist() == [True, False]


def test_legacy_config_migrates_viewed_ids(viewer, tmp_path):
    paths = make_experiment(tmp_path, positions=1)
    configuration = config(paths[0])
    configuration['version'] = 1
    configuration.pop('input_options')
    configuration['viewed_cells'] = {
        measurements_viewer.os.path.normcase(str(tmp_path / 'Position_1')): [2]
    }
    path = tmp_path / 'legacy.json'
    path.write_text(json.dumps(configuration))
    assert viewer.loadConfiguration(filepath=str(path))
    assert viewer.acdc_df[VISITED_COLUMN].all()
    assert 'viewed_cells' not in viewer.configuration()


def test_new_experiment_input_is_retained_on_save(viewer, tmp_path, monkeypatch):
    original = make_experiment(tmp_path / 'original', positions=1)
    other = make_experiment(tmp_path / 'other', positions=1)
    configuration = config(original[0])
    path = tmp_path / 'workflow.json'
    path.write_text(json.dumps(configuration))
    monkeypatch.setattr(
        viewer, 'selectConfigurationInput', lambda config: input_option(other[0])
    )
    assert viewer.loadConfiguration(filepath=str(path))
    assert viewer.saveConfiguration()
    saved = json.loads(path.read_text())
    assert saved['input_options'] == [
        input_option(original[0]), input_option(other[0])
    ]
    assert saved['images_paths'] == other


def test_saved_input_restores_channels_and_overlays(viewer, tmp_path, monkeypatch):
    # Exceptions raised inside Qt slots (e.g. `overlay_cb` when reloading
    # with overlays active) only reach sys.excepthook
    slotErrors = []
    monkeypatch.setattr(
        sys, 'excepthook', lambda *exc_info: slotErrors.append(exc_info)
    )
    paths = make_experiment(tmp_path, positions=1)
    images = tmp_path / 'Position_1' / 'Images'
    tifffile.imwrite(
        images / 'test_GFP.tif',
        np.arange(900, dtype=np.uint16).reshape(30, 30) * 2
    )
    configuration = config(paths[0])
    configuration['overlay_channels'] = ['GFP']
    configuration['visible_channels'] = ['GFP']
    configuration['input_options'][0].update(
        overlay_channels=['GFP'], visible_channels=['GFP']
    )
    second = {
        **input_option(paths[0]), 'channel': 'GFP',
        'visible_channels': ['GFP']
    }
    configuration['input_options'].append(second)
    path = tmp_path / 'workflow.json'
    path.write_text(json.dumps(configuration))
    assert viewer.loadConfiguration(filepath=str(path))
    assert viewer.checkedOverlayChannels == {'GFP'}
    assert viewer.allOverlayToolbuttons['GFP'].isChecked()
    assert not viewer.allOverlayToolbuttons['phase'].isChecked()
    monkeypatch.setattr(viewer, 'selectConfigurationInput', lambda config: second)
    # Discard the unsaved replayed columns of the first load.
    monkeypatch.setattr(viewer, 'askSaveBeforeLoading', Mock(return_value=True))
    assert viewer.loadConfiguration(filepath=str(path))
    assert not slotErrors
    assert viewer.user_ch_name == 'GFP'
    assert not viewer.checkedOverlayChannels
    assert viewer.allOverlayToolbuttons['GFP'].isChecked()
    assert viewer.saveConfiguration()
    assert json.loads(path.read_text())['input_options'] == configuration['input_options']


def test_saving_unsorted_overlays_does_not_duplicate_input(viewer, tmp_path):
    paths = make_experiment(tmp_path, positions=1)
    images = tmp_path / 'Position_1' / 'Images'
    for factor, name in enumerate(('mCherry', 'GFP'), start=2):
        tifffile.imwrite(
            images / f'test_{name}.tif',
            np.arange(900, dtype=np.uint16).reshape(30, 30) * factor
        )
    configuration = config(paths[0])
    # Not in alphabetical (nor toolbutton) order
    channels = {
        'overlay_channels': ['mCherry', 'GFP'],
        'visible_channels': ['mCherry', 'phase', 'GFP'],
    }
    configuration.update(channels)
    configuration['input_options'][0].update(copy.deepcopy(channels))
    path = tmp_path / 'workflow.json'
    path.write_text(json.dumps(configuration))
    assert viewer.loadConfiguration(filepath=str(path))
    assert viewer.checkedOverlayChannels == {'mCherry', 'GFP'}
    assert len(viewer.configuration()['input_options']) == 1
    assert viewer.saveConfiguration()
    saved = json.loads(path.read_text())
    assert len(saved['input_options']) == 1
    option = saved['input_options'][0]
    assert sorted(option['overlay_channels']) == ['GFP', 'mCherry']
    assert sorted(option['visible_channels']) == ['GFP', 'mCherry', 'phase']
    assert len(MeasurementsViewerWin.configurationInputOptions(saved)) == 1


def test_toggling_layer_visibility_updates_input_in_place(viewer, tmp_path):
    paths = make_experiment(tmp_path, positions=1)
    images = tmp_path / 'Position_1' / 'Images'
    tifffile.imwrite(
        images / 'test_GFP.tif',
        np.arange(900, dtype=np.uint16).reshape(30, 30) * 2
    )
    configuration = config(paths[0])
    channels = {
        'overlay_channels': ['GFP'], 'visible_channels': ['phase', 'GFP']
    }
    configuration.update(channels)
    configuration['input_options'][0].update(copy.deepcopy(channels))
    path = tmp_path / 'workflow.json'
    path.write_text(json.dumps(configuration))
    assert viewer.loadConfiguration(filepath=str(path))
    viewer.allOverlayToolbuttons['GFP'].setChecked(False)
    assert viewer.saveConfiguration()
    viewer.allOverlayToolbuttons['GFP'].setChecked(True)
    viewer.allOverlayToolbuttons['phase'].setChecked(False)
    assert viewer.saveConfiguration()
    saved = json.loads(path.read_text())
    assert len(saved['input_options']) == 1
    option = saved['input_options'][0]
    assert option['overlay_channels'] == ['GFP']
    assert option['visible_channels'] == ['GFP']
    assert len(MeasurementsViewerWin.configurationInputOptions(saved)) == 1


def test_layer_change_updates_loaded_input_not_first_same_experiment(
        viewer, tmp_path):
    paths = make_experiment(tmp_path, positions=1)
    images = tmp_path / 'Position_1' / 'Images'
    tifffile.imwrite(
        images / 'test_GFP.tif',
        np.arange(900, dtype=np.uint16).reshape(30, 30) * 2
    )
    configuration = config(paths[0])
    channels = {
        'overlay_channels': ['GFP'], 'visible_channels': ['phase', 'GFP']
    }
    configuration.update(channels)
    configuration['input_options'][0].update(copy.deepcopy(channels))
    path = tmp_path / 'workflow.json'
    path.write_text(json.dumps(configuration))
    assert viewer.loadConfiguration(filepath=str(path))
    loaded = copy.deepcopy(viewer.configurationInputs[0])
    # Another saved option of the same experiment listed before the loaded one
    other = {**copy.deepcopy(loaded), 'visible_channels': ['GFP']}
    viewer.configurationInputs = [other, loaded]
    viewer.allOverlayToolbuttons['GFP'].setChecked(False)
    options = viewer.configuration()['input_options']
    assert len(options) == 2
    assert options[0] == other
    assert options[1]['overlay_channels'] == ['GFP']
    assert options[1]['visible_channels'] == ['phase']
    # Later saves keep tracking the updated option
    assert viewer.saveConfiguration()
    viewer.allOverlayToolbuttons['GFP'].setChecked(True)
    options = viewer.configuration()['input_options']
    assert len(options) == 2
    assert options[0] == other
    assert options[1]['visible_channels'] == ['phase', 'GFP']


def test_input_labels_show_overlays(viewer, tmp_path, monkeypatch):
    paths = make_experiment(tmp_path, positions=1)
    configuration = config(paths[0])
    configuration['input_options'].append({
        **input_option(paths[0]), 'overlay_channels': ['GFP'],
        'visible_channels': ['GFP']
    })
    select = Mock(return_value=None)
    monkeypatch.setattr(viewer, 'selectItem', select)
    MeasurementsViewerWin.selectConfigurationInput(viewer, configuration)
    labels = select.call_args.args[2][:-1]
    assert len(labels) == 2 and len(set(labels)) == 2
    assert 'overlays' not in labels[0]
    assert labels[1].endswith('; overlays: GFP)')


def test_loading_configuration_removes_previous_custom_annotations(
        viewer, tmp_path, monkeypatch):
    first = make_experiment(tmp_path / 'first', positions=1)
    second = make_experiment(tmp_path / 'second')
    configuration = config(first[0])
    configuration['measurement_processing'] = []
    configuration['combined_measurements'] = []
    configuration['time_course_columns'] = ['area']
    configuration['custom_annotations'] = [{
        'name': 'Dead', 'symbolColor': [255, 0, 0, 255], 'shortcut': 'Q',
        'predictor': {'type': 'equation', 'equation': 'area < 0'},
    }]
    path = tmp_path / 'first.json'
    path.write_text(json.dumps(configuration))
    assert viewer.loadConfiguration(filepath=str(path))
    assert 'Dead' in viewer.customAnnotNames()
    assert viewer.saveAcdcDfs(viewer.unsavedAcdcDfPositions())

    other = copy.deepcopy(configuration)
    other.pop('custom_annotations')
    other['images_paths'] = second
    other['input_options'] = [{**input_option(second[0]), 'images_paths': second}]
    identifier = TimepointIdentifier('Start', (255, 0, 0), 'Q')
    other['timepoint_identifiers'] = [identifier.to_dict()]
    path = tmp_path / 'second.json'
    path.write_text(json.dumps(other))
    assert viewer.loadConfiguration(filepath=str(path))
    viewer.configurationError.assert_not_called()
    assert not viewer.customAnnotNames()
    for posData in viewer.data:
        assert 'Dead' not in posData.acdc_df.columns
    assert 'Dead' not in viewer.acdc_df.columns
    assert viewer.saveAcdcDfs(viewer.unsavedAcdcDfPositions())
    for images in second:
        saved = pd.read_csv(os.path.join(images, 'test_acdc_output.csv'))
        assert 'Dead' not in saved.columns
    loaded, = viewer.timepointIdentifiers
    autoAction = viewer.timepointAutoActions[loaded.uid]
    assert autoAction.shortcut().toString() == 'Shift+Q'


def test_exported_manual_assignments_are_not_predicted_again(viewer, tmp_path):
    paths = make_experiment(tmp_path, positions=1)
    images = tmp_path / 'Position_1' / 'Images'
    tablepath = images / 'test_acdc_output.csv'
    table = pd.read_csv(tablepath)
    table[VISITED_COLUMN] = [True, False]
    table[f'{TIMEPOINT_PREFIX}Start'] = [False, False]
    table['Flag'] = [1, 0]
    table.to_csv(tablepath, index=False)
    configuration = config(paths[0])
    configuration['timepoint_identifiers'] = [TimepointIdentifier(
        'Start', (255, 0, 0), 'Q',
        {'type': 'equation', 'equation': 'final > 0', 'mode': 'first_true'},
        autoApply=True
    ).to_dict()]
    configuration['custom_annotations'] = [{
        'name': 'Flag', 'symbolColor': [255, 0, 0, 255],
        'predictor': {'type': 'equation', 'equation': 'final < 0'},
        'autoApply': True
    }]
    path = tmp_path / 'workflow.json'
    path.write_text(json.dumps(configuration))
    assert viewer.loadConfiguration(filepath=str(path))
    assert not viewer.acdc_df[f'{TIMEPOINT_PREFIX}Start'].any()
    assert viewer.acdc_df['Flag'].tolist() == [1, 0]
    viewer.setSelectedID(2)
    assert viewer.acdc_df[f'{TIMEPOINT_PREFIX}Start'].tolist() == [False, True]
    assert viewer.acdc_df['Flag'].tolist() == [1, 0]


def premade_config():
    return {
        'version': 2,
        'channel_placeholders': {'MAIN': 'main channel', 'OTHER': ''},
        'channel': '{MAIN}', 'segm_endname': '',
        'overlay_channels': ['{OTHER}'],
        'visible_channels': ['{MAIN}'],
        'combined_measurements': [
            {'order': 0, 'equations': {'{OTHER}_double': 'area * 2'}}
        ],
        'measurements_mode': 0,
        'time_course_columns': ['{OTHER}_double'],
        'measurements_axes': {'X': '', 'Y': ''},
        'normalize_measurements': False,
    }


def test_substitute_channel_placeholders():
    obj = {
        '{RED}_x': ['{RED}', 'RED_y', {'a': '{RED}/{GREEN}'}],
        'n': 1, 'b': None
    }
    result = MeasurementsViewerWin.substituteChannelPlaceholders(
        obj, {'RED': 'mCherry', 'GREEN': 'GFP'}
    )
    assert result == {
        'mCherry_x': ['mCherry', 'RED_y', {'a': 'mCherry/GFP'}],
        'n': 1, 'b': None
    }
    assert obj['{RED}_x'][0] == '{RED}'
    assert MeasurementsViewerWin.unresolvedPlaceholders(
        {'a': '{RED}'}, ['RED', 'GREEN']
    ) == ['RED']


def test_quoted_placeholders_keep_backticks_only_if_needed():
    equation = '`{RED}_mean` / `{GREEN}_mean` + `other col` + {RED}_x'
    quoted = set()
    result = MeasurementsViewerWin.substituteChannelPlaceholders(
        {'eq': equation}, {'RED': 'mKO2-hCdt1', 'GREEN': 'GFP'},
        quotedNames=quoted
    )
    # Unrelated quoted names and unquoted tokens are left as written
    assert result == {
        'eq': '`mKO2-hCdt1_mean` / GFP_mean + `other col` + mKO2-hCdt1_x'
    }
    assert quoted == {'mKO2-hCdt1_mean'}


@pytest.mark.parametrize('name, valid', [
    ('GFP_mean', True), ('mKO2-hCdt1_mean', True), ('_561_mean', True),
    ('561_mean', False), ('GFP-488_mean', False), ('GFP.488_mean', False),
    ('', False), ('a`b', False),
])
def test_is_equation_name_rule(name, valid):
    # Explicit rule, independent of the interpreter's tokenizer (Python
    # 3.10/3.11 split `561_mean` silently, 3.12+ reject it)
    assert MeasurementsViewerWin.isEquationName(name) is valid


@pytest.mark.parametrize('channel, valid', [
    ('GFP', True), ('mKO2-hCdt1', True), ('_561', True),
    ('561', False), ('GFP-488', False),
])
def test_premade_equation_channel_names(channel, valid):
    template = premade_config()
    template['combined_measurements'].append({
        'order': 1, 'equations': {'final': '`{OTHER}_double` + 1'}
    })
    mapping = {'MAIN': 'phase', 'OTHER': channel}
    if not valid:
        with pytest.raises(ValueError, match='must start with a letter'):
            MeasurementsViewerWin.resolvePremadeTemplate(template, mapping)
        return
    result = MeasurementsViewerWin.resolvePremadeTemplate(template, mapping)
    posData = SimpleNamespace(SizeT=1, acdc_df=pd.DataFrame(
        {'frame_i': [0, 0], 'Cell_ID': [1, 2], 'area': [1., 2.]}
    ).set_index(['frame_i', 'Cell_ID']))
    MeasurementsViewerWin.replayMeasurementSteps(posData, result)
    assert posData.acdc_df['final'].tolist() == [3., 5.]
    assert result['combined_measurements'][1]['equations']['final'] == (
        f'`{channel}_double` + 1' if channel == 'mKO2-hCdt1'
        else f'{channel}_double + 1'
    )


@pytest.mark.parametrize('bad', [
    None, {}, [], {'': 'x'}, {'RE D': 'x'}, {'{RED}': 'x'}, {'RED': 1}
])
def test_invalid_channel_placeholders_are_rejected(bad):
    with pytest.raises(ValueError):
        MeasurementsViewerWin.validateChannelPlaceholders(bad)


def patch_premade_dialogs(viewer, monkeypatch, folder, saved, mapping):
    monkeypatch.setattr(measurements_viewer, 'MEASUREMENTS_CONFIGURATIONS_PATH', str(saved))
    monkeypatch.setattr(
        measurements_viewer.QFileDialog, 'getExistingDirectory',
        lambda *args, **kwargs: str(folder)
    )
    offered = []

    def select(placeholders, channels, inputs=None, validate=None, **kwargs):
        offered.append(channels)
        if mapping is None:
            return None
        if validate is not None:
            validate(mapping, {})
        return mapping, {}

    monkeypatch.setattr(viewer, 'selectChannelMapping', select)
    monkeypatch.setattr(
        viewer, 'selectConfigurationOverlayChannels', lambda c, text=None: []
    )
    monkeypatch.setattr(
        viewer, 'askPremadeCopyFilename',
        lambda name: str(saved / f'{name}.json')
    )
    # Keep the template's base channel unless a test overrides this
    monkeypatch.setattr(
        viewer, 'selectPremadeBaseChannel',
        lambda channels, default=None: default
    )
    return offered


def test_premade_configuration_is_copied_and_loaded(viewer, tmp_path, monkeypatch):
    experiment = tmp_path / 'exp'
    paths = make_experiment(experiment)
    for images in paths:
        add_channel(images, 'GFP')
    saved = tmp_path / 'saved'
    saved.mkdir()
    premade = tmp_path / 'template.json'
    premade.write_text(json.dumps(premade_config()))
    offered = patch_premade_dialogs(
        viewer, monkeypatch, experiment, saved,
        {'MAIN': 'phase', 'OTHER': 'GFP'}
    )
    assert viewer.loadConfiguration(filepath=str(premade))
    assert offered == [['GFP', 'phase']]
    copy_path = saved / 'template_phase_GFP.json'
    assert viewer.configurationFilepath == str(copy_path)
    text = copy_path.read_text()
    assert '{MAIN}' not in text and '{OTHER}' not in text
    written = json.loads(text)
    assert 'channel_placeholders' not in written
    assert written['premade'] == {
        'source': 'template.json',
        'channel_mapping': {'MAIN': 'phase', 'OTHER': 'GFP'}
    }
    assert written['channel'] == 'phase'
    assert written['segm_endname'] == 'segm'
    assert written['overlay_channels'] == ['GFP']
    assert written['images_paths'] == paths
    assert written['combined_measurements'][0]['equations'] == {
        'GFP_double': 'area * 2'
    }
    assert viewer.user_ch_name == 'phase'
    assert 'GFP_double' in viewer.data[0].acdc_df
    # The template is untouched
    assert json.loads(premade.read_text()) == premade_config()


def test_premade_selected_display_channel_is_base(viewer, tmp_path, monkeypatch):
    experiment = tmp_path / 'exp'
    paths = make_experiment(experiment, positions=1)
    add_channel(paths[0], 'GFP')
    add_channel(paths[0], 'combined', 3)
    saved = tmp_path / 'saved'
    saved.mkdir()
    premade = tmp_path / 'template.json'
    premade.write_text(json.dumps(premade_config()))
    patch_premade_dialogs(
        viewer, monkeypatch, experiment, saved,
        {'MAIN': 'phase', 'OTHER': 'GFP'}
    )
    asked = []

    def select_base(channels, default=None):
        asked.append((channels, default))
        return 'combined'

    monkeypatch.setattr(viewer, 'selectPremadeBaseChannel', select_base)
    assert viewer.loadConfiguration(filepath=str(premade))
    assert asked == [(['GFP', 'combined', 'phase'], 'phase')]
    written = json.loads(next(saved.iterdir()).read_text())
    assert written['channel'] == 'combined'
    # The mapped channel of the template's base becomes an overlay
    assert written['overlay_channels'] == ['phase', 'GFP']
    assert written['visible_channels'] == ['combined', 'phase']
    assert viewer.user_ch_name == 'combined'
    assert list(viewer.allOverlayToolbuttons)[0] == 'combined'
    # Masks and contours are drawn above the overlay channel layers
    overlayZ = max(
        items[0].zValue() for items in viewer.overlayLayersItems.values()
    )
    assert viewer.labelsLayerImg1.zValue() > overlayZ
    assert viewer.ax1_contoursImageItem.zValue() > overlayZ
    assert viewer.ax1_contoursImageItem.zValue() < 1  # below text IDs


def test_set_configuration_base_channel():
    configuration = {
        'channel': 'GFP', 'overlay_channels': ['mCherry', 'BF'],
        'visible_channels': ['GFP', 'mCherry']
    }
    MeasurementsViewerWin.setConfigurationBaseChannel(configuration, 'BF')
    assert configuration == {
        'channel': 'BF', 'overlay_channels': ['GFP', 'mCherry'],
        'visible_channels': ['BF', 'GFP', 'mCherry']
    }
    MeasurementsViewerWin.setConfigurationBaseChannel(configuration, 'BF')
    assert configuration['overlay_channels'] == ['GFP', 'mCherry']


def test_premade_mapping_cancel_writes_nothing(viewer, tmp_path, monkeypatch):
    experiment = tmp_path / 'exp'
    paths = make_experiment(experiment, positions=1)
    add_channel(paths[0], 'GFP')
    saved = tmp_path / 'saved'
    saved.mkdir()
    premade = tmp_path / 'template.json'
    premade.write_text(json.dumps(premade_config()))
    patch_premade_dialogs(viewer, monkeypatch, experiment, saved, None)
    assert not viewer.loadConfiguration(filepath=str(premade))
    assert list(saved.iterdir()) == []
    assert not viewer.isDataLoaded


def test_premade_needs_enough_channels(viewer, tmp_path, monkeypatch):
    experiment = tmp_path / 'exp'
    make_experiment(experiment, positions=1)
    saved = tmp_path / 'saved'
    saved.mkdir()
    premade = tmp_path / 'template.json'
    premade.write_text(json.dumps(premade_config()))
    patch_premade_dialogs(
        viewer, monkeypatch, experiment, saved, {'MAIN': 'phase', 'OTHER': 'x'}
    )
    assert not viewer.loadConfiguration(filepath=str(premade))
    assert 'channels in common' in str(viewer.configurationError.call_args)


def test_shipped_premade_configurations_are_valid():
    folder = measurements_viewer.PREMADE_CONFIGURATIONS_PATH
    files = [f for f in os.listdir(folder) if f.endswith('.json')]
    assert files
    for filename in files:
        with open(os.path.join(folder, filename), encoding='utf-8') as file:
            template = json.load(file)
        placeholders = template['channel_placeholders']
        MeasurementsViewerWin.validateChannelPlaceholders(placeholders)
        # Dashed channel names (e.g. 'mKO2-hCdt1') are valid in equations
        for prefix in ('ch', 'ch-x'):
            mapping = {
                name: f'{prefix}{i}' for i, name in enumerate(placeholders)
            }
            result = MeasurementsViewerWin.resolvePremadeTemplate(
                template, mapping
            )
            if not result['channel']:
                # Asked when instantiating: e.g. a phase contrast channel
                MeasurementsViewerWin.setConfigurationBaseChannel(
                    result, 'phase'
                )
            result['segm_endname'] = 'segm'
            result['images_paths'] = ['unused']
            MeasurementsViewerWin.validateConfiguration(result)


def test_channel_mapping_dialog(app, monkeypatch):
    dialog = measurements_viewer.ChannelPlaceholderMappingDialog(
        {'RED': 'red channel', 'GREEN': ''}, ['GFP', 'mCherry_red', 'phase']
    )
    # Matching names are preselected, the rest get the first unused channel
    assert dialog.currentMapping() == {'RED': 'mCherry_red', 'GREEN': 'GFP'}
    warnings = []
    monkeypatch.setattr(
        measurements_viewer.widgets, 'myMessageBox',
        lambda **kwargs: SimpleNamespace(warning=lambda *a: warnings.append(a))
    )
    dialog.comboboxes['GREEN'].setCurrentText('mCherry_red')
    dialog.ok_cb()
    assert warnings and dialog.cancel and dialog.mapping is None
    dialog.comboboxes['GREEN'].setCurrentText('phase')
    dialog.ok_cb()
    assert not dialog.cancel
    assert dialog.mapping == {'RED': 'mCherry_red', 'GREEN': 'phase'}


def test_channel_mapping_dialog_rejects_invalid_equation_channels(
        app, monkeypatch
    ):
    with open(os.path.join(
            measurements_viewer.PREMADE_CONFIGURATIONS_PATH,
            'fucci_cell_cycle.json'), encoding='utf-8') as file:
        template = json.load(file)
    dialog = measurements_viewer.ChannelPlaceholderMappingDialog(
        template['channel_placeholders'], ['488', 'mKO2-hCdt1', 'phase'],
        inputs=template['input_placeholders'],
        validate=partial(MeasurementsViewerWin.resolvePremadeTemplate, template)
    )
    warnings = []
    monkeypatch.setattr(
        measurements_viewer.widgets, 'myMessageBox',
        lambda **kwargs: SimpleNamespace(warning=lambda *a: warnings.append(a))
    )
    dialog.comboboxes['GEMININ'].setCurrentText('488')
    dialog.comboboxes['CDT1'].setCurrentText('mKO2-hCdt1')
    dialog.ok_cb()
    assert dialog.cancel and dialog.mapping is None
    assert '488_ramp' in warnings[0][2]
    # The selection is kept so that the user can fix it
    assert dialog.comboboxes['CDT1'].currentText() == 'mKO2-hCdt1'
    dialog.comboboxes['GEMININ'].setCurrentText('phase')
    dialog.ok_cb()
    assert not dialog.cancel
    assert dialog.mapping == {'GEMININ': 'phase', 'CDT1': 'mKO2-hCdt1'}


# --- Conditional formatting ---------------------------------------------------

@pytest.mark.parametrize('bad', [
    [], {'a': {}}, {'': []}, {'a': [{'condition': 'x > 1'}]},
    {'a': [{'condition': ' ', 'color': '#ff0000'}]},
    {'a': [{'condition': 'x > 1', 'color': 'red'}]},
    {'a': [{'condition': 'x > 1', 'color': 'lighter:50'}]},
    {'a': [{'condition': 'x > 1', 'color': 'brighter'}]},
    {'a': [{'condition': 'x > 1', 'color': '#ff0000', 'extra': 1}]},
])
def test_invalid_conditional_formatting_is_rejected(bad):
    with pytest.raises(ValueError):
        MeasurementsViewerWin.validateConditionalFormatting(bad)


def test_evaluate_conditional_formatting_first_true_rule_wins():
    df_ID = pd.DataFrame(
        {'ramp': [1, 1, -1, np.nan, 0], 'value': [1., 5., 3., 2., 9.]},
        index=pd.Index([0, 1, 2, 3, 5], name='frame_i')
    )
    rules = [
        {'condition': 'value == MAX(value)', 'color': '#000000'},
        {'condition': 'ramp == 1', 'color': '#00ff00'},
        {'condition': 'ramp == -1', 'color': '#ff0000'},
    ]
    matched = MeasurementsViewerWin.evaluateConditionalFormatting(
        df_ID, rules, ID=1, SizeT=6
    )
    assert matched.tolist() == [1, 1, 2, -1, 0]
    # Scalars apply to the whole cell; NaN counts as False; SizeT is known
    scalar = [{'condition': 'any(ramp == -1) & (SizeT == 6)', 'color': '#000000'}]
    assert MeasurementsViewerWin.evaluateConditionalFormatting(
        df_ID, scalar, SizeT=6
    ).tolist() == [0] * 5
    nan = [{'condition': 'ramp', 'color': '#000000'}]
    assert MeasurementsViewerWin.evaluateConditionalFormatting(
        df_ID.iloc[3:4], nan
    ).tolist() == [-1]
    for condition in ('value', 'missing > 0', 'value >'):
        with pytest.raises(Exception):
            MeasurementsViewerWin.evaluateConditionalFormatting(
                df_ID, [{'condition': condition, 'color': '#000000'}]
            )


def test_conditional_formatting_colors_points_and_segments(viewer, tmp_path):
    load_clean_workflow(viewer, tmp_path)
    viewer.acdc_df = pd.DataFrame({
        'frame_i': [0, 1, 2, 3, 4], 'Cell_ID': 1,
        'final': [1., 2., 3., 4., np.nan], 'ramp': [1, 1, 0, 0, -1],
    }).set_index(['frame_i', 'Cell_ID'])
    viewer.selectedID = 1
    rules = [
        {'condition': 'ramp == 1', 'color': '#00ff00'},
        {'condition': 'ramp == -1', 'color': '#ff0000'},
    ]
    viewer.setConditionalFormatting('final', rules)
    items = viewer.measurementsPlot.listDataItems()
    main = next(item for item in items if item.name() == 'final')
    brushes = [b.color().name() for b in main.scatter.data['brush']]
    default = pg_color_name(viewer.getSignalColor('final'))
    assert brushes == ['#00ff00', '#00ff00', default, default, '#ff0000']
    # The main curve keeps the unformatted segment 3->4 (frame numbers);
    # nothing connects to the missing last value
    assert main.opts['connect'].tolist() == [False, False, True, False, False]
    segments = [
        item for item in items if isinstance(item.opts['connect'], str)
        and item.opts['connect'] == 'pairs'
    ]
    assert len(segments) == 1  # the falling ramp has no finite segment
    assert segments[0].xData.tolist() == [1, 2, 2, 3]
    tip = viewer.measurementPointTip(1, 1, main.scatter.data['data'][0])
    assert 'Formatting: ramp == 1' in tip
    assert viewer.configuration()['conditional_formatting'] == {
        'final': rules
    }
    assert viewer.hasUnsavedConfiguration()
    # Errors leave the curve unformatted and are shown in the status bar
    viewer.setConditionalFormatting('final', [
        {'condition': 'missing == 1', 'color': '#00ff00'}
    ])
    main = next(
        item for item in viewer.measurementsPlot.listDataItems()
        if item.name() == 'final'
    )
    assert {b.color().name() for b in main.scatter.data['brush']} == {default}
    assert 'Conditional formatting of "final" failed' in (
        viewer.statusBar().currentMessage()
    )
    viewer.setConditionalFormatting('final', [])
    assert 'final' not in viewer.configuration()['conditional_formatting']


def pg_color_name(color):
    import pyqtgraph as pg
    return pg.mkColor(color).name()


def test_conditional_formatting_is_saved_and_loaded(viewer, tmp_path, monkeypatch):
    _, path = load_clean_workflow(viewer, tmp_path)
    rules = [{'condition': 'area > 20', 'color': '#123456'}]
    viewer.setConditionalFormatting('final', rules)
    assert viewer.saveConfiguration()
    assert json.loads(path.read_text())['conditional_formatting'] == {
        'final': rules
    }
    viewer.conditionalFormatting = {}
    # Discard the unsaved changes when reloading the saved configuration.
    monkeypatch.setattr(viewer, 'askSaveBeforeLoading', Mock(return_value=True))
    assert viewer.loadConfiguration(filepath=str(path))
    assert viewer.conditionalFormatting == {'final': rules}
    assert not viewer.hasUnsavedConfiguration()


def test_custom_signal_colors_are_saved_and_loaded(
        viewer, tmp_path, monkeypatch):
    _, path = load_clean_workflow(viewer, tmp_path)
    monkeypatch.setattr(
        measurements_viewer.QColorDialog, 'getColor',
        Mock(return_value=measurements_viewer.QColor('#123456'))
    )
    viewer.changeSignalColor('final')
    assert viewer.configuration()['measurements_colors'] == {
        'final': '#123456'
    }
    assert viewer.hasUnsavedConfiguration()
    assert viewer.saveConfiguration()
    assert json.loads(path.read_text())['measurements_colors'] == {
        'final': '#123456'
    }
    viewer.measurementsColors = {}
    monkeypatch.setattr(viewer, 'askSaveBeforeLoading', Mock(return_value=True))
    assert viewer.loadConfiguration(filepath=str(path))
    assert pg_color_name(viewer.getSignalColor('final')) == '#123456'
    assert not viewer.hasUnsavedConfiguration()


@pytest.mark.parametrize('bad', [
    [], {'': '#00ff00'}, {'a': 'green'}, {'a': '#00ff0'}, {'a': [0, 255, 0]},
])
def test_invalid_signal_colors_are_rejected(viewer, tmp_path, bad):
    load_clean_workflow(viewer, tmp_path)
    configuration = viewer.configuration()
    configuration['measurements_colors'] = bad
    with pytest.raises(ValueError):
        MeasurementsViewerWin.validateConfiguration(configuration)


def test_conditional_formatting_dialog(app, monkeypatch):
    warnings = []
    monkeypatch.setattr(
        measurements_viewer.widgets, 'myMessageBox',
        lambda **kwargs: SimpleNamespace(warning=lambda *a: warnings.append(a))
    )

    def validate(condition):
        if 'bad' in condition:
            raise ValueError('invalid')

    dialog = measurements_viewer.ConditionalFormattingDialog(
        'x', [{'condition': 'a == 1', 'color': '#00ff00'}],
        validateCondition=validate, editCondition=lambda text: 'b == 2'
    )
    dialog.addRule()
    assert len(dialog.rows) == 2
    dialog.ok_cb()  # the new rule is empty
    assert warnings and dialog.cancel
    dialog.rows[1]['edit'].setText('bad')
    dialog.ok_cb()
    assert len(warnings) == 2 and dialog.cancel
    dialog.editRule(1)
    dialog.moveRule(1, -1)
    assert [rule['condition'] for rule in dialog.rules] == ['b == 2', 'a == 1']
    dialog.removeRule(1)
    dialog.ok_cb()
    assert not dialog.cancel
    assert dialog.rules == [{'condition': 'b == 2', 'color': 'lighter'}]


def test_fucci_template_formats_reporters_by_ramp():
    with open(os.path.join(
            measurements_viewer.PREMADE_CONFIGURATIONS_PATH,
            'fucci_cell_cycle.json'), encoding='utf-8') as file:
        template = json.load(file)
    result = MeasurementsViewerWin.resolvePremadeTemplate(
        template, {'GEMININ': 'GFP', 'CDT1': 'mCherry'}
    )
    formatting = result['conditional_formatting']
    assert set(formatting) == {'GFP_fucci', 'mCherry_fucci'}
    assert formatting['mCherry_fucci'] == [
        {'condition': 'mCherry_ramp == 1', 'color': 'lighter'},
        {'condition': 'mCherry_ramp == -1', 'color': 'darker'},
    ]
    # Geminin (GFP) reporter plotted in green, Cdt1 (mCherry) in red
    assert result['measurements_colors'] == {
        'GFP_fucci': '#2ecc40', 'mCherry_fucci': '#e8384f'
    }


def test_formatting_condition_editor(app):
    acdc_df = pd.DataFrame({
        'frame_i': [0, 1, 0], 'Cell_ID': [1, 1, 2], 'area': [12., 30., 24.]
    }).set_index(['frame_i', 'Cell_ID'])
    dialog = measurements_viewer._FormattingConditionDialog(
        ['phase'], False, False, acdc_df=acdc_df, SizeT=2,
        condition='area > MEAN(area)'
    )
    assert dialog.windowTitle() == 'Conditional formatting condition'
    # Conditions see only the selected cell: no other-cell inputs
    assert not hasattr(dialog, 'otherMeasurementsTreeWidget')
    assert not any(
        name.endswith('_other') for name in dialog.variableNames
    )
    for condition in ('area_other > 0', 'MAX_OTHER(area) > 0'):
        success, message = dialog.validateEquation(condition)
        assert not success
        assert 'not available in conditional formatting' in message
    dialog.ok_cb()
    assert not dialog.cancel and dialog.equation == 'area > MEAN(area)'


def test_formatting_condition_editor_missing_values_are_false(app):
    """The condition editor accepts conditions with missing values like
    the viewer does (missing values are False; regression)."""
    acdc_df = pd.DataFrame({
        'frame_i': [0, 1, 2, 3, 0, 1], 'Cell_ID': [1, 1, 1, 1, 2, 2],
        'flag': [0., 0., 1., 1., 1., np.nan],
    }).set_index(['frame_i', 'Cell_ID'])
    dialog = measurements_viewer._FormattingConditionDialog(
        ['phase'], False, False, acdc_df=acdc_df, SizeT=4,
        condition='SHIFT(flag, 1)'
    )
    dialog.test_cb()
    assert dialog.okButton.isEnabled()
    mask = dialog.evaluateEquation(
        dialog.getTestData(), 'SHIFT(flag, 1)', enforce_dtype_output=bool
    )
    cell = acdc_df.loc[pd.IndexSlice[:, 1], :].droplevel('Cell_ID')
    expected = MeasurementsViewerWin.evaluateConditionalFormatting(
        cell, [{'condition': 'SHIFT(flag, 1)'}], ID=1, SizeT=4
    ) == 0
    assert mask.tolist()[:4] == expected.tolist()
    assert mask.tolist() == [False, False, False, True, False, True]
    # Values other than 0/1 are still rejected
    dialog.formulaEditWidget.setText('flag + 2')
    dialog.test_cb()
    assert not dialog.okButton.isEnabled()
    assert '0/1' in dialog.statusLabel.text()
    dialog.deleteLater()


def test_relative_formatting_colors(app, viewer, tmp_path):
    base = measurements_viewer.QColor('#ff0000')
    lighter = measurements_viewer.formatting_color('lighter', base)
    darker = measurements_viewer.formatting_color('darker', base)
    assert lighter.lightness() > base.lightness() > darker.lightness()
    assert lighter.hue() == darker.hue() == base.hue()
    assert measurements_viewer.formatting_color('darker:300', base).value() < (
        darker.value()
    )
    assert measurements_viewer.formatting_color('#123456', base).name() == (
        '#123456'
    )
    MeasurementsViewerWin.validateConditionalFormatting({'a': [
        {'condition': 'x', 'color': c}
        for c in ('lighter', 'darker', 'darker:250', '#00ff00')
    ]})
    # Plotted shades follow the column color
    load_clean_workflow(viewer, tmp_path)
    viewer.acdc_df = pd.DataFrame({
        'frame_i': [0, 1, 2], 'Cell_ID': 1,
        'final': [1., 2., 3.], 'ramp': [1, -1, 0],
    }).set_index(['frame_i', 'Cell_ID'])
    viewer.selectedID = 1
    viewer.measurementsColors['final'] = base
    viewer.setConditionalFormatting('final', [
        {'condition': 'ramp == 1', 'color': 'lighter'},
        {'condition': 'ramp == -1', 'color': 'darker'},
    ])
    items = viewer.measurementsPlot.listDataItems()
    main = next(item for item in items if item.name() == 'final')
    assert [b.color().name() for b in main.scatter.data['brush']] == [
        lighter.name(), darker.name(), '#ff0000'
    ]
    pens = {
        item.opts['pen'].color().name() for item in items
        if isinstance(item.opts['connect'], str)
        and item.opts['connect'] == 'pairs'
    }
    assert pens == {lighter.name(), darker.name()}


def test_formatting_dialog_keeps_relative_modes(app):
    dialog = measurements_viewer.ConditionalFormattingDialog('x', [
        {'condition': 'a', 'color': 'darker:250'},
        {'condition': 'b', 'color': '#00ff00'},
    ])
    assert dialog.rows[0]['mode'].currentText() == 'Darker shade'
    assert not dialog.rows[0]['color'].isEnabled()
    assert dialog.rows[1]['color'].isEnabled()
    dialog.syncRules()
    assert [rule['color'] for rule in dialog.rules] == ['darker:250', '#00ff00']
    dialog.rows[1]['mode'].setCurrentIndex(0)
    dialog.syncRules()
    assert dialog.rules[1]['color'] == 'lighter'


# --- Premade input placeholders ----------------------------------------------
def premade_inputs():
    return {
        'BG': {'type': 'float', 'default': 0.5, 'description': 'bg', 'info': 'i'},
        'N': {'type': 'int', 'default': 3},
        'LABEL': {'type': 'str', 'default': 'x'},
    }


def test_resolve_premade_template_with_inputs():
    template = dict(
        premade_config(), input_placeholders=premade_inputs(),
        measurement_processing=[{
            'order': 1, 'columns': ['{MAIN}_mean'],
            'recipe': [{'method': 'Smooth measurement', 'kwargs': {
                'window': '{N}', 'bg': '{BG}'
            }}],
            'output_columns': ['{MAIN}_{LABEL}'],
        }],
        combined_measurements=[
            {'order': 0, 'equations': {'{OTHER}_shift': 'area - {BG}'}}
        ],
    )
    original = copy.deepcopy(template)
    result = MeasurementsViewerWin.resolvePremadeTemplate(
        template, {'MAIN': 'GFP', 'OTHER': 'RFP'}, {'BG': 2.0}
    )
    assert template == original
    assert 'input_placeholders' not in result
    assert 'channel_placeholders' not in result
    kwargs = result['measurement_processing'][0]['recipe'][0]['kwargs']
    # Whole-token strings become typed values, defaults fill missing inputs
    assert kwargs == {'window': 3, 'bg': 2.0}
    assert result['measurement_processing'][0]['output_columns'] == ['GFP_x']
    assert result['combined_measurements'][0]['equations'] == {
        'RFP_shift': 'area - 2.0'
    }


@pytest.mark.parametrize('bad', [
    [], {'MAIN': {'type': 'float', 'default': 0.0}},
    {'A B': {'type': 'float', 'default': 0.0}},
    {'A': {'type': 'float'}}, {'A': {'type': 'complex', 'default': 0}},
    {'A': {'type': 'int', 'default': 0.5}}, {'A': {'type': 'float', 'default': True}},
    {'A': {'type': 'str', 'default': 'x', 'extra': 1}},
    {'A': {'type': 'str', 'default': 'x', 'info': 1}},
])
def test_invalid_input_placeholders_are_rejected(bad):
    with pytest.raises(ValueError):
        MeasurementsViewerWin.validateInputPlaceholders(bad, ['MAIN'])


def test_setup_dialog_asks_for_inputs(app, monkeypatch):
    dialog = measurements_viewer.ChannelPlaceholderMappingDialog(
        {'MAIN': ''}, ['GFP'], inputs=premade_inputs()
    )
    assert dialog.inputLineEdits['BG'].text() == '0.5'
    warnings = []
    monkeypatch.setattr(
        measurements_viewer.widgets, 'myMessageBox',
        lambda **kwargs: SimpleNamespace(warning=lambda *a: warnings.append(a))
    )
    dialog.inputLineEdits['N'].setText('2.5')
    dialog.ok_cb()
    assert warnings and dialog.cancel
    dialog.inputLineEdits['N'].setText(' 4 ')
    dialog.inputLineEdits['BG'].setText('1e3')
    dialog.ok_cb()
    assert not dialog.cancel
    assert dialog.mapping == {'MAIN': 'GFP'}
    assert dialog.values == {'BG': 1000.0, 'N': 4, 'LABEL': 'x'}


def test_premade_input_values_are_written(viewer, tmp_path, monkeypatch):
    experiment = tmp_path / 'exp'
    paths = make_experiment(experiment)
    for images in paths:
        add_channel(images, 'GFP')
    saved = tmp_path / 'saved'
    saved.mkdir()
    premade = tmp_path / 'template.json'
    template = premade_config()
    template['input_placeholders'] = premade_inputs()
    template['combined_measurements'][0]['equations'] = {
        '{OTHER}_double': 'area * {N}'
    }
    premade.write_text(json.dumps(template))
    patch_premade_dialogs(
        viewer, monkeypatch, experiment, saved,
        {'MAIN': 'phase', 'OTHER': 'GFP'}
    )
    monkeypatch.setattr(
        viewer, 'selectChannelMapping',
        lambda placeholders, channels, inputs=None, validate=None, **kw: (
            {'MAIN': 'phase', 'OTHER': 'GFP'}, {'BG': 0.5, 'N': 5, 'LABEL': 'x'}
        )
    )
    assert viewer.loadConfiguration(filepath=str(premade))
    written = json.loads(next(saved.glob('*.json')).read_text())
    assert 'input_placeholders' not in written
    assert written['premade']['input_values'] == {'BG': 0.5, 'N': 5, 'LABEL': 'x'}
    assert written['combined_measurements'][0]['equations'] == {
        'GFP_double': 'area * 5'
    }


# --- Premade background values ------------------------------------------------
def fucci_template():
    with open(os.path.join(
            measurements_viewer.PREMADE_CONFIGURATIONS_PATH,
            'fucci_cell_cycle.json'), encoding='utf-8') as file:
        return json.load(file)


def background_summary(**overrides):
    """Summary table as saved by the "Compute background intensity"
    utility (`background_values.normalization_constants` + settings)."""
    rows = [
        {'channel': 'GFP', 'constant_background': 36.0,
         'constant_min': 10.5, 'constant_max': 210.25},
        {'channel': 'mCherry', 'constant_background': 142.0,
         'constant_min': 3.0, 'constant_max': 98.0},
    ]
    df = pd.DataFrame(rows)
    df.insert(1, 'background_statistic', 'median')
    df.insert(2, 'low_percentile', 5.0)
    df.insert(3, 'high_percentile', 95.0)
    df['num_cell_values'] = 100
    df['num_background_values'] = 50
    for col, values in overrides.items():
        df[col] = values
    return df


def fucci_setup_dialog(**kwargs):
    template = fucci_template()
    spec = MeasurementsViewerWin.validateBackgroundValuesSpec(
        template['background_values'], template['channel_placeholders'],
        template['input_placeholders']
    )
    dialog = measurements_viewer.ChannelPlaceholderMappingDialog(
        template['channel_placeholders'], ['GFP', 'mCherry', 'phase'],
        inputs=template['input_placeholders'], backgroundValues=spec,
        **kwargs
    )
    dialog.comboboxes['GEMININ'].setCurrentText('GFP')
    dialog.comboboxes['CDT1'].setCurrentText('mCherry')
    return dialog


def capture_warnings(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        measurements_viewer.widgets, 'myMessageBox',
        lambda **kwargs: SimpleNamespace(warning=lambda *a: warnings.append(a))
    )
    return warnings


def test_fucci_background_values_spec():
    template = fucci_template()
    spec = MeasurementsViewerWin.validateBackgroundValuesSpec(
        template['background_values'], template['channel_placeholders'],
        template['input_placeholders']
    )
    assert spec['statistic'] == 'median'
    assert (spec['low_percentile'], spec['high_percentile']) == (5.0, 95.0)
    # Every normalization input of the template is linked
    assert set(spec['inputs']) == set(template['input_placeholders'])
    # Defaults of the optional settings
    minimal = MeasurementsViewerWin.validateBackgroundValuesSpec(
        {'inputs': {'GEMININ_MIN': {
            'channel': 'GEMININ', 'value': 'constant_min'
        }}},
        template['channel_placeholders'], template['input_placeholders']
    )
    assert minimal['statistic'] == 'median'
    assert minimal['exclude_zero_pixels'] is True
    # Setup-only section: not written to the configuration
    result = MeasurementsViewerWin.resolvePremadeTemplate(
        template, {'GEMININ': 'GFP', 'CDT1': 'mCherry'}
    )
    assert 'background_values' not in result


@pytest.mark.parametrize('bad', [
    [], {}, {'inputs': {}}, {'inputs': {'BG': {'channel': 'MAIN',
                                                'value': 'constant_min'}},
                             'extra': 1},
    {'statistic': 'max', 'inputs': {'BG': {'channel': 'MAIN',
                                           'value': 'constant_min'}}},
    {'low_percentile': 95, 'high_percentile': 5,
     'inputs': {'BG': {'channel': 'MAIN', 'value': 'constant_min'}}},
    {'low_percentile': True,
     'inputs': {'BG': {'channel': 'MAIN', 'value': 'constant_min'}}},
    {'exclude_zero_pixels': 1,
     'inputs': {'BG': {'channel': 'MAIN', 'value': 'constant_min'}}},
    # Unknown channel placeholder, unknown value, non-float input
    {'inputs': {'BG': {'channel': 'OTHER', 'value': 'constant_min'}}},
    {'inputs': {'BG': {'channel': 'MAIN', 'value': 'num_cell_values'}}},
    # The per-frame values are disabled for now
    {'inputs': {'BG': {'channel': 'MAIN', 'value': 'per_frame_min'}}},
    {'inputs': {'N': {'channel': 'MAIN', 'value': 'constant_min'}}},
    {'inputs': {'BG': {'channel': 'MAIN'}}},
])
def test_invalid_background_values_spec_is_rejected(bad):
    with pytest.raises(ValueError):
        MeasurementsViewerWin.validateBackgroundValuesSpec(
            bad, {'MAIN': ''}, premade_inputs()
        )


def test_setup_dialog_fills_values_from_summary(app, monkeypatch):
    dialog = fucci_setup_dialog()
    warnings = capture_warnings(monkeypatch)
    # Missing values (e.g. min/max without corrected acdc_output tables)
    # are left unchanged and reported
    summary = background_summary(constant_max=[210.25, np.nan])
    dialog.inputLineEdits['CDT1_MAX'].setText('7')
    filled = dialog.fillBackgroundValues(summary, 'test')
    assert filled == {
        'GEMININ_BACKGROUND': 36.0, 'GEMININ_MIN': 10.5,
        'GEMININ_MAX': 210.25, 'CDT1_BACKGROUND': 142.0, 'CDT1_MIN': 3.0,
    }
    assert dialog.inputLineEdits['GEMININ_MAX'].text() == '210.25'
    assert dialog.inputLineEdits['CDT1_MAX'].text() == '7'
    assert len(warnings) == 1 and 'CDT1_MAX' in warnings[0][2]
    assert 'Filled in 5 of 6' in dialog.backgroundSourceLabel.text()
    dialog.ok_cb()
    assert not dialog.cancel
    assert dialog.values['GEMININ_MIN'] == 10.5
    assert dialog.values['CDT1_MAX'] == 7.0


def test_setup_dialog_asks_table_row_and_reports_other_settings(
        app, monkeypatch
    ):
    dialog = fucci_setup_dialog()
    warnings = capture_warnings(monkeypatch)
    asked = []

    def askTableChannel(placeholder, channel, tableChannels):
        asked.append((placeholder, channel, tableChannels))
        return 'GFP2' if placeholder == 'GEMININ' else None

    monkeypatch.setattr(dialog, 'askTableChannel', askTableChannel)
    # Another experiment with other channel names and percentiles
    summary = background_summary(
        channel=['GFP2', 'mCherry2'], low_percentile=10.0,
        high_percentile=90.0
    )
    filled = dialog.fillBackgroundValues(summary, 'test')
    assert asked == [
        ('GEMININ', 'GFP', ['GFP2', 'mCherry2']),
        ('CDT1', 'mCherry', ['GFP2', 'mCherry2']),
    ]
    assert set(filled) == {'GEMININ_BACKGROUND', 'GEMININ_MIN', 'GEMININ_MAX'}
    assert dialog.inputLineEdits['CDT1_MIN'].text() == '0.0'
    text = warnings[0][2]
    assert 'CDT1_MIN' in text and '10th/90th percentiles' in text


def test_setup_dialog_loads_summary_table(app, monkeypatch, tmp_path):
    dialog = fucci_setup_dialog(tableStartDir=str(tmp_path))
    warnings = capture_warnings(monkeypatch)
    # Channels that CSV reading would turn into a number stay text
    table = tmp_path / 'background_values_summary.csv'
    background_summary(channel=['488', 'mCherry']).to_csv(table, index=False)
    monkeypatch.setattr(dialog, 'askTableChannel', lambda *args: '488')
    started = []

    def getOpenFileName(parent, caption, folder, filters):
        started.append(folder)
        return str(table), ''

    monkeypatch.setattr(
        measurements_viewer.apps_measurements_viewer.QFileDialog,
        'getOpenFileName', getOpenFileName
    )
    dialog.loadBackgroundTableButton.click()
    assert started == [str(tmp_path)]
    assert not warnings
    assert dialog.inputLineEdits['GEMININ_MAX'].text() == '210.25'
    assert dialog.inputLineEdits['CDT1_BACKGROUND'].text() == '142'
    # A per-Position background table is refused with a hint
    other = tmp_path / 'exp_background_values.csv'
    pd.DataFrame({'frame_i': [0], 'num_objects': [1]}).to_csv(
        other, index=False
    )
    table = other
    dialog.loadBackgroundTableButton.click()
    assert 'single Position' in warnings[0][2]


def test_setup_dialog_computes_background_values(app, monkeypatch):
    calls = []
    results = [None, pd.DataFrame(), background_summary()]

    def compute(channels, settings, parent=None):
        calls.append((channels, settings['statistic'], parent))
        return results.pop(0)

    dialog = fucci_setup_dialog(computeBackgroundValues=compute)
    warnings = capture_warnings(monkeypatch)
    dialog.comboboxes['GEMININ'].setCurrentText('mCherry')
    dialog.comboboxes['CDT1'].setCurrentText('GFP')
    dialog.computeBackgroundButton.click()   # cancelled: nothing happens
    assert not warnings and dialog.inputLineEdits['CDT1_MIN'].text() == '0.0'
    dialog.computeBackgroundButton.click()   # no Position processed
    assert len(warnings) == 1
    dialog.computeBackgroundButton.click()
    assert calls == [(['mCherry', 'GFP'], 'median', dialog)] * 3
    # Rows follow the selected channels
    assert dialog.inputLineEdits['CDT1_MIN'].text() == '10.5'
    assert dialog.inputLineEdits['GEMININ_MIN'].text() == '3'


def test_setup_dialog_refills_when_linked_channel_changes(app, monkeypatch):
    dialog = fucci_setup_dialog()
    warnings = capture_warnings(monkeypatch)
    dialog.fillBackgroundValues(background_summary(), 'test')
    assert dialog.inputLineEdits['GEMININ_MIN'].text() == '10.5'
    # The values belonged to GFP: switching the reporters refills them
    dialog.comboboxes['GEMININ'].setCurrentText('mCherry')
    dialog.comboboxes['CDT1'].setCurrentText('GFP')
    assert dialog.inputLineEdits['GEMININ_MIN'].text() == '3'
    assert dialog.inputLineEdits['CDT1_MAX'].text() == '210.25'
    assert 'CDT1 = GFP' in dialog.backgroundSourceLabel.text()
    assert not warnings


def test_setup_dialog_reports_other_zero_pixel_setting(app, monkeypatch):
    dialog = fucci_setup_dialog()
    warnings = capture_warnings(monkeypatch)
    dialog.fillBackgroundValues(
        background_summary(exclude_zero_pixels=['False', 'True']), 'test'
    )
    assert len(warnings) == 1
    text = warnings[0][2]
    assert 'pixels with intensity 0 included' in text
    assert 'mCherry' not in text


def test_setup_dialog_is_disabled_while_computing(app, monkeypatch):
    states = []

    def compute(channels, settings, parent=None):
        states.append(parent.isEnabled())
        parent.close()   # e.g. cancelled while the experiments are selected
        return background_summary()

    dialog = fucci_setup_dialog(computeBackgroundValues=compute)
    dialog.computeBackgroundButton.click()
    assert states == [False]
    assert dialog.isEnabled()
    # Closed meanwhile: nothing is filled
    assert dialog.inputLineEdits['GEMININ_MIN'].text() == '0.0'
    assert dialog.backgroundFill is None


def test_fucci_hints_preselect_reporter_channels_for_calculation(app):
    template = fucci_template()
    spec = MeasurementsViewerWin.validateBackgroundValuesSpec(
        template['background_values'], template['channel_placeholders'],
        template['input_placeholders']
    )
    hints = MeasurementsViewerWin.validateChannelHints(
        template['channel_hints'], template['channel_placeholders']
    )
    calls = []
    dialog = measurements_viewer.ChannelPlaceholderMappingDialog(
        template['channel_placeholders'],
        ['BF', 'GFP', 'combined', 'combined_preprocessed', 'mCherry'],
        inputs=template['input_placeholders'], backgroundValues=spec,
        channelHints=hints,
        computeBackgroundValues=lambda channels, *a, **k: calls.append(
            channels
        )
    )
    assert dialog.currentMapping() == {'GEMININ': 'GFP', 'CDT1': 'mCherry'}
    dialog.computeBackgroundButton.click()
    assert calls == [['GFP', 'mCherry']]
    # Placeholder names in channel names win over the hints
    guess = measurements_viewer.ChannelPlaceholderMappingDialog.guessChannels
    assert guess(
        template['channel_placeholders'],
        ['GFP', 'mAG1-hGeminin', 'mKO2-hCdt1'], hints
    ) == {'GEMININ': 'mAG1-hGeminin', 'CDT1': 'mKO2-hCdt1'}
    assert guess(
        template['channel_placeholders'], ['mAG1', 'mKO2', 'phase'], hints
    ) == {'GEMININ': 'mAG1', 'CDT1': 'mKO2'}
    # Without hints: first unused channels, as before
    assert guess({'A': '', 'B': ''}, ['x', 'y']) == {'A': 'x', 'B': 'y'}
    result = MeasurementsViewerWin.resolvePremadeTemplate(
        template, {'GEMININ': 'GFP', 'CDT1': 'mCherry'}
    )
    assert 'channel_hints' not in result


@pytest.mark.parametrize('bad', [
    [], {'OTHER': ['gfp']}, {'MAIN': 'gfp'}, {'MAIN': ['']}, {'MAIN': [1]},
])
def test_invalid_channel_hints_are_rejected(bad):
    with pytest.raises(ValueError):
        MeasurementsViewerWin.validateChannelHints(bad, {'MAIN': ''})


def test_setup_dialog_without_background_values_has_no_buttons(app):
    dialog = measurements_viewer.ChannelPlaceholderMappingDialog(
        {'MAIN': ''}, ['GFP'], inputs=premade_inputs()
    )
    assert dialog.backgroundSourceLabel is None
    assert not hasattr(dialog, 'loadBackgroundTableButton')


def test_compute_background_summary_presets_utility(viewer, monkeypatch):
    from cellacdc.utils import computeBackgroundValues as utilsBackground
    selected, created = [], []
    selections = [None, {'exp': ['Position_1']}]

    def select_positions(parent, experimentPositions):
        selected.append(experimentPositions)
        return selections.pop(0)

    class _FakeUtil:
        def __init__(self, expPaths, app, *args, parent=None,
                     presetParams=None, presetNote=''):
            created.append((expPaths, parent, presetParams))
            self.summaryDf = None

        def exec_(self):
            self.summaryDf = background_summary()

    monkeypatch.setattr(utilsBackground, 'select_positions', select_positions)
    monkeypatch.setattr(
        utilsBackground, 'ComputeBackgroundValuesUtil', _FakeUtil
    )
    settings = MeasurementsViewerWin.validateBackgroundValuesSpec(
        {'statistic': 'mean', 'low_percentile': 2,
         'inputs': {'BG': {'channel': 'MAIN', 'value': 'constant_min'}}},
        {'MAIN': ''}, premade_inputs()
    )
    experiment = {'exp': ['Position_1']}
    assert viewer.computeBackgroundSummary(
        experiment, ['GFP'], settings
    ) is None
    assert not created
    summary = viewer.computeBackgroundSummary(experiment, ['GFP'], settings)
    assert summary['channel'].tolist() == ['GFP', 'mCherry']
    assert selected == [experiment, experiment]
    assert created == [({'exp': ['Position_1']}, viewer, {
        'channels': ['GFP'], 'statistic': 'mean',
        'exclude_zero_pixels': True, 'use_acdc_output': True,
        'low_percentile': 2.0, 'high_percentile': 95.0,
    })]


def test_premade_setup_gets_background_values_and_experiment(
        viewer, tmp_path, monkeypatch
    ):
    experiment = tmp_path / 'exp'
    paths = make_experiment(experiment)
    for images in paths:
        add_channel(images, 'GFP')
    saved = tmp_path / 'saved'
    saved.mkdir()
    premade = tmp_path / 'template.json'
    template = premade_config()
    template['input_placeholders'] = premade_inputs()
    template['background_values'] = {
        'inputs': {'BG': {'channel': 'OTHER', 'value': 'constant_background'}}
    }
    premade.write_text(json.dumps(template))
    patch_premade_dialogs(
        viewer, monkeypatch, experiment, saved,
        {'MAIN': 'phase', 'OTHER': 'GFP'}
    )
    received = []

    def select(placeholders, channels, inputs=None, validate=None, **kwargs):
        received.append(kwargs)
        return {'MAIN': 'phase', 'OTHER': 'GFP'}, {'BG': 36.0}

    monkeypatch.setattr(viewer, 'selectChannelMapping', select)
    assert viewer.loadConfiguration(filepath=str(premade))
    spec = received[0]['backgroundValues']
    assert spec['inputs'] == template['background_values']['inputs']
    positions = received[0]['experimentPositions']
    assert {os.path.normpath(p): sorted(pos) for p, pos in positions.items()} == {
        os.path.normpath(experiment): ['Position_1', 'Position_2']
    }
    written = json.loads(next(saved.glob('*.json')).read_text())
    assert 'background_values' not in written
    assert written['premade']['input_values'] == {'BG': 36.0}


# --- Empty display channel ----------------------------------------------------
def test_fucci_template_has_both_reporters_as_overlays():
    with open(os.path.join(
            measurements_viewer.PREMADE_CONFIGURATIONS_PATH,
            'fucci_cell_cycle.json'), encoding='utf-8') as file:
        template = json.load(file)
    result = MeasurementsViewerWin.resolvePremadeTemplate(
        template, {'GEMININ': 'GFP', 'CDT1': 'mCherry'}
    )
    assert result['channel'] == ''
    assert result['overlay_channels'] == ['GFP', 'mCherry']
    assert result['visible_channels'] == ['GFP', 'mCherry']


def test_premade_without_channel_asks_unmapped_first(viewer, tmp_path, monkeypatch):
    experiment = tmp_path / 'exp'
    paths = make_experiment(experiment, positions=1)
    add_channel(paths[0], 'GFP')
    add_channel(paths[0], 'mCherry', 3)
    saved = tmp_path / 'saved'
    saved.mkdir()
    template = premade_config()
    template.update(
        channel='', overlay_channels=['{MAIN}', '{OTHER}'],
        visible_channels=['{MAIN}', '{OTHER}']
    )
    premade = tmp_path / 'template.json'
    premade.write_text(json.dumps(template))
    patch_premade_dialogs(
        viewer, monkeypatch, experiment, saved,
        {'MAIN': 'mCherry', 'OTHER': 'GFP'}
    )
    asked = []

    def select_base(channels, default=None):
        asked.append((channels, default))
        return channels[0]

    monkeypatch.setattr(viewer, 'selectPremadeBaseChannel', select_base)
    assert viewer.loadConfiguration(filepath=str(premade))
    assert asked == [(['phase', 'GFP', 'mCherry'], None)]
    written = json.loads(next(saved.iterdir()).read_text())
    # The selected channel is saved in the instantiated configuration
    assert written['channel'] == 'phase'
    assert written['overlay_channels'] == ['mCherry', 'GFP']
    assert written['visible_channels'] == ['phase', 'mCherry', 'GFP']
    assert viewer.user_ch_name == 'phase'


def test_configuration_without_channel_asks_when_loading(viewer, tmp_path):
    paths = make_experiment(tmp_path, positions=2)
    for images in paths:
        add_channel(images, 'GFP')
    configuration = config('unused')
    configuration.update(
        images_paths=[str(path) for path in paths], channel='',
        overlay_channels=['GFP'], visible_channels=['GFP']
    )
    MeasurementsViewerWin.validateConfiguration(configuration)
    with pytest.raises(ValueError):
        MeasurementsViewerWin.validateConfiguration(
            dict(configuration, visible_channels=[''])
        )
    viewer.selectItem = Mock(return_value='GFP')
    # The current input (empty channel) follows the stored input_options
    selectedInput = MeasurementsViewerWin.configurationInputOptions(configuration)[-1]
    assert selectedInput['channel'] == ''
    resolved = viewer.resolveInputChannel(selectedInput)
    # Non-overlay channels are offered first
    assert viewer.selectItem.call_args.args[2] == ['phase', 'GFP']
    assert resolved['channel'] == 'GFP'
    assert resolved['overlay_channels'] == []
    assert resolved['visible_channels'] == ['GFP']
    assert selectedInput['channel'] == ''
    viewer.selectItem = Mock(return_value=None)
    assert viewer.resolveInputChannel(selectedInput) is None
    # New experiments ask as well
    viewer.selectItem = Mock(return_value='phase')
    viewer.selectConfigurationOverlayChannels = Mock(return_value=[])
    option = viewer.newExperimentInput(configuration, paths)
    assert option['channel'] == 'phase'
    assert option['overlay_channels'] == ['GFP']
    assert option['visible_channels'] == ['phase', 'GFP']
