import copy
import json
import os
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import tifffile
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication

from cellacdc import apps, measurements_viewer
from cellacdc.timepoint_predictors import TIMEPOINT_PREFIX
from cellacdc.measurements_viewer import MeasurementsViewerWin, VISITED_COLUMN
from cellacdc.timepoint_predictors import TimepointIdentifier


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
        assert option['visible_channels'] == []


def test_configuration_actions_in_menu_and_toolbar(viewer):
    actions = (
        viewer.loadConfigurationAction, viewer.saveConfigurationAction,
        viewer.saveConfigurationAsAction
    )
    for action in actions:
        assert action in viewer.fileToolBar.actions()
        assert action in viewer.fileMenu.actions()
        assert not action.icon().isNull()
    viewer.setEnabledFileToolbar(False)
    assert viewer.loadConfigurationAction.isEnabled()
    assert not viewer.saveConfigurationAction.isEnabled()
    assert not viewer.saveConfigurationAsAction.isEnabled()


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
    assert viewer.loadConfiguration(filepath=str(path))
    assert viewer.user_ch_name == 'GFP'
    assert not viewer.checkedOverlayChannels
    assert viewer.allOverlayToolbuttons['GFP'].isChecked()
    assert viewer.saveConfiguration()
    assert json.loads(path.read_text())['input_options'] == configuration['input_options']


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

    def select(placeholders, channels, inputs=None):
        offered.append(channels)
        return None if mapping is None else (mapping, {})

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
        mapping = {name: f'ch{i}' for i, name in enumerate(placeholders)}
        result = MeasurementsViewerWin.resolvePremadeTemplate(template, mapping)
        if not result['channel']:
            # Asked when instantiating: e.g. a phase contrast channel
            MeasurementsViewerWin.setConfigurationBaseChannel(result, 'phase')
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


def test_conditional_formatting_is_saved_and_loaded(viewer, tmp_path):
    _, path = load_clean_workflow(viewer, tmp_path)
    rules = [{'condition': 'area > 20', 'color': '#123456'}]
    viewer.setConditionalFormatting('final', rules)
    assert viewer.saveConfiguration()
    assert json.loads(path.read_text())['conditional_formatting'] == {
        'final': rules
    }
    viewer.conditionalFormatting = {}
    assert viewer.loadConfiguration(filepath=str(path))
    assert viewer.conditionalFormatting == {'final': rules}
    assert not viewer.hasUnsavedConfiguration()


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


def test_formatting_condition_editor(app):
    acdc_df = pd.DataFrame({
        'frame_i': [0, 1, 0], 'Cell_ID': [1, 1, 2], 'area': [12., 30., 24.]
    }).set_index(['frame_i', 'Cell_ID'])
    dialog = measurements_viewer._FormattingConditionDialog(
        ['phase'], False, False, acdc_df=acdc_df, SizeT=2,
        condition='area > MEAN(area)'
    )
    assert dialog.windowTitle() == 'Conditional formatting condition'
    dialog.ok_cb()
    assert not dialog.cancel and dialog.equation == 'area > MEAN(area)'


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
        lambda placeholders, channels, inputs=None: (
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
