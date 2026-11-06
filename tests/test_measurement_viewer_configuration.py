import json
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
