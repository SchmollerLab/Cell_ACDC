"""Tests of the per-test isolation provided by ``tests/conftest.py``.

- The user's settings folder and files (``settings.csv``, ``shortcuts.ini``,
  custom annotations, metadata entries, recent paths, ``QSettings``) are
  redirected to a temporary folder.
- Widgets released with ``deleteLater()`` are destroyed after each test.
"""
import os

import numpy as np
import pandas as pd
import pytest
import tifffile
from qtpy.QtWidgets import QApplication, QWidget

import cellacdc
from cellacdc import gui_core, gui_custom_annotations, gui_viewer_tools
from cellacdc import load, measurements_viewer, myutils, prompts
from cellacdc.measurements_viewer import MeasurementsViewerWin
from tests.conftest import flushDeferredDeletes

# Captured at import (collection) time, before any test redirects it
REAL_SETTINGS_FOLDERPATH = cellacdc.settings_folderpath


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def _snapshot(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as file:
        content = file.read()
    return content, os.path.getmtime(path)


def _isInside(path, folderpath):
    path = os.path.normcase(os.path.abspath(path))
    folderpath = os.path.normcase(os.path.abspath(folderpath))
    return os.path.commonpath([path, folderpath]) == folderpath


def test_settings_paths_point_to_isolated_folder(_isolate_user_settings):
    folderpath = _isolate_user_settings
    assert folderpath is not None
    for module, name in (
            (cellacdc, 'settings_csv_path'),
            (measurements_viewer, 'settings_csv_path'),
            (gui_viewer_tools, 'settings_csv_path'),
            (gui_viewer_tools, 'shortcut_filepath'),
            (gui_custom_annotations, 'custom_annot_path'),
            (measurements_viewer, 'MEASUREMENTS_CONFIGURATIONS_PATH'),
            (load, 'last_entries_metadata_path'),
            (load, 'additional_metadata_path'),
            (load, 'last_selected_measurements_ini_path'),
            (myutils, 'recentPaths_path'),
            (cellacdc, 'recentPaths_path'),
        ):
        assert _isInside(getattr(module, name), folderpath), (module, name)
    # The settings folder itself is redirected too (load.py and prompts.py
    # join file names to it at call time)
    for module in (cellacdc, load, prompts, measurements_viewer):
        assert module.settings_folderpath == folderpath, module
    assert not _isInside(REAL_SETTINGS_FOLDERPATH, folderpath)


def _makePosition(root):
    images = root / 'Position_1' / 'Images'
    images.mkdir(parents=True)
    tifffile.imwrite(
        images / 'test_phase.tif',
        np.arange(900, dtype=np.uint16).reshape(30, 30)
    )
    labels = np.zeros((30, 30), dtype=np.uint16)
    labels[2:5, 3:7] = 1
    np.savez_compressed(images / 'test_segm.npz', labels)
    pd.DataFrame({
        'frame_i': [0], 'Cell_ID': [1], 'area': [12.]
    }).to_csv(images / 'test_acdc_output.csv', index=False)
    return str(images)


def test_loading_position_does_not_touch_real_settings(
        tmp_path, _isolate_user_settings
    ):
    folderpath = _isolate_user_settings
    realFiles = [
        os.path.join(REAL_SETTINGS_FOLDERPATH, filename) for filename in (
            'last_entries_metadata.csv', 'additional_metadata.json',
            'recentPaths.csv'
        )
    ]
    before = {path: _snapshot(path) for path in realFiles}
    # An unquoted multi-comma value: loadLastEntriesMetadata rewrites the
    # file with the value quoted (parse_metadata_csv_file), proving that it
    # read the isolated file
    lastEntriesPath = os.path.join(folderpath, 'last_entries_metadata.csv')
    with open(lastEntriesPath, 'w') as file:
        file.write('Description,values\nchannel_0_name,a,b\n')

    posData = measurements_viewer.load_position(
        _makePosition(tmp_path), 'phase'
    )

    assert posData.last_md_df is not None
    with open(lastEntriesPath) as file:
        assert 'channel_0_name,"ab"' in file.read()
    after = {path: _snapshot(path) for path in realFiles}
    assert after == before


def test_viewer_does_not_touch_real_settings(app, _isolate_user_settings):
    folderpath = _isolate_user_settings
    realFolder = REAL_SETTINGS_FOLDERPATH
    realFiles = [
        os.path.join(realFolder, filename) for filename in (
            'settings.csv', 'shortcuts.ini', 'custom_annotations.json'
        )
    ]
    before = {path: _snapshot(path) for path in realFiles}

    window = MeasurementsViewerWin()
    window._startupConfigurationPrompted = True
    try:
        assert _isInside(window.settings_csv_path, folderpath)
        # Both toggles persist followID through _set_setting_value
        window.followIDcheckbox.setChecked(
            not window.followIDcheckbox.isChecked()
        )
        window.followIDcheckbox.setChecked(
            not window.followIDcheckbox.isChecked()
        )
        window.saveWindowGeometry('_conftest_isolation')
    finally:
        window.deleteLater()

    after = {path: _snapshot(path) for path in realFiles}
    assert after == before
    with open(os.path.join(folderpath, 'settings.csv')) as file:
        assert 'followID' in file.read()
    assert os.path.exists(os.path.join(folderpath, 'shortcuts.ini'))
    # QSettings('schmollerlab', 'acdc_gui') wrote an INI file, not the
    # registry/user config
    settings = gui_core.QSettings('schmollerlab', 'acdc_gui')
    assert _isInside(settings.fileName(), folderpath)
    assert settings.value('geometry_conftest_isolation') is not None


_released = {}


def test_deleted_later_widget_released(app):
    widget = QWidget()
    widget.show()
    _released['destroyed'] = False
    widget.destroyed.connect(lambda *args: _released.update(destroyed=True))
    widget.deleteLater()
    app.processEvents()
    # processEvents() does not deliver DeferredDelete outside an event loop
    assert not _released['destroyed']
    _released['name'] = 'conftest_isolation_released_widget'
    widget.setObjectName(_released['name'])


def test_deleted_later_widget_destroyed_after_previous_test(app):
    # End-to-end check of the autouse fixture's teardown; the self-contained
    # check is test_flush_deferred_deletes_destroys_widget
    if 'name' not in _released:
        pytest.skip('requires test_deleted_later_widget_released to run first')
    assert _released.get('destroyed') is True
    names = [widget.objectName() for widget in QApplication.topLevelWidgets()]
    assert _released['name'] not in names


def test_flush_deferred_deletes_destroys_widget(app):
    widget = QWidget()
    widget.show()
    destroyed = []
    widget.destroyed.connect(lambda *args: destroyed.append(True))
    widget.deleteLater()
    app.processEvents()
    # processEvents() does not deliver DeferredDelete outside an event loop
    assert not destroyed
    flushDeferredDeletes()
    assert destroyed
