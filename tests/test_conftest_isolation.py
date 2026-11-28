"""Tests of the per-test isolation provided by ``tests/conftest.py``.

- The user's settings folder and files (``settings.csv``, ``shortcuts.ini``,
  custom annotations, metadata entries, recent paths, ``QSettings``) and the
  log folder are redirected to a temporary folder, where the code reads
  them (checked through functions that read them, wherever they are
  defined), also in modules first imported during a test.
- Widgets released with ``deleteLater()`` are destroyed after each test.
- The garbage of each test (also its fixture windows) is collected after it,
  although the objects alive after earlier tests are frozen.
- The pending events of a test run after it with its settings still
  isolated.

Checks that span several tests run in their own pytest session
(``test_conftest_cleans_up_after_each_test``).
"""
import inspect
import os
import subprocess
import sys
import weakref

import numpy as np
import pandas as pd
import pytest
import tifffile
from qtpy.QtWidgets import QApplication, QWidget

import cellacdc
from cellacdc import gui_custom_annotations, gui_viewer_tools
from cellacdc import load, measurements_viewer, myutils, prompts, widgets
from cellacdc.measurements_viewer import MeasurementsViewerWin
from cellacdc.promptable_models import saved_models
from tests import conftest
from tests.conftest import flushDeferredDeletes

pytestmark = pytest.mark.infra

# Captured at import (collection) time, before any test redirects them
REAL_SETTINGS_FOLDERPATH = cellacdc.settings_folderpath
REAL_LOGS_PATH = cellacdc.logs_path


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
    try:
        return os.path.commonpath([path, folderpath]) == folderpath
    except ValueError:
        # Different drives on Windows (CI checks out on D:, tmp is on C:)
        return False


def _readGlobal(function, name):
    """The global `name` as `function` reads it at call time: from the
    namespace of the module that defines the (undecorated) function."""
    return inspect.unwrap(function).__globals__[name]


def test_settings_paths_point_to_isolated_folder(_isolate_user_settings):
    folderpath = _isolate_user_settings
    assert folderpath is not None
    for module, name in (
            (cellacdc, 'settings_csv_path'),
            (cellacdc, 'shortcut_filepath'),
            (load, 'last_entries_metadata_path'),
            (load, 'additional_metadata_path'),
            (load, 'last_selected_measurements_ini_path'),
            (cellacdc, 'recentPaths_path'),
            (cellacdc, 'saved_promptable_models_path'),
            (saved_models, 'saved_promptable_models_path'),
            (cellacdc, 'promptable_models_list_file_path'),
        ):
        assert _isInside(getattr(module, name), folderpath), (module, name)
    # Where the code reads them (the gui mixins' own methods: the viewer
    # overrides some of them)
    colorScheme = gui_viewer_tools.ColorSchemeGuiElements
    hoverInfo = gui_viewer_tools.HoverInfoGuiElements
    shortcuts = gui_viewer_tools.ShortcutsGuiElements
    customAnnots = gui_custom_annotations.CustomAnnotationsGuiElements
    for function, name in (
            (MeasurementsViewerWin.loadSettings, 'settings_csv_path'),
            (colorScheme.onToggleColorScheme, 'settings_csv_path'),
            (hoverInfo.showMirroredCursorToggled, 'settings_csv_path'),
            (shortcuts.initShortcuts, 'shortcut_filepath'),
            (customAnnots.saveCustomAnnot, 'custom_annot_path'),
            (
                MeasurementsViewerWin.selectConfigurationFilepath,
                'MEASUREMENTS_CONFIGURATIONS_PATH'
            ),
            (myutils.addToRecentPaths, 'recentPaths_path'),
            (
                myutils.get_list_of_promptable_models,
                'promptable_models_list_file_path'
            ),
            (widgets.getCustomGradients, 'custom_cmaps_filepath'),
        ):
        assert _isInside(_readGlobal(function, name), folderpath), (
            function.__qualname__, name
        )
    # Every binding of every imported cellacdc module
    names = [*conftest.USER_SETTINGS_PATHS, conftest.SETTINGS_FOLDERPATH_NAME]
    for module in conftest._importedCellacdcModules():
        for name in names:
            value = vars(module).get(name)
            if isinstance(value, str):
                assert _isInside(value, folderpath), (module.__name__, name)
    # The settings folder itself is redirected too (load.py and prompts.py
    # join file names to it at call time)
    for module in (cellacdc, load, prompts):
        assert module.settings_folderpath == folderpath, module
    assert not _isInside(REAL_SETTINGS_FOLDERPATH, folderpath)
    # myutils.setupLogger (e.g. creating a utility window) writes a log file
    # into, and deletes week-old logs from, the folder of get_logs_path()
    assert _isInside(myutils.get_logs_path(), folderpath)
    assert not _isInside(REAL_LOGS_PATH, folderpath)


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
    QSettings = _readGlobal(
        MeasurementsViewerWin.saveWindowGeometry, 'QSettings'
    )
    settings = QSettings('schmollerlab', 'acdc_gui')
    assert _isInside(settings.fileName(), folderpath)
    assert settings.value('geometry_conftest_isolation') is not None


def test_flush_collects_the_cyclic_garbage_of_the_test(app):
    # As after each test, the objects alive now are frozen (gc.freeze): the
    # ones created afterwards are still collected
    conftest.freezeSurvivingObjects()
    widget = QWidget()
    # The Python wrapper in a reference cycle: only gc.collect() frees it
    widget.cycle = widget
    ref = weakref.ref(widget)
    del widget
    assert ref() is not None
    flushDeferredDeletes()
    assert ref() is None


# Run in its own pytest session (with the conftest as plugin): the checks
# span several tests, and a test must not depend on an earlier test of the
# suite (deselection would skip it)
_BETWEEN_TESTS_SESSION = '''
import weakref

import pytest
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication, QDialog, QPushButton, QWidget

import cellacdc

REAL_SETTINGS_FOLDERPATH = cellacdc.settings_folderpath
APP = QApplication.instance() or QApplication([])
state = {}


class ProgressLikeDialog(QDialog):
    """Refuses to close, like the progress window of a running worker."""

    def closeEvent(self, event):
        event.ignore()


@pytest.fixture
def window_with_slot():
    window = QWidget()
    child = QPushButton(window)
    # Python wrapper in a reference cycle, also through a PyQt slot proxy
    window.cycle = window
    child.clicked.connect(lambda: window.objectName())
    yield window
    window.deleteLater()


def test_fixture_window(window_with_slot):
    state['window'] = weakref.ref(window_with_slot)


def test_pending_timer():
    def record():
        state['settings_folderpath'] = cellacdc.settings_folderpath
    QTimer.singleShot(0, record)


def test_deleted_later_widget():
    widget = QWidget()
    widget.setObjectName('released_by_a_test')
    widget.show()
    state['destroyed'] = False
    widget.destroyed.connect(lambda *args: state.update(destroyed=True))
    widget.deleteLater()


@pytest.mark.filterwarnings('ignore::tests.conftest.LeftOpenDialogWarning')
def test_dialog_left_open():
    # e.g. the leaked "Computing" progress window of a background utility
    # test, which used to reach the watchdog tests in CI
    dialog = ProgressLikeDialog()
    dialog.setWindowTitle('Left open by a test')
    dialog.show()
    state['dialog'] = dialog


def test_after_the_other_tests():
    # pytest holds the fixture values until after the fixture teardowns: the
    # conftest collects (and freezes) after that, else the window would stay
    # alive until the end of the module
    assert state['window']() is None
    # The event-loop pass after a test runs while its settings are isolated
    assert state['settings_folderpath'] != REAL_SETTINGS_FOLDERPATH
    # Widgets released with deleteLater() are destroyed after the test
    # (processEvents() never delivers DeferredDelete outside an event loop)
    assert state['destroyed'] is True
    names = [widget.objectName() for widget in QApplication.topLevelWidgets()]
    assert 'released_by_a_test' not in names
    # Dialogs left open by a test are hidden after it
    assert not state['dialog'].isVisible()
'''


@pytest.mark.slow
def test_conftest_cleans_up_after_each_test(tmp_path):
    """End-to-end checks of the conftest's cleanup between tests.

    The self-contained parts are checked in the push tier by the other
    tests of this file and of ``test_conftest_dialog_watchdog.py``.
    """
    testFile = tmp_path / 'test_between_tests.py'
    testFile.write_text(_BETWEEN_TESTS_SESSION)
    repoPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {
        **os.environ, 'PYTHONPATH': repoPath, 'QT_QPA_PLATFORM': 'offscreen'
    }
    env.pop('ACDC_TESTS_FULL_GC', None)
    result = subprocess.run(
        [
            sys.executable, '-m', 'pytest', str(testFile), '-q',
            '-p', 'tests.conftest', '-p', 'no:cacheprovider',
            '--basetemp', str(tmp_path / 'basetemp'),
        ],
        cwd=repoPath, env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr


# Run in their own pytest session (with the conftest as plugin): the first
# test imports cellacdc, the last one checks what is left after the others
_FRESH_IMPORTS_COMMON = '''
import os
import sys


def isInside(path, folderpath):
    path = os.path.normcase(os.path.abspath(path))
    folderpath = os.path.normcase(os.path.abspath(folderpath))
    try:
        return os.path.commonpath([path, folderpath]) == folderpath
    except ValueError:
        return False


PROBE_NAME = '_conftest_isolation_probe'
'''

_FRESH_IMPORTS_SESSION = {
    'test_a_fresh_imports.py': _FRESH_IMPORTS_COMMON + '''
import importlib

PROBE = """
import os
from qtpy.QtCore import QSettings

settings_folderpath = os.path.join(os.path.dirname(__file__), 'real')
shortcut_filepath = os.path.join(settings_folderpath, 'shortcuts.ini')
custom_cmaps_filepath = os.path.join(
    settings_folderpath, 'custom_colormaps.ini'
)
other_filepath = os.path.join(settings_folderpath, 'other.ini')
"""


def test_cellacdc_first_imported_in_the_test_body(
        _isolate_user_settings, tmp_path_factory
    ):
    # The conftest imports cellacdc only when a test did
    assert 'cellacdc' not in sys.modules
    assert _isolate_user_settings is None
    import cellacdc
    folderpath = cellacdc.settings_folderpath
    assert isInside(folderpath, str(tmp_path_factory.getbasetemp()))
    assert cellacdc.settings_csv_path == os.path.join(
        folderpath, 'settings.csv'
    )
    assert os.path.exists(cellacdc.settings_csv_path)
    assert isInside(cellacdc.logs_path, folderpath)
    from tests import conftest
    names = [*conftest.USER_SETTINGS_PATHS, conftest.SETTINGS_FOLDERPATH_NAME]
    modules = conftest._importedCellacdcModules()
    assert len(modules) > 1
    for module in modules:
        for name in names:
            value = vars(module).get(name)
            if isinstance(value, str):
                assert isInside(value, folderpath), (module.__name__, name)


def test_module_first_imported_in_the_test_body(
        _isolate_user_settings, tmp_path
    ):
    import cellacdc
    from qtpy.QtCore import QSettings
    folderpath = _isolate_user_settings
    assert folderpath is not None
    assert PROBE_NAME not in dir(cellacdc)
    (tmp_path / f'{PROBE_NAME}.py').write_text(PROBE)
    importlib.invalidate_caches()
    cellacdc.__path__.append(str(tmp_path))
    try:
        probe = importlib.import_module(f'cellacdc.{PROBE_NAME}')
    finally:
        cellacdc.__path__.remove(str(tmp_path))
    assert probe.settings_folderpath == folderpath
    assert probe.shortcut_filepath == os.path.join(folderpath, 'shortcuts.ini')
    assert probe.custom_cmaps_filepath == os.path.join(
        folderpath, 'custom_colormaps.ini'
    )
    # Only the names of the settings paths
    assert not isInside(probe.other_filepath, folderpath)
    assert probe.QSettings is not QSettings
    settings = probe.QSettings('schmollerlab', 'acdc_conftest_probe')
    assert isInside(settings.fileName(), folderpath)
    # The module keeps its own loader
    assert type(probe.__loader__).__name__ == 'SourceFileLoader'
    assert probe.__spec__.loader is probe.__loader__
''',
    'test_b_after_the_tests.py': _FRESH_IMPORTS_COMMON + '''
import pytest


@pytest.fixture(autouse=True)
def _isolate_user_settings():
    # Overrides the conftest's fixture: the state between the tests
    yield None


def test_nothing_is_left_after_the_tests(tmp_path_factory):
    from qtpy.QtCore import QSettings
    from tests import conftest
    import cellacdc
    assert not [
        finder for finder in sys.meta_path
        if isinstance(finder, conftest._RedirectFreshImports)
    ]
    probe = sys.modules[f'cellacdc.{PROBE_NAME}']
    real = os.path.join(os.path.dirname(probe.__file__), 'real')
    assert probe.settings_folderpath == real
    assert probe.shortcut_filepath == os.path.join(real, 'shortcuts.ini')
    assert probe.QSettings is QSettings
    basetemp = str(tmp_path_factory.getbasetemp())
    assert not isInside(cellacdc.settings_folderpath, basetemp)
    assert not isInside(cellacdc.settings_csv_path, basetemp)
''',
}


@pytest.mark.slow
def test_modules_first_imported_during_a_test_are_redirected(tmp_path):
    """cellacdc modules first imported inside a test body (also cellacdc
    itself) get the settings redirection, undone after the test."""
    sessionPath = tmp_path / 'session'
    sessionPath.mkdir()
    for filename, source in _FRESH_IMPORTS_SESSION.items():
        (sessionPath / filename).write_text(source)
    repoPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {
        **os.environ, 'PYTHONPATH': repoPath, 'QT_QPA_PLATFORM': 'offscreen'
    }
    result = subprocess.run(
        [
            sys.executable, '-m', 'pytest', str(sessionPath), '-q',
            '-p', 'tests.conftest', '-p', 'no:cacheprovider',
            '--basetemp', str(tmp_path / 'basetemp'),
        ],
        cwd=repoPath, env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '3 passed' in result.stdout, result.stdout


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
