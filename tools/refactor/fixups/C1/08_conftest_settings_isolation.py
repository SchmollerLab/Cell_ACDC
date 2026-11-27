"""C1: the tests' settings isolation (tests/conftest.py, synthesis 6F)
works at every layout:

- `_isolate_user_settings` reads each module's own bindings with
  `vars(module)` instead of getattr, so it never runs the module-level
  `__getattr__` of compat modules (old import paths from C2 on; their
  MovedWarning becomes an error in the tests from C13 on). The modules are
  found by name (USER_SETTINGS_PATHS maps global names, not modules), so
  the redirection follows the moves of C3-C11 without changes;
- while a test runs, a sys.meta_path entry (`_RedirectFreshImports`)
  applies the same redirection to cellacdc modules first imported during
  the test (e.g. inside its body), right after they are executed; it is
  removed at teardown, the monkeypatch undoes its patches. This also
  isolates a test that imports cellacdc only in its body (the settings
  folder is then created on the first redirection);
- `custom_cmaps_filepath` (custom_colormaps.ini, written by the LUT menus,
  module global of widgets.py until C5) joins USER_SETTINGS_PATHS.

tests/test_conftest_isolation.py checks the globals where the code reads
them (`function.__globals__` of a reader, resolved through the classes and
modules that the C12 rewrite updates) instead of naming the module that
binds each path, plus every binding of every imported cellacdc module; a
new subprocess session checks the fresh-import redirection (a probe module
in a temporary folder on cellacdc.__path__, a test importing cellacdc in
its body) and that nothing is left after the test.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import replace, run  # noqa: E402

CONFTEST = 'tests/conftest.py'
ISOLATION_TEST = 'tests/test_conftest_isolation.py'


DOC_OLD = """\
- The user's settings used by the GUI, the measurements viewer and data
  loading are never read or written: for every test, the module-level
  ``settings_folderpath`` and the paths ``settings_csv_path``,
  ``shortcut_filepath``, ``custom_annot_path``,
  ``MEASUREMENTS_CONFIGURATIONS_PATH``, ``last_entries_metadata_path``,
  ``additional_metadata_path``, ``last_selected_measurements_ini_path``,
  ``recentPaths_path``, the saved and custom promptable models
  ``saved_promptable_models_path`` and ``promptable_models_list_file_path``
  and the log folder ``logs_path`` (written
  and pruned by ``myutils.setupLogger``, e.g. when a utility window is
  created) (``USER_SETTINGS_PATHS``) of ``cellacdc`` and its
  already imported submodules point into a fresh temporary settings folder
  (``settings.csv`` holds only the current ``colorScheme``), and
  ``QSettings(organization, application)`` in those modules writes an INI
  file there instead of the registry/user config. Other module-level paths
  built from the settings folder at import time (e.g.
  ``widgets.custom_cmaps_filepath``) are not redirected; add their names to
  ``USER_SETTINGS_PATHS`` when tests use them. Modules first imported inside
  a test body are not redirected, so import GUI modules at the top of the
  test file.
"""

DOC_NEW = """\
- The user's settings used by the GUI, the measurements viewer and data
  loading are never read or written: for every test, the module-level
  ``settings_folderpath`` and the paths ``settings_csv_path``,
  ``shortcut_filepath``, ``custom_annot_path``,
  ``MEASUREMENTS_CONFIGURATIONS_PATH``, ``last_entries_metadata_path``,
  ``additional_metadata_path``, ``last_selected_measurements_ini_path``,
  ``recentPaths_path``, the saved and custom promptable models
  ``saved_promptable_models_path`` and ``promptable_models_list_file_path``,
  the colormaps saved by users ``custom_cmaps_filepath`` and the log folder
  ``logs_path`` (written and pruned by ``myutils.setupLogger``, e.g. when a
  utility window is created) (``USER_SETTINGS_PATHS``) of ``cellacdc`` and
  its submodules point into a fresh temporary settings folder
  (``settings.csv`` holds only the current ``colorScheme``), and
  ``QSettings(organization, application)`` in those modules writes an INI
  file there instead of the registry/user config. Every module is
  redirected where it binds a name (``vars(module)``: a module-level
  ``__getattr__``, e.g. of a module that forwards moved names, never runs),
  which is where its code reads the global at call time. Modules imported
  when the test starts are redirected before it; modules first imported
  during the test (e.g. inside its body, also ``cellacdc`` itself) right
  after they are executed, by a ``sys.meta_path`` entry installed while
  the test runs (``_RedirectFreshImports``). Other module-level paths built
  from the settings folder at import time are not redirected; add their
  names to ``USER_SETTINGS_PATHS`` when tests use them.
"""

PATHS_OLD = """\
    # Custom promptable models (read by myutils.get_list_of_promptable_models)
    'promptable_models_list_file_path': 'custom_promptable_models_paths.ini',
}
"""

PATHS_NEW = """\
    # Custom promptable models (read by myutils.get_list_of_promptable_models)
    'promptable_models_list_file_path': 'custom_promptable_models_paths.ini',
    # Colormaps saved by users from the LUT menus (read when the widgets
    # create their gradients, written when a colormap is saved or removed)
    'custom_cmaps_filepath': 'custom_colormaps.ini',
}
"""

FIXTURE_OLD = """\
@pytest.fixture(autouse=True)
def _isolate_user_settings(tmp_path_factory, monkeypatch):
    \"\"\"Point the user's settings files to a temporary folder.

    See the module docstring. Yields the temporary settings folder (``None``
    when ``cellacdc`` is not imported, i.e. nothing can touch the settings).
    \"\"\"
    if 'cellacdc' not in sys.modules:
        yield None
        return
    import cellacdc
    folderpath = str(tmp_path_factory.mktemp('acdc_settings'))
    settingsCsvPath = os.path.join(folderpath, USER_SETTINGS_PATHS[
        'settings_csv_path'
    ])
    # Same content as a fresh install (cellacdc/__init__.py), keeping the
    # color scheme of the already loaded Qt resources
    scheme = getattr(cellacdc, 'scheme', 'light')
    with open(settingsCsvPath, 'w') as file:
        file.write(f'setting,value\\ncolorScheme,{scheme}\\n')
    modules = _importedCellacdcModules()
    for module in modules:
        for name, filename in USER_SETTINGS_PATHS.items():
            if isinstance(getattr(module, name, None), str):
                monkeypatch.setattr(
                    module, name, os.path.join(folderpath, filename),
                    raising=False
                )
        if isinstance(getattr(module, SETTINGS_FOLDERPATH_NAME, None), str):
            monkeypatch.setattr(
                module, SETTINGS_FOLDERPATH_NAME, folderpath, raising=False
            )
    if 'qtpy' in sys.modules:
        QSettings, IsolatedQSettings = _isolatedQSettingsClass(folderpath)
        for module in modules:
            if getattr(module, 'QSettings', None) is QSettings:
                monkeypatch.setattr(
                    module, 'QSettings', IsolatedQSettings, raising=False
                )
    yield folderpath
"""

FIXTURE_NEW = """\
class _SettingsRedirection:
    \"\"\"Points the user settings of ``cellacdc`` modules into the isolated
    settings folder of one test (see the module docstring), with the test's
    ``monkeypatch`` (undone after the test).

    The folder is created on the first use (``folder``), so that a test
    that imports ``cellacdc`` only inside its body is isolated too.
    \"\"\"

    def __init__(self, monkeypatch, makeFolder):
        self.monkeypatch = monkeypatch
        self._makeFolder = makeFolder
        self._folderpath = None
        self._qsettingsClasses = None
        # Cleared when the test ends: an import that started during the test
        # (e.g. in a worker thread) and ends later is not patched any more,
        # the monkeypatch would never undo it
        self.active = True

    def folder(self):
        if self._folderpath is None:
            self._folderpath = self._makeFolder()
        return self._folderpath

    def redirect(self, module):
        if not self.active or module is None:
            return
        # The module's own bindings (where its code reads them): getattr
        # would run the module-level `__getattr__` of modules that forward
        # moved names to their new modules (with a warning)
        namespace = vars(module)
        folderpath = self.folder()
        for name, filename in USER_SETTINGS_PATHS.items():
            if isinstance(namespace.get(name), str):
                self.monkeypatch.setattr(
                    module, name, os.path.join(folderpath, filename),
                    raising=False
                )
        if isinstance(namespace.get(SETTINGS_FOLDERPATH_NAME), str):
            self.monkeypatch.setattr(
                module, SETTINGS_FOLDERPATH_NAME, folderpath, raising=False
            )
        if 'qtpy' not in sys.modules:
            return
        if self._qsettingsClasses is None:
            self._qsettingsClasses = _isolatedQSettingsClass(folderpath)
        QSettings, IsolatedQSettings = self._qsettingsClasses
        if namespace.get('QSettings') is QSettings:
            self.monkeypatch.setattr(
                module, 'QSettings', IsolatedQSettings, raising=False
            )


class _RedirectingLoader:
    \"\"\"Loader of a ``cellacdc`` module first imported while a test runs:
    executes the module with its own loader, then redirects its settings.

    Everything else is delegated to the module's loader, which replaces
    this one on the module and its spec once the module is executed.
    \"\"\"

    def __init__(self, loader, redirection):
        self._loader = loader
        self._redirection = redirection

    def __getattr__(self, name):
        return getattr(self._loader, name)

    def exec_module(self, module):
        spec = module.__spec__
        try:
            self._loader.exec_module(module)
        finally:
            if getattr(spec, 'loader', None) is self:
                spec.loader = self._loader
            if module.__dict__.get('__loader__') is self:
                module.__loader__ = self._loader
        # A module may replace itself in sys.modules (e.g. an old import
        # path by the module at its new path)
        name = spec.name if spec is not None else module.__name__
        self._redirection.redirect(sys.modules.get(name, module))


class _RedirectFreshImports:
    \"\"\"``sys.meta_path`` entry installed while a test runs: a ``cellacdc``
    module first imported during the test gets the settings redirection
    right after it is executed (``_RedirectingLoader``).

    The spec comes from the other finders, as without this entry.
    \"\"\"

    def __init__(self, redirection):
        self.redirection = redirection

    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'cellacdc' and not fullname.startswith('cellacdc.'):
            return None
        for finder in sys.meta_path:
            if isinstance(finder, _RedirectFreshImports):
                continue
            findSpec = getattr(finder, 'find_spec', None)
            if findSpec is None:
                continue
            spec = findSpec(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        if hasattr(spec.loader, 'exec_module'):
            spec.loader = _RedirectingLoader(spec.loader, self.redirection)
        return spec


@pytest.fixture(autouse=True)
def _isolate_user_settings(tmp_path_factory, monkeypatch):
    \"\"\"Point the user's settings files to a temporary folder.

    See the module docstring. Yields the temporary settings folder (``None``
    when ``cellacdc`` is not imported yet: the folder is created when the
    test imports it).
    \"\"\"
    def makeFolder():
        folderpath = str(tmp_path_factory.mktemp('acdc_settings'))
        settingsCsvPath = os.path.join(folderpath, USER_SETTINGS_PATHS[
            'settings_csv_path'
        ])
        # Same content as a fresh install (cellacdc/__init__.py), keeping the
        # color scheme of the already loaded Qt resources
        scheme = getattr(sys.modules.get('cellacdc'), 'scheme', 'light')
        with open(settingsCsvPath, 'w') as file:
            file.write(f'setting,value\\ncolorScheme,{scheme}\\n')
        return folderpath

    redirection = _SettingsRedirection(monkeypatch, makeFolder)
    folderpath = None
    if 'cellacdc' in sys.modules:
        folderpath = redirection.folder()
        for module in _importedCellacdcModules():
            redirection.redirect(module)
    finder = _RedirectFreshImports(redirection)
    sys.meta_path.insert(0, finder)
    try:
        yield folderpath
    finally:
        redirection.active = False
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
"""


# ------------------------------------------- tests/test_conftest_isolation.py
TEST_DOC_OLD = """\
- The user's settings folder and files (``settings.csv``, ``shortcuts.ini``,
  custom annotations, metadata entries, recent paths, ``QSettings``) and the
  log folder are redirected to a temporary folder.
"""

TEST_DOC_NEW = """\
- The user's settings folder and files (``settings.csv``, ``shortcuts.ini``,
  custom annotations, metadata entries, recent paths, ``QSettings``) and the
  log folder are redirected to a temporary folder, where the code reads
  them (checked through functions that read them, wherever they are
  defined), also in modules first imported during a test.
"""

# (the end of the module docstring is part of the old text: idempotency)
TEST_IMPORTS_OLD = '''\
"""
import os
import subprocess
'''

TEST_IMPORTS_NEW = '''\
"""
import inspect
import os
import subprocess
'''

TEST_CELLACDC_IMPORTS_OLD = """\
from cellacdc import gui_core, gui_custom_annotations, gui_viewer_tools
from cellacdc import load, measurements_viewer, myutils, prompts
"""

TEST_CELLACDC_IMPORTS_NEW = """\
from cellacdc import gui_custom_annotations, gui_viewer_tools
from cellacdc import load, measurements_viewer, myutils, prompts, widgets
"""

TEST_PATHS_OLD = """\
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
            (cellacdc, 'saved_promptable_models_path'),
            (saved_models, 'saved_promptable_models_path'),
            (cellacdc, 'promptable_models_list_file_path'),
            (myutils, 'promptable_models_list_file_path'),
        ):
        assert _isInside(getattr(module, name), folderpath), (module, name)
    # The settings folder itself is redirected too (load.py and prompts.py
    # join file names to it at call time)
    for module in (cellacdc, load, prompts, measurements_viewer):
        assert module.settings_folderpath == folderpath, module
"""

TEST_PATHS_NEW = """\
def _readGlobal(function, name):
    \"\"\"The global `name` as `function` reads it at call time: from the
    namespace of the module that defines the (undecorated) function.\"\"\"
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
"""

TEST_QSETTINGS_OLD = """\
    # QSettings('schmollerlab', 'acdc_gui') wrote an INI file, not the
    # registry/user config
    settings = gui_core.QSettings('schmollerlab', 'acdc_gui')
"""

TEST_QSETTINGS_NEW = """\
    # QSettings('schmollerlab', 'acdc_gui') wrote an INI file, not the
    # registry/user config
    QSettings = _readGlobal(
        MeasurementsViewerWin.saveWindowGeometry, 'QSettings'
    )
    settings = QSettings('schmollerlab', 'acdc_gui')
"""

TEST_SESSION_ANCHOR = """\
def test_flush_deferred_deletes_destroys_widget(app):
"""

TEST_SESSION_NEW = '''\
# Run in their own pytest session (with the conftest as plugin): the first
# test imports cellacdc, the last one checks what is left after the others
_FRESH_IMPORTS_COMMON = \'\'\'
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
\'\'\'

_FRESH_IMPORTS_SESSION = {
    'test_a_fresh_imports.py': _FRESH_IMPORTS_COMMON + \'\'\'
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
\'\'\',
    'test_b_after_the_tests.py': _FRESH_IMPORTS_COMMON + \'\'\'
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
\'\'\',
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
'''


def main(repo):
    items = [CONFTEST + '::__doc__']
    replace(repo, CONFTEST, DOC_OLD, DOC_NEW, items=items)
    replace(repo, CONFTEST, PATHS_OLD, PATHS_NEW,
            items=[CONFTEST + '::USER_SETTINGS_PATHS'])
    replace(repo, CONFTEST, FIXTURE_OLD, FIXTURE_NEW,
            items=[CONFTEST + '::_isolate_user_settings',
                   CONFTEST + '::_SettingsRedirection',
                   CONFTEST + '::_RedirectingLoader',
                   CONFTEST + '::_RedirectFreshImports'])

    test = ISOLATION_TEST
    replace(repo, test, TEST_DOC_OLD, TEST_DOC_NEW, items=[test + '::__doc__'])
    replace(repo, test, TEST_IMPORTS_OLD, TEST_IMPORTS_NEW,
            items=[test + '::import:import os'])
    replace(repo, test, TEST_CELLACDC_IMPORTS_OLD, TEST_CELLACDC_IMPORTS_NEW,
            items=[test + '::import:from cellacdc import gui_core, '
                   'gui_custom_annotations, gui_viewer_tools'])
    replace(repo, test, TEST_PATHS_OLD, TEST_PATHS_NEW,
            items=[test + '::test_settings_paths_point_to_isolated_folder',
                   test + '::_readGlobal'])
    replace(repo, test, TEST_QSETTINGS_OLD, TEST_QSETTINGS_NEW,
            items=[test + '::test_viewer_does_not_touch_real_settings'])
    if '_FRESH_IMPORTS_SESSION' not in _text(repo, test):
        replace(repo, test, TEST_SESSION_ANCHOR, TEST_SESSION_NEW,
                items=[test + '::_FRESH_IMPORTS_SESSION',
                       test + '::test_modules_first_imported_during_a_test_are_redirected'])


def _text(repo, relpath):
    with open(os.path.join(repo, relpath), encoding='utf-8') as file:
        return file.read()


run(main)
