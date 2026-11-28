"""Shared pytest setup for the Qt (UI) tests.

- Qt runs on the ``offscreen`` platform, so no windows appear and tests work
  without a display. Set ``QT_QPA_PLATFORM`` explicitly (e.g. ``windows``)
  before running pytest to watch the UI while debugging.
- Blocking dialogs never hang a test: while a nested event loop is running
  (``exec_`` of a ``QDialog``, ``QBaseDialog``/``myMessageBox`` loops, static
  ``QFileDialog``/``QInputDialog`` getters), a watchdog answers the newest
  visible dialog after a short grace period, so timers that the test
  scheduled itself act first. It clicks the dialog's OK/default button (or
  calls ``ok_cb``/``accept``), which keeps the preselected choice, e.g. the
  current item or the default value; a plain ``QMessageBox`` gets its
  default button, else its first accepting (Accept/Yes/Apply role) button.
  Dialogs that stay open after two attempts (e.g. failed validation), or
  whose answering raises, are rejected/closed. Answers run outside the
  watchdog's timer slot, so a dialog opened while another one is being
  answered (e.g. a validation warning shown by ``ok_cb``) is answered too.
  Each answered dialog emits an
  ``AutoAnsweredDialogWarning`` naming the dialog by its title (a
  ``QMessageBox`` by its text when the title is empty: macOS ignores
  message box titles), so it is listed in the pytest summary; tests
  should still monkeypatch dialogs whose answer matters. Dialogs are tracked
  by a key stored as a Qt property on the dialog, not by ``id()``, which is
  unstable for dialogs created by Qt (e.g. static ``QMessageBox.question``);
  keys are unique across tests.
- Top-level dialogs that a test opened and left visible without releasing
  them with ``deleteLater()`` (e.g. a progress window whose ``closeEvent``
  refuses to close) are hidden after the test (after the deferred deletes
  below), with a ``LeftOpenDialogWarning``: otherwise they stay visible for
  the rest of the pytest session (CI runs all files in one process) and the
  watchdog of a later test may answer them instead of that test's own
  dialog.
- Widgets released with ``deleteLater()`` are destroyed after each test
  (posted ``DeferredDelete`` events are flushed, then ``gc.collect()``):
  tests never run the main loop and ``processEvents()`` does not deliver
  ``DeferredDelete``, so leaked windows used to pile up and crash the
  interpreter at exit. After the test's fixture teardowns, a single
  event-loop pass (``QTest.qWait(0)``, still with the isolated settings and
  the watchdog below) runs the test's timers already due and queued slots and
  releases PyQt's slot proxies of deleted widgets. Timers with a delay still
  pending fire later, in the event loop of another test (with its settings):
  a test that starts one waits for it or stops it.
  Once pytest released the test's fixture values, its garbage is collected
  again and the objects still alive are frozen (``gc.freeze()``,
  ``pytest_runtest_protocol``; not after the last test), so the next test's
  collections scan only the objects created since: a collection of the whole
  heap takes up to 0.5 s late in the session and used to be half of the
  suite's run time. After the last test of each module the whole heap is
  collected (``gc.unfreeze()``). Set ``ACDC_TESTS_FULL_GC=1`` to collect the
  whole heap after every test.
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
- Icons (``QIcon(':<name>.svg')``) come from the committed
  ``cellacdc/qrc_resources_<scheme>.py``: ``cellacdc`` registers its runtime
  copy (``cellacdc/qrc_resources.py`` or ``~/acdc-appdata/qrc_resources.py``),
  which is created only when missing and refreshed only at app start
  (``_run._setup_app``), so in an existing checkout it can lack icons added
  since. When it differs from the committed module, its resources are
  replaced by those of the committed module for the session (in memory, no
  file is changed).
- The ``dummy_processor`` fixture registers the test-only identity
  measurement processor ``'Dummy'`` (``tests/utils/measurement_processing.py``),
  which is not shipped with Cell-ACDC.
"""
import gc
import itertools
import os
import sys
import time
import warnings
from functools import partial

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
if sys.platform.startswith('win'):
    _fonts = os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts')
    if os.path.isdir(_fonts):
        os.environ.setdefault('QT_QPA_FONTDIR', _fonts)

GRACE_SECONDS = 0.3
CHECK_INTERVAL_MS = 100
MAX_ATTEMPTS = 2


class AutoAnsweredDialogWarning(UserWarning):
    """A blocking dialog was answered automatically by the test watchdog."""


class LeftOpenDialogWarning(UserWarning):
    """A test left a top-level dialog visible; it was hidden after the test."""


# Keys of all watchdogs (see `_DialogWatchdog.key`): a dialog that outlives
# its test keeps its key property, so a per-watchdog counter would give a
# later dialog the same key
_DIALOG_KEYS = itertools.count(1)


def _visibleTopLevelDialogs():
    from qtpy.QtWidgets import QApplication, QDialog
    return [
        widget for widget in QApplication.topLevelWidgets()
        if isinstance(widget, QDialog) and widget.isVisible()
    ]


class _DialogWatchdog:
    # Qt dynamic property holding the key of a dialog (see `key`)
    KEY_PROPERTY = '_testWatchdogKey'

    def __init__(self):
        from qtpy.QtCore import QTimer
        self.firstSeen = {}
        self.attempts = {}
        self.answered = []
        # keys of dialogs whose answer is scheduled or running
        self.pending = set()
        self.stopped = False
        # Dialogs visible before the test (e.g. of a module-scoped fixture)
        # are not hidden by `hideDialogsLeftOpen`
        self.keysAtStart = {
            self.key(dialog) for dialog in _visibleTopLevelDialogs()
        }
        self.timer = QTimer()
        self.timer.setInterval(CHECK_INTERVAL_MS)
        self.timer.timeout.connect(self.check)
        self.timer.start()

    def stop(self):
        # Idempotent: tests that run a real worker thread may stop it early
        if self.stopped:
            return
        self.stopped = True
        self.timer.stop()
        self.timer.deleteLater()

    @staticmethod
    def isBlocking():
        from qtpy.QtCore import QThread
        # Tests never run the main loop: level > 0 means a nested loop,
        # i.e. code is waiting for a dialog to be answered
        return QThread.currentThread().loopLevel() > 0

    def key(self, dialog):
        """Key of `dialog` in the watchdog state, unique per dialog.

        Stored as a Qt dynamic property on the C++ object. Not ``id()``:
        dialogs created by Qt (e.g. static ``QMessageBox.question``) get a
        new, short-lived Python wrapper from every ``topLevelWidgets()``
        call, so their ``id()`` can change between checks (the grace period
        then never ends), and a new dialog can reuse a freed one's ``id()``
        (no grace period, predecessor's attempt count). Unique across
        watchdogs (`_DIALOG_KEYS`).
        """
        key = dialog.property(self.KEY_PROPERTY)
        if key is None:
            key = next(_DIALOG_KEYS)
            dialog.setProperty(self.KEY_PROPERTY, key)
        return key

    @staticmethod
    def label(dialog):
        """Title of `dialog`, else the text of a ``QMessageBox``.

        macOS ignores ``QMessageBox.setWindowTitle`` (Qt follows the macOS
        guidelines), so message boxes have no title there.
        """
        from qtpy.QtWidgets import QMessageBox
        title = dialog.windowTitle()
        if not title and isinstance(dialog, QMessageBox):
            title = dialog.text()
        return title

    def forget(self, keys):
        """Drop the grace-period and attempt state of the dialogs `keys`."""
        for key in keys:
            self.firstSeen.pop(key, None)
            self.attempts.pop(key, None)

    def visibleDialogs(self):
        """Visible dialogs as ``(dialog, key)``, oldest first."""
        now = time.monotonic()
        dialogs = [
            (widget, self.key(widget)) for widget in _visibleTopLevelDialogs()
        ]
        visibleKeys = {key for _, key in dialogs}
        self.forget(
            set(self.firstSeen).union(self.attempts).difference(visibleKeys)
        )
        for _, key in dialogs:
            self.firstSeen.setdefault(key, now)
        return sorted(dialogs, key=lambda item: self.firstSeen[item[1]])

    def check(self):
        # An exception escaping a Qt slot can abort the interpreter (PyQt5)
        # or be swallowed silently, so never let one leave the watchdog
        from qtpy.QtCore import QTimer
        try:
            if not self.isBlocking():
                return
            dialogs = self.visibleDialogs()
            if not dialogs:
                return
            dialog, key = dialogs[-1]
            if key in self.pending:
                return
            if time.monotonic() - self.firstSeen[key] < GRACE_SECONDS:
                return
            # Answer outside this slot: Qt does not fire a timer again while
            # its slot runs, so a dialog opened by the answer (e.g. ok_cb
            # showing a validation warning) would never be answered
            self.pending.add(key)
            QTimer.singleShot(0, partial(self.answerScheduled, dialog, key))
        except Exception as err:
            warnings.warn(
                f'Blocking-dialog watchdog failed: {err!r}',
                AutoAnsweredDialogWarning, stacklevel=1
            )

    def answerScheduled(self, dialog, key):
        """Answer `dialog` from a zero-timer scheduled by ``check``."""
        try:
            if self.stopped or not dialog.isVisible():
                return
            self.answer(dialog, key)
        except Exception as err:
            # Also a Qt slot: never let an exception leave it
            warnings.warn(
                f'Blocking-dialog watchdog failed: {err!r}',
                AutoAnsweredDialogWarning, stacklevel=1
            )
        finally:
            self.pending.discard(key)

    @staticmethod
    def dialogButtons(dialog):
        """Buttons of `dialog` except its cancel button, as a list.

        ``myMessageBox`` stores its buttons in a ``buttons`` list attribute,
        while ``QMessageBox``/``QDialogButtonBox`` expose a ``buttons()``
        method; anything else (missing, ``None``, not iterable) gives ``[]``.
        """
        from qtpy.QtWidgets import QAbstractButton
        buttons = getattr(dialog, 'buttons', None)
        if callable(buttons):
            buttons = buttons()
        try:
            buttons = list(buttons or [])
        except TypeError:
            buttons = []
        cancelButton = getattr(dialog, 'cancelButton', None)
        return [
            button for button in buttons
            if isinstance(button, QAbstractButton)
            and button is not cancelButton
        ]

    @staticmethod
    def messageBoxButton(messageBox):
        """Button of a ``QMessageBox`` that accepts it (``None`` if none).

        The default button wins, then the first button with an accepting
        role (Accept, Yes, Apply), then the first button.
        """
        from qtpy.QtWidgets import QMessageBox
        button = messageBox.defaultButton()
        if button is not None:
            return button
        buttons = messageBox.buttons()
        for roleName in ('AcceptRole', 'YesRole', 'ApplyRole'):
            # Unscoped on PyQt5 (qtpy promotes them on PyQt6), scoped on PySide6
            role = getattr(QMessageBox, roleName, None)
            if role is None:
                role = getattr(QMessageBox.ButtonRole, roleName)
            for button in buttons:
                if messageBox.buttonRole(button) == role:
                    return button
        return buttons[0] if buttons else None

    def answerOk(self, dialog):
        """Answer `dialog` with its OK/default choice; return the action."""
        from qtpy.QtWidgets import QAbstractButton, QMessageBox
        okButton = getattr(dialog, 'okButton', None)
        if isinstance(okButton, QAbstractButton) and okButton.isEnabled():
            okButton.click()
            return f'clicked "{okButton.text().strip()}"'
        if isinstance(dialog, QMessageBox):
            button = self.messageBoxButton(dialog)
            if button is None:
                dialog.accept()
                return 'accepted'
            button.click()
            return f'clicked "{button.text().strip()}"'
        buttons = self.dialogButtons(dialog)
        if buttons:
            buttons[0].click()
            return f'clicked "{buttons[0].text().strip()}"'
        if callable(getattr(dialog, 'ok_cb', None)):
            dialog.ok_cb()
            return 'called ok_cb()'
        dialog.accept()
        return 'accepted'

    @staticmethod
    def forceClose(dialog):
        """Reject/close/hide `dialog` until it is no longer visible."""
        steps = (
            dialog.reject,
            # myMessageBox ignores closeEvent unless allowClose is True
            lambda: setattr(dialog, 'allowClose', True),
            dialog.close,
            dialog.hide,
        )
        for step in steps:
            if not dialog.isVisible():
                return
            try:
                step()
            except Exception:
                pass

    def hideDialogsLeftOpen(self):
        """Hide the dialogs opened during the test and still visible.

        Only ``hide()``: no ``closeEvent`` or ``finished``/``rejected``
        slots run, since the test's patches and the settings isolation are
        already undone at this point. Returns ``(type name, title)`` of the
        hidden dialogs, each also reported as a ``LeftOpenDialogWarning``.
        """
        hidden = []
        for dialog in _visibleTopLevelDialogs():
            if self.key(dialog) in self.keysAtStart:
                continue
            name, title = type(dialog).__name__, self.label(dialog)
            dialog.hide()
            hidden.append((name, title))
            warnings.warn(
                f'Hid dialog {name} "{title}" left open by the test',
                LeftOpenDialogWarning, stacklevel=1
            )
        return hidden

    def answer(self, dialog, key):
        self.attempts[key] = self.attempts.get(key, 0) + 1
        title = self.label(dialog)
        name = type(dialog).__name__
        if self.attempts[key] > MAX_ATTEMPTS:
            action = 'closed (still open after answering)'
            self.forceClose(dialog)
        else:
            try:
                action = self.answerOk(dialog)
            except Exception as err:
                # Never leave a dialog blocking because answering it failed
                action = f'closed (answering raised {err!r})'
                self.forceClose(dialog)
        # Custom loops (QBaseDialog) only exit from closeEvent
        loop = getattr(dialog, 'loop', None)
        if not dialog.isVisible() and loop is not None and loop.isRunning():
            loop.exit()
        if not dialog.isVisible():
            self.forget((key,))
        self.answered.append((name, title, action))
        warnings.warn(
            f'Auto-answered blocking dialog {name} "{title}": {action}',
            AutoAnsweredDialogWarning, stacklevel=1
        )


def flushDeferredDeletes():
    """Destroy the objects released with ``deleteLater()`` so far.

    Without a running event loop ``processEvents()`` never delivers
    ``DeferredDelete`` events, so windows that fixtures ``deleteLater()``
    would stay alive until the interpreter exits (and crash it there).
    """
    if 'qtpy' not in sys.modules:
        return
    from qtpy.QtCore import QCoreApplication, QEvent
    if QCoreApplication.instance() is None:
        return
    deferredDelete = getattr(QEvent, 'DeferredDelete', None)
    if deferredDelete is None:
        deferredDelete = QEvent.Type.DeferredDelete
    QCoreApplication.sendPostedEvents(None, deferredDelete)
    # Python wrappers collected now may post more deleteLater() calls
    gc.collect()
    QCoreApplication.sendPostedEvents(None, deferredDelete)


@pytest.fixture(autouse=True)
def _auto_answer_blocking_dialogs():
    """Answer blocking dialogs of Qt tests (see the module docstring).

    Also destroys ``deleteLater()``'d widgets and then hides the dialogs the
    test left visible after each test. Being autouse, this fixture
    is set up before (and torn down after) the test's own fixtures, so the
    windows they release are flushed too.
    """
    if 'qtpy' not in sys.modules:
        # Not a Qt test (or Qt not imported yet): nothing can block
        try:
            yield None
        finally:
            # The test may have imported Qt and created widgets itself
            flushDeferredDeletes()
        return
    from qtpy.QtWidgets import QApplication
    global _QAPP
    # Keep a reference: an unreferenced QApplication is destroyed right away
    # and the watchdog's QTimer would be created without an application
    _QAPP = QApplication.instance() or QApplication([])
    watchdog = _DialogWatchdog()
    try:
        yield watchdog
    finally:
        watchdog.stop()
        # First destroy the dialogs released with deleteLater(): only the
        # ones still alive afterwards were left open
        flushDeferredDeletes()
        watchdog.hideDialogsLeftOpen()


# Set ACDC_TESTS_FULL_GC=1 to collect the whole heap after every test
FREEZE_SURVIVING_OBJECTS = os.environ.get('ACDC_TESTS_FULL_GC') != '1'


def freezeSurvivingObjects():
    """Exclude the objects alive now from the following collections.

    ``gc.freeze()`` moves them to the permanent generation, so the
    ``gc.collect()`` of the next ``flushDeferredDeletes`` scans only the
    objects created since: a collection of the whole heap takes up to 0.5 s
    once the session has imported and created millions of objects (it used
    to be half of the suite's run time).
    """
    if FREEZE_SURVIVING_OBJECTS:
        gc.freeze()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    """After each test, collect its garbage and freeze what survives.

    This runs after pytest released the test's fixture values
    (``item.funcargs``): the fixture teardowns are too early, the windows of
    the test would be frozen while still referenced, then never collected.
    PyQt released the slot proxies of the deleted widgets before, in the
    event-loop pass of ``_process_pending_events_of_test``. After the last
    test of a module (its module fixtures are torn down then), everything
    frozen during the module is unfrozen and the whole heap collected. A
    widget of a module fixture with a Python slot connected is freed only
    one module later (its slot proxy is released in the next test's pass,
    run with that test's isolated settings); no module fixture creates
    widgets so far.
    """
    yield
    if nextitem is None:
        # End of the session: nothing to speed up any more, and objects left
        # in the permanent generation are skipped by the final collection
        gc.unfreeze()
        flushDeferredDeletes()
        return
    if getattr(nextitem, 'module', None) is not getattr(item, 'module', None):
        gc.unfreeze()
    # No event-loop pass here: the settings isolation, the monkeypatches and
    # the dialog watchdog are already undone, so pending timers and queued
    # slots of the test must run before (_process_pending_events_of_test)
    flushDeferredDeletes()
    freezeSurvivingObjects()


# Module-level names that bind paths inside the user's settings folder,
# mapped to the file/folder name used in the isolated settings folder
USER_SETTINGS_PATHS = {
    'settings_csv_path': 'settings.csv',
    'shortcut_filepath': 'shortcuts.ini',
    'custom_annot_path': 'custom_annotations.json',
    'MEASUREMENTS_CONFIGURATIONS_PATH': 'measurements_viewer_configurations',
    'last_entries_metadata_path': 'last_entries_metadata.csv',
    'additional_metadata_path': 'additional_metadata.json',
    'last_selected_measurements_ini_path': 'last_selected_measurements.ini',
    'recentPaths_path': 'recentPaths.csv',
    # myutils.setupLogger writes there and deletes logs older than 7 days
    'logs_path': 'logs',
    # Trained promptable models saved from the GUI (`cellacdc.
    # promptable_models.saved_models` imports the name)
    'saved_promptable_models_path': 'saved_promptable_models',
    # Custom promptable models (read by myutils.get_list_of_promptable_models)
    'promptable_models_list_file_path': 'custom_promptable_models_paths.ini',
    # Colormaps saved by users from the LUT menus (read when the widgets
    # create their gradients, written when a colormap is saved or removed)
    'custom_cmaps_filepath': 'custom_colormaps.ini',
}
# Module-level name of the settings folder itself, redirected to the
# isolated folder (e.g. load.loadLastEntriesMetadata joins file names to it)
SETTINGS_FOLDERPATH_NAME = 'settings_folderpath'


def _importedCellacdcModules():
    return [
        module for name, module in list(sys.modules.items())
        if module is not None
        and (name == 'cellacdc' or name.startswith('cellacdc.'))
    ]


def _isolatedQSettingsClass(folderpath):
    """``QSettings`` subclass storing ``QSettings(org, app)`` in `folderpath`.

    Other constructor signatures (e.g. an explicit file name) are unchanged.
    """
    from qtpy.QtCore import QSettings
    iniFormat = getattr(QSettings, 'IniFormat', None)
    if iniFormat is None:
        iniFormat = QSettings.Format.IniFormat

    class IsolatedQSettings(QSettings):
        def __init__(self, *args, **kwargs):
            isOrgApp = (
                len(args) == 2 and not kwargs
                and all(isinstance(arg, str) for arg in args)
            )
            if isOrgApp:
                filename = '_'.join(args) + '.ini'
                args = (os.path.join(folderpath, filename), iniFormat)
            super().__init__(*args, **kwargs)

    return QSettings, IsolatedQSettings


class _SettingsRedirection:
    """Points the user settings of ``cellacdc`` modules into the isolated
    settings folder of one test (see the module docstring), with the test's
    ``monkeypatch`` (undone after the test).

    The folder is created on the first use (``folder``), so that a test
    that imports ``cellacdc`` only inside its body is isolated too.
    """

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
    """Loader of a ``cellacdc`` module first imported while a test runs:
    executes the module with its own loader, then redirects its settings.

    Everything else is delegated to the module's loader, which replaces
    this one on the module and its spec once the module is executed.
    """

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
    """``sys.meta_path`` entry installed while a test runs: a ``cellacdc``
    module first imported during the test gets the settings redirection
    right after it is executed (``_RedirectingLoader``).

    The spec comes from the other finders, as without this entry.
    """

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
    """Point the user's settings files to a temporary folder.

    See the module docstring. Yields the temporary settings folder (``None``
    when ``cellacdc`` is not imported yet: the folder is created when the
    test imports it).
    """
    def makeFolder():
        folderpath = str(tmp_path_factory.mktemp('acdc_settings'))
        settingsCsvPath = os.path.join(folderpath, USER_SETTINGS_PATHS[
            'settings_csv_path'
        ])
        # Same content as a fresh install (cellacdc/__init__.py), keeping the
        # color scheme of the already loaded Qt resources
        scheme = getattr(sys.modules.get('cellacdc'), 'scheme', 'light')
        with open(settingsCsvPath, 'w') as file:
            file.write(f'setting,value\ncolorScheme,{scheme}\n')
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


@pytest.fixture(autouse=True)
def _process_pending_events_of_test():
    """Destroy the test's released widgets and run its pending events.

    A single event-loop pass after the test's own fixtures are torn down:
    PyQt releases the slot proxies of deleted widgets only in an event loop,
    and a proxy still alive keeps its widget's Python wrapper alive through
    the collection of ``pytest_runtest_protocol`` (which then freezes it).
    The pass also runs the test's timers already due (e.g.
    ``QTimer.singleShot(0, ...)``) and queued slots; timers with a delay
    still pending fire later, in another test's event loop. Autouse
    fixtures of this file are set up in name order, so this teardown runs
    before the ones of ``_isolate_user_settings`` and
    ``_auto_answer_blocking_dialogs``: those slots still see the isolated
    settings and the dialog watchdog.

    ``QTest.qWait(0)`` processes the events and the deferred deletes once;
    ``qWait(1)`` would then sleep until the timer tick, about 15 ms per test
    on Windows.
    """
    yield
    if 'qtpy' not in sys.modules:
        return
    from qtpy.QtCore import QCoreApplication
    if QCoreApplication.instance() is None:
        return
    from qtpy.QtTest import QTest
    flushDeferredDeletes()
    QTest.qWait(0)
    # Destroy what the pass released while the settings are still isolated
    flushDeferredDeletes()


@pytest.fixture(scope='session', autouse=True)
def _register_committed_qrc_resources():
    """Use the icons of the committed ``qrc_resources_<scheme>.py``.

    See the module docstring: the runtime copy that ``cellacdc`` registers
    on import is refreshed only when the app starts.
    """
    if 'cellacdc' not in sys.modules:
        return
    import filecmp
    import importlib
    import cellacdc
    runtime = getattr(cellacdc, 'acdc_qrc_resources', None)
    runtimePath = getattr(runtime, '__file__', None)
    if not runtimePath or not os.path.exists(runtimePath):
        return
    scheme = 'dark' if getattr(cellacdc, 'scheme', 'light') == 'dark' else (
        'light'
    )
    committedPath = getattr(cellacdc, f'qrc_resources_{scheme}_path')
    if filecmp.cmp(runtimePath, committedPath, shallow=False):
        return
    runtime.qCleanupResources()
    moduleName = f'cellacdc.qrc_resources_{scheme}'
    wasImported = moduleName in sys.modules
    # Importing it registers its resources
    committed = importlib.import_module(moduleName)
    if wasImported:
        committed.qInitResources()


@pytest.fixture
def dummy_processor(monkeypatch):
    """Register the test-only identity measurement processor ``'Dummy'``.

    It is not shipped with Cell-ACDC; see ``tests/utils/measurement_processing.py``.
    Yields its ``Model`` class.
    """
    from tests.utils.measurement_processing import register_dummy
    yield register_dummy(monkeypatch)
