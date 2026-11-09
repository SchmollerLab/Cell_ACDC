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
  ``AutoAnsweredDialogWarning`` so it is listed in the pytest summary; tests
  should still monkeypatch dialogs whose answer matters.
- Widgets released with ``deleteLater()`` are destroyed after each test
  (posted ``DeferredDelete`` events are flushed, then ``gc.collect()``):
  tests never run the main loop and ``processEvents()`` does not deliver
  ``DeferredDelete``, so leaked windows used to pile up and crash the
  interpreter at exit.
- The user's settings used by the GUI, the measurements viewer and data
  loading are never read or written: for every test, the module-level
  ``settings_folderpath`` and the paths ``settings_csv_path``,
  ``shortcut_filepath``, ``custom_annot_path``,
  ``MEASUREMENTS_CONFIGURATIONS_PATH``, ``last_entries_metadata_path``,
  ``additional_metadata_path``, ``last_selected_measurements_ini_path`` and
  ``recentPaths_path`` (``USER_SETTINGS_PATHS``) of ``cellacdc`` and its
  already imported submodules point into a fresh temporary settings folder
  (``settings.csv`` holds only the current ``colorScheme``), and
  ``QSettings(organization, application)`` in those modules writes an INI
  file there instead of the registry/user config. Other module-level paths
  built from the settings folder at import time (e.g.
  ``widgets.custom_cmaps_filepath``) are not redirected; add their names to
  ``USER_SETTINGS_PATHS`` when tests use them. Modules first imported inside
  a test body are not redirected, so import GUI modules at the top of the
  test file.
- The ``dummy_processor`` fixture registers the test-only identity
  measurement processor ``'Dummy'`` (``tests/utils/measurement_processing.py``),
  which is not shipped with Cell-ACDC.
"""
import gc
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


class _DialogWatchdog:
    def __init__(self):
        from qtpy.QtCore import QTimer
        self.firstSeen = {}
        self.attempts = {}
        self.answered = []
        # ids of dialogs whose answer is scheduled or running
        self.pending = set()
        self.stopped = False
        self.timer = QTimer()
        self.timer.setInterval(CHECK_INTERVAL_MS)
        self.timer.timeout.connect(self.check)
        self.timer.start()

    def stop(self):
        self.stopped = True
        self.timer.stop()
        self.timer.deleteLater()

    @staticmethod
    def isBlocking():
        from qtpy.QtCore import QThread
        # Tests never run the main loop: level > 0 means a nested loop,
        # i.e. code is waiting for a dialog to be answered
        return QThread.currentThread().loopLevel() > 0

    def forget(self, keys):
        """Drop the grace-period and attempt state of the dialog ids `keys`.

        State is keyed by ``id(dialog)``, which CPython reuses for a new
        dialog once the old one is freed: stale entries would give the new
        dialog no grace period and its predecessor's attempt count.
        """
        for key in keys:
            self.firstSeen.pop(key, None)
            self.attempts.pop(key, None)

    def visibleDialogs(self):
        from qtpy.QtWidgets import QApplication, QDialog
        now = time.monotonic()
        dialogs = [
            widget for widget in QApplication.topLevelWidgets()
            if isinstance(widget, QDialog) and widget.isVisible()
        ]
        visibleIds = {id(dialog) for dialog in dialogs}
        self.forget(
            set(self.firstSeen).union(self.attempts).difference(visibleIds)
        )
        for dialog in dialogs:
            self.firstSeen.setdefault(id(dialog), now)
        return sorted(dialogs, key=lambda d: self.firstSeen[id(d)])

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
            dialog = dialogs[-1]
            key = id(dialog)
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
            self.answer(dialog)
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

    def answer(self, dialog):
        key = id(dialog)
        self.attempts[key] = self.attempts.get(key, 0) + 1
        title = dialog.windowTitle()
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

    Also destroys ``deleteLater()``'d widgets after each test. Being autouse,
    this fixture is set up before (and torn down after) the test's own
    fixtures, so the windows they release are flushed too.
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
        flushDeferredDeletes()


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


@pytest.fixture(autouse=True)
def _isolate_user_settings(tmp_path_factory, monkeypatch):
    """Point the user's settings files to a temporary folder.

    See the module docstring. Yields the temporary settings folder (``None``
    when ``cellacdc`` is not imported, i.e. nothing can touch the settings).
    """
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
        file.write(f'setting,value\ncolorScheme,{scheme}\n')
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


@pytest.fixture
def dummy_processor(monkeypatch):
    """Register the test-only identity measurement processor ``'Dummy'``.

    It is not shipped with Cell-ACDC; see ``tests/utils/measurement_processing.py``.
    Yields its ``Model`` class.
    """
    from tests.utils.measurement_processing import register_dummy
    yield register_dummy(monkeypatch)
