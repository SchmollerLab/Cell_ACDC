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
  current item or the default value. Dialogs that stay open after two
  attempts (e.g. failed validation) are closed. Each answered dialog emits an
  ``AutoAnsweredDialogWarning`` so it is listed in the pytest summary; tests
  should still monkeypatch dialogs whose answer matters.
"""
import os
import sys
import time
import warnings

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
        self.timer = QTimer()
        self.timer.setInterval(CHECK_INTERVAL_MS)
        self.timer.timeout.connect(self.check)
        self.timer.start()

    def stop(self):
        self.timer.stop()
        self.timer.deleteLater()

    @staticmethod
    def isBlocking():
        from qtpy.QtCore import QThread
        # Tests never run the main loop: level > 0 means a nested loop,
        # i.e. code is waiting for a dialog to be answered
        return QThread.currentThread().loopLevel() > 0

    def visibleDialogs(self):
        from qtpy.QtWidgets import QApplication, QDialog
        now = time.monotonic()
        dialogs = [
            widget for widget in QApplication.topLevelWidgets()
            if isinstance(widget, QDialog) and widget.isVisible()
        ]
        for dialog in dialogs:
            self.firstSeen.setdefault(id(dialog), now)
        return sorted(dialogs, key=lambda d: self.firstSeen[id(d)])

    def check(self):
        if not self.isBlocking():
            return
        dialogs = self.visibleDialogs()
        if not dialogs:
            return
        dialog = dialogs[-1]
        if time.monotonic() - self.firstSeen[id(dialog)] < GRACE_SECONDS:
            return
        self.answer(dialog)

    def answer(self, dialog):
        from qtpy.QtWidgets import QAbstractButton, QMessageBox
        key = id(dialog)
        self.attempts[key] = self.attempts.get(key, 0) + 1
        title = dialog.windowTitle()
        name = type(dialog).__name__
        if self.attempts[key] > MAX_ATTEMPTS:
            action = 'closed (still open after answering)'
            dialog.close()
        else:
            okButton = getattr(dialog, 'okButton', None)
            buttons = [
                button for button in getattr(dialog, 'buttons', [])
                if isinstance(button, QAbstractButton)
                and button is not getattr(dialog, 'cancelButton', None)
            ]
            if isinstance(okButton, QAbstractButton) and okButton.isEnabled():
                action = f'clicked "{okButton.text().strip()}"'
                okButton.click()
            elif buttons:
                action = f'clicked "{buttons[0].text().strip()}"'
                buttons[0].click()
            elif isinstance(dialog, QMessageBox):
                button = dialog.defaultButton() or (dialog.buttons() or [None])[0]
                if button is None:
                    action = 'accepted'
                    dialog.accept()
                else:
                    action = f'clicked "{button.text().strip()}"'
                    button.click()
            elif callable(getattr(dialog, 'ok_cb', None)):
                action = 'called ok_cb()'
                dialog.ok_cb()
            else:
                action = 'accepted'
                dialog.accept()
        # Custom loops (QBaseDialog) only exit from closeEvent
        loop = getattr(dialog, 'loop', None)
        if not dialog.isVisible() and loop is not None and loop.isRunning():
            loop.exit()
        self.answered.append((name, title, action))
        warnings.warn(
            f'Auto-answered blocking dialog {name} "{title}": {action}',
            AutoAnsweredDialogWarning, stacklevel=1
        )


@pytest.fixture(autouse=True)
def _auto_answer_blocking_dialogs():
    """Answer blocking dialogs of Qt tests (see the module docstring)."""
    if 'qtpy' not in sys.modules:
        # Not a Qt test (or Qt not imported yet): nothing can block
        yield None
        return
    from qtpy.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    watchdog = _DialogWatchdog()
    try:
        yield watchdog
    finally:
        watchdog.stop()
