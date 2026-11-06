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
  whose answering raises, are rejected/closed. Each answered dialog emits an
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
        # An exception escaping a Qt slot can abort the interpreter (PyQt5)
        # or be swallowed silently, so never let one leave the watchdog
        try:
            if not self.isBlocking():
                return
            dialogs = self.visibleDialogs()
            if not dialogs:
                return
            dialog = dialogs[-1]
            if time.monotonic() - self.firstSeen[id(dialog)] < GRACE_SECONDS:
                return
            self.answer(dialog)
        except Exception as err:
            warnings.warn(
                f'Blocking-dialog watchdog failed: {err!r}',
                AutoAnsweredDialogWarning, stacklevel=1
            )

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
    global _QAPP
    # Keep a reference: an unreferenced QApplication is destroyed right away
    # and the watchdog's QTimer would be created without an application
    _QAPP = QApplication.instance() or QApplication([])
    watchdog = _DialogWatchdog()
    try:
        yield watchdog
    finally:
        watchdog.stop()
