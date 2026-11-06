"""Tests of the blocking-dialog watchdog in ``tests/conftest.py``.

Each test opens a dialog that blocks in a nested event loop and relies on the
autouse watchdog fixture to answer it. A safety timer closes any dialog that
is still open after a few seconds, so a broken watchdog fails the test instead
of hanging the test session.
"""
import pytest
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication, QDialog, QMessageBox

from cellacdc import widgets
from tests.conftest import AutoAnsweredDialogWarning, _DialogWatchdog

SAFETY_TIMEOUT_MS = 5000


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def safety_net(app):
    """Force-close dialogs the watchdog failed to answer; report if it did."""
    fired = []

    def closeAll():
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, QDialog) and widget.isVisible():
                fired.append(type(widget).__name__)
                _DialogWatchdog.forceClose(widget)
                loop = getattr(widget, 'loop', None)
                if loop is not None and loop.isRunning():
                    loop.exit()

    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(closeAll)
    timer.start(SAFETY_TIMEOUT_MS)
    yield fired
    timer.stop()


def _standard(name):
    return getattr(QMessageBox, name, None) or getattr(
        QMessageBox.StandardButton, name
    )


def test_dialog_buttons_normalizes_method_and_list(app):
    # QMessageBox.buttons is a method: iterating it directly used to raise
    # TypeError before the QMessageBox branch could answer the dialog
    msg = QMessageBox()
    msg.setStandardButtons(_standard('Ok') | _standard('Cancel'))
    assert len(_DialogWatchdog.dialogButtons(msg)) == 2

    myMsg = widgets.myMessageBox()
    okButton = myMsg.addButton('Ok')
    assert _DialogWatchdog.dialogButtons(myMsg) == [okButton]

    dialog = QDialog()
    dialog.buttons = None
    assert _DialogWatchdog.dialogButtons(dialog) == []


def test_plain_qmessagebox_exec_is_auto_answered(
        app, safety_net, _auto_answer_blocking_dialogs
    ):
    msg = QMessageBox()
    msg.setWindowTitle('Plain message box')
    msg.setText('Proceed?')
    msg.setStandardButtons(_standard('Ok') | _standard('Cancel'))
    with pytest.warns(AutoAnsweredDialogWarning, match='Plain message box'):
        msg.exec()
    assert not safety_net
    assert msg.clickedButton() is msg.button(_standard('Ok'))
    assert _auto_answer_blocking_dialogs.answered[-1][0] == 'QMessageBox'


def test_qmessagebox_default_button_is_preferred(
        app, safety_net, _auto_answer_blocking_dialogs
    ):
    msg = QMessageBox()
    msg.setWindowTitle('Default button')
    msg.setText('Overwrite?')
    msg.setStandardButtons(_standard('Yes') | _standard('No'))
    msg.setDefaultButton(_standard('No'))
    with pytest.warns(AutoAnsweredDialogWarning, match='Default button'):
        msg.exec()
    assert not safety_net
    assert msg.clickedButton() is msg.button(_standard('No'))
    # Clicked as the default, not closed (which also gives the escape "No")
    action = _auto_answer_blocking_dialogs.answered[-1][2]
    assert action.replace('&', '') == 'clicked "No"'  # '&' = mnemonic


def test_static_qmessagebox_question_is_auto_answered(app, safety_net):
    with pytest.warns(AutoAnsweredDialogWarning, match='Static question'):
        answer = QMessageBox.question(None, 'Static question', 'Continue?')
    assert not safety_net
    assert answer == _standard('Yes')


def test_my_message_box_is_auto_answered(app, safety_net):
    msg = widgets.myMessageBox(wrapText=False)
    with pytest.warns(AutoAnsweredDialogWarning, match='Custom message box'):
        msg.warning(
            None, 'Custom message box', 'Something happened',
            buttonsTexts=('Cancel', 'Ok')
        )
    assert not safety_net
    assert not msg.cancel
    assert msg.clickedButton.text().strip() == 'Ok'


class _FailingDialog(QDialog):
    def ok_cb(self):
        raise RuntimeError('ok_cb failed')


def test_dialog_whose_answer_raises_is_rejected(app, safety_net):
    dialog = _FailingDialog()
    dialog.setWindowTitle('Failing dialog')
    with pytest.warns(AutoAnsweredDialogWarning, match='answering raised'):
        result = dialog.exec()
    assert not safety_net
    assert result == 0  # QDialog.Rejected
