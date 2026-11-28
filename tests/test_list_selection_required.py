"""Single-choice lists (`MeasurementsViewerWin.selectItem`, e.g. the base
layer of a premade configuration, every single choice whose answer is read
by index) and the overlay pickers require a selection: OK with nothing
selected warns and keeps the list open instead of ending the calling flow
without any message. Additional buttons (e.g. "Do not track") answer without
a selection. `QDialogListbox.ok_cb` never reads `sender()`."""
import ast
import inspect
import os
import pathlib
import warnings
from unittest.mock import Mock

import pytest
from qtpy.QtCore import Qt, QTimer
from qtpy.QtTest import QTest
from qtpy.QtWidgets import (
    QAbstractItemView, QApplication, QPushButton, QWidget
)

import cellacdc
from cellacdc import widgets
from cellacdc.gui import guiWin
from cellacdc.gui_overlay import (
    OverlayChannelsGuiElements, OverlayLabelsGuiElements
)
from cellacdc.measurements_viewer import MeasurementsViewerWin
from cellacdc.utils import concat

pytestmark = pytest.mark.gui


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def ok_button(dialog):
    buttons = [
        button for button in dialog.findChildren(QPushButton)
        if button.text().strip().lower() == 'ok'
    ]
    assert len(buttons) == 1
    return buttons[0]


def test_select_item_keeps_the_list_open_until_an_item_is_selected(
        app, monkeypatch
    ):
    viewer = MeasurementsViewerWin()
    viewer._startupConfigurationPrompted = True
    warnings = []
    monkeypatch.setattr(
        widgets.QDialogListbox, 'warnSelectionEmpty',
        lambda dialog: warnings.append(dialog.isVisible())
    )
    steps = []

    def answer():
        # Record only: never assert inside a timer slot
        dialogs = [
            widget for widget in app.topLevelWidgets()
            if isinstance(widget, widgets.QDialogListbox)
            and widget.isVisible()
        ]
        if not dialogs:
            QTimer.singleShot(20, answer)
            return
        dialog, = dialogs
        # E.g., Ctrl+Click on the selected item: nothing selected
        dialog.listBox.clearSelection()
        ok_button(dialog).click()
        steps.append(('empty OK, still open', dialog.isVisible()))
        dialog.listBox.item(1).setSelected(True)
        ok_button(dialog).click()
        steps.append(('selected OK, still open', dialog.isVisible()))

    QTimer.singleShot(0, answer)
    try:
        selected = viewer.selectItem(
            'Select channel', 'Select channel to display (base layer):',
            ['BF', 'GFP', 'mCherry']
        )
    finally:
        viewer.deleteLater()

    assert warnings == [True]
    assert steps == [
        ('empty OK, still open', True), ('selected OK, still open', False)
    ]
    assert selected == 'GFP'


@pytest.mark.parametrize('multiSelection', [False, True])
def test_empty_selection_warning_matches_the_selection_mode(
        app, monkeypatch, multiSelection
    ):
    texts = []
    monkeypatch.setattr(
        widgets.myMessageBox, 'warning',
        lambda self, parent, title, text, *args, **kwargs: texts.append(text)
    )
    dialog = widgets.QDialogListbox(
        'Select', 'Select:', ['a', 'b'], multiSelection=multiSelection,
        allowEmptySelection=False
    )
    dialog.listBox.clearSelection()
    ok_button(dialog).click()
    assert dialog.cancel
    assert len(texts) == 1
    # Ctrl/Shift+Click hints only where several items can be selected
    assert ('Ctrl+Click' in texts[0]) == multiSelection
    dialog.listBox.item(0).setSelected(True)
    ok_button(dialog).click()
    assert not dialog.cancel and dialog.selectedItemsText == ['a']
    assert dialog.clickedButton is ok_button(dialog) is dialog.okButton
    dialog.deleteLater()


def no_sender(monkeypatch):
    """Record any `QDialogListbox.sender()` call: outside a slot PyQt returns
    its last proxy sender, possibly a deleted object, and touching it
    segfaults (Ubuntu PyQt5 CI)."""
    calls = []
    monkeypatch.setattr(
        widgets.QDialogListbox, 'sender',
        lambda self: calls.append(self), raising=False
    )
    return calls


def test_ok_cb_called_directly_reports_the_ok_button(app, monkeypatch):
    senderCalls = no_sender(monkeypatch)
    dialog = widgets.QDialogListbox(
        'Select', 'Select:', ['a', 'b'], multiSelection=False,
        allowEmptySelection=False
    )
    # Outside any slot, e.g. by code, tests or the conftest watchdog
    dialog.ok_cb()
    assert not dialog.cancel and dialog.selectedItemsText == ['a']
    assert dialog.clickedButton is dialog.okButton
    assert ok_button(dialog) is dialog.okButton
    dialog.deleteLater()

    # Double-click on an item = OK
    dialog = widgets.QDialogListbox(
        'Select', 'Select:', ['a', 'b'], multiSelection=False
    )
    dialog.listBox.itemDoubleClicked.emit(dialog.listBox.item(1))
    assert not dialog.cancel
    assert dialog.clickedButton is dialog.okButton
    dialog.deleteLater()
    assert senderCalls == []


def test_additional_buttons_answer_without_a_selection(app, monkeypatch):
    senderCalls = no_sender(monkeypatch)
    warnings = []
    monkeypatch.setattr(
        widgets.QDialogListbox, 'warnSelectionEmpty',
        lambda dialog: warnings.append(dialog)
    )
    # As segm.py's tracker list: "Do not track" needs no tracker
    dialog = widgets.QDialogListbox(
        'Track objects?', 'Select tracker:', ['A', 'B'],
        additionalButtons=['Do not track'], multiSelection=False,
        allowEmptySelection=False
    )
    doNotTrack, = dialog._additionalButtons
    dialog.listBox.clearSelection()
    doNotTrack.click()
    assert warnings == []
    assert not dialog.cancel and dialog.selectedItemsText == []
    assert dialog.clickedButton is doNotTrack
    dialog.deleteLater()

    # OK still needs a selection
    dialog = widgets.QDialogListbox(
        'Track objects?', 'Select tracker:', ['A', 'B'],
        additionalButtons=['Do not track'], multiSelection=False,
        allowEmptySelection=False
    )
    dialog.listBox.clearSelection()
    dialog.okButton.click()
    assert warnings == [dialog] and dialog.cancel
    dialog.deleteLater()
    assert senderCalls == []


def test_concat_no_button_skips_copying_without_a_selection(
        app, monkeypatch, tmp_path
    ):
    # The "No" button is passed as a widget, connected by the caller
    senderCalls = no_sender(monkeypatch)
    (tmp_path / 'Pos1_acdc_output.csv').write_text('')
    clicked = []

    def exec_(dialog):
        # "No, do not copy" with nothing selected (Ctrl+Click)
        dialog.listBox.clearSelection()
        noButton, = [
            button for button in dialog.findChildren(QPushButton)
            if button.text() == 'No, do not copy'
        ]
        noButton.click()
        clicked.append((dialog, noButton))

    monkeypatch.setattr(widgets.QDialogListbox, 'exec_', exec_)
    host = QWidget()
    worker = host.worker = Mock(abort=False)
    try:
        concat.ConcatWin.askCopyCcaFromAcdcOutput(host, str(tmp_path))
        (dialog, noButton), = clicked
        assert dialog.clickedButton is noButton
    finally:
        host.deleteLater()
    assert not worker.abort
    worker.setAcdcOutputEndname.assert_not_called()
    worker.waitCond.wakeAll.assert_called_once()
    assert senderCalls == []


def _single_choice_lists_read_without_a_selection_check():
    """`QDialogListbox(..., multiSelection=False)` assignments in cellacdc
    whose `selectedItemsText` is indexed in the same function (or module)
    without a `not <dialog>.selectedItemsText` check, as
    `(path, line, dialog, allowEmptySelection keyword value or None)`
    (`path`: posix path relative to the package folder)."""
    package = os.path.dirname(cellacdc.__file__)
    found = []
    for dirpath, _, filenames in os.walk(package):
        for filename in filenames:
            if not filename.endswith('.py') or filename.startswith('qrc_'):
                continue
            path = os.path.join(dirpath, filename)
            with open(path, encoding='utf-8') as file:
                source = file.read()
            if 'QDialogListbox(' not in source:
                continue
            with warnings.catch_warnings():
                # Invalid escape sequences of old code
                warnings.simplefilter('ignore', SyntaxWarning)
                tree = ast.parse(source)
            scopes = [tree] + [
                node for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            relpath = pathlib.Path(os.path.relpath(path, package)).as_posix()
            for scope in scopes:
                for row in _unchecked_single_choices(scope):
                    found.append((relpath, *row))
    return found


def _scope_nodes(scope):
    """Nodes of `scope` without those of nested functions."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stack.extend(ast.iter_child_nodes(node))


def _keyword_value(call, name):
    for keyword in call.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            return keyword.value.value
    return None


def _unchecked_single_choices(scope):
    nodes = list(_scope_nodes(scope))
    for node in nodes:
        if not isinstance(node, ast.Assign):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        name = getattr(func, 'attr', getattr(func, 'id', None))
        if name != 'QDialogListbox':
            continue
        if _keyword_value(call, 'multiSelection') is not False:
            continue
        dialog = ast.unparse(node.targets[0])
        selection = f'{dialog}.selectedItemsText'
        indexed = any(
            isinstance(other, ast.Subscript)
            and ast.unparse(other.value) == selection
            for other in nodes
        )
        checked = any(
            isinstance(other, ast.UnaryOp) and isinstance(other.op, ast.Not)
            and ast.unparse(other.operand) == selection
            for other in nodes
        )
        if indexed and not checked:
            yield (
                node.lineno, dialog,
                _keyword_value(call, 'allowEmptySelection')
            )


@pytest.mark.slow
def test_single_choices_read_by_index_require_a_selection():
    # Ctrl+Click on the selected item of a single-choice list empties it: an
    # OK then returned `selectedItemsText == []` and `[0]` raised IndexError
    found = _single_choice_lists_read_without_a_selection_check()
    # The scan finds the callers (e.g. the main GUI's model selection,
    # wherever the method is defined)
    method = inspect.unwrap(guiWin.autoSegm_cb)
    path = pathlib.Path(os.path.relpath(
        inspect.getsourcefile(method), os.path.dirname(cellacdc.__file__)
    )).as_posix()
    lines, start = inspect.getsourcelines(method)
    assert [
        row for row in found
        if row[0] == path and start <= row[1] < start + len(lines)
        and row[2] == 'win'
    ], found
    missing = [row for row in found if row[-1] is not False]
    assert missing == []


def test_tracker_selection_requires_a_tracker(app, monkeypatch):
    # `selectTrackerGUI` overrides ok_cb; `guiWin.repeatTrackingVideo` reads
    # `selectedItemsText[0]`
    warned = []
    monkeypatch.setattr(
        widgets.QDialogListbox, 'warnSelectionEmpty',
        lambda dialog: warned.append(dialog)
    )
    monkeypatch.setattr(
        widgets.myutils, 'get_list_of_trackers', lambda: ['A', 'B']
    )
    dialog = widgets.selectTrackerGUI(3)
    dialog.listBox.clearSelection()
    dialog.okButton.click()
    assert warned == [dialog] and dialog.cancel
    dialog.listBox.item(1).setSelected(True)
    dialog.okButton.click()
    assert not dialog.cancel and dialog.selectedItemsText == ['B']
    assert dialog.clickedButton is dialog.okButton
    assert (dialog.startFrame, dialog.stopFrame) == (1, 3)
    dialog.deleteLater()


def test_escape_cancels_a_single_choice_and_ctrl_keeps_it_single(app):
    steps = []

    def answer():
        dialogs = [
            widget for widget in app.topLevelWidgets()
            if isinstance(widget, widgets.QDialogListbox)
            and widget.isVisible()
        ]
        if not dialogs:
            QTimer.singleShot(20, answer)
            return
        dialog, = dialogs
        QTest.keyPress(dialog, Qt.Key_Control, Qt.ControlModifier)
        steps.append(dialog.listBox.selectionMode())
        QTest.keyClick(dialog, Qt.Key_Escape)
        steps.append(dialog.isVisible())

    dialog = widgets.QDialogListbox(
        'Select', 'Select:', ['a', 'b'], multiSelection=False,
        allowEmptySelection=False
    )
    QTimer.singleShot(0, answer)
    dialog.exec_()
    assert steps == [QAbstractItemView.SelectionMode.SingleSelection, False]
    assert dialog.cancel and dialog.selectedItemsText is None
    dialog.deleteLater()

    # Multiple choice: Escape still only clears the selection
    dialog = widgets.QDialogListbox(
        'Select', 'Select:', ['a', 'b'], multiSelection=True
    )
    dialog.show()
    QTest.keyClick(dialog, Qt.Key_Escape)
    assert dialog.isVisible() and not dialog.listBox.selectedItems()
    dialog.close()
    dialog.deleteLater()


def test_overlay_pickers_without_items_report_instead_of_opening(
        app, monkeypatch
    ):
    class Host(OverlayChannelsGuiElements, OverlayLabelsGuiElements):
        pass

    host = Host()
    host.ch_names = ['phase']
    host.user_ch_name = 'phase'
    host.existingSegmEndNames = set()
    reported = []
    monkeypatch.setattr(
        Host, 'warnNoItemsToOverlay', lambda self, what: reported.append(what)
    )
    opened = []
    monkeypatch.setattr(
        widgets.QDialogListbox, 'exec_', lambda self: opened.append(self)
    )
    assert host.askSelectOverlayChannel() is None
    assert host.askLabelsToOverlay() is None
    assert reported == ['channels', 'segmentation files']
    assert opened == []
