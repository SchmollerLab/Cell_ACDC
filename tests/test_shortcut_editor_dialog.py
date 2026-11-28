"""Shortcut editor dialog and the shared `ShortcutsGuiElements` host hook."""
import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QAction, QKeySequence
from qtpy.QtWidgets import QApplication, QWidget

from cellacdc.data_structure.dialogs import metadata as data_structure_dialogs_metadata
from cellacdc.gui.dialogs import data_io as gui_dialogs_data_io
from cellacdc.gui.mixins import dialogs as gui_mixin_dialogs
from cellacdc.shared.dialogs import (
    generic as dialogs_generic,
    model_selection as dialogs_model_selection,
)
from cellacdc.shared.widgets import (
    line_edits as widgets_line_edits,
    lut_items as widgets_lut_items,
    message_box as widgets_message_box,
)
from cellacdc.gui.mixins.window_core import ShortcutsGuiElements

pytestmark = pytest.mark.gui


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def make_actions(parent):
    nextAction = QAction('Next', parent)
    nextAction.setShortcut('Right')
    prevAction = QAction('Previous', parent)
    prevAction.setShortcut('Left')
    return {'Next': nextAction, 'Previous': prevAction}


@pytest.mark.parametrize('show', [False, True])
def test_ok_works_with_and_without_delete_object_row(app, show):
    parent = QWidget()
    dialog = gui_mixin_dialogs.ShortcutEditorDialog(
        make_actions(parent), delObjectKey='',
        delObjectButton='Middle click', zoomOutKeyValue=Qt.Key.Key_H,
        hard_shortcuts={'Ctrl+F': 'Foo'},
        show_delete_object_shortcut=show
    )
    assert ('Delete object' in dialog.shortcutLineEdits) == show
    dialog.ok_cb()
    assert not dialog.cancel
    assert set(dialog.customShortcuts) == {'Next', 'Previous'}
    assert 'Delete object' not in dialog.customShortcuts
    if show:
        assert dialog.delObjAction == (None, Qt.MouseButton.MiddleButton)
    else:
        assert dialog.delObjAction is None
    dialog.deleteLater()
    parent.deleteLater()


@pytest.mark.parametrize('isMac', [False, True])
def test_conflicts_are_detected_with_mac_key_names(app, monkeypatch, isMac):
    # On macOS the line edits display 'Command+O' (textChanged passes it)
    # while ShortcutLineEdit.text() returns 'Ctrl+O': conflicts between
    # two line edits were never detected there
    # moved: is_mac is read in each of these modules
    monkeypatch.setattr(data_structure_dialogs_metadata, 'is_mac', isMac)
    monkeypatch.setattr(gui_dialogs_data_io, 'is_mac', isMac)
    monkeypatch.setattr(dialogs_generic, 'is_mac', isMac)
    monkeypatch.setattr(dialogs_model_selection, 'is_mac', isMac)
    monkeypatch.setattr(widgets_line_edits, 'is_mac', isMac)
    monkeypatch.setattr(widgets_lut_items, 'is_mac', isMac)
    monkeypatch.setattr(widgets_message_box, 'is_mac', isMac)
    parent = QWidget()
    dialog = gui_mixin_dialogs.ShortcutEditorDialog(
        make_actions(parent), zoomOutKeyValue=Qt.Key.Key_H,
        hard_shortcuts={'Ctrl+F': 'Foo'}, show_delete_object_shortcut=False
    )
    nextEdit = dialog.shortcutLineEdits['Next']
    prevEdit = dialog.shortcutLineEdits['Previous']
    nextEdit.setText('Ctrl+O')
    assert nextEdit.displayText() == ('Command+O' if isMac else 'Ctrl+O')
    assert nextEdit.text() == 'Ctrl+O'
    assert not prevEdit.warnConflictLabel.text()
    prevEdit.setText('Ctrl+O')
    assert 'Conflict with' in prevEdit.warnConflictLabel.text()
    assert 'Next' in prevEdit.warnConflictLabel.text()
    prevEdit.setText('Ctrl+F')
    assert 'Foo' in prevEdit.warnConflictLabel.text()
    dialog.deleteLater()
    parent.deleteLater()


KEPT_DEL_OBJ_ACTION = (QKeySequence('Ctrl+D'), Qt.MouseButton.LeftButton)


class _Host(QWidget, ShortcutsGuiElements):
    def __init__(self, showDeleteObjectShortcut):
        super().__init__()
        self.showDeleteObjectShortcut = showDeleteObjectShortcut
        self.widgetsWithShortcut = make_actions(self)
        self.delObjAction = KEPT_DEL_OBJ_ACTION
        self.zoomOutKeyValue = Qt.Key.Key_H
        self.mouseBindings = {}
        self.setShortcutsCalls = []

    def getHardcodedShortcuts(self):
        return {}

    def setShortcuts(self, shortcuts, save=True):
        self.setShortcutsCalls.append(shortcuts)


@pytest.mark.parametrize('show', [False, True])
def test_edit_shortcuts_applies_changes_and_keeps_hidden_delete_action(
        app, monkeypatch, show
    ):
    seen = {}

    def accept(dialog):
        seen['show'] = dialog.show_delete_object_shortcut
        dialog.ok_cb()

    monkeypatch.setattr(gui_mixin_dialogs.ShortcutEditorDialog, 'exec_', accept)
    host = _Host(show)
    host.editShortcuts_cb()
    assert seen['show'] == show
    assert len(host.setShortcutsCalls) == 1
    assert not host.showingShortcutEditorDialog
    if show:
        # Row shown: action comes from the dialog (pre-filled from host)
        keySequence, button = host.delObjAction
        # Compare key combinations: KeySequenceFromText.toString() gives
        # the macOS display text ('Command+D') on macOS
        assert keySequence == QKeySequence('Ctrl+D')
        assert button == Qt.MouseButton.LeftButton
        assert host.delObjAction is not KEPT_DEL_OBJ_ACTION
    else:
        # Host without a delete-object tool keeps its own action
        assert host.delObjAction is KEPT_DEL_OBJ_ACTION
    host.deleteLater()


def test_main_gui_shows_delete_object_row_and_viewer_hides_it():
    from cellacdc.gui.main_window import window as gui_window
    from cellacdc.measurements_viewer.main_window import window as viewer_window
    assert ShortcutsGuiElements.showDeleteObjectShortcut is True
    assert gui_window.guiWin.showDeleteObjectShortcut is True
    assert (
        viewer_window.MeasurementsViewerWin.showDeleteObjectShortcut
        is False
    )
