"""Shortcut editor dialog and the shared `ShortcutsGuiElements` host hook."""
import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QAction, QKeySequence
from qtpy.QtWidgets import QApplication, QWidget

from cellacdc import apps
from cellacdc.gui_viewer_tools import ShortcutsGuiElements


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
    dialog = apps.ShortcutEditorDialog(
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

    monkeypatch.setattr(apps.ShortcutEditorDialog, 'exec_', accept)
    host = _Host(show)
    host.editShortcuts_cb()
    assert seen['show'] == show
    assert len(host.setShortcutsCalls) == 1
    assert not host.showingShortcutEditorDialog
    if show:
        # Row shown: action comes from the dialog (pre-filled from host)
        keySequence, button = host.delObjAction
        assert keySequence.toString() == 'Ctrl+D'
        assert button == Qt.MouseButton.LeftButton
        assert host.delObjAction is not KEPT_DEL_OBJ_ACTION
    else:
        # Host without a delete-object tool keeps its own action
        assert host.delObjAction is KEPT_DEL_OBJ_ACTION
    host.deleteLater()


def test_main_gui_shows_delete_object_row_and_viewer_hides_it():
    from cellacdc import gui, measurements_viewer
    assert ShortcutsGuiElements.showDeleteObjectShortcut is True
    assert gui.guiWin.showDeleteObjectShortcut is True
    assert (
        measurements_viewer.MeasurementsViewerWin.showDeleteObjectShortcut
        is False
    )
