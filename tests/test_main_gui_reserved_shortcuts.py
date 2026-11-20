"""Keys of custom annotations and points layers of the main GUI must not
clash with the keys already used in the window.

Two enabled shortcuts with the same key trigger neither, and in snapshots
the edit tools (e.g., "Annotate cell as dead", D) are shown next to the
custom annotations. `guiWin.reservedShortcuts` lists the used keys (any
mode), passed to `apps.customAnnotationDialog`
(`guiWin.customAnnotationDialogKwargs`) and `apps.AddPointsLayerDialog`,
which refuse them on OK. The methods are tested on a small host window
(building `guiWin` takes seconds).
"""
import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import QAction, QApplication, QMainWindow, QToolButton

from cellacdc import apps, gui, widgets
from cellacdc.gui_custom_annotations import CustomAnnotationsGuiElements


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


class _Host(QMainWindow):
    KEY_EVENT_KEYS = gui.guiWin.KEY_EVENT_KEYS
    reservedShortcuts = gui.guiWin.reservedShortcuts
    userKeyButtons = gui.guiWin.userKeyButtons
    customAnnotationDialogKwargs = gui.guiWin.customAnnotationDialogKwargs
    customAnnotButtonByName = (
        CustomAnnotationsGuiElements.customAnnotButtonByName
    )

    def __init__(self):
        super().__init__()
        self.zoomOutKeyValue = Qt.Key.Key_H
        toolbar = widgets.ToolBar('Edit', self)
        self.addToolBar(toolbar)
        self.ripCellButton = QToolButton(self)
        self.ripCellButton.setShortcut('D')
        self.ripCellButton.setToolTip('Name: Annotate cell as dead\n\nMore')
        toolbar.addWidget(self.ripCellButton)
        self.brushButton = QToolButton(self)
        self.brushButton.keyPressShortcut = Qt.Key.Key_B
        self.widgetsWithShortcut = {
            'Annotate cell as dead': self.ripCellButton,
            'Brush': self.brushButton,
        }
        self.saveAction = QAction('Save...', self)
        self.saveAction.setShortcut('Ctrl+Alt+S')

        self.annotateToolbar = widgets.ToolBar('Custom annotations', self)
        self.addToolBar(self.annotateToolbar)
        self.annotButton = QToolButton(self)
        self.annotButton.setShortcut('Q')
        self.annotateToolbar.addWidget(self.annotButton)
        self.customAnnotDict = {
            self.annotButton: {'state': {'name': 'budding'}}
        }

        pointsToolbar = widgets.ToolBar('Points layers', self)
        self.addToolBar(pointsToolbar)
        self.pointsButton = widgets.PointsLayerToolButton('o', parent=self)
        self.pointsButton.setShortcut('J')
        action = pointsToolbar.addWidget(self.pointsButton)
        action.layerType = 'Spots'
        self.pointsLayersToolbars = [pointsToolbar]

        # A dialog's buttons are not keys of the window
        self.dialog = apps.QBaseDialog(self)
        dialogButton = QToolButton(self.dialog)
        dialogButton.setShortcut('Ctrl+Shift+Y')


@pytest.fixture
def host(app):
    host = _Host()
    yield host
    host.dialog.deleteLater()
    host.deleteLater()


def test_reserved_shortcuts_of_main_window(host):
    reserved = host.reservedShortcuts()
    assert reserved['D'] == 'Annotate cell as dead'
    assert reserved['B'] == 'Brush'
    assert reserved['Ctrl+Alt+S'] == 'Save'
    assert reserved['H'] == 'Zoom out'
    assert reserved['Q'] == 'Annotation "budding"'
    assert reserved['J'] == 'Points layer "Spots"'
    for key in ('Left', 'Right', 'Up', 'Down', 'Space', 'Esc', 'Z'):
        assert QKeySequence(key).toString() in reserved
    assert 'Ctrl+Shift+Y' not in reserved
    assert 'Q' not in host.reservedShortcuts(host.annotButton)


def test_custom_annotation_dialog_gets_keys_used_by_others(host):
    # New annotation: every used key, edited annotation: not its own key
    kwargs = host.customAnnotationDialogKwargs()
    assert {'D', 'Q', 'J'} <= set(kwargs['reservedShortcuts'])
    kwargs = host.customAnnotationDialogKwargs(state={'name': 'budding'})
    assert 'Q' not in kwargs['reservedShortcuts']
    assert 'D' in kwargs['reservedShortcuts']


@pytest.mark.parametrize('shortcut, refused', [('D', True), ('Shift+D', False)])
def test_add_points_layer_dialog_refuses_used_key(
        host, monkeypatch, shortcut, refused
    ):
    dialog = apps.AddPointsLayerDialog(
        parent=host, reservedShortcuts=host.reservedShortcuts()
    )
    warnings = []
    monkeypatch.setattr(
        dialog, 'warnShortcutAlreadyUsed',
        lambda shortcut, usedBy: warnings.append((shortcut, usedBy))
    )
    dialog.clickEntryRadiobutton.setChecked(True)
    dialog.appearanceGroupbox.shortcutWidget.widget.setText(shortcut)
    dialog.ok_cb()
    if refused:
        assert warnings == [('D', 'Annotate cell as dead')]
        assert dialog.cancel
    else:
        assert warnings == []
        assert not dialog.cancel
    dialog.deleteLater()
