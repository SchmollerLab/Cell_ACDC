"""The main GUI launches (`guiWin.run`, as `_main.mainWin.launchGui`)."""
import pytest
from qtpy.QtWidgets import QApplication, QToolButton

from cellacdc.gui.main_window import window as gui_window

pytestmark = [pytest.mark.gui, pytest.mark.slow]


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.mark.smoke
def test_gui_launches_with_right_click_menus(app):
    win = gui_window.guiWin(app)
    try:
        win.run()

        # `QToolBar.addAction(action)` returns None (PyQt5 and PyQt6): the
        # registered widgets must be the buttons (`widgetForAction`)
        assert win.widgetsForActions
        for name, widget in win.widgetsForActions.items():
            assert isinstance(widget, QToolButton), name

        # The "Set shortcut..." menu is on the action's button
        for name in ('Repeat segmentation', 'Repeat Tracking',
                     'Delete all objects outside segm'):
            button = win.widgetsForActions[name]
            assert button.rightClickMenu is not None, name
            assert 'Set shortcut...' in [
                action.text() for action in button.rightClickMenu.actions()
            ], name
    finally:
        win.close()
