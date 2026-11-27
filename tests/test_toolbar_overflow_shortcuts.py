"""Shortcuts of toolbar buttons hidden by an overflowing toolbar.

Qt triggers the shortcut of a button (`QAbstractButton.setShortcut`) only
while the button is visible, and `QToolBar` hides the widgets that do not
fit (e.g., on smaller screens). `widgets.ToolBar` keeps the shortcuts of
buttons added with `addWidget` working while they are hidden by the
overflow (user report: "Edit ID" (N) did not work until the button was
moved where it is always visible), without making them active when the
button is hidden on purpose (hidden action, disabled button, hidden
toolbar), where another button can use the same key (e.g., N of
"Edit ID" and of "No tool (lineage tree)").
"""
import pytest
from qtpy.QtCore import Qt
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QMainWindow, QToolButton, QWidget

from cellacdc import widgets
from cellacdc.gui_viewer_tools import ShortcutsGuiElements

pytestmark = pytest.mark.gui

KEYS = 'ABCDEFGH'


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


class _Window(QMainWindow, ShortcutsGuiElements):
    """Main window with a toolbar of fixed size buttons (keys `KEYS`) and
    the shared `setShortcuts` of the main GUI and the viewer."""
    def __init__(self, width, keys=KEYS):
        super().__init__()
        self.toolbar = widgets.ToolBar('Edit', self)
        self.addToolBar(self.toolbar)
        self.buttons = {}
        self.actions = {}
        self.widgetsWithShortcut = {}
        for key in keys:
            button = self.newButton(key)
            self.actions[key] = self.toolbar.addWidget(button)
            self.buttons[key] = button
            self.widgetsWithShortcut[f'Tool {key}'] = button
        self.setCentralWidget(QWidget())
        self.resize(width, 300)

    def newButton(self, key):
        button = QToolButton(self)
        button.setFixedSize(60, 30)
        button.setCheckable(True)
        button.setShortcut(key)
        button.setToolTip(f'Tool {key}\n\nShortcut: "{key}"')
        return button

    def addOtherToolbar(self, key):
        """Toolbar on its own row with one visible button using `key`."""
        self.addToolBarBreak()
        other = widgets.ToolBar('Other', self)
        self.addToolBar(other)
        button = self.newButton(key)
        other.addWidget(button)
        return button

    def showActive(self, app):
        self.show()
        self.activateWindow()
        self.setFocus()
        app.processEvents()


@pytest.fixture
def make_window(app):
    windows = []

    def make(width, keys=KEYS):
        window = _Window(width, keys=keys)
        windows.append(window)
        return window

    yield make
    for window in windows:
        window.hide()
        window.deleteLater()


def press_key(window, app, key):
    window.activateWindow()
    app.processEvents()
    QTest.keyClick(window, key)
    app.processEvents()


def wait_checked(button, checked=True):
    # The button's own shortcut clicks with `animateClick` (100 ms delay)
    for _ in range(50):
        if button.isChecked() == checked:
            return True
        QTest.qWait(10)
    return button.isChecked() == checked


def test_shortcut_of_button_hidden_by_overflow_clicks_it(make_window, app):
    window = make_window(width=200)
    window.showActive(app)
    button = window.buttons['H']
    assert not button.isVisible()
    assert window.toolbar.extendButton.isVisible()

    press_key(window, app, Qt.Key.Key_H)
    assert button.isChecked()
    press_key(window, app, Qt.Key.Key_H)
    assert not button.isChecked()

    # The visible first button keeps working with its own shortcut
    first = window.buttons['A']
    assert first.isVisible()
    press_key(window, app, Qt.Key.Key_A)
    assert wait_checked(first)


def test_visible_buttons_have_no_overflow_shortcut(make_window, app):
    window = make_window(width=2000, keys='ABC')
    window.showActive(app)
    assert all(button.isVisible() for button in window.buttons.values())
    overflowShortcuts = window.toolbar._overflowShortcuts.values()
    assert len(overflowShortcuts) == 3
    assert not any(s.shortcut.isEnabled() for s in overflowShortcuts)

    # Two enabled shortcuts with the same key would trigger neither
    button = window.buttons['C']
    press_key(window, app, Qt.Key.Key_C)
    assert wait_checked(button)


@pytest.mark.parametrize(
    'hiddenBy', ['action', 'disabled button', 'hidden toolbar']
)
def test_button_hidden_on_purpose_does_not_take_the_key(
        make_window, app, hiddenBy
    ):
    # As the "Edit ID" (N) button in "Normal division: Lineage tree" mode,
    # where "No tool (lineage tree)" uses N
    window = make_window(width=200)
    otherButton = window.addOtherToolbar('H')
    window.showActive(app)
    button = window.buttons['H']
    assert not button.isVisible()
    if hiddenBy == 'action':
        window.actions['H'].setVisible(False)
    elif hiddenBy == 'disabled button':
        button.setDisabled(True)
    else:
        window.toolbar.setVisible(False)
    app.processEvents()
    assert otherButton.isVisible()

    press_key(window, app, Qt.Key.Key_H)
    assert wait_checked(otherButton)
    assert not button.isChecked()

    # Visible again (and still hidden by the overflow): the key clicks it
    window.actions['H'].setVisible(True)
    button.setDisabled(False)
    window.toolbar.setVisible(True)
    otherButton.setDisabled(True)
    app.processEvents()
    assert not button.isVisible()
    press_key(window, app, Qt.Key.Key_H)
    assert button.isChecked()


def test_shortcut_edited_while_hidden_by_overflow(make_window, app):
    window = make_window(width=200)
    window.showActive(app)
    button = window.buttons['H']
    assert not button.isVisible()

    window.setShortcuts({'Tool H': ('J', widgets.KeySequenceFromText('J'))}, save=False)
    press_key(window, app, Qt.Key.Key_H)
    assert not button.isChecked()
    press_key(window, app, Qt.Key.Key_J)
    assert button.isChecked()

    # Without update the old key does nothing and the new one is set
    button.setShortcut('K')
    press_key(window, app, Qt.Key.Key_J)
    assert button.isChecked()
    press_key(window, app, Qt.Key.Key_K)
    assert not button.isChecked()


def test_is_widget_hidden_by_overflow(make_window, app):
    # Used by the brush/eraser keys of the main GUI (handled in keyPressEvent)
    window = make_window(width=200)
    window.showActive(app)
    toolbar = window.toolbar
    first, last = window.buttons['A'], window.buttons['H']
    assert first.isVisible()
    assert not toolbar.isWidgetHiddenByOverflow(first)
    assert toolbar.isWidgetHiddenByOverflow(last)
    window.actions['H'].setVisible(False)
    assert not toolbar.isWidgetHiddenByOverflow(last)
    window.actions['H'].setVisible(True)
    last.setDisabled(True)
    assert not toolbar.isWidgetHiddenByOverflow(last)
    assert not toolbar.isWidgetHiddenByOverflow(QWidget())


def test_removed_button_drops_its_overflow_shortcut(make_window, app):
    window = make_window(width=200)
    window.showActive(app)
    button = window.buttons['H']
    window.toolbar.removeAction(window.actions['H'])
    assert window.actions['H'] not in window.toolbar._overflowShortcuts
    press_key(window, app, Qt.Key.Key_H)
    assert not button.isChecked()
