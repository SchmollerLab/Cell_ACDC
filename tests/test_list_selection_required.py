"""Single-choice lists (`MeasurementsViewerWin.selectItem`, e.g. the base
layer of a premade configuration) and the overlay pickers require a
selection: OK with nothing selected warns and keeps the list open instead of
ending the calling flow without any message."""
import pytest
from qtpy.QtCore import Qt, QTimer
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QAbstractItemView, QApplication, QPushButton

from cellacdc import widgets
from cellacdc.gui_overlay import (
    OverlayChannelsGuiElements, OverlayLabelsGuiElements
)
from cellacdc.measurements_viewer import MeasurementsViewerWin


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
    dialog.ok_cb()
    assert dialog.cancel
    assert len(texts) == 1
    # Ctrl/Shift+Click hints only where several items can be selected
    assert ('Ctrl+Click' in texts[0]) == multiSelection
    dialog.listBox.item(0).setSelected(True)
    dialog.ok_cb()
    assert not dialog.cancel and dialog.selectedItemsText == ['a']
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
