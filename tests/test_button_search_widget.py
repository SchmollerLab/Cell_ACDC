from types import SimpleNamespace

import pytest

from cellacdc import apps, qutils, widgets
from cellacdc.gui import guiWin

from qtpy.QtCore import Qt
from qtpy.QtTest import QTest
from qtpy.QtWidgets import (
    QAction, QApplication, QMainWindow, QPushButton, QToolBar, QWidget,
)

@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def search(app):
    widget = widgets.ButtonSearchWidget(loadingType=None)
    yield widget
    app.removeEventFilter(widget)
    widget.popup.deleteLater()
    widget.deleteLater()
    app.processEvents()


@pytest.mark.parametrize('group_type', [list, tuple])
def test_grouped_search_selection_and_metadata(search, group_type):
    first = QPushButton()
    first.setToolTip('First control')
    second = QPushButton()
    second.setToolTip('Second control')
    second.setShortcut('Ctrl+K')
    group = group_type((first, second))
    search.addItems([('Grouped controls', group)])
    item = search._items_by_name['grouped controls']
    tooltip = item.data(widgets.ButtonSearchCompleter.TOOLTIP_ROLE)
    assert 'First control' in tooltip
    assert 'Second control' in tooltip
    assert 'Shortcut: Ctrl+K' in tooltip
    assert item.data(Qt.UserRole + 2) == group

    emitted = []
    search.sigTriggerBlink.connect(emitted.append)
    search.confirm_selection(search.proxy_model.index(0, 0))
    assert emitted == [group]
    assert search.highlightTargets(emitted[0]) == [first, second]


def test_toolbar_widgets_actions_and_nested_controls_include_opener(search):
    window = QMainWindow()
    toolbar = QToolBar(window)
    window.addToolBar(toolbar)
    toolbar.hide()
    opener = QPushButton(window)
    search.registerToolbarTargets(toolbar, opener)
    container = QWidget()
    child = QPushButton(container)
    toolbar.addWidget(container)
    action = QAction('Toolbar action', window)
    toolbar.addAction(action)
    search.guiWin = SimpleNamespace(child=child, action=action)

    assert search.highlightTargets('child') == [child, opener]
    assert search.highlightTargets('action') == [action, opener]
    assert search.highlightTargets([child, action, opener, child]) == [
        child, opener, action,
    ]
    assert toolbar.isHidden()
    window.deleteLater()


def test_shared_toolbar_and_cyclic_links_are_deduplicated(search):
    toolbar = QToolBar()
    first = QPushButton()
    second = QPushButton()
    toolbar.addWidget(first)
    toolbar.addWidget(second)
    search.registerToolbarTargets(toolbar, (first, second))
    assert search.highlightTargets(first) == [first, second]
    toolbar.deleteLater()


def test_blink_group_uses_one_timer_and_restores_individual_styles(app):
    window = QMainWindow()
    first_toolbar = QToolBar(window)
    second_toolbar = QToolBar(window)
    window.addToolBar(first_toolbar)
    window.addToolBar(second_toolbar)
    action = QAction('Shared action', window)
    first_toolbar.addAction(action)
    second_toolbar.addAction(action)
    first = first_toolbar.widgetForAction(action)
    second = second_toolbar.widgetForAction(action)
    standalone = QPushButton(window)
    controls = [first, second, standalone]
    styles = ['color: red', 'color: blue', 'color: green']
    for control, style in zip(controls, styles):
        control.setStyleSheet(style)

    blinker = qutils.QControlBlink([action, standalone, first], qparent=window)
    assert blinker._widgets == controls
    blinker.start()
    assert blinker.timer.isActive()
    assert blinker.stopTimer.isActive()
    blinker.timerCallback()
    blinker.timerCallback()
    assert all(
        control.styleSheet() == 'background-color: orange'
        for control in controls
    )
    blinker.stop()
    assert [control.styleSheet() for control in controls] == styles
    assert not blinker.timer.isActive()
    assert not blinker.stopTimer.isActive()
    window.deleteLater()


@pytest.mark.parametrize('group_type', [None, list, tuple])
def test_gui_search_handler_blinks_toolbar_control_and_opener(search, group_type):
    window = QMainWindow()
    toolbar = QToolBar(window)
    window.addToolBar(toolbar)
    toolbar.hide()
    action = QAction('Child action', window)
    toolbar.addAction(action)
    opener = QPushButton(window)
    search.guiWin = window
    search.registerToolbarTargets(toolbar, opener)
    window.searchWidget = search
    window.childAction = action
    search.sigTriggerBlink.connect(
        lambda target: guiWin.onSearchTriggerBlink(window, target)
    )
    target = (
        'childAction' if group_type is None
        else group_type(('childAction', opener))
    )
    search.addItems([('Child action', target)])
    search.confirm_selection(search.proxy_model.index(0, 0))

    blinker, = window.findChildren(qutils.QControlBlink)
    assert blinker._widgets == [toolbar.widgetForAction(action), opener]
    assert blinker.timer.isActive()
    assert toolbar.isHidden()
    blinker.stop()
    window.deleteLater()


def test_single_target_and_documented_id_selection_remain_supported(search):
    control = QPushButton()
    search.addItems([('Single control', control)])
    assert search.highlightTargets(control) == [control]
    emitted = []
    search.sigTriggerBlink.connect(emitted.append)
    search.confirm_selection(search.proxy_model.index(0, 0))
    search.on_button_selected('Documented control (controlId)')
    assert emitted == [control, 'controlId']
    blinker = qutils.QControlBlink(control, qparent=search)
    blinker.start()
    blinker.stop()


@pytest.mark.parametrize(
    ('query', 'label', 'expected'),
    [
        ('bkgr', 'Background mean intensity', True),
        ('background', 'bkgrVal_mean', True),
        ('backgroud', 'Background mean intensity', False),
        (' MEAN ', 'Background mean intensity', True),
        ('bkgr', 'Cell area', False),
        ('', 'Background mean intensity', False),
    ],
)
def test_metric_highlighting_matches_literal_terms_and_synonyms(query, label, expected):
    assert widgets._matchesSearchText(query, label) is expected


def test_custom_search_numeric_query_does_not_offer_ids(search):
    control = QPushButton()
    search.addItems([('cell_volume_3D', control)])
    search.on_search_text_changed('3')
    assert search.proxy_model.rowCount() == 1
    assert search.proxy_model.index(0, 0).data(Qt.UserRole + 2) is control


def test_dropdown_retains_fuzzy_matching_without_checkbox_highlighting(search):
    control = QPushButton()
    search.addItems([('Background mean intensity', control)])
    search.on_search_text_changed('backgroud')
    assert search.proxy_model.rowCount() == 1
    assert search.proxy_model.index(0, 0).data(Qt.UserRole + 2) is control
    assert not widgets._matchesSearchText('backgroud', 'Background mean intensity')


@pytest.fixture
def measurements_dialog(app):
    dialog = apps.SetMeasurementsDialog(
        ['GFP', 'RFP'], [], isZstack=False, isSegm3D=False
    )
    dialog.show()
    app.processEvents()
    yield dialog
    dialog.stopMeasurementSearchBlink()
    app.removeEventFilter(dialog.searchWidget)
    dialog.searchWidget.popup.deleteLater()
    dialog.deleteLater()
    app.processEvents()


@pytest.mark.parametrize('selection_method', ['keyboard', 'mouse'])
def test_measurement_popup_search_and_selection(
        measurements_dialog, app, selection_method
    ):
    dialog = measurements_dialog
    search = dialog.searchWidget
    checkboxes = [
        checkbox
        for group, _ in dialog.measurementSearchGroups()
        for checkbox in group.checkBoxes
    ]
    assert search.model.rowCount() == len(checkboxes)
    search.search_input.setFocus()
    search.search_input.setText('background')
    search.on_search_text_changed('background')
    app.processEvents()
    assert search.popup.isVisible()
    assert search.proxy_model.rowCount() > 0
    index = search.proxy_model.index(0, 0)
    checkbox = index.data(Qt.UserRole + 2)
    states = [control.isChecked() for control in checkboxes]
    if selection_method == 'keyboard':
        QTest.keyClick(search.search_input, Qt.Key_Down)
        assert search.popup.currentIndex().row() == 1
        QTest.keyClick(search.search_input, Qt.Key_Up)
        assert search.popup.currentIndex().row() == 0
        QTest.keyClick(search.search_input, Qt.Key_Return)
    else:
        QTest.mouseClick(
            search.popup.viewport(), Qt.LeftButton,
            pos=search.popup.visualRect(index).center(),
        )
    app.processEvents()
    blinker, = dialog.findChildren(qutils.QControlBlink)
    assert blinker._widgets == [checkbox]
    assert blinker.timer.isActive()
    assert not search.popup.isVisible()
    assert not search.search_input.text()
    assert [control.isChecked() for control in checkboxes] == states
    search.search_input.setText('area')
    assert not blinker.timer.isActive()


def test_measurement_search_excludes_deleted_checkbox(measurements_dialog):
    dialog = measurements_dialog
    group, _ = next(dialog.measurementSearchGroups())
    checkbox = group.checkBoxes[0]
    checkbox.hide()
    dialog.updateMeasurementSearchRecords()
    targets = [
        dialog.searchWidget.model.item(row).data(Qt.UserRole + 2)
        for row in range(dialog.searchWidget.model.rowCount())
    ]
    assert checkbox not in targets
