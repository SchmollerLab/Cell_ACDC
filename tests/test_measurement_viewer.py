import json
import os
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import pyqtgraph as pg
import pyqtgraph.exporters

from qtpy.QtCore import QEvent, QPointF, Qt, QTimer
from qtpy.QtGui import QColor, QMouseEvent
from qtpy.QtTest import QTest
from qtpy.QtWidgets import (
    QApplication, QAbstractItemView, QCheckBox, QStyle, QStyleOptionViewItem,
    QLineEdit, QWidget,
)

from cellacdc.measurements_viewer import MeasurementsViewerWin
from cellacdc import annotate, measurements_viewer, qutils, widgets
from cellacdc import measurements_viewer_utils

pytestmark = pytest.mark.viewer


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def viewer(app, monkeypatch):
    window = MeasurementsViewerWin()
    # Skip the startup configuration picker that showEvent would schedule
    window._startupConfigurationPrompted = True
    labels = np.zeros((3, 30, 30), dtype=np.uint16)
    labels[:2, 2:5, 3:7] = 1
    labels[1:, 15:19, 20:25] = 2
    window.data = [
        SimpleNamespace(SizeT=3, SizeZ=1, frame_i=0, segm_data=labels)
    ]
    window.isSegm3D = False
    window.acdc_df = pd.DataFrame(
        {
            'frame_i': [0, 1, 1, 2],
            'Cell_ID': [1, 1, 2, 2],
            'area': [10.0, 20.0, 30.0, 40.0],
            'intensity': [100.0, 250.0, 150.0, 300.0],
        }
    ).set_index(['frame_i', 'Cell_ID'])
    monkeypatch.setattr(window, 'updateImages', Mock())
    monkeypatch.setattr(window, 'setViewRangeToObjBbox', Mock())
    window.centralWidget().setEnabled(True)
    window.initFramesNavigation()
    window.initIDsNavigation(ID=1)
    window.updateColumnsList()
    window.restrictFramesCheckbox.setChecked(True)
    window.followIDcheckbox.setChecked(False)
    window.updateFramesRange()
    window.plotMeasurements()
    yield window
    window.deleteLater()
    app.processEvents()


def column(viewer, name):
    """Column leaf of the grouped measurements tree."""
    item, = [
        item for item in viewer.columnsModel.measurementItems()
        if item.text() == name
    ]
    return item


def column_items(viewer):
    return list(viewer.columnsModel.measurementItems())


def proxy_index(viewer, name):
    """Visible (search proxy) index of the column leaf `name`."""
    proxy = viewer.columnsSearchProxy
    index, = proxy.match(
        proxy.index(0, 0), Qt.DisplayRole, name, -1,
        Qt.MatchExactly | Qt.MatchRecursive
    )
    return index


def wait_navigation():
    # Navigation after clicking a point is deferred with a QTimer
    QTest.qWait(300)


def comparison(viewer):
    viewer.measurementsModeCombobox.setCurrentIndex(1)
    column(viewer, 'area').setCheckState(Qt.Checked)
    column(viewer, 'intensity').setCheckState(Qt.Checked)
    return viewer.measurementsPlot.listDataItems()[0]


def test_comparison_plots_every_cell_and_frame(viewer):
    item = comparison(viewer)
    np.testing.assert_array_equal(item.xData, [10, 20, 30, 40])
    np.testing.assert_array_equal(item.yData, [100, 250, 150, 300])
    assert [
        (point.data()['frame_i'], point.data()['Cell_ID'])
        for point in item.scatter.points()
    ] == [(0, 1), (1, 1), (1, 2), (2, 2)]
    assert item.opts['pen'] is None
    assert not viewer.currentFrameLine.isVisible()
    assert not viewer.timeCourseColumnsWidget.isHidden()
    assert column(viewer, 'area').data(Qt.AccessibleDescriptionRole) == 'X'
    assert column(viewer, 'intensity').data(Qt.AccessibleDescriptionRole) == 'Y'
    assert column(viewer, 'area').isCheckable()
    assert viewer.checkedColumns() == ['area', 'intensity']
    assert viewer.measurementsPlot.getAxis('bottom').labelText == 'area'
    assert viewer.measurementsPlot.getAxis('left').labelText == 'intensity'


def test_switching_modes_preserves_time_course_selection(viewer):
    comparison(viewer)
    viewer.measurementsModeCombobox.setCurrentIndex(0)
    item, = viewer.measurementsPlot.listDataItems()
    np.testing.assert_array_equal(item.xData, [1, 2])
    np.testing.assert_array_equal(item.yData, [10, 20])
    assert viewer.checkedColumns() == ['area']
    assert viewer.currentFrameLine.isVisible()
    assert viewer.measurementsPlot.getAxis('bottom').labelText == 'Frame n.'
    assert viewer.measurementsPlot.getAxis('left').labelText == 'Value'
    assert not viewer.timeCourseColumnsWidget.isHidden()
    assert column(viewer, 'area').isCheckable()
    assert not column(viewer, 'area').icon().isNull()
    assert column(viewer, 'intensity').icon().isNull()


@pytest.mark.parametrize('mode', [0, 1])
def test_points_keep_metadata_and_hover_details(viewer, mode):
    comparison(viewer)
    viewer.measurementsModeCombobox.setCurrentIndex(mode)
    viewer.normalizeCheckbox.setChecked(True)
    item, = viewer.measurementsPlot.listDataItems()
    points = item.scatter.points()
    assert len(points) == (2 if mode == 0 else 4)
    assert item.scatter.opts['hoverable']
    point = points[0]
    tip = item.scatter.opts['tip'](
        point.pos().x(), point.pos().y(), point.data()
    )
    assert 'Cell ID: 1' in tip
    assert 'Frame: 1' in tip
    assert 'area: 10' in tip
    assert 'area (normalized): 0' in tip
    if mode == 1:
        assert 'intensity: 100' in tip
        np.testing.assert_allclose(item.xData, [0, 1/3, 2/3, 1])
        np.testing.assert_allclose(item.yData, [0, 0.75, 0.25, 1])


@pytest.mark.slow
def test_comparison_click_selects_frame_and_forces_zoom(viewer, app):
    item = comparison(viewer)
    event = SimpleNamespace(button=lambda: Qt.LeftButton)
    item.scatter.sigClicked.emit(
        item.scatter, [item.scatter.points()[-1]], event
    )
    # Navigation is deferred so the clicked item survives click dispatch
    assert viewer.selectedID == 1
    wait_navigation()
    assert viewer.selectedID == 2
    assert viewer.idSpinBox.value() == 2
    assert viewer.frame_i == 2
    assert viewer.navSpinBox.value() == 3
    assert viewer.navigateScrollBar.minimum() == 2
    viewer.setViewRangeToObjBbox.assert_called_with(
        (15, 20, 19, 25), padding=10
    )


def test_comparison_keeps_scatter_and_zoom_when_selecting_cell(viewer):
    item = comparison(viewer)
    viewer.measurementsPlot.setXRange(12, 18, padding=0)
    xRange = viewer.measurementsPlot.viewRange()[0]
    viewer.setSelectedID(2)
    assert viewer.measurementsPlot.listDataItems() == [item]
    assert viewer.measurementsPlot.viewRange()[0] == xRange

    # Values changed while selecting (e.g., automatic predictions) replot
    def ensureTimepointPredictions():
        viewer.acdc_df.loc[(2, 2), 'area'] = 50.0
    viewer.ensureTimepointPredictions = ensureTimepointPredictions
    viewer.setSelectedID(1)
    newItem, = viewer.measurementsPlot.listDataItems()
    assert newItem is not item
    np.testing.assert_array_equal(newItem.xData, [10, 20, 30, 50])


def test_comparison_zoom_kept_when_selected_cell_gets_phases(viewer, app):
    # Automatic phase evaluation of a newly selected cell redraws the
    # scatter (phase colors): only changing the axes autoscales it
    viewer.data[0].acdc_df = viewer.acdc_df
    viewer.phaseTranslatorRules = [{
        'name': 'All', 'predictor': {'type': 'none'}, 'timepoints': [],
        'annotations': [], 'phases': ['Whole'],
    }]
    viewer.autoApplyPhases = True
    comparison(viewer)
    vb = viewer.measurementsPlot.getViewBox()
    viewer.measurementsPlot.setRange(xRange=(12, 18), yRange=(0, 5), padding=0)
    zoomed = viewer.measurementsPlot.viewRange()
    viewer.setSelectedID(2)
    app.processEvents()
    assert viewer.acdc_df.xs(2, level='Cell_ID')['phase'].tolist() == [
        'Whole', 'Whole'
    ]
    assert vb.autoRangeEnabled() == [False, False]
    assert viewer.measurementsPlot.viewRange() == zoomed
    # Toggling normalization changes the axes: autoscaled
    viewer.normalizeCheckbox.setChecked(True)
    assert vb.autoRangeEnabled() == [True, True]


def test_frames_restricted_to_selected_cell_update_the_3d_viewer(viewer):
    # Cell 1: frames 0-1, cell 2: frames 1-2 ("Only frames with selected
    # cell" is checked by the fixture)
    renderer = Mock()
    renderer.are_frames_synced.return_value = True
    viewer._volume_renderer = renderer
    try:
        viewer.setFrame(0)
        renderer.set_frame_number.reset_mock()
        viewer.setSelectedID(2)
        assert viewer.frame_i == 1
        renderer.set_frame_number.assert_called_once_with(2)
        # Checking the option moves the frame into the cell's frames too
        viewer.restrictFramesCheckbox.setChecked(False)
        viewer.setFrame(2)
        viewer.setSelectedID(1)
        renderer.set_frame_number.reset_mock()
        viewer.restrictFramesCheckbox.setChecked(True)
        assert viewer.frame_i == 1
        renderer.set_frame_number.assert_called_once_with(2)
    finally:
        viewer._volume_renderer = None


def test_time_course_replots_keep_x_zoom(viewer):
    viewer.rescalePlotCheckbox.setChecked(False)
    column(viewer, 'area').setCheckState(Qt.Checked)
    viewer.measurementsPlot.setXRange(1.5, 2.5, padding=0)
    xRange = viewer.measurementsPlot.viewRange()[0]

    viewer.setSelectedID(2)
    item, = viewer.measurementsPlot.listDataItems()
    np.testing.assert_array_equal(item.xData, [2, 3])
    assert viewer.measurementsPlot.viewRange()[0] == xRange

    # Replot after a table edit read by the plot (e.g., time point added)
    viewer.refreshPlotForColumns(['area'])
    assert viewer.measurementsPlot.viewRange()[0] == xRange

    # Switching back from comparison mode autoscales the time course (as
    # the plot's "A" button) to the frames of cell 2 (2 and 3)
    viewer.measurementsModeCombobox.setCurrentIndex(1)
    viewer.measurementsModeCombobox.setCurrentIndex(0)
    vb = viewer.measurementsPlot.getViewBox()
    assert vb.autoRangeEnabled() == [True, True]
    vb.updateAutoRange()
    xMin, xMax = viewer.measurementsPlot.viewRange()[0]
    assert 1.5 < xMin <= 2 and xMax >= 3


@pytest.mark.slow
def test_time_course_click_navigates_to_measurement(viewer, app):
    item, = viewer.measurementsPlot.listDataItems()
    item.scatter.sigClicked.emit(
        item.scatter, [item.scatter.points()[-1]],
        SimpleNamespace(button=lambda: Qt.LeftButton)
    )
    wait_navigation()
    assert viewer.selectedID == 1
    assert viewer.frame_i == 1
    viewer.setViewRangeToObjBbox.assert_called_with(
        (2, 3, 5, 7), padding=10
    )


@pytest.mark.slow
def test_right_click_does_not_navigate(viewer, app):
    item = comparison(viewer)
    item.scatter.sigClicked.emit(
        item.scatter, [item.scatter.points()[-1]],
        SimpleNamespace(button=lambda: Qt.RightButton)
    )
    wait_navigation()
    assert viewer.selectedID == 1
    assert viewer.frame_i == 0
    viewer.setViewRangeToObjBbox.assert_not_called()


def test_hardcoded_shortcuts_are_listed_and_conflict_in_editor(viewer):
    from cellacdc import apps
    hardShortcuts = viewer.getHardcodedShortcuts()
    # Same format as guiWin: {key sequence: (name, widget)}
    name, widget = hardShortcuts['Ctrl+O']
    assert widget is viewer.openFolderAction
    assert name == 'Load Folder'
    dialog = apps.ShortcutEditorDialog(
        viewer.widgetsWithShortcut, zoomOutKeyValue=viewer.zoomOutKeyValue,
        mouseBindings=viewer.mouseBindings, hard_shortcuts=hardShortcuts,
        show_delete_object_shortcut=False, parent=viewer
    )
    hardTexts = [
        shortcut for shortcuts in dialog.new_hard_shortcuts.values()
        for shortcut in shortcuts.values()
    ]
    # The editor shows macOS key names on macOS ('Command+O')
    assert widgets.windowsShortcutToMac('Ctrl+O') in hardTexts
    lineEdit = dialog.shortcutLineEdits['Next']
    lineEdit.setText('Ctrl+O')
    assert 'Conflict with' in lineEdit.warnConflictLabel.text()
    dialog.deleteLater()


def test_change_and_reset_signal_color(viewer, monkeypatch):
    from qtpy.QtGui import QColor
    import pyqtgraph as pg
    calls = []
    returned = [QColor('#123456')]

    # Same signature as QColorDialog.getColor (catches a duplicated parent)
    def getColor(initial=None, parent=None, title='', options=None):
        calls.append(parent)
        return returned[0]

    monkeypatch.setattr(
        measurements_viewer, 'QColorDialog', SimpleNamespace(getColor=getColor)
    )
    col, = viewer.checkedColumns()

    def penColor():
        item, = viewer.measurementsPlot.listDataItems()
        return pg.mkPen(item.opts['pen']).color().name()

    viewer.changeSignalColor(col)
    assert calls == [viewer]
    assert viewer.measurementsColors[col].name() == '#123456'
    assert penColor() == '#123456'
    # Cancel keeps the color
    returned[0] = QColor()
    viewer.changeSignalColor(col)
    assert viewer.measurementsColors[col].name() == '#123456'
    viewer.resetSignalColor(col)
    assert col not in viewer.measurementsColors
    assert penColor() != '#123456'


def test_id_spinbox_arrows_step_to_existing_ids(viewer):
    viewer.acdc_df = pd.DataFrame(
        {'frame_i': [0, 0, 1], 'Cell_ID': [1, 2, 5], 'area': [1.0, 2.0, 3.0]}
    ).set_index(['frame_i', 'Cell_ID'])
    viewer.initIDsNavigation(ID=2)
    # The arrow first changes the value by +1, then emits sigUpClicked
    viewer.idSpinBox.setValueNoEmit(3)
    viewer.idSpinBox.sigUpClicked.emit()
    assert viewer.selectedID == 5
    assert viewer.idSpinBox.value() == 5
    viewer.idSpinBox.setValueNoEmit(4)
    viewer.idSpinBox.sigDownClicked.emit()
    assert viewer.selectedID == 2
    assert viewer.idSpinBox.value() == 2
    viewer.idSpinBox.sigDownClicked.emit()
    assert viewer.selectedID == 1
    # Down from the first ID stays on it (spinbox resynced)
    viewer.idSpinBox.setValueNoEmit(0)
    viewer.idSpinBox.sigDownClicked.emit()
    assert viewer.selectedID == 1
    assert viewer.idSpinBox.value() == 1


def test_next_previous_step_frames_on_2d_data(viewer, app):
    # The z-slice checkbox is hidden for 2D data: the Z key does nothing
    QTest.keyClick(viewer, Qt.Key_Z)
    assert not viewer.zSliceCheckbox.isChecked()
    # Even if checked (e.g., from a z-stack loaded before), frames are stepped
    viewer.zSliceCheckbox.setChecked(True)
    viewer.nextAction.trigger()
    app.processEvents()
    assert viewer.frame_i == 1
    viewer.prevAction.trigger()
    app.processEvents()
    assert viewer.frame_i == 0


def test_auto_zoom_toggle_not_in_edit_menu(viewer):
    from qtpy.QtWidgets import QMenu
    menuActions = [
        action for menu in viewer.menuBar().findChildren(QMenu)
        for action in menu.actions()
    ]
    assert viewer.enableAutoZoomToCellsAction not in menuActions


def test_max_projection_toggle_disabled_without_data(app, monkeypatch):
    window = MeasurementsViewerWin()
    window._startupConfigurationPrompted = True
    errors = []
    monkeypatch.setattr(sys, 'excepthook', lambda *args: errors.append(args))
    try:
        assert not window.MaxProjToggleAction.isEnabled()
        window.MaxProjToggleAction.trigger()
        app.processEvents()
        assert not errors
    finally:
        window.deleteLater()
        app.processEvents()


def test_mode_switching_without_data(app):
    window = MeasurementsViewerWin()
    window.measurementsModeCombobox.setCurrentIndex(1)
    assert not window.measurementsPlot.listDataItems()
    assert window.measurementsPlot.plotItem.titleLabel.text == 'No data loaded'
    window.measurementsModeCombobox.setCurrentIndex(0)
    assert window.currentFrameLine.isVisible()
    window.deleteLater()
    app.processEvents()


@pytest.mark.slow
@pytest.mark.parametrize('mode', [0, 1])
def test_mouse_hover_shows_popup_and_click_navigates(
        viewer, app, mode, monkeypatch):
    # Exceptions raised inside Qt event handlers only reach sys.excepthook
    errors = []
    monkeypatch.setattr(
        sys, 'excepthook', lambda *exc_info: errors.append(exc_info)
    )
    comparison(viewer)
    viewer.measurementsModeCombobox.setCurrentIndex(mode)
    viewer.show()
    app.processEvents()
    viewer.measurementsPlot.autoRange()
    app.processEvents()
    item, = viewer.measurementsPlot.listDataItems()
    point = item.scatter.points()[-1]
    scenePos = viewer.measurementsPlot.plotItem.vb.mapViewToScene(
        QPointF(point.pos())
    )
    viewportPos = viewer.measurementsPlot.mapFromScene(scenePos)
    QApplication.sendEvent(
        viewer.measurementsPlot.viewport(),
        QMouseEvent(
            QEvent.MouseMove, QPointF(viewportPos),
            Qt.NoButton, Qt.NoButton, Qt.NoModifier
        )
    )
    QTest.qWait(150)
    app.processEvents()
    tip = viewer.measurementsPlot.plotItem.vb.toolTip()
    assert f'Cell ID: {point.data()["Cell_ID"]}' in tip
    assert f'Frame: {point.data()["frame_i"]+1}' in tip
    # Release the test's references so the replot really deletes the item
    del item, point
    QTest.mouseClick(
        viewer.measurementsPlot.viewport(), Qt.LeftButton, pos=viewportPos
    )
    wait_navigation()
    assert not errors, errors
    assert viewer.frame_i == (1 if mode == 0 else 2)
    assert viewer.selectedID == (1 if mode == 0 else 2)
    assert viewer.setViewRangeToObjBbox.called


def test_comparison_filters_unplottable_pairs_without_losing_identity(viewer):
    viewer.acdc_df.loc[(1, 1), 'area'] = np.nan
    viewer.acdc_df.loc[(1, 2), 'intensity'] = np.inf
    item = comparison(viewer)
    np.testing.assert_array_equal(item.xData, [10, 40])
    assert [
        (p.data()['frame_i'], p.data()['Cell_ID'])
        for p in item.scatter.points()
    ] == [(0, 1), (2, 2)]
    viewer.acdc_df['area'] = np.nan
    viewer.plotMeasurements()
    assert not viewer.measurementsPlot.listDataItems()
    assert viewer.measurementsPlot.plotItem.titleLabel.text == (
        'No finite paired measurements'
    )


def test_axes_refresh_preserves_choices_and_clears_missing_columns(viewer):
    comparison(viewer)
    column(viewer, 'area').setCheckState(Qt.Unchecked)
    column(viewer, 'intensity').setCheckState(Qt.Unchecked)
    column(viewer, 'intensity').setCheckState(Qt.Checked)
    column(viewer, 'area').setCheckState(Qt.Checked)
    viewer.updateColumnsList()
    assert viewer.measurementsAxes == {'X': 'intensity', 'Y': 'area'}
    viewer.acdc_df = viewer.acdc_df.drop(columns='area')
    viewer.updateColumnsList()
    viewer.plotMeasurements()
    assert viewer.measurementsAxes == {'X': 'intensity', 'Y': ''}
    assert column(viewer, 'intensity').data(Qt.AccessibleDescriptionRole) == 'X'
    assert viewer.checkedColumns() == ['intensity']
    assert not viewer.measurementsPlot.listDataItems()
    viewer.acdc_df = None
    viewer.updateColumnsList()
    viewer.plotMeasurements()
    assert viewer.measurementsAxes == {'X': '', 'Y': ''}
    assert not viewer.measurementsPlot.listDataItems()


@pytest.mark.slow
def test_filtered_checkbox_toggles_axis_and_preserves_time_course(viewer, app):
    comparison(viewer)
    viewer.show()
    viewer.columnsSearchLineEdit.setText('intensity')
    app.processEvents()
    index = proxy_index(viewer, 'intensity')
    assert index.data() == 'intensity'
    rect = viewer.columnsListWidget.visualRect(index)
    pos = rect.center()
    pos.setX(rect.left()+8)

    def click():
        QTest.mouseClick(viewer.columnsListWidget.viewport(), Qt.LeftButton, pos=pos)
        app.processEvents()

    click()
    assert viewer.measurementsAxes == {'X': 'area', 'Y': ''}
    assert viewer.checkedColumns() == ['area']
    assert not viewer.measurementsPlot.listDataItems()
    click()
    assert viewer.measurementsAxes == {'X': 'area', 'Y': 'intensity'}
    assert viewer.checkedColumns() == ['area', 'intensity']
    viewer.measurementsModeCombobox.setCurrentIndex(0)
    item, = viewer.measurementsPlot.listDataItems()
    np.testing.assert_array_equal(item.yData, [10, 20])


@pytest.mark.slow
def test_comparison_context_menu_toggles_check(viewer, app):
    comparison(viewer)
    viewer.show()
    viewer.columnsSearchLineEdit.setText('intensity')
    app.processEvents()
    index = proxy_index(viewer, 'intensity')
    pos = viewer.columnsListWidget.visualRect(index).center()
    viewer.showColumnsContextMenu(pos)
    menu = QApplication.activePopupWidget()
    assert menu is not None
    actions = menu.actions()
    assert [action.text() for action in actions] == ['Plot']
    assert actions[0].isChecked()
    actions[0].trigger()
    menu.hide()
    assert viewer.measurementsAxes == {'X': 'area', 'Y': ''}
    assert column(viewer, 'intensity').icon().isNull()
    assert viewer.checkedColumns() == ['area']


def test_modes_have_independent_checks(viewer):
    viewer.measurementsModeCombobox.setCurrentIndex(1)
    assert viewer.checkedColumns() == []
    assert viewer.measurementsAxes == {'X': '', 'Y': ''}
    assert all(item.icon().isNull() for item in column_items(viewer))
    column(viewer, 'intensity').setCheckState(Qt.Checked)
    assert viewer.measurementsAxes == {'X': 'intensity', 'Y': ''}
    viewer.measurementsModeCombobox.setCurrentIndex(0)
    assert viewer.checkedColumns() == ['area']
    viewer.measurementsModeCombobox.setCurrentIndex(1)
    assert viewer.checkedColumns() == ['intensity']
    assert column(viewer, 'intensity').data(Qt.AccessibleDescriptionRole) == 'X'


def test_mode_change_repaints_indicators(viewer, monkeypatch):
    viewer.measurementsModeCombobox.setCurrentIndex(1)
    column(viewer, 'intensity').setCheckState(Qt.Checked)
    viewport = viewer.columnsListWidget.viewport()
    repaint = Mock()
    monkeypatch.setattr(viewport, 'update', repaint)
    itemChanged = Mock()
    viewer.columnsModel.itemChanged.connect(itemChanged)
    for mode, icons in ((0, {'area'}), (1, {'intensity'})):
        repaint.reset_mock()
        viewer.measurementsModeCombobox.setCurrentIndex(mode)
        repaint.assert_called()
        assert {
            item.text() for item in column_items(viewer)
            if not item.icon().isNull()
        } == icons
    itemChanged.assert_not_called()
    assert viewer.measurementsAxes == {'X': 'intensity', 'Y': ''}


def test_comparison_requires_unchecking_before_replacing_axis(viewer):
    viewer.acdc_df['third'] = [1, 2, 3, 4]
    viewer.updateColumnsList()
    comparison(viewer)
    column(viewer, 'third').setCheckState(Qt.Checked)
    assert column(viewer, 'third').checkState() == Qt.Unchecked
    assert 'Two measurements are already selected' in viewer.statusBar().currentMessage()
    column(viewer, 'area').setCheckState(Qt.Unchecked)
    column(viewer, 'third').setCheckState(Qt.Checked)
    assert viewer.measurementsAxes == {'X': 'third', 'Y': 'intensity'}
    assert column(viewer, 'third').data(Qt.AccessibleDescriptionRole) == 'X'


@pytest.mark.slow
@pytest.mark.parametrize('mode', [0, 1])
@pytest.mark.parametrize('filtered', [False, True])
def test_indicator_replaces_tick_without_shifting_text(viewer, app, mode, filtered):
    viewer.measurementsModeCombobox.setCurrentIndex(mode)
    viewer.show()
    if filtered:
        viewer.columnsSearchLineEdit.setText('intensity')
    app.processEvents()
    index = proxy_index(viewer, 'intensity')
    delegate = viewer.columnsHighlightDelegate

    def text_rect():
        option = QStyleOptionViewItem()
        option.rect = viewer.columnsListWidget.visualRect(index)
        option.widget = viewer.columnsListWidget
        delegate.initStyleOption(option, index)
        assert not option.features & QStyleOptionViewItem.HasCheckIndicator
        assert not option.features & QStyleOptionViewItem.HasDecoration
        return viewer.columnsListWidget.style().subElementRect(
            QStyle.SE_ItemViewItemText, option, viewer.columnsListWidget
        )

    before = text_rect()
    rect = viewer.columnsListWidget.visualRect(index)
    pos = rect.center()
    pos.setX(rect.left()+8)
    QTest.mouseClick(viewer.columnsListWidget.viewport(), Qt.LeftButton, pos=pos)
    app.processEvents()
    assert 'intensity' in viewer.checkedColumns()
    assert text_rect() == before
    assert not index.data(Qt.DecorationRole).isNull()
    QTest.mouseClick(viewer.columnsListWidget.viewport(), Qt.LeftButton, pos=pos)
    app.processEvents()
    assert 'intensity' not in viewer.checkedColumns()
    assert text_rect() == before


@pytest.mark.slow
def test_clicking_column_name_toggles_selection(viewer, app):
    viewer.show()
    app.processEvents()

    def click_name(name):
        index = proxy_index(viewer, name)
        rect = viewer.columnsListWidget.visualRect(index)
        pos = rect.center()
        # Well to the right of the fixed-width indicator, on the name
        pos.setX(rect.left() + viewer.columnsHighlightDelegate.indicatorWidth + 20)
        QTest.mouseClick(viewer.columnsListWidget.viewport(), Qt.LeftButton, pos=pos)
        app.processEvents()

    # Time course: add/remove plotted columns
    click_name('intensity')
    assert viewer.checkedColumns() == ['area', 'intensity']
    assert len(viewer.measurementsPlot.listDataItems()) == 2
    click_name('area')
    assert viewer.checkedColumns() == ['intensity']

    # Comparison: set X then Y, clicking again frees the axis
    viewer.measurementsModeCombobox.setCurrentIndex(1)
    click_name('intensity')
    assert viewer.measurementsAxes == {'X': 'intensity', 'Y': ''}
    click_name('area')
    assert viewer.measurementsAxes == {'X': 'intensity', 'Y': 'area'}
    assert len(viewer.measurementsPlot.listDataItems()) == 1
    click_name('intensity')
    assert viewer.measurementsAxes == {'X': '', 'Y': 'area'}

    viewer.measurementsModeCombobox.setCurrentIndex(0)
    assert viewer.checkedColumns() == ['intensity']


def test_column_names_cannot_be_edited(viewer, app):
    viewer.show()
    app.processEvents()
    assert viewer.columnsListWidget.editTriggers() == QAbstractItemView.NoEditTriggers
    for item in column_items(viewer):
        assert not item.isEditable()
    index = proxy_index(viewer, 'area')
    name = index.data()
    rect = viewer.columnsListWidget.visualRect(index)
    QTest.mouseDClick(
        viewer.columnsListWidget.viewport(), Qt.LeftButton, pos=rect.center()
    )
    QTest.keyClick(viewer.columnsListWidget, Qt.Key_F2)
    app.processEvents()
    assert not viewer.columnsListWidget.findChildren(QLineEdit)
    assert index.data() == name


@pytest.mark.parametrize(
    ('values', 'expected'),
    [
        ([np.nan, np.inf], [np.nan, np.nan]),
        ([5, 5, np.nan], [0, 0, np.nan]),
        ([10, np.nan, 20, np.inf], [0, np.nan, 1, np.nan]),
    ]
)
def test_normalization_handles_nonfinite_and_constant_measurements(values, expected):
    np.testing.assert_allclose(
        MeasurementsViewerWin._normalize(np.array(values, dtype=float)),
        expected, equal_nan=True
    )


def configuration_data(viewer, tmp_path):
    images = tmp_path / 'Images'
    images.mkdir()
    pos = viewer.data[0]
    pos.images_path = str(images)
    pos.basename = 'test_'
    pos.segm_npz_path = str(images / 'test_segm_custom.npz')
    viewer.user_ch_name = 'phase'
    viewer.isDataLoaded = True
    viewer.allOverlayToolbuttons = {
        'phase': SimpleNamespace(isChecked=lambda: True, setChecked=Mock())
    }
    return images


def mock_filename_dialog(monkeypatch, folder, *filenames):
    """Answer the save dialogs with `filenames` (None cancels) in `folder`,
    which replaces the configurations folder of the settings."""
    monkeypatch.setattr(
        measurements_viewer, 'MEASUREMENTS_CONFIGURATIONS_PATH', str(folder)
    )
    answers = iter(filenames)

    def create_dialog(*args, **kwargs):
        filename = next(answers)
        return SimpleNamespace(
            exec_=lambda: None, cancel=filename is None, doNotSave=False,
            filename=filename or ''
        )

    dialog = Mock(side_effect=create_dialog)
    monkeypatch.setattr(measurements_viewer.apps, 'filenameDialog', dialog)
    return dialog


@pytest.mark.slow
@pytest.mark.parametrize('mode', [0, 1])
def test_configuration_roundtrip(viewer, tmp_path, monkeypatch, mode):
    images = configuration_data(viewer, tmp_path)
    comparison(viewer)
    # Reverse X/Y to prove order is restored, not inferred from column order.
    viewer.measurementsAxes = {'X': 'intensity', 'Y': 'area'}
    viewer.measurementsModeCombobox.setCurrentIndex(mode)
    viewer.normalizeCheckbox.setChecked(True)
    filepath = str(tmp_path / 'state.json')
    mock_filename_dialog(monkeypatch, tmp_path, 'state.json')
    assert viewer.saveConfiguration()
    saved = json.loads((tmp_path / 'state.json').read_text())
    assert saved['images_paths'] == [os.path.abspath(images)]
    assert saved['channel'] == 'phase'
    assert saved['segm_endname'] == 'segm_custom'
    assert saved['time_course_columns'] == ['area']
    assert saved['measurements_axes'] == {'X': 'intensity', 'Y': 'area'}

    viewer.measurementsModeCombobox.setCurrentIndex(0)
    column(viewer, 'area').setCheckState(Qt.Unchecked)
    column(viewer, 'intensity').setCheckState(Qt.Checked)
    viewer.measurementsAxes = {'X': '', 'Y': ''}
    viewer.normalizeCheckbox.setChecked(False)
    load_position = Mock(return_value=viewer.data[0])
    monkeypatch.setattr(measurements_viewer_utils, 'load_position', load_position)
    monkeypatch.setattr(
        measurements_viewer, 'get_channels_and_segm_endnames',
        Mock(return_value=(['phase'], ['segm_custom']))
    )
    monkeypatch.setattr(
        viewer, 'selectConfigurationInput',
        lambda config: viewer.configurationInputOptions(config)[0]
    )
    monkeypatch.setattr(viewer, 'setData', Mock())
    monkeypatch.setattr(viewer, 'setOverlayItemsOpacities', Mock())
    # Discard the unsaved changes made above
    monkeypatch.setattr(viewer, 'askSaveBeforeLoading', Mock(return_value=True))
    assert viewer.loadConfiguration(filepath=filepath)
    load_position.assert_called_once_with(str(images), 'phase', 'segm_custom')
    assert viewer.configurationFilepath == filepath
    assert viewer.measurementsModeCombobox.currentIndex() == mode
    assert viewer.normalizeCheckbox.isChecked()
    assert viewer.measurementsAxes == {'X': 'intensity', 'Y': 'area'}
    assert viewer.checkedColumns() == (
        ['area'] if mode == 0 else ['area', 'intensity']
    )
    viewer.measurementsModeCombobox.setCurrentIndex(1-mode)
    assert viewer.checkedColumns() == (
        ['area', 'intensity'] if mode == 0 else ['area']
    )


def test_save_as_and_save_reuse_filepath(viewer, tmp_path, monkeypatch):
    configuration_data(viewer, tmp_path)
    first, second = tmp_path / 'first.json', tmp_path / 'second.json'
    dialog = mock_filename_dialog(monkeypatch, tmp_path, 'first', 'second.json')
    assert viewer.saveConfiguration()
    viewer.normalizeCheckbox.setChecked(True)
    assert viewer.saveConfiguration()
    assert dialog.call_count == 1
    assert json.loads(first.read_text())['normalize_measurements']
    assert viewer.saveConfigurationAs()
    assert viewer.configurationFilepath == str(second)
    assert second.exists()
    assert dialog.call_count == 2


@pytest.mark.parametrize('failure', ['invalid', 'missing_folder', 'missing_channel',
                                     'missing_segmentation', 'load_error'])
def test_loading_failure_preserves_current_data(
        viewer, tmp_path, monkeypatch, failure):
    configuration_data(viewer, tmp_path)
    configuration = viewer.configuration()
    if failure == 'invalid':
        configuration['measurements_axes'] = ['area', 'intensity']
    elif failure == 'missing_folder':
        configuration['images_paths'] = [str(tmp_path / 'missing')]
    channels = [] if failure == 'missing_channel' else ['phase']
    segmentations = [] if failure == 'missing_segmentation' else ['segm_custom']
    monkeypatch.setattr(
        measurements_viewer, 'get_channels_and_segm_endnames',
        Mock(return_value=(channels, segmentations))
    )
    monkeypatch.setattr(
        measurements_viewer_utils, 'load_position', Mock(side_effect=OSError('Cannot read'))
    )
    error = Mock()
    set_data = Mock()
    monkeypatch.setattr(viewer, 'configurationError', error)
    monkeypatch.setattr(viewer, 'setData', set_data)
    monkeypatch.setattr(
        viewer, 'selectConfigurationInput',
        lambda config: viewer.configurationInputOptions(config)[0]
    )
    monkeypatch.setattr(viewer, 'askSaveBeforeLoading', Mock(return_value=True))
    path = tmp_path / 'invalid.json'
    path.write_text(json.dumps(configuration))
    viewer.configurationFilepath = 'previous.json'
    assert not viewer.loadConfiguration(filepath=str(path))
    error.assert_called_once()
    set_data.assert_not_called()
    assert viewer.configurationFilepath == 'previous.json'


def test_save_cancel_and_write_failure(viewer, tmp_path, monkeypatch):
    configuration_data(viewer, tmp_path)
    mock_filename_dialog(monkeypatch, tmp_path, None)
    assert not viewer.saveConfiguration()
    assert viewer.configurationFilepath is None
    viewer.configurationFilepath = str(tmp_path / 'missing' / 'state.json')
    error = Mock()
    monkeypatch.setattr(viewer, 'configurationError', error)
    assert not viewer.saveConfiguration()
    error.assert_called_once()


@pytest.mark.slow
def test_startup_prompt_only_once(app, monkeypatch):
    prompt = Mock()
    monkeypatch.setattr(MeasurementsViewerWin, 'loadConfiguration', prompt)
    window = MeasurementsViewerWin()
    window.show()
    QTest.qWait(50)
    window.hide()
    window.show()
    QTest.qWait(50)
    prompt.assert_called_once_with()
    window.deleteLater()
    app.processEvents()


def test_startup_opens_configuration_picker_directly(app, monkeypatch):
    window = MeasurementsViewerWin()
    picker = Mock(return_value=None)
    monkeypatch.setattr(window, 'selectConfigurationFilepath', picker)
    window.show()
    QTest.qWait(50)
    # Skipping the picker starts the viewer without data
    picker.assert_called_once_with()
    assert not window.isDataLoaded
    window.deleteLater()
    app.processEvents()


@pytest.mark.slow
def test_geometry_is_restored_only_on_first_show(app, monkeypatch):
    window = MeasurementsViewerWin()
    window._startupConfigurationPrompted = True
    readSettings = Mock()
    monkeypatch.setattr(window, 'readSettings', readSettings)
    window.show()
    QTest.qWait(20)
    # E.g., restoring the window from minimized sends another show event
    window.hide()
    window.show()
    QTest.qWait(20)
    readSettings.assert_called_once_with(setting_suffix='_measurements_viewer')
    window.deleteLater()
    app.processEvents()


@pytest.mark.slow
@pytest.mark.parametrize('folder', [True, False])
def test_run_with_folder_skips_startup_prompt(app, tmp_path, monkeypatch, folder):
    from cellacdc import _run
    fakeApp = SimpleNamespace(exec_=lambda: None)
    monkeypatch.setattr(_run, '_setup_app', lambda *args, **kwargs: (fakeApp, None))
    openFolder = Mock()
    prompt = Mock()
    monkeypatch.setattr(MeasurementsViewerWin, 'openFolder', openFolder)
    monkeypatch.setattr(MeasurementsViewerWin, 'loadConfiguration', prompt)
    created = []
    init = MeasurementsViewerWin.__init__

    def recordInit(self, *args, **kwargs):
        init(self, *args, **kwargs)
        created.append(self)

    monkeypatch.setattr(MeasurementsViewerWin, '__init__', recordInit)
    measurements_viewer.run(str(tmp_path) if folder else None)
    QTest.qWait(20)
    if folder:
        openFolder.assert_called_once_with(folderpath=str(tmp_path))
        prompt.assert_not_called()
    else:
        openFolder.assert_not_called()
        prompt.assert_called_once_with()
    for window in created:
        window.hide()
        window.deleteLater()
    app.processEvents()


@pytest.mark.parametrize('has_configuration', [False, True])
@pytest.mark.parametrize('choice', ['skip', 'select', 'browse'])
def test_configuration_picker_buttons(
        viewer, tmp_path, monkeypatch, has_configuration, choice):
    from cellacdc import apps
    from qtpy.QtWidgets import QPushButton

    premade = tmp_path / 'premade'
    premade.mkdir()
    monkeypatch.setattr(
        measurements_viewer, 'MEASUREMENTS_CONFIGURATIONS_PATH', str(tmp_path)
    )
    monkeypatch.setattr(
        measurements_viewer, 'PREMADE_CONFIGURATIONS_PATH', str(premade)
    )
    path = tmp_path / 'saved.json'
    if has_configuration:
        path.write_text('{}')
    external = str(tmp_path / 'external.json')
    monkeypatch.setattr(
        measurements_viewer.QFileDialog, 'getOpenFileName',
        Mock(return_value=(external, ''))
    )
    original_dialog = apps.QTreeDialog
    # OK-button state seen while the dialog is open. Asserted after it
    # returns: an AssertionError raised inside the Qt slot would abort the
    # whole pytest process instead of failing this test
    seen = []

    def create_dialog(*args, **kwargs):
        dialog = original_dialog(*args, **kwargs)
        buttons = dialog.findChildren(QPushButton)
        skip, = [
            button for button in buttons
            if button.text() == 'Skip/New configuration'
        ]
        browse, = [
            button for button in buttons if button.text() == 'Browse configuration...'
        ]
        ok, = [
            button for button in buttons
            if button.text().replace('&', '').strip().lower() == 'ok'
        ]

        def interact():
            seen.append(ok.isEnabled())
            if choice == 'browse':
                browse.click()
            elif choice == 'select' and has_configuration:
                ok.click()
            else:
                skip.click()

        measurements_viewer.QTimer.singleShot(0, interact)
        return dialog

    monkeypatch.setattr(apps, 'QTreeDialog', create_dialog)
    expected = (
        external if choice == 'browse'
        else str(path) if choice == 'select' and has_configuration else None
    )
    assert viewer.selectConfigurationFilepath() == expected
    assert seen == [has_configuration]


@pytest.mark.parametrize('choice', ['cancel', 'dismiss', 'discard', 'save'])
@pytest.mark.parametrize('saved', [False, True])
def test_close_prompts_and_respects_cancel(
        viewer, tmp_path, monkeypatch, choice, saved):
    # Loaded data with a configuration that was never saved
    configuration_data(viewer, tmp_path)
    assert viewer.hasUnsavedConfiguration()
    # Cancel, Don't save, Save configuration...
    buttons = [object(), object(), object()]
    msg = Mock()
    msg.warning.return_value = buttons
    msg.cancel = choice in ('cancel', 'dismiss')
    msg.clickedButton = {
        'cancel': buttons[0], 'dismiss': None,
        'discard': buttons[1], 'save': buttons[2]
    }[choice]
    monkeypatch.setattr(measurements_viewer.widgets, 'myMessageBox', lambda **kw: msg)
    save = Mock(return_value=saved)
    monkeypatch.setattr(viewer, 'saveConfiguration', save)
    cleanup = Mock()
    monkeypatch.setattr(viewer, 'saveAnnotationOptions', cleanup)
    from qtpy.QtGui import QCloseEvent
    event = QCloseEvent()
    viewer.closeEvent(event)
    accepted = choice == 'discard' or (choice == 'save' and saved)
    msg.warning.assert_called_once()
    assert event.isAccepted() == accepted
    assert cleanup.called == accepted
    assert save.called == (choice == 'save')


@pytest.mark.slow
@pytest.mark.smoke
def test_configuration_restores_real_images_and_overlay(app, tmp_path, monkeypatch):
    import tifffile

    images = tmp_path / 'Position_1' / 'Images'
    images.mkdir(parents=True)
    image = np.arange(900, dtype=np.uint16).reshape(30, 30)
    tifffile.imwrite(images / 'test_phase.tif', image)
    tifffile.imwrite(images / 'test_GFP.tif', image * 2)
    labels = np.zeros((30, 30), dtype=np.uint16)
    labels[2:5, 3:7] = 1
    np.savez_compressed(images / 'test_segm_custom.npz', labels)
    pd.DataFrame({
        'frame_i': [0], 'Cell_ID': [1], 'area': [12.0], 'intensity': [20.0]
    }).to_csv(images / 'test_acdc_output_custom.csv', index=False)
    window = MeasurementsViewerWin()
    window._startupConfigurationPrompted = True
    errors = []
    monkeypatch.setattr(window, 'configurationError', lambda *args: errors.append(args))
    monkeypatch.setattr(
        window, 'selectConfigurationInput',
        lambda config: window.configurationInputOptions(config)[0]
    )
    configuration = {
        'version': 1, 'images_paths': [str(images)], 'channel': 'phase',
        'segm_endname': 'segm_custom', 'overlay_channels': ['GFP'],
        'visible_channels': ['GFP'], 'measurements_mode': 1,
        'time_course_columns': ['intensity'],
        'measurements_axes': {'X': 'intensity', 'Y': 'area'},
        'normalize_measurements': True
    }
    path = tmp_path / 'real.json'
    path.write_text(json.dumps(configuration))
    try:
        assert window.loadConfiguration(filepath=str(path)), errors
        assert not errors
        assert window.checkedOverlayChannels == {'GFP'}
        assert window.overlayButton.isChecked()
        assert window.allOverlayToolbuttons['GFP'].isChecked()
        assert not window.allOverlayToolbuttons['phase'].isChecked()
        np.testing.assert_array_equal(window.posData().img_data[0], image)
        np.testing.assert_array_equal(window.posData().segm_data[0], labels)
        # The ruler and the mirrored cursor stay above the overlay layers
        # (added when loading) and the annotation layers
        below = [
            window.rgbaImg1, window.labelsLayerImg1,
            window.ax1_contoursImageItem, window.searchedIDitemLeft,
            *(items[0] for items in window.overlayLayersItems.values()),
        ]
        topItems = (
            window.ax1_rulerPlotItem, window.ax1_rulerPlotItem.labelItem,
            window.ax1_rulerAnchorsItem, window.ax1_cursor,
        )
        for item in topItems:
            assert item.zValue() > max(other.zValue() for other in below)
        # Version 1 files are upgraded to version 2 on saving
        restored = window.configuration()
        assert restored['version'] == 2
        assert {
            key: value for key, value in restored.items()
            if key in configuration and key != 'version'
        } == {
            key: value for key, value in configuration.items()
            if key != 'version'
        }
        assert window.saveConfiguration()
        assert json.loads(path.read_text()) == restored
        assert len(window.measurementsPlot.listDataItems()) == 1
    finally:
        window.deleteLater()
        app.processEvents()


def make_timelapse_position(
        root, pos=1, SizeT=3, SizeZ=1, extra_cells=(), cca_frames=None
    ):
    """Write a real time-lapse position (cell 1 in every frame, plus the 
    `(frame_i, ID)` of `extra_cells`). Cell cycle annotations are written 
    for the frames in `cca_frames` and are NaN in the others."""
    import tifffile

    images = root / f'Position_{pos}' / 'Images'
    images.mkdir(parents=True)
    shape = (SizeT, SizeZ, 30, 30) if SizeZ > 1 else (SizeT, 30, 30)
    image = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    tifffile.imwrite(images / 'test_phase.tif', image)
    labels = np.zeros((SizeT, 30, 30), dtype=np.uint16)
    labels[:, 2:5, 3:7] = 1
    rows = [(frame_i, 1) for frame_i in range(SizeT)]
    for frame_i, ID in extra_cells:
        labels[frame_i, 15:19, 20:25] = ID
        rows.append((frame_i, ID))
    np.savez_compressed(images / 'test_segm.npz', labels)
    df = pd.DataFrame(rows, columns=['frame_i', 'Cell_ID'])
    df['area'] = 12.0
    if cca_frames is not None:
        isAnnotated = df['frame_i'].isin(cca_frames)
        df['cell_cycle_stage'] = np.where(isAnnotated, 'G1', None)
        for col in ('generation_num', 'relative_ID', 'emerg_frame_i', 
                'division_frame_i', 'corrected_on_frame_i', 'will_divide'):
            df[col] = np.where(isAnnotated, 1, np.nan)
        df['relationship'] = np.where(isAnnotated, 'mother', None)
        df['is_history_known'] = np.where(isAnnotated, True, None)
    df.to_csv(images / 'test_acdc_output.csv', index=False)
    pd.DataFrame({
        'Description': ['SizeT', 'SizeZ', 'basename'],
        'values': [str(SizeT), str(SizeZ), 'test_']
    }).to_csv(images / 'test_metadata.csv', index=False)
    return str(images)


@pytest.fixture
def real_viewer(app, monkeypatch):
    """Viewer that fails on exceptions raised inside Qt slots"""
    errors = []
    monkeypatch.setattr(
        sys, 'excepthook', lambda *exc_info: errors.append(exc_info)
    )
    window = MeasurementsViewerWin()
    window._startupConfigurationPrompted = True
    window.slotErrors = errors
    yield window
    window.deleteLater()
    app.processEvents()


def load_real(window, paths, **kwargs):
    window.setData(
        measurements_viewer.load_positions(paths, 'phase'), **kwargs
    )


def focus_in_active_window(app, window, widget):
    window.show()
    window.activateWindow()
    QTest.qWaitForWindowActive(window, 1000)
    widget.setFocus()
    app.processEvents()
    assert QApplication.focusWidget() is widget


@pytest.mark.slow
def test_arrow_keys_navigate_whatever_widget_has_the_focus(
        real_viewer, app, tmp_path
    ):
    # Buttons and check boxes use the arrow keys to move the focus (e.g.,
    # the first toolbar button, focused when the window is activated again
    # after a dialog): the viewer's event filter navigates instead
    images = make_timelapse_position(
        tmp_path, SizeT=4, extra_cells=[(frame_i, 2) for frame_i in range(4)]
    )
    load_real(real_viewer, [images])
    for widget in (real_viewer.zoomRectButton, real_viewer.followIDcheckbox):
        focus_in_active_window(app, real_viewer, widget)
        frame_i = real_viewer.frame_i
        QTest.keyClick(widget, Qt.Key_Right)
        assert real_viewer.frame_i == frame_i + 1
        QTest.keyClick(widget, Qt.Key_Left)
        assert real_viewer.frame_i == frame_i
        assert real_viewer.currID() == 1
        QTest.keyClick(widget, Qt.Key_Up)
        assert real_viewer.currID() == 2
        QTest.keyClick(widget, Qt.Key_Down)
        assert real_viewer.currID() == 1
        # The focus stays where it was
        assert QApplication.focusWidget() is widget
    assert not real_viewer.slotErrors


@pytest.mark.slow
def test_arrow_keys_stay_with_text_inputs(real_viewer, app, tmp_path):
    images = make_timelapse_position(tmp_path, SizeT=4)
    load_real(real_viewer, [images])
    lineEdit = QLineEdit(real_viewer.measurementsPlot.parentWidget())
    lineEdit.setText('abc')
    lineEdit.show()
    focus_in_active_window(app, real_viewer, lineEdit)
    lineEdit.setCursorPosition(0)
    QTest.keyClick(lineEdit, Qt.Key_Right)
    assert lineEdit.cursorPosition() == 1
    assert real_viewer.frame_i == 0
    assert not real_viewer.slotErrors


@pytest.mark.slow
def test_arrow_keys_do_nothing_while_the_window_is_disabled(
        real_viewer, app, tmp_path
    ):
    # `runInThread` disables the window during loads: application event
    # filters see its key presses before Qt drops them (disabled widgets)
    images = make_timelapse_position(
        tmp_path, SizeT=4, extra_cells=[(frame_i, 2) for frame_i in range(4)]
    )
    load_real(real_viewer, [images])
    focus_in_active_window(app, real_viewer, real_viewer.zoomRectButton)
    handle = real_viewer.windowHandle()
    recorded = {}

    def pressArrowKeys():
        recorded['enabled'] = real_viewer.isEnabled()
        for key in (Qt.Key_Right, Qt.Key_Up):
            QTest.keyClick(handle, key)
        recorded['frame_i'] = real_viewer.frame_i
        recorded['ID'] = real_viewer.currID()

    QTimer.singleShot(0, pressArrowKeys)
    real_viewer.runInThread(time.sleep, 0.2, text='Waiting...')
    assert recorded == {'enabled': False, 'frame_i': 0, 'ID': 1}
    # Enabled again (the focus was cleared): the keys navigate
    assert real_viewer.isEnabled()
    QTest.keyClick(handle, Qt.Key_Right)
    QTest.keyClick(handle, Qt.Key_Up)
    assert (real_viewer.frame_i, real_viewer.currID()) == (1, 2)
    assert not real_viewer.slotErrors


def close_popups(app):
    for _ in range(5):
        popup = QApplication.activePopupWidget()
        if popup is None:
            break
        popup.close()
        app.processEvents()


@pytest.mark.slow
def test_arrow_keys_switch_the_menus_of_an_open_menu(
        real_viewer, app, tmp_path
    ):
    # An open menu passes Left/Right on to its menu bar (switch menus),
    # while the focus widget is still the one focused before
    images = make_timelapse_position(tmp_path, SizeT=4)
    load_real(real_viewer, [images])
    menuBar = real_viewer.menuBar()
    # Windows/Linux menu bar (macOS: native, outside the window)
    menuBar.setNativeMenuBar(False)
    focus_in_active_window(app, real_viewer, real_viewer.zoomRectButton)
    fileAction, editAction = menuBar.actions()[:2]
    try:
        QTest.mouseClick(
            menuBar, Qt.LeftButton,
            pos=menuBar.actionGeometry(fileAction).center()
        )
        popup = QApplication.activePopupWidget()
        assert popup is fileAction.menu()
        assert QApplication.focusWidget() is real_viewer.zoomRectButton
        QTest.keyClick(popup, Qt.Key_Right)
        assert menuBar.activeAction() is editAction
        assert QApplication.activePopupWidget() is editAction.menu()
        assert real_viewer.frame_i == 0
        QTest.keyClick(editAction.menu(), Qt.Key_Left)
        assert menuBar.activeAction() is fileAction
        assert real_viewer.frame_i == 0
    finally:
        close_popups(app)
    assert not real_viewer.slotErrors


@pytest.mark.slow
def test_arrow_keys_navigate_the_menu_bar_after_alt(
        real_viewer, app, tmp_path
    ):
    images = make_timelapse_position(
        tmp_path, SizeT=4, extra_cells=[(frame_i, 2) for frame_i in range(4)]
    )
    load_real(real_viewer, [images])
    menuBar = real_viewer.menuBar()
    menuBar.setNativeMenuBar(False)
    altNavigation = menuBar.style().styleHint(
        QStyle.StyleHint.SH_MenuBar_AltKeyNavigation, None, menuBar
    )
    if not altNavigation:
        pytest.skip('The style has no keyboard navigation of the menu bar')
    focus_in_active_window(app, real_viewer, real_viewer.zoomRectButton)
    fileAction, editAction = menuBar.actions()[:2]
    handle = real_viewer.windowHandle()
    try:
        QTest.keyClick(handle, Qt.Key_Alt)
        assert QApplication.focusWidget() is menuBar
        assert menuBar.activeAction() is fileAction
        QTest.keyClick(handle, Qt.Key_Right)
        assert menuBar.activeAction() is editAction
        # Down opens the menu
        QTest.keyClick(handle, Qt.Key_Down)
        assert QApplication.activePopupWidget() is editAction.menu()
        assert (real_viewer.frame_i, real_viewer.currID()) == (0, 1)
    finally:
        close_popups(app)
    assert not real_viewer.slotErrors


def test_user_guide_limits_alt_menu_navigation_to_windows_and_linux():
    # macOS: the menu bar is native (outside the window) and its style has
    # no Alt navigation, so the arrow keys keep navigating frames/cells
    rst_path = os.path.join(
        os.path.dirname(measurements_viewer.__file__),
        'docs', 'source', 'measurements-viewer.rst'
    )
    with open(rst_path, encoding='utf-8') as file:
        paragraphs = file.read().split('\n\n')
    arrows, = [p for p in paragraphs if p.startswith('The arrow keys change')]
    arrows = ' '.join(arrows.split())
    assert 'menu bar after pressing :kbd:`Alt` (Windows and Linux;' in arrows
    assert 'on macOS the menu bar is outside the window' in arrows


def test_key_events_of_other_windows_are_not_for_the_window(app):
    # The main GUI's app event filter handles the arrow keys only for its
    # own windows (and the windows they own), not, e.g., for the viewer
    window = QWidget()
    child = QLineEdit(window)
    owned = QWidget(window, Qt.WindowType.Window)
    other = QWidget()
    otherChild = QLineEdit(other)
    try:
        assert qutils.is_key_event_for_window(window, window)
        assert qutils.is_key_event_for_window(window, child)
        assert qutils.is_key_event_for_window(window, owned)
        assert not qutils.is_key_event_for_window(window, other)
        assert not qutils.is_key_event_for_window(window, otherChild)
        # Seen on the QWindow first: the focus widget tells the target
        focus_in_active_window(app, other, otherChild)
        handle = other.windowHandle()
        assert not qutils.is_key_event_for_window(window, handle)
        assert qutils.is_key_event_for_window(other, handle)
    finally:
        for widget in (window, owned, other):
            widget.hide()
            widget.deleteLater()
        app.processEvents()


@pytest.mark.slow
def test_loading_shorter_timelapse_resets_navigation(real_viewer, tmp_path):
    longer = make_timelapse_position(tmp_path / 'A', SizeT=10)
    shorter = make_timelapse_position(tmp_path / 'B', SizeT=3)
    load_real(real_viewer, [longer])
    real_viewer.setFrame(9)
    assert real_viewer.navigateScrollBar.value() == 10
    load_real(real_viewer, [shorter])
    assert not real_viewer.slotErrors
    assert real_viewer.frame_i == real_viewer.posData().frame_i == 0
    assert real_viewer.navigateScrollBar.value() == 1
    assert real_viewer.navigateScrollBar.maximum() == 3


@pytest.mark.slow
def test_loading_channel_shorter_than_segmentation(real_viewer, tmp_path):
    # Image and segmentation are cut to the shorter of both (per-frame
    # structures such as allData_li have SizeT items)
    import tifffile

    images = make_timelapse_position(tmp_path, SizeT=5)
    tifffile.imwrite(
        os.path.join(images, 'test_short.tif'),
        np.zeros((3, 30, 30), dtype=np.uint16)
    )
    real_viewer.setData(
        measurements_viewer.load_display_positions([images], 'short')
    )
    posData = real_viewer.posData()
    assert posData.SizeT == 3
    assert len(posData.img_data) == len(posData.segm_data) == 3
    assert len(posData.allData_li) == 3
    assert real_viewer.navigateScrollBar.maximum() == 3
    assert not real_viewer.slotErrors


@pytest.mark.slow
def test_loading_smaller_zstack_resets_z_navigation(real_viewer, tmp_path):
    larger = make_timelapse_position(tmp_path / 'A', SizeT=2, SizeZ=10)
    smaller = make_timelapse_position(tmp_path / 'B', SizeT=2, SizeZ=4)
    load_real(real_viewer, [larger])
    assert real_viewer.posData().SizeZ == 10
    real_viewer.zSliceScrollBar.setSliderPosition(9)
    load_real(real_viewer, [smaller])
    assert not real_viewer.slotErrors
    assert real_viewer.zSliceScrollBar.maximum() == 3
    assert real_viewer.zSliceScrollBar.value() == 2


@pytest.mark.slow
def test_loading_2d_data_resets_viewed_plane(real_viewer, tmp_path):
    # A zy/zx plane left from a z-stack must not stay active (hidden) for
    # 2D data, where it would silently disable custom annotations.
    zstack = make_timelapse_position(tmp_path / 'A', SizeT=2, SizeZ=4)
    flat = make_timelapse_position(tmp_path / 'B', SizeT=2)
    load_real(real_viewer, [zstack])
    real_viewer.switchPlaneCombobox.blockSignals(True)
    real_viewer.switchPlaneCombobox.setPlane('zy')
    real_viewer.switchPlaneCombobox.blockSignals(False)
    real_viewer.zProjComboBox.setDisabled(True)
    assert real_viewer.switchPlaneCombobox.depthAxes() == 'x'
    load_real(real_viewer, [flat])
    assert not real_viewer.slotErrors
    assert real_viewer.posData().SizeZ == 1
    assert real_viewer.switchPlaneCombobox.depthAxes() == 'z'
    assert real_viewer.zProjComboBox.isEnabled()


def make_zstack_overlay_position(root, SizeT=2, SizeZ=5):
    """z-stack position with 3D segmentation and a GFP channel"""
    import tifffile

    images = make_timelapse_position(root, SizeT=SizeT, SizeZ=SizeZ)
    shape = (SizeT, SizeZ, 30, 30)
    tifffile.imwrite(
        os.path.join(images, 'test_GFP.tif'),
        np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    )
    labels = np.zeros(shape, dtype=np.uint16)
    labels[:, 1:4, 2:5, 3:7] = 1
    np.savez_compressed(os.path.join(images, 'test_segm.npz'), labels)
    return images


@pytest.mark.slow
def test_overlay_is_turned_off_on_zy_zx_planes(
        real_viewer, tmp_path, monkeypatch
    ):
    # As in the main GUI: `getOlImg` would index the overlay z-stack with
    # the y/x index of the z-slice scrollbar (IndexError on every update)
    load_real(real_viewer, [make_zstack_overlay_position(tmp_path)])
    assert real_viewer.isSegm3D
    monkeypatch.setattr(
        real_viewer, 'askSelectOverlayChannel', lambda: ['GFP']
    )
    button = real_viewer.overlayButton
    button.setChecked(True)
    assert real_viewer.configuration()['overlay_channels'] == ['GFP']
    real_viewer.switchPlaneCombobox.setPlane('zy')
    assert not real_viewer.slotErrors
    assert not button.isChecked() and not button.isEnabled()
    assert real_viewer.img1.image.shape == (5, 30)
    real_viewer.zSliceScrollBar.setSliderPosition(20)
    # zy -> zx keeps the state of the xy plane, also in configurations
    real_viewer.switchPlaneCombobox.setPlane('zx')
    assert real_viewer.configuration()['overlay_channels'] == ['GFP']
    real_viewer.switchPlaneCombobox.setPlane('xy')
    assert not real_viewer.slotErrors
    assert button.isChecked() and button.isEnabled()
    overlayImage = real_viewer.overlayLayersItems['GFP'][0].image
    assert overlayImage.shape == real_viewer.img1.image.shape == (30, 30)
    # Reloading the position resets the plane: the overlay stays off but
    # can be turned on again
    real_viewer.switchPlaneCombobox.setPlane('zy')
    real_viewer.loadPosition(0)
    assert real_viewer.switchPlaneCombobox.depthAxes() == 'z'
    assert button.isEnabled() and not button.isChecked()
    assert real_viewer.configuration()['overlay_channels'] == []
    button.setChecked(True)
    assert not real_viewer.slotErrors


@pytest.mark.slow
def test_loading_fewer_positions_resets_position(real_viewer, tmp_path):
    paths = [
        make_timelapse_position(tmp_path / 'A', pos=pos) for pos in (1, 2)
    ]
    load_real(real_viewer, paths)
    real_viewer.posCombobox.setCurrentIndex(1)
    assert real_viewer.pos_i == 1
    single = make_timelapse_position(tmp_path / 'B')
    load_real(real_viewer, [single], pos_i=1)
    assert not real_viewer.slotErrors
    assert real_viewer.pos_i == 0
    assert real_viewer.posData().images_path == single


# --- Status bar: what is loaded (as `gui.guiWin.setStatusBarLabel`) ---------
def make_overlay_position(root, pos=1, SizeT=3):
    """2D time-lapse position (`make_timelapse_position`) with a GFP
    channel"""
    import tifffile

    images = make_timelapse_position(root, pos=pos, SizeT=SizeT)
    tifffile.imwrite(
        os.path.join(images, 'test_GFP.tif'),
        np.ones((SizeT, 30, 30), dtype=np.uint16)
    )
    return images


def unwrap_tooltip(tooltip):
    """`tooltip` with the line breaks of `qutils.wrap_tooltip` (at spaces)
    and of `qutils.wrap_path` (after path separators, on indented lines)
    undone"""
    text = ''
    for line in tooltip.split('\n'):
        if text.endswith(('\\', '/')):
            text += line.strip()
        else:
            text = f'{text} {line.strip()}' if text else line.strip()
    return text


def unwrapped_tooltip(widget):
    """Tool tip of `widget` with its line breaks undone (`unwrap_tooltip`)"""
    return unwrap_tooltip(widget.toolTip())


def assert_tooltip_lines_fit(tooltip):
    lines = tooltip.split('\n')
    assert max(map(len, lines)) <= qutils.TOOLTIP_MAX_LINE_LENGTH, lines


def test_loaded_data_status_lists_what_is_displayed():
    images = os.path.join('data', 'exp', 'Position_2', 'Images')
    posData = SimpleNamespace(
        images_path=images, pos_foldername='Position_2',
        exp_path=os.path.join('data', 'exp'), basename='test_',
        user_ch_name='phase',
        segm_npz_path=os.path.join(images, 'test_segm_custom.npz')
    )
    status = measurements_viewer_utils.loaded_data_status
    text, tooltip = status(posData)
    assert text == (
        'Experiment: exp || Position_2 || Basename: test_ || '
        'Channel: phase || Segmentation file name: segm_custom.npz'
    )
    assert tooltip.split('\n') == [
        f'Images folder: {images}', 'Position_2', 'Basename: test_',
        'Channel: phase', 'Segmentation file name: segm_custom.npz'
    ]
    config = os.path.join('settings', 'fucci.json')
    text, tooltip = status(
        posData, pos_i=1, num_positions=3, channel='GFP',
        overlay_channels=['mCherry', 'DAPI'], configuration_filepath=config
    )
    assert text == (
        'Experiment: exp || Position_2 (2/3) || Basename: test_ || '
        'Channel: GFP || Overlay channels: mCherry, DAPI || '
        'Segmentation file name: segm_custom.npz || Configuration: fucci.json'
    )
    assert tooltip.split('\n')[0] == f'Images folder: {images}'
    assert tooltip.split('\n')[-1] == f'Configuration file: {config}'
    # Synthetic data without paths and names, no data
    assert status(SimpleNamespace()) == ('', '')
    assert status(None) == ('', '')


def test_loaded_data_status_breaks_long_paths_after_separators():
    # Review: `qutils.wrap_tooltip` never splits a path (no spaces), so the
    # tool tip had lines as wide as the Images folder and configuration
    # paths (a saved configuration's path alone is often > 80 characters)
    images = os.path.join(
        os.sep, 'data', *['long_experiment_folder_name']*4, 'exp',
        'Position_2', 'Images'
    )
    config = os.path.join(
        os.sep, 'Users', 'someone', 'acdc-appdata', '.acdc-settings',
        'measurements_viewer_configurations', 'fucci_cell_cycle.json'
    )
    posData = SimpleNamespace(
        images_path=images, pos_foldername='Position_2', basename='test_',
        user_ch_name='phase'
    )
    _, tooltip = measurements_viewer_utils.loaded_data_status(
        posData, configuration_filepath=config
    )
    assert_tooltip_lines_fit(tooltip)
    # Kept by the wrapping of the other lines
    assert qutils.wrap_tooltip(tooltip) == tooltip
    lines = tooltip.split('\n')
    assert lines[0] == 'Images folder:'
    position = lines.index('Position_2')
    config_label = lines.index('Configuration file:')
    imagesLines, configLines = lines[1:position], lines[config_label+1:]
    assert len(imagesLines) > 1 and len(configLines) > 1
    assert ''.join(line[4:] for line in imagesLines) == images
    assert ''.join(line[4:] for line in configLines) == config
    for line in imagesLines + configLines:
        assert line.startswith('    ') and line[4] != ' '
    for line in imagesLines[:-1] + configLines[:-1]:
        assert line.endswith(os.sep)
    assert f'Images folder: {images}' in unwrap_tooltip(tooltip)
    assert f'Configuration file: {config}' in unwrap_tooltip(tooltip)
    # Items that fit stay on one line
    path_lines = measurements_viewer_utils.path_tooltip_lines
    assert path_lines('Images folder', os.path.join('exp', 'Images')) == [
        f'Images folder: {os.path.join("exp", "Images")}'
    ]


@pytest.mark.slow
def test_status_bar_shows_the_loaded_data(real_viewer, tmp_path, capsys):
    # Reviewer request: as in the main GUI, the status bar tells what is
    # loaded, also after switching position; logged once per load/switch
    paths = [
        make_timelapse_position(tmp_path / 'exp', pos=pos) for pos in (1, 2)
    ]
    label = real_viewer.statusBarLabel
    assert label.text() == label.toolTip() == ''
    capsys.readouterr()
    load_real(real_viewer, paths)
    expected = (
        'Experiment: exp || Position_1 (1/2) || Basename: test_ || '
        'Channel: phase || Segmentation file name: segm.npz'
    )
    assert label.text() == expected
    assert f'Images folder: {paths[0]}' in unwrapped_tooltip(label)
    # The pytest tmp folder alone gives a path line longer than 80
    # characters: broken after separators
    assert_tooltip_lines_fit(label.toolTip())
    assert capsys.readouterr().out.count('Experiment: exp') == 1
    real_viewer.posCombobox.setCurrentIndex(1)
    expected = expected.replace('Position_1 (1/2)', 'Position_2 (2/2)')
    assert label.text() == expected
    assert f'Images folder: {paths[1]}' in unwrapped_tooltip(label)
    assert capsys.readouterr().out.count('Experiment: exp') == 1
    # Not logged per frame
    real_viewer.setFrame(1)
    assert 'Experiment: exp' not in capsys.readouterr().out
    assert label.text() == expected
    real_viewer.setData([])
    assert label.text() == label.toolTip() == ''
    assert not real_viewer.slotErrors


@pytest.mark.slow
def test_status_bar_lists_overlay_channels_and_keeps_the_loaded_data(
        real_viewer, tmp_path, monkeypatch
    ):
    load_real(real_viewer, [make_overlay_position(tmp_path / 'exp')])
    label = real_viewer.statusBarLabel
    loaded = label.text()
    assert loaded.startswith('Experiment: exp || Position_1 || ')
    assert 'Overlay' not in loaded
    monkeypatch.setattr(
        real_viewer, 'askSelectOverlayChannel', lambda: ['GFP']
    )
    real_viewer.overlayButton.setChecked(True)
    withOverlay = loaded.replace(
        'Channel: phase', 'Channel: phase || Overlay channels: GFP'
    )
    assert label.text() == withOverlay
    assert 'Overlay channels: GFP' in label.toolTip()
    # Channel toggled from the overlay button context menu
    # (`overlayChannelToggled` reads `sender()`)
    action, = [
        action for action in real_viewer.overlayContextMenu.actions()
        if action.text() == 'GFP'
    ]
    action.setChecked(False)
    assert label.text() == loaded
    action.setChecked(True)
    assert label.text() == withOverlay
    real_viewer.overlayButton.setChecked(False)
    assert label.text() == loaded
    # Exporting reports the file in a temporary message
    export = tmp_path / 'export.csv'
    monkeypatch.setattr(
        measurements_viewer.QFileDialog, 'getSaveFileName',
        Mock(return_value=(str(export), 'CSV (*.csv)'))
    )
    real_viewer.exportMeasurements()
    assert export.exists()
    assert label.text() == loaded
    assert str(export) in real_viewer.statusBar().currentMessage()
    # The mixin writes the color scheme restart notice into the label
    gui_viewer_tools = measurements_viewer.gui_viewer_tools
    monkeypatch.setattr(
        gui_viewer_tools._warnings, 'warnRestartCellACDCcolorModeToggled',
        Mock()
    )
    monkeypatch.setattr(
        gui_viewer_tools.load, 'rename_qrc_resources_file', Mock()
    )
    real_viewer.onToggleColorScheme()
    assert label.text() == loaded
    assert 'Restart' in real_viewer.statusBar().currentMessage()
    assert not real_viewer.slotErrors


@pytest.mark.slow
def test_status_bar_shows_the_configuration_file(
        real_viewer, tmp_path, monkeypatch, capsys
    ):
    images = make_overlay_position(tmp_path / 'exp')
    errors = []
    monkeypatch.setattr(
        real_viewer, 'configurationError', lambda *args: errors.append(args)
    )
    monkeypatch.setattr(
        real_viewer, 'selectConfigurationInput',
        lambda config: real_viewer.configurationInputOptions(config)[0]
    )
    configuration = {
        'version': 1, 'images_paths': [images], 'channel': 'phase',
        'segm_endname': 'segm', 'overlay_channels': ['GFP'],
        'visible_channels': ['GFP'], 'measurements_mode': 1,
        'time_course_columns': ['area'],
        'measurements_axes': {'X': '', 'Y': ''},
        'normalize_measurements': False
    }
    path = tmp_path / 'workflow.json'
    path.write_text(json.dumps(configuration))
    capsys.readouterr()
    assert real_viewer.loadConfiguration(filepath=str(path)), errors
    assert not errors
    expected = (
        'Experiment: exp || Position_1 || Basename: test_ || '
        'Channel: phase || Overlay channels: GFP || '
        'Segmentation file name: segm.npz || Configuration: workflow.json'
    )
    label = real_viewer.statusBarLabel
    assert label.text() == expected
    assert f'Configuration file: {path}' in unwrapped_tooltip(label)
    assert_tooltip_lines_fit(label.toolTip())
    # Logged once, with the configuration file
    out = capsys.readouterr().out
    assert out.count('Experiment: exp') == 1
    assert expected in out
    # Saved under another name
    mock_filename_dialog(monkeypatch, tmp_path, 'copy')
    assert real_viewer.saveConfigurationAs()
    assert label.text() == expected.replace('workflow.json', 'copy.json')
    assert not real_viewer.slotErrors


@pytest.mark.slow
def test_cca_table_on_frames_without_annotations(
        real_viewer, tmp_path, monkeypatch
    ):
    from cellacdc import widgets

    path = make_timelapse_position(tmp_path, cca_frames=[0])
    load_real(real_viewer, [path])
    assert real_viewer.viewCcaTableAction.isEnabled()
    real_viewer.viewCcaTable()
    assert real_viewer.ccaTableWin is not None
    real_viewer.setFrame(1)
    assert real_viewer.posData().cca_df is None
    assert real_viewer.ccaTableWin.tableView.model().rowCount() == 0
    real_viewer.onSigUpdateCcaTableWindow()
    real_viewer.ccaTableWin.close()
    real_viewer.ccaTableWin = None
    
    warnings = []
    monkeypatch.setattr(
        widgets.myMessageBox, 'warning', 
        lambda self, *args, **kwargs: warnings.append(args)
    )
    real_viewer.viewCcaTable()
    assert len(warnings) == 1
    assert real_viewer.ccaTableWin is None
    assert not real_viewer.slotErrors


@pytest.mark.slow
def test_cca_table_zoomed_on_cell_without_annotations(real_viewer, tmp_path):
    # The viewer's cca tables keep only cells with complete annotations,
    # while the zoomed IDs come from the label image
    path = make_timelapse_position(
        tmp_path, SizeT=2, extra_cells=[(0, 2), (1, 2)], cca_frames=[0]
    )
    csv = os.path.join(path, 'test_acdc_output.csv')
    df = pd.read_csv(csv)
    df.loc[(df.frame_i == 0) & (df.Cell_ID == 2), 'will_divide'] = np.nan
    df.to_csv(csv, index=False)
    load_real(real_viewer, [path])
    assert list(real_viewer.posData().cca_df.index) == [1]
    real_viewer.ax1.setRange(xRange=(12, 29), yRange=(10, 25), padding=0)
    assert real_viewer.getZoomIDs() == [2]
    real_viewer.viewCcaTable()
    assert real_viewer.ccaTableWin is not None
    assert real_viewer.ccaTableWin.tableView.model().rowCount() == 0
    assert not real_viewer.slotErrors


@pytest.mark.slow
def test_search_id_in_later_frame_with_unvisited_positions(
        real_viewer, tmp_path, monkeypatch
    ):
    paths = [
        make_timelapse_position(tmp_path, pos=1, extra_cells=[(2, 3)]),
        make_timelapse_position(tmp_path, pos=2),
    ]
    load_real(real_viewer, paths)
    assert not hasattr(real_viewer.data[1], 'allData_li')
    critical, notFound, found = [], [], []
    monkeypatch.setattr(real_viewer, 'workerCritical', critical.append)
    monkeypatch.setattr(real_viewer, 'warnIDnotFound', notFound.append)
    monkeypatch.setattr(
        real_viewer, 'askGoToFrameFoundID', 
        lambda ID, frame_i: found.append(frame_i) or True
    )
    real_viewer.onSearchId(3)
    assert not critical
    assert not notFound
    assert found == [2]
    assert real_viewer.posData().frame_i == 2
    assert not real_viewer.slotErrors


@pytest.mark.slow
def test_count_objects_snapshots_with_unvisited_positions(
        real_viewer, tmp_path
    ):
    paths = [
        make_timelapse_position(tmp_path, pos=pos, SizeT=1) for pos in (1, 2)
    ]
    load_real(real_viewer, paths)
    assert not hasattr(real_viewer.data[1], 'allData_li')
    real_viewer.countObjsButton.setChecked(True)
    assert not real_viewer.slotErrors
    assert real_viewer.countObjsWindow is not None
    counts = real_viewer.countObjectsSnapshots()
    assert counts['In all loaded positions'] == 2
    assert counts['In all visited positions (previous sessions)'] == 2
    real_viewer.countObjsButton.setChecked(False)
    assert not real_viewer.slotErrors


@pytest.mark.slow
def test_live_object_counts_compute_undisplayed_positions_once(
        real_viewer, tmp_path, monkeypatch
    ):
    # Live counts run on every image update: the regionprops of positions
    # without `allData_li` must not be recomputed every time
    paths = [
        make_timelapse_position(tmp_path, pos=pos, SizeT=1)
        for pos in (1, 2, 3)
    ]
    load_real(real_viewer, paths)
    undisplayed = real_viewer.data[1:]
    assert not any(hasattr(posData, 'allData_li') for posData in undisplayed)
    computed = []
    acdcRegionProps = real_viewer._acdcRegionProps

    def countingRegionProps(lab, *args, **kwargs):
        for posData in undisplayed:
            if np.shares_memory(lab, posData.segm_data):
                computed.append(posData)
        return acdcRegionProps(lab, *args, **kwargs)

    monkeypatch.setattr(real_viewer, '_acdcRegionProps', countingRegionProps)
    real_viewer.countObjsButton.setChecked(True)
    assert real_viewer.countObjsWindow.isVisible()
    for _ in range(2):
        real_viewer.updateImages()
    assert len(computed) == len(undisplayed)
    counts = real_viewer.countObjectsSnapshots()
    assert counts['In all loaded positions'] == 3
    real_viewer.countObjsButton.setChecked(False)
    assert not real_viewer.slotErrors


@pytest.mark.slow
def test_slideshow_shows_the_viewer_z_slice(real_viewer, tmp_path):
    # Without segmInfo.csv (`segmInfo_df` None). The slideshow's own z-slider
    # wrote `segmInfo_df` (AttributeError) and did not change the image
    load_real(real_viewer, [make_timelapse_position(tmp_path, SizeZ=3)])
    posData = real_viewer.posData()
    assert getattr(posData, 'segmInfo_df', None) is None
    real_viewer.slideshowButton.setChecked(True)
    assert not real_viewer.slotErrors
    win = real_viewer.slideshowWin
    assert not win.zSliceScrollBar.isVisibleTo(win)
    assert not win.zSliceScrollBar.isEnabled()
    zstack = posData.img_data[0]
    np.testing.assert_array_equal(win.img.image, zstack[1])
    real_viewer.zSliceScrollBar.setSliderPosition(0)
    np.testing.assert_array_equal(win.img.image, zstack[0])
    real_viewer.zProjComboBox.setCurrentText('max z-projection')
    np.testing.assert_array_equal(win.img.image, zstack.max(axis=0))
    # Not linked to the frames: it keeps its frame, but shows the z-slice
    win.linkWindowCheckbox.setChecked(False)
    real_viewer.setFrame(1)
    real_viewer.zProjComboBox.setCurrentText('single z-slice')
    np.testing.assert_array_equal(win.img.image, zstack[0])
    real_viewer.slideshowButton.setChecked(False)
    assert real_viewer.slideshowWin is None
    assert not real_viewer.slotErrors


@pytest.mark.slow
def test_slideshow_of_snapshots_survives_image_updates(real_viewer, tmp_path):
    # Single frames: the slideshow has no "Link" checkbox, while the viewer
    # calls `updateViewerWindow` on every `updateImages`
    paths = [
        make_timelapse_position(tmp_path, pos=pos, SizeT=1) for pos in (1, 2)
    ]
    load_real(real_viewer, paths)
    real_viewer.slideshowButton.setChecked(True)
    real_viewer.updateImages()
    real_viewer.loadPosition(1)
    assert real_viewer.pos_i == 1
    real_viewer.slideshowButton.setChecked(False)
    assert not real_viewer.slotErrors


@pytest.mark.slow
def test_restore_default_settings_un_inverts_black_white(
        real_viewer, tmp_path
    ):
    # `loadPreset` of the labels LUT rebinds `df_settings`: restoring from
    # the stale table re-inverted black/white (real LUT widgets needed)
    import pyqtgraph as pg
    path = make_timelapse_position(tmp_path, SizeT=2)
    load_real(real_viewer, [path])
    real_viewer._defaultSignalColorIndices['area'] = (
        measurements_viewer_utils.SIGNAL_COLORS.index('#F0E442')
    )
    column(real_viewer, 'area').setCheckState(Qt.Checked)
    real_viewer.invertBwAction.setChecked(True)
    real_viewer.restoreDefaultSettings()
    assert not real_viewer.invertBwAction.isChecked()
    assert not real_viewer.imgGrad.invertBwAction.isChecked()
    assert not real_viewer.labelsGrad.invertBwAction.isChecked()
    saved = pd.read_csv(real_viewer.settings_csv_path, index_col='setting')
    assert saved.at['is_bw_inverted', 'value'] == 'No'
    assert saved.at['labels_bkgrColor', 'value'] == '(25, 25, 25)'
    # Default colors follow the dark background again (yellow, not black)
    item, = real_viewer.measurementsPlot.listDataItems()
    assert pg.mkPen(item.opts['pen']).color().name() == '#f0e442'
    assert not real_viewer.slotErrors


@pytest.mark.slow
def test_line_weight_actions_after_restore_default_settings(
        real_viewer, tmp_path
    ):
    # The line weight helpers no longer read `self.sender()` (PyQt5 returned
    # a deleted sender outside slots: segfault in `restoreDefaultSettings`);
    # choosing a weight from the LUT menu must still apply and check it
    path = make_timelapse_position(tmp_path, SizeT=2)
    load_real(real_viewer, [path])
    real_viewer.restoreDefaultSettings()
    lut = real_viewer.imgGrad
    groups = {
        'contLineWeight': lut.contLineWightActionGroup,
        'mothBudLineSize': lut.mothBudLineWightActionGroup,
    }
    for key, group in groups.items():
        action = next(a for a in group.actions() if a.lineWeight == 3)
        action.trigger()
        assert int(real_viewer.df_settings.at[key, 'value']) == 3
        checked = [a.lineWeight for a in group.actions() if a.isChecked()]
        assert checked == [3]
    assert real_viewer.contLineWeight == 3
    assert real_viewer.mothBudLineWeight == 3
    assert not real_viewer.slotErrors


@pytest.mark.slow
def test_inverted_text_colors_survive_position_switch(real_viewer, tmp_path):
    # The viewer recreates the text items on every position switch
    # (`gui_setTextAnnotColors`): the S-phase/G1 cell cycle texts must keep
    # the inverted colors too, not only the IDs (regression)
    paths = [
        make_timelapse_position(tmp_path, pos=pos, SizeT=2, cca_frames=[0])
        for pos in (1, 2)
    ]
    load_real(real_viewer, paths)

    def textColors():
        colors = real_viewer.textAnnot[0].colors()
        return {
            name: tuple(colors[name][:3])
            for name in annotate.INVERTIBLE_COLOR_NAMES
        }

    dark = textColors()
    real_viewer.invertBwAction.setChecked(True)
    inverted = textColors()
    assert inverted == {
        name: tuple(255 - value for value in rgb)
        for name, rgb in dark.items()
    }
    real_viewer.loadPosition(1)
    assert textColors() == inverted
    real_viewer.invertBwAction.setChecked(False)
    assert textColors() == dark
    real_viewer.loadPosition(0)
    assert textColors() == dark
    assert not real_viewer.slotErrors


def test_text_resolution_toggle_keeps_text_above_annotations(
        real_viewer, tmp_path
    ):
    # `changeResolution` replaces the text item: it must keep z-value 1
    # above the annotation layers (0.5-0.6), as when it was created
    load_real(real_viewer, [make_timelapse_position(tmp_path, SizeT=2)])
    layers = (
        real_viewer.labelsLayerImg1, real_viewer.ax1_contoursImageItem,
        real_viewer.searchedIDitemLeft
    )
    top = max(item.zValue() for item in layers)
    item = real_viewer.textAnnot[0].item
    for _ in range(2):
        real_viewer.highLowResAction.trigger()
        assert real_viewer.textAnnot[0].item is not item
        item = real_viewer.textAnnot[0].item
        assert item.zValue() > top
    assert not real_viewer.slotErrors


def annotation_state(name, predictor=None):
    state = {
        'name': name, 'symbolColor': [255, 0, 0, 255], 'autoApply': True
    }
    if predictor is not None:
        state['predictor'] = predictor
    return state


@pytest.mark.slow
def test_opening_folder_keeps_saved_annotations_of_visited_cells(
        real_viewer, tmp_path, monkeypatch
    ):
    monkeypatch.setattr(
        real_viewer, 'askSaveBeforeLoading', Mock(return_value=True)
    )
    first = make_timelapse_position(tmp_path / 'A')
    second = make_timelapse_position(
        tmp_path / 'B', extra_cells=[(1, 2), (2, 2)]
    )
    tablepath = os.path.join(second, 'test_acdc_output.csv')
    table = pd.read_csv(tablepath)
    table['Flag'] = 0
    table[measurements_viewer.VISITED_COLUMN] = table['Cell_ID'] == 1
    table.to_csv(tablepath, index=False)
    real_viewer.loadImagesPaths([first], channel='phase')
    real_viewer.setCustomAnnotations([annotation_state(
        'Flag', {'type': 'equation', 'equation': 'area > 0'}
    )])
    assert real_viewer.acdc_df['Flag'].all()
    real_viewer.loadImagesPaths([second], channel='phase')
    acdc_df = real_viewer.acdc_df
    isCell1 = acdc_df.index.get_level_values('Cell_ID') == 1
    # The saved assignment of the visited cell is not predicted again ...
    assert not acdc_df.loc[isCell1, 'Flag'].any()
    # ... while a cell viewed for the first time is
    real_viewer.setSelectedID(2)
    acdc_df = real_viewer.acdc_df
    assert not acdc_df.loc[isCell1, 'Flag'].any()
    assert acdc_df.loc[~isCell1, 'Flag'].all()
    assert not real_viewer.slotErrors


@pytest.mark.slow
def test_unbound_mouse_side_button_over_image(real_viewer, tmp_path, app):
    load_real(real_viewer, [make_timelapse_position(tmp_path, SizeT=3)])
    assert real_viewer.isDataLoaded
    real_viewer.show()
    app.processEvents()
    view = real_viewer.graphLayout
    scenePos = real_viewer.img1.mapToScene(QPointF(15, 15))
    viewportPos = view.mapFromScene(scenePos)
    assert view.viewport().rect().contains(viewportPos)

    def sideButtonPress():
        QTest.mousePress(
            view.viewport(), Qt.MouseButton.XButton1,
            Qt.KeyboardModifier.NoModifier, viewportPos
        )
        QTest.mouseRelease(
            view.viewport(), Qt.MouseButton.XButton1,
            Qt.KeyboardModifier.NoModifier, viewportPos
        )
        app.processEvents()

    # No binding: ignored (the window's mousePressEvent does not accept
    # scene events)
    sideButtonPress()
    assert not real_viewer.slotErrors
    assert real_viewer.frame_i == 0
    # Bound buttons trigger the action
    real_viewer.mouseBindings['Next'] = Qt.MouseButton.XButton1
    sideButtonPress()
    assert real_viewer.frame_i == 1
    assert not real_viewer.slotErrors
    real_viewer.hide()


@pytest.mark.slow
def test_on_demand_ax1_items_stay_above_overlays_after_reload(
        real_viewer, tmp_path, monkeypatch
    ):
    import tifffile
    from cellacdc import apps

    monkeypatch.setattr(apps.ScaleBarPropertiesDialog, 'exec_', Mock())
    monkeypatch.setattr(apps.TimestampPropertiesDialog, 'exec_', Mock())
    paths = []
    for folder in ('A', 'B'):
        path = make_timelapse_position(tmp_path / folder, SizeT=3)
        tifffile.imwrite(
            os.path.join(path, 'test_GFP.tif'),
            np.ones((3, 30, 30), dtype=np.uint16)
        )
        paths.append(path)
    load_real(real_viewer, [paths[0]])
    tracksCheckbox = real_viewer.annotObjectTracksCheckbox(0)
    tracksCheckbox.blockSignals(True)
    tracksCheckbox.setChecked(True)
    tracksCheckbox.blockSignals(False)
    real_viewer.setFrame(2)
    real_viewer.annotateAllObjectTracks()
    # The overlay layers are re-added (z-value 0) on every load (which
    # removes the scale bar and the timestamp)
    load_real(real_viewer, [paths[1]])
    real_viewer.addScaleBar(True)
    real_viewer.addTimestamp(True)
    assert not real_viewer.slotErrors
    below = [
        real_viewer.rgbaImg1,
        *(items[0] for items in real_viewer.overlayLayersItems.values()),
    ]
    assert len(below) > 1
    onDemandItems = (
        real_viewer.tracksPlotItems[0], real_viewer.scaleBar.plotItem,
        real_viewer.scaleBar.labelItem, real_viewer.timestamp,
    )
    for item in onDemandItems:
        assert item.zValue() > max(other.zValue() for other in below)


def accept_scale_bar_and_timestamp_dialogs(monkeypatch):
    from cellacdc import apps

    dialogs = (apps.ScaleBarPropertiesDialog, apps.TimestampPropertiesDialog)
    for dialog in dialogs:
        monkeypatch.setattr(
            dialog, 'exec_', lambda self, *args, **kwargs: self.ok_cb()
        )


@pytest.mark.slow
def test_loading_data_removes_scale_bar_and_timestamp(
        real_viewer, tmp_path, monkeypatch
    ):
    # As the main GUI's `reInitGui`: they keep the pixel size and time
    # increment of the data they were added to (wrong values on new data)
    accept_scale_bar_and_timestamp_dialogs(monkeypatch)
    paths = [
        make_timelapse_position(tmp_path / folder, SizeT=3)
        for folder in ('A', 'B')
    ]
    load_real(real_viewer, [paths[0]])
    real_viewer.addScaleBarAction.setChecked(True)
    real_viewer.addTimestampAction.setChecked(True)
    items = (
        real_viewer.scaleBar.plotItem, real_viewer.scaleBar.labelItem,
        real_viewer.timestamp
    )
    assert all(item in real_viewer.ax1.items for item in items)
    load_real(real_viewer, [paths[1]])
    assert not real_viewer.slotErrors
    assert not real_viewer.addScaleBarAction.isChecked()
    assert not real_viewer.addTimestampAction.isChecked()
    assert not real_viewer.imgGrad.addScaleBarAction.isChecked()
    assert not real_viewer.imgGrad.addTimestampAction.isChecked()
    assert not any(item in real_viewer.ax1.items for item in items)
    # They can be added again for the new data
    real_viewer.addScaleBarAction.setChecked(True)
    assert real_viewer.scaleBar.plotItem in real_viewer.ax1.items
    assert not real_viewer.slotErrors


@pytest.mark.slow
def test_loading_data_while_scale_bar_and_timestamp_dialogs_are_open(
        real_viewer, tmp_path, monkeypatch
    ):
    # Their dialogs are not modal: data can be loaded while one is open
    from cellacdc import apps

    paths = [
        make_timelapse_position(tmp_path / folder, SizeT=3)
        for folder in ('A', 'B')
    ]
    load_real(real_viewer, [paths[0]])
    closedByLoad = []

    def loadWhileOpen(dialog, *args, **kwargs):
        load_real(real_viewer, [paths[1]])
        closedByLoad.append(not dialog.isVisible())
        dialog.ok_cb()

    dialogs = (apps.ScaleBarPropertiesDialog, apps.TimestampPropertiesDialog)
    for dialog in dialogs:
        monkeypatch.setattr(dialog, 'exec_', loadWhileOpen)
    real_viewer.addScaleBarAction.setChecked(True)
    real_viewer.addTimestampAction.setChecked(True)
    assert not real_viewer.slotErrors
    assert closedByLoad == [True, True]
    for action in (
            real_viewer.addScaleBarAction, real_viewer.addTimestampAction,
            real_viewer.imgGrad.addScaleBarAction,
            real_viewer.imgGrad.addTimestampAction
        ):
        assert not action.isChecked()
    items = (real_viewer.scaleBar.plotItem, real_viewer.timestamp)
    assert not any(item in real_viewer.ax1.items for item in items)
    assert real_viewer.scaleBarDialog is None
    assert real_viewer.timestampDialog is None


@pytest.mark.slow
def test_scale_bar_keeps_the_zoom(real_viewer, tmp_path, monkeypatch):
    # The bar is placed in the zoomed view: rescaling would move it to the
    # middle of the image and lose the user's zoom
    accept_scale_bar_and_timestamp_dialogs(monkeypatch)
    load_real(real_viewer, [make_timelapse_position(tmp_path, SizeT=2)])
    real_viewer.followIDcheckbox.setChecked(False)
    real_viewer.ax1.setRange(xRange=(5, 15), yRange=(5, 15), padding=0)
    # Settle the aspect ratio lock
    QTest.qWait(20)
    zoomed = np.array(real_viewer.ax1.viewRange())
    assert zoomed.min() > 0 and zoomed.max() < 25
    for checked in (True, False):
        real_viewer.addScaleBarAction.setChecked(checked)
        QTest.qWait(20)
        np.testing.assert_allclose(real_viewer.ax1.viewRange(), zoomed)
    assert not real_viewer.slotErrors


def formatted_brushes(window, column):
    item = next(
        item for item in window.measurementsPlot.listDataItems()
        if item.name() == column
    )
    return [brush.color().name() for brush in item.scatter.data['brush']]


@pytest.mark.slow
def test_time_point_and_annotation_edits_refresh_formatting(
        real_viewer, tmp_path
    ):
    import pyqtgraph as pg
    from cellacdc.timepoint_predictors import TimepointIdentifier

    load_real(real_viewer, [make_timelapse_position(tmp_path, SizeT=3)])
    identifier = TimepointIdentifier('Bud', (255, 0, 0), 'Q')
    real_viewer.addTimepointIdentifierItem(identifier)
    real_viewer.updateColumnsList()
    column(real_viewer, 'area').setCheckState(Qt.Checked)
    green = '#00ff00'
    real_viewer.setConditionalFormatting(
        'area', [{'condition': 'timepoint_Bud == 1', 'color': green}]
    )
    default = pg.mkColor(real_viewer.getSignalColor('area')).name()
    assert formatted_brushes(real_viewer, 'area') == [default]*3

    assert real_viewer.addTimepoint(identifier.uid, 1)
    assert formatted_brushes(real_viewer, 'area') == [default, green, default]

    line, = real_viewer.timepointLines
    line.setValue(3)  # frame numbers are 1-based
    real_viewer.timepointLineMoved(line)
    QTest.qWait(50)
    assert formatted_brushes(real_viewer, 'area') == [default, default, green]

    line, = real_viewer.timepointLines
    real_viewer.timepointLineRemoveRequested(line)
    assert formatted_brushes(real_viewer, 'area') == [default]*3

    real_viewer.setCustomAnnotations([annotation_state('Mitotic')])
    real_viewer.setConditionalFormatting(
        'area', [{'condition': 'Mitotic == 1', 'color': green}]
    )
    assert formatted_brushes(real_viewer, 'area') == [default]*3
    button, = real_viewer.customAnnotDict
    real_viewer.customAnnotShortcutTriggered(button)
    assert formatted_brushes(real_viewer, 'area') == [green]*3
    assert not real_viewer.slotErrors


def test_time_point_shortcut_over_plot_without_data(app, monkeypatch):
    window = MeasurementsViewerWin()
    window._startupConfigurationPrompted = True
    try:
        window.show()
        app.processEvents()
        identifier = window.defaultTimepointIdentifier()
        window.addTimepointIdentifierItem(identifier)
        plot = window.measurementsPlot
        assert plot.isVisible()
        center = plot.mapToGlobal(plot.rect().center())
        monkeypatch.setattr(measurements_viewer.QCursor, 'pos', lambda: center)
        window.timepointActionTriggered(identifier.uid)
        assert window.statusBar().currentMessage() == (
            'Load data to add time points.'
        )
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_formatting_dialog_warning_escapes_condition_and_error(
        app, monkeypatch
    ):
    # Threshold conditions contain '<': the rich-text warning must escape the
    # condition and the error so that neither is cut off as a fake tag.
    from cellacdc import widgets

    def validateCondition(condition):
        raise TypeError("'<' not supported between 'str' and 'int'")

    warning = Mock()
    monkeypatch.setattr(widgets.myMessageBox, 'warning', warning)
    dialog = measurements_viewer.ConditionalFormattingDialog(
        'area', [{'condition': 'area < 500', 'color': '#ff0000'}],
        validateCondition=validateCondition
    )
    try:
        dialog.ok_cb()
        assert dialog.cancel
        text = warning.call_args.args[2]
        assert '<code>area &lt; 500</code>' in text
        assert '&lt;&#x27; not supported' in text
    finally:
        dialog.deleteLater()


def test_formatting_dialog_is_disabled_while_condition_editor_runs(app):
    # The condition editor is not modal: removing or moving rules meanwhile
    # made editRule raise IndexError (aborting the app) or edit another rule
    states = []
    dialog = measurements_viewer.ConditionalFormattingDialog(
        'x', [{'condition': 'a > 1', 'color': 'lighter'},
              {'condition': 'b > 1', 'color': 'darker'}],
        editCondition=lambda text: states.append(dialog.isEnabled()) or 'b > 5'
    )
    try:
        assert dialog.isEnabled()
        dialog.rows[1]['editor'].click()
        assert states == [False]
        assert dialog.isEnabled()
        assert dialog.rows[1]['edit'].text() == 'b > 5'
        assert dialog.rows[0]['edit'].text() == 'a > 1'
    finally:
        dialog.deleteLater()


# --- Time course y axes ------------------------------------------------------

def open_column_menu(viewer, app, name):
    viewer.show()
    app.processEvents()
    pos = viewer.columnsListWidget.visualRect(proxy_index(viewer, name)).center()
    viewer.showColumnsContextMenu(pos)
    menu = QApplication.activePopupWidget()
    assert menu is not None
    return menu


def menu_action(menu, text):
    action, = [action for action in menu.actions() if action.text() == text]
    return action


def add_to_axis_actions(menu):
    return menu_action(menu, 'Add to y-axis of...').menu().actions()


def axes_columns(viewer):
    """Curves of the time course on each displayed y axis (left first)."""
    plotItem = viewer.measurementsPlot.getPlotItem()
    return [
        [item.name() for item in plotItem.listDataItems()],
        *[
            [item.name() for item in items]
            for items in plotItem.extraYAxesDataItems()
        ]
    ]


@pytest.mark.slow
def test_y_axis_menu_creates_and_joins_axes(viewer, app):
    column(viewer, 'intensity').setCheckState(Qt.Checked)
    plotItem = viewer.measurementsPlot.getPlotItem()
    menu = open_column_menu(viewer, app, 'intensity')
    assert [action.text() for action in menu.actions()][:4] == [
        'Plot', '', 'Create new y axis for this', 'Add to y-axis of...'
    ]
    newAxis = menu_action(menu, 'Create new y axis for this')
    assert newAxis.isEnabled()
    # The column's own axis is not listed: empty with a single axis
    assert not add_to_axis_actions(menu)
    assert not menu_action(menu, 'Add to y-axis of...').isEnabled()
    newAxis.trigger()
    menu.hide()
    assert viewer.timeCourseYAxes == {'intensity': 2}
    assert axes_columns(viewer) == [['area'], ['intensity']]
    extraAxis, = plotItem.shownExtraAxes()
    assert extraAxis.isVisible()
    assert extraAxis.labelText == 'intensity'
    assert viewer.measurementsPlot.getAxis('left').labelText == 'area'
    assert [label.text for _, label in viewer.measurementsLegend.items] == [
        'area', 'intensity'
    ]

    menu = open_column_menu(viewer, app, 'intensity')
    # Alone on its axis: it already has its own axis
    assert not menu_action(menu, 'Create new y axis for this').isEnabled()
    assert [action.text() for action in add_to_axis_actions(menu)] == ['area']
    menu.hide()
    menu = open_column_menu(viewer, app, 'area')
    addTo, = add_to_axis_actions(menu)
    assert addTo.text() == 'intensity'
    addTo.trigger()
    menu.hide()
    # area was the last column of axis 1: intensity's axis becomes axis 1
    assert viewer.timeCourseYAxes == {}
    # One axis left: shown on the left as before
    assert axes_columns(viewer) == [['area', 'intensity']]
    assert not plotItem.shownExtraAxes()
    assert not extraAxis.isVisible()
    assert viewer.measurementsPlot.getAxis('left').labelText == 'Value'
    menu = open_column_menu(viewer, app, 'area')
    assert not add_to_axis_actions(menu)
    menu.hide()


@pytest.mark.slow
def test_new_y_axis_plots_unplotted_column_and_is_kept(viewer, app):
    plotItem = viewer.measurementsPlot.getPlotItem()
    assert viewer.checkedColumns() == ['area']
    menu = open_column_menu(viewer, app, 'intensity')
    assert not add_to_axis_actions(menu)
    menu_action(menu, 'Create new y axis for this').trigger()
    menu.hide()
    assert viewer.checkedColumns() == ['area', 'intensity']
    assert axes_columns(viewer) == [['area'], ['intensity']]
    # Unplotting hides the axis but keeps the column's axis number
    column(viewer, 'intensity').setCheckState(Qt.Unchecked)
    assert viewer.timeCourseYAxes == {'intensity': 2}
    assert plotItem.extraYAxesCount == 0
    assert axes_columns(viewer) == [['area']]
    column(viewer, 'intensity').setCheckState(Qt.Checked)
    assert axes_columns(viewer) == [['area'], ['intensity']]
    # A new axis comes after all used numbers (the rightmost axis); the
    # lowest displayed axis is the left one
    viewer.plotOnNewYAxis('area')
    assert viewer.timeCourseYAxes == {'intensity': 2, 'area': 3}
    assert axes_columns(viewer) == [['intensity'], ['area']]
    # Comparison mode has no extra axes; they come back in the time course
    viewer.measurementsModeCombobox.setCurrentIndex(1)
    assert plotItem.extraYAxesCount == 0
    viewer.measurementsModeCombobox.setCurrentIndex(0)
    assert axes_columns(viewer) == [['intensity'], ['area']]
    # Renaming the column keeps its axis
    viewer.acdc_df = viewer.acdc_df.rename(columns={'area': 'size'})
    viewer.renameMeasurementColumnReferences('area', 'size')
    viewer.plotMeasurements()
    assert viewer.timeCourseYAxes == {'intensity': 2, 'size': 3}
    assert axes_columns(viewer) == [['intensity'], ['size']]


def test_new_y_axis_skips_numbers_of_unplotted_columns(viewer):
    viewer.acdc_df['volume'] = [1.0, 2.0, 3.0, 4.0]
    viewer.updateColumnsList()
    column(viewer, 'volume').setCheckState(Qt.Checked)
    assert viewer.checkedColumns() == ['area', 'volume']
    viewer.plotOnNewYAxis('intensity')
    assert viewer.timeCourseYAxes == {'intensity': 2}
    column(viewer, 'intensity').setCheckState(Qt.Unchecked)
    # intensity keeps axis 2 while unplotted: the new axis is not axis 2,
    # else plotting intensity again would put it on volume's axis
    viewer.plotOnNewYAxis('volume')
    column(viewer, 'intensity').setCheckState(Qt.Checked)
    assert viewer.timeCourseYAxes == {'intensity': 2, 'volume': 3}
    assert axes_columns(viewer) == [['area'], ['intensity'], ['volume']]


def legend_texts(viewer):
    """Legend entries of the measurements plot as displayed."""
    return [
        label.item.toPlainText() for _, label in viewer.measurementsLegend.items
    ]


def test_y_axis_labels_show_column_names_verbatim(viewer, tmp_path):
    # Axis labels and legend entries are HTML: names from the CSV file are
    # shown as text, while the curves keep the raw names (CSV export)
    viewer.acdc_df['size<5um'] = viewer.acdc_df['area']
    viewer.acdc_df['a&lt;b'] = viewer.acdc_df['intensity']
    viewer.updateColumnsList()
    column(viewer, 'size<5um').setCheckState(Qt.Checked)
    assert sorted(legend_texts(viewer)) == ['area', 'size<5um']
    assert sorted(
        item.name() for item in viewer.timeCourseDataItems()
    ) == ['area', 'size<5um']
    column(viewer, 'size<5um').setCheckState(Qt.Unchecked)
    viewer.plotOnNewYAxis('size<5um')
    viewer.plotOnNewYAxis('a&lt;b')
    plotItem = viewer.measurementsPlot.getPlotItem()
    assert [
        axis.label.toPlainText().strip()
        for axis in [viewer.measurementsPlot.getAxis('left'),
                     *plotItem.shownExtraAxes()]
    ] == ['area', 'size<5um', 'a&lt;b']
    expected = sorted(['area', 'size<5um', 'a&lt;b'])
    assert sorted(legend_texts(viewer)) == expected
    assert sorted(
        item.name() for item in viewer.timeCourseDataItems()
    ) == expected
    path = tmp_path / 'plot.csv'
    pg.exporters.CSVExporter(plotItem).export(fileName=str(path))
    header = path.read_text().splitlines()[0]
    assert 'size<5um_x' in header and 'a&lt;b_x' in header

    viewer.measurementsModeCombobox.setCurrentIndex(1)
    column(viewer, 'size<5um').setCheckState(Qt.Checked)
    column(viewer, 'a&lt;b').setCheckState(Qt.Checked)
    assert viewer.measurementsAxes == {'X': 'size<5um', 'Y': 'a&lt;b'}
    assert [
        viewer.measurementsPlot.getAxis(side).label.toPlainText().strip()
        for side in ('bottom', 'left')
    ] == ['size<5um', 'a&lt;b']


def test_phase_legend_shows_phase_names_verbatim(viewer, monkeypatch):
    monkeypatch.setattr(viewer, 'phaseColorMap', lambda: {
        'G1<S': QColor('#56b4e9'), 'a&lt;b': QColor('#009e73')
    })
    viewer.measurementsLegend.clear()
    viewer.comparisonPhaseBrushes(['G1<S', 'a&lt;b', ''])
    assert legend_texts(viewer) == ['G1<S', 'a&lt;b', 'No phase']


class _WheelEvent:
    """Wheel event as `ViewBox.wheelEvent` reads it."""
    def __init__(self, pos, delta=120):
        self._pos, self._delta = pos, delta

    def pos(self):
        return self._pos

    def delta(self):
        return self._delta

    def accept(self):
        pass

    def ignore(self):
        pass


def relative_change(before, after):
    span = before[1] - before[0]
    return [(after[0] - before[0]) / span, (after[1] - before[0]) / span]


@pytest.mark.slow
def test_y_axes_autoscale_separately_and_zoom_together(viewer, app):
    viewer.plotOnNewYAxis('intensity')
    viewer.show()
    app.processEvents()
    plotItem = viewer.measurementsPlot.getPlotItem()
    mainVb = plotItem.vb
    extraVb, = plotItem.shownExtraViewBoxes()
    # Same geometry, x-linked, under the main view box
    assert extraVb.sceneBoundingRect() == mainVb.sceneBoundingRect()
    assert extraVb.zValue() < mainVb.zValue()
    mainVb.updateAutoRange()
    extraVb.updateAutoRange()
    # Cell 1: area 10, 20 (left), intensity 100, 250 (right)
    (xMin, xMax), (yMin, yMax) = mainVb.viewRange()
    assert xMin <= 1 and xMax >= 2
    assert yMin <= 10 and 20 <= yMax < 100
    (exMin, exMax), (eyMin, eyMax) = extraVb.viewRange()
    assert (exMin, exMax) == pytest.approx((xMin, xMax))
    assert 20 < eyMin <= 100 and eyMax >= 250

    # Zooming in the plot area zooms every axis in the same way
    mainBefore, extraBefore = mainVb.viewRange()[1], extraVb.viewRange()[1]
    mainVb.wheelEvent(_WheelEvent(mainVb.rect().center()))
    mainAfter, extraAfter = mainVb.viewRange()[1], extraVb.viewRange()[1]
    assert mainAfter != mainBefore
    assert relative_change(extraBefore, extraAfter) == pytest.approx(
        relative_change(mainBefore, mainAfter)
    )
    assert not extraVb.autoRangeEnabled()[1]

    # "A" autoscales every axis
    plotItem.autoBtnClicked()
    assert all(mainVb.autoRangeEnabled())
    assert extraVb.autoRangeEnabled()[1]
    # Zooming on the right axis changes only that axis and shows "A"
    mainBefore = mainVb.viewRange()[1]
    extraVb.wheelEvent(_WheelEvent(extraVb.rect().center()), axis=1)
    assert mainVb.viewRange()[1] == mainBefore
    assert not extraVb.autoRangeEnabled()[1]
    plotItem.mouseHovering = True
    plotItem.updateButtons()
    assert plotItem.autoBtn.isVisible()
    # Every automatic rescale includes the extra axes
    viewer.autoscaleMeasurementsPlot()
    assert extraVb.autoRangeEnabled()[1]


def show_two_axes(viewer, app):
    """Show area (left) and intensity (right axis), both autoscaled; return
    the plot item, the main and the extra view box."""
    viewer.plotOnNewYAxis('intensity')
    viewer.show()
    app.processEvents()
    plotItem = viewer.measurementsPlot.getPlotItem()
    extraVb, = plotItem.shownExtraViewBoxes()
    plotItem.vb.updateAutoRange()
    extraVb.updateAutoRange()
    app.processEvents()
    return plotItem, plotItem.vb, extraVb


def send_mouse(viewer, eventType, scenePos, button=Qt.NoButton,
               buttons=Qt.NoButton):
    viewportPos = viewer.measurementsPlot.mapFromScene(scenePos)
    QApplication.sendEvent(
        viewer.measurementsPlot.viewport(),
        QMouseEvent(
            eventType, QPointF(viewportPos), button, buttons, Qt.NoModifier
        )
    )
    return viewportPos


def hover_point(viewer, vb, point):
    # pyqtgraph applies a new view range to the hit tests (transform of
    # the curves) at the next paint, which may not have happened offscreen
    plotItem = viewer.measurementsPlot.getPlotItem()
    for viewBox in (plotItem.vb, *plotItem.shownExtraViewBoxes()):
        viewBox.updateMatrix()
    scenePos = vb.mapViewToScene(QPointF(point.pos()))
    viewportPos = send_mouse(viewer, QEvent.MouseMove, scenePos)
    QTest.qWait(50)
    return viewportPos


@pytest.mark.slow
def test_points_of_extra_y_axis_show_tip_and_navigate(viewer, app, monkeypatch):
    errors = []
    monkeypatch.setattr(
        sys, 'excepthook', lambda *exc_info: errors.append(exc_info)
    )
    # Cell 1: the right-axis point of frame 2 (intensity 100, bottom) does
    # not lie on the left-axis point of frame 2 (area 20, top)
    viewer.acdc_df.loc[(0, 1), 'intensity'] = 250.0
    viewer.acdc_df.loc[(1, 1), 'intensity'] = 100.0
    plotItem, mainVb, extraVb = show_two_axes(viewer, app)
    (item,), = plotItem.extraYAxesDataItems()
    point = item.scatter.points()[-1]
    assert point.data()['frame_i'] == 1
    viewportPos = hover_point(viewer, extraVb, point)
    # The tip of every axis is on the main view box, above the others
    assert 'intensity: 100' in mainVb.toolTip()
    assert 'area' not in mainVb.toolTip()
    assert not extraVb.toolTip()
    # A replot (e.g. another cell) drops the tip of the removed points
    viewer.plotMeasurements()
    assert not mainVb.toolTip()
    del item, point
    assert viewer.frame_i == 0
    QTest.mouseClick(
        viewer.measurementsPlot.viewport(), Qt.LeftButton, pos=viewportPos
    )
    wait_navigation()
    assert not errors, errors
    assert viewer.frame_i == 1


@pytest.mark.slow
def test_coinciding_points_of_all_axes_share_the_tip(viewer, app):
    # Cell 1 over frames 1-3, middle values in the middle of both axes
    # (each axis fits its data), so the frame-2 points coincide
    viewer.acdc_df.loc[(1, 1), ['area', 'intensity']] = [15.0, 175.0]
    viewer.acdc_df.loc[(2, 1), ['area', 'intensity']] = [20.0, 250.0]
    plotItem, mainVb, extraVb = show_two_axes(viewer, app)
    (extraItem,), = plotItem.extraYAxesDataItems()
    point = extraItem.scatter.points()[1]
    assert point.data()['frame_i'] == 1
    hover_point(viewer, extraVb, point)
    tip = mainVb.toolTip()
    assert 'area: 15' in tip and 'intensity: 175' in tip
    # Leaving the points clears it
    send_mouse(viewer, QEvent.MouseMove, mainVb.mapViewToScene(QPointF(
        1.5, sum(mainVb.viewRange()[1]) / 2
    )))
    QTest.qWait(50)
    assert not mainVb.toolTip()


@pytest.mark.slow
def test_horizontal_drag_freezes_every_y_axis(viewer, app):
    viewer.rescalePlotCheckbox.setChecked(False)
    plotItem, mainVb, extraVb = show_two_axes(viewer, app)
    extraRange = extraVb.viewRange()[1]
    rect = mainVb.mapRectToScene(mainVb.rect())
    start = QPointF(rect.center().x(), rect.center().y())
    send_mouse(viewer, QEvent.MouseButtonPress, start, Qt.LeftButton,
               Qt.LeftButton)
    for dx in (10, 20, 30):
        QTest.qWait(30)
        send_mouse(viewer, QEvent.MouseMove, start + QPointF(dx, 0),
                   Qt.NoButton, Qt.LeftButton)
    send_mouse(viewer, QEvent.MouseButtonRelease, start + QPointF(30, 0),
               Qt.LeftButton, Qt.NoButton)
    app.processEvents()
    assert not any(mainVb.autoRangeEnabled())
    # The right axis keeps its range, as the left one (zoom kept)
    assert not extraVb.autoRangeEnabled()[1]
    extraVb.updateAutoRange()
    assert extraVb.viewRange()[1] == pytest.approx(extraRange)


@pytest.mark.slow
def test_view_all_and_right_click_on_extra_axis(viewer, app):
    plotItem, mainVb, extraVb = show_two_axes(viewer, app)
    extraAxis, = plotItem.shownExtraAxes()
    assert extraAxis.linkedView() is extraVb
    mainVb.setYRange(15, 16)
    extraVb.setYRange(500, 600)
    mainVb.menu.viewAll.trigger()
    # Every axis fits its data once (intensity 100, 250)
    low, high = extraVb.viewRange()[1]
    assert low <= 100 and 250 <= high < 500
    assert not extraVb.autoRangeEnabled()[1]
    # A right-click on the right axis opens the plot menu
    scenePos = extraAxis.mapToScene(extraAxis.rect().center())
    viewportPos = viewer.measurementsPlot.mapFromScene(scenePos)
    QTest.mouseClick(
        viewer.measurementsPlot.viewport(), Qt.RightButton, pos=viewportPos
    )
    app.processEvents()
    menu = QApplication.activePopupWidget()
    assert menu is mainVb.menu
    menu.hide()


@pytest.mark.slow
def test_extra_y_axes_follow_moves_of_main_view_box(viewer, app):
    # Opposite magnitudes: switching cell swaps the tick label widths of
    # the left and right axis, which moves the main view box without
    # resizing it
    viewer.acdc_df['area'] = [10.0, 20.0, 100000.0, 200000.0]
    viewer.acdc_df['intensity'] = [100000.0, 200000.0, 10.0, 20.0]
    plotItem, mainVb, extraVb = show_two_axes(viewer, app)
    x = mainVb.geometry().x()
    viewer.idScrollBar.setValue(1)
    for _ in range(3):
        mainVb.updateAutoRange()
        extraVb.updateAutoRange()
        app.processEvents()
        QTest.qWait(10)
    assert viewer.selectedID == 2
    assert mainVb.geometry().x() != x
    assert extraVb.geometry() == mainVb.geometry()
    frame = QPointF(2, 0)
    assert extraVb.mapViewToScene(frame).x() == pytest.approx(
        mainVb.mapViewToScene(frame).x(), abs=0.5
    )


def test_moving_last_left_column_merges_into_axis_1(viewer, app):
    viewer.acdc_df['volume'] = [1.0, 2.0, 3.0, 4.0]
    viewer.updateColumnsList()
    viewer.plotOnNewYAxis('intensity')
    viewer.addToYAxisOf('area', 2)
    # Axis 2 became the default axis 1
    assert viewer.timeCourseYAxes == {}
    assert [sorted(cols) for cols in axes_columns(viewer)] == [
        ['area', 'intensity']
    ]
    column(viewer, 'volume').setCheckState(Qt.Checked)
    assert [sorted(cols) for cols in axes_columns(viewer)] == [
        ['area', 'intensity', 'volume']
    ]
    # Moving a column that is not the last one of axis 1 creates no merge
    viewer.plotOnNewYAxis('volume')
    viewer.addToYAxisOf('area', 2)
    assert viewer.timeCourseYAxes == {'volume': 2, 'area': 2}


@pytest.mark.slow
def test_extra_y_axes_curves_are_exported_and_shaded_under(
        viewer, app, tmp_path):
    plotItem, mainVb, extraVb = show_two_axes(viewer, app)
    path = tmp_path / 'plot.csv'
    pg.exporters.CSVExporter(plotItem).export(fileName=str(path))
    header = path.read_text().splitlines()[0]
    assert 'area_x' in header and 'intensity_x' in header
    # The shading of unanalyzed frames is under the curves of every axis
    assert viewer.unanalyzedRegion.getViewBox() is extraVb
    assert extraVb.zValue() < mainVb.zValue()
    viewer.addToYAxisOf('intensity', 1)
    assert viewer.unanalyzedRegion.getViewBox() is mainVb
    # Removed curves are no longer exported
    assert len(plotItem.curves) == len(plotItem.listDataItems())


def test_log_mode_applies_to_extra_y_axes(viewer):
    viewer.plotOnNewYAxis('intensity')
    plotItem = viewer.measurementsPlot.getPlotItem()
    plotItem.ctrl.logYCheck.setChecked(True)
    try:
        extraAxis, = plotItem.shownExtraAxes()
        (item,), = plotItem.extraYAxesDataItems()
        assert extraAxis.logMode
        assert item.opts['logMode'][1]
        # Replotted curves keep it
        viewer.plotMeasurements()
        (item,), = plotItem.extraYAxesDataItems()
        assert item.opts['logMode'][1]
    finally:
        plotItem.ctrl.logYCheck.setChecked(False)
    assert not extraAxis.logMode


def test_other_transforms_autoscale_extra_y_axes(app):
    # Plot item alone, with plain curves: the viewer disables the transforms
    # that change the number of points (`disableUnsupportedPlotTransforms`)
    widget = pg.PlotWidget(plotItem=measurements_viewer._TimeCoursePlotItem())
    try:
        plotItem = widget.getPlotItem()
        frames = [1, 2, 3, 4]
        plotItem.plot(frames, [1.0, 2.0, 3.0, 4.0])
        plotItem.setExtraYAxesCount(1)
        plotItem.addExtraYAxisItem(
            0, pg.PlotDataItem(frames, [100.0, 400.0, 900.0, 1600.0])
        )
        extraVb, = plotItem.shownExtraViewBoxes()
        names = ['fftCheck', 'derivativeCheck', 'phasemapCheck']
        # "Subtract Mean" exists from pyqtgraph 0.14 (0.13.7 is supported)
        if hasattr(plotItem.ctrl, 'subtractMeanCheck'):
            names.append('subtractMeanCheck')
        for name in names:
            check = getattr(plotItem.ctrl, name)
            # Range set by hand, as after a pan or zoom in the plot area
            extraVb.setYRange(5000, 6000)
            assert not extraVb.autoRangeEnabled()[1]
            check.setChecked(True)
            try:
                assert extraVb.autoRangeEnabled()[1], name
                if name == 'phasemapCheck':
                    # x = y: the extra curve is outside the main x range
                    continue
                plotItem.vb.updateAutoRange()
                extraVb.updateAutoRange()
                assert extraVb.viewRange()[1][1] < 5000, name
            finally:
                check.setChecked(False)
    finally:
        widget.deleteLater()


@pytest.mark.parametrize('mode', [0, 1])
# pyqtgraph 0.13.7 computes 10**range of the linear range when log turns on
@pytest.mark.filterwarnings('ignore:overflow encountered in power')
def test_available_plot_transforms_work_on_viewer_curves(
        viewer, monkeypatch, mode):
    # FFT, dy/dx and Y vs Y' have fewer points than the data: pyqtgraph
    # applied the per-point brushes and hover data of the viewer's curves
    # to the transformed points and raised (with FFT on, on every replot)
    errors = []
    monkeypatch.setattr(
        sys, 'excepthook', lambda *excInfo: errors.append(excInfo[1])
    )
    plotItem = viewer.measurementsPlot.getPlotItem()
    ctrl = plotItem.ctrl
    for name in viewer.UNSUPPORTED_PLOT_TRANSFORMS:
        check = getattr(ctrl, name)
        assert not check.isEnabled() and not check.isChecked()
        assert 'Not available' in check.toolTip()
    if mode == 0:
        viewer.plotOnNewYAxis('intensity')
    else:
        comparison(viewer)
    extraViewBoxes = plotItem.shownExtraViewBoxes()
    assert len(extraViewBoxes) == (1 if mode == 0 else 0)
    checks = [
        check for check in ctrl.transformGroup.findChildren(QCheckBox)
        if check.isEnabled()
    ]
    assert ctrl.logYCheck in checks
    for check in checks:
        for vb in extraViewBoxes:
            # Range set by hand, as after a pan or zoom in the plot area
            # (small: the axis computes 10**range in log mode)
            vb.setYRange(50, 60)
        check.setChecked(True)
        try:
            viewer.plotMeasurements()
            assert not errors, (check.text(), errors)
            for vb in extraViewBoxes:
                assert vb.autoRangeEnabled()[1], check.text()
        finally:
            check.setChecked(False)
    assert not errors, errors


@pytest.mark.slow
@pytest.mark.parametrize('mode', [0, 1])
def test_plot_downsampling_options_are_disabled(viewer, monkeypatch, mode):
    # "Downsample" (factor 2: one point every 2 frames, or the mean of 2
    # points) and "Clip to View" (zoomed-in x range) plot fewer points than
    # the data: pyqtgraph applied the per-point brushes and hover data of the
    # viewer's curves to them and raised on every replot (and pan/zoom)
    errors = []
    monkeypatch.setattr(
        sys, 'excepthook', lambda *excInfo: errors.append(excInfo[1])
    )
    plotItem = viewer.measurementsPlot.getPlotItem()
    ctrl = plotItem.ctrl
    for name in viewer.UNSUPPORTED_PLOT_DOWNSAMPLING:
        widget = getattr(ctrl, name)
        assert not widget.isEnabled(), name
        assert 'Not available' in widget.toolTip(), name
    assert not ctrl.downsampleCheck.isChecked()
    assert not ctrl.clipToViewCheck.isChecked()
    # Keeps every point
    assert ctrl.maxTracesCheck.isEnabled()
    if mode == 0:
        viewer.plotOnNewYAxis('intensity')
    else:
        comparison(viewer)
    # What the options could still change: subsample by 2, clip to the view
    if ctrl.downsampleSpin.isEnabled():
        ctrl.downsampleSpin.setValue(2)
    for name in ('subsampleRadio', 'downsampleCheck', 'clipToViewCheck'):
        check = getattr(ctrl, name)
        if check.isEnabled():
            check.setChecked(True)
    assert plotItem.downsampleMode()[:2] == (1, False)
    assert not plotItem.clipToViewMode()
    plotItem.setXRange(1.5, 2.5, padding=0)
    app = QApplication.instance()
    app.processEvents()
    viewer.plotMeasurements()
    app.processEvents()
    assert not errors, errors
    items = viewer.timeCourseDataItems() if mode == 0 else (
        viewer.measurementsPlot.listDataItems()
    )
    assert len(items) == (2 if mode == 0 else 1)
    for item in items:
        assert len(item.scatter.points()) == len(item.xData) > 1


def test_extra_y_axes_work_without_subtract_mean_option(viewer, monkeypatch):
    # pyqtgraph 0.13.7 (still supported) has no "Subtract Mean" option
    viewer.plotOnNewYAxis('intensity')
    plotItem = viewer.measurementsPlot.getPlotItem()
    extraVb, = plotItem.shownExtraViewBoxes()
    monkeypatch.delattr(plotItem.ctrl, 'subtractMeanCheck', raising=False)
    monkeypatch.delattr(pg.PlotDataItem, 'setSubtractMeanMode', raising=False)
    monkeypatch.delattr(pg.PlotItem, 'updateSubtractMeanMode', raising=False)
    item = pg.PlotDataItem([1, 2], [5.0, 6.0])
    plotItem.addExtraYAxisItem(0, item)
    assert item.getViewBox() is extraVb
    assert item in plotItem.items and item in plotItem.curves
    assert plotItem.extraYAxesDataItems()[0][-1] is item
    extraVb.setYRange(1000, 2000)
    plotItem.updateSubtractMeanMode()
    assert extraVb.autoRangeEnabled()[1]
    # An item that cannot be added is not registered
    failing = pg.PlotDataItem([1, 2], [5.0, 6.0])
    monkeypatch.setattr(
        failing, 'setFftMode', Mock(side_effect=RuntimeError('failed'))
    )
    with pytest.raises(RuntimeError, match='failed'):
        plotItem.addExtraYAxisItem(0, failing)
    assert failing not in plotItem.items
    assert failing not in plotItem.curves
    assert failing not in plotItem.itemMeta
    assert failing not in plotItem.extraYAxesDataItems()[0]


@pytest.mark.slow
def test_formatting_and_normalization_on_extra_y_axis(viewer):
    # Both points of cell 1 (intensity 100, 250) match: the segment between
    # them is drawn by the rule, the main curve draws only the points
    viewer.conditionalFormatting['intensity'] = [
        {'condition': 'intensity > 50', 'color': '#0000ff'}
    ]
    viewer.plotOnNewYAxis('intensity')
    plotItem = viewer.measurementsPlot.getPlotItem()
    extraVb, = plotItem.shownExtraViewBoxes()
    extraAxis, = plotItem.shownExtraAxes()
    leftAxis = viewer.measurementsPlot.getAxis('left')
    for isNormalized in (False, True):
        viewer.normalizeCheckbox.setChecked(isNormalized)
        assert [item.name() for item in plotItem.listDataItems()] == ['area']
        (segments, curve), = plotItem.extraYAxesDataItems()
        assert isinstance(segments.opts['connect'], str)
        assert segments.opts['connect'] == 'pairs'
        assert curve.name() == 'intensity'
        assert not np.any(curve.opts['connect'])
        assert segments.getViewBox() is extraVb
        assert curve.getViewBox() is extraVb
        expected = [0.0, 1.0] if isNormalized else [100.0, 250.0]
        np.testing.assert_allclose(segments.yData, expected)
        suffix = ' (normalized)' if isNormalized else ''
        assert leftAxis.labelText == 'area' + suffix
        assert extraAxis.labelText == 'intensity' + suffix


@pytest.mark.slow
def test_phase_band_matches_extra_y_axes_width(viewer, app):
    viewer.show()
    app.processEvents()
    phaseRight = viewer.phaseBandPlot.getAxis('right')
    assert not phaseRight.isVisible()
    viewer.plotOnNewYAxis('intensity')
    app.processEvents()
    extraAxis, = viewer.measurementsPlot.getPlotItem().shownExtraAxes()
    assert extraAxis.width() > 0
    assert phaseRight.isVisible()
    # Fixed width (the band itself is hidden without phases)
    assert phaseRight.maximumWidth() == pytest.approx(extraAxis.width(), abs=1)
    assert phaseRight.minimumWidth() == phaseRight.maximumWidth()
    viewer.addToYAxisOf('intensity', 1)
    app.processEvents()
    assert not phaseRight.isVisible()
