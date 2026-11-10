import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from qtpy.QtCore import QEvent, QPointF, Qt
from qtpy.QtGui import QMouseEvent
from qtpy.QtTest import QTest
from qtpy.QtWidgets import (
    QApplication, QAbstractItemView, QStyle, QStyleOptionViewItem, QLineEdit,
)

from cellacdc.measurements_viewer import MeasurementsViewerWin
from cellacdc import measurements_viewer, widgets


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

    # Switching back from comparison mode resets the frame range
    viewer.measurementsModeCombobox.setCurrentIndex(1)
    viewer.measurementsModeCombobox.setCurrentIndex(0)
    xMin, xMax = viewer.measurementsPlot.viewRange()[0]
    assert xMin < 1 and xMax > 3


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
    monkeypatch.setattr(measurements_viewer, 'load_position', load_position)
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
        measurements_viewer, 'load_position', Mock(side_effect=OSError('Cannot read'))
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


def annotation_state(name, predictor=None):
    state = {
        'name': name, 'symbolColor': [255, 0, 0, 255], 'autoApply': True
    }
    if predictor is not None:
        state['predictor'] = predictor
    return state


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
    real_viewer.addScaleBar(True)
    real_viewer.addTimestamp(True)
    tracksCheckbox = real_viewer.annotObjectTracksCheckbox(0)
    tracksCheckbox.blockSignals(True)
    tracksCheckbox.setChecked(True)
    tracksCheckbox.blockSignals(False)
    real_viewer.setFrame(2)
    real_viewer.annotateAllObjectTracks()
    # The overlay layers are re-added (z-value 0) on every load
    load_real(real_viewer, [paths[1]])
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


def formatted_brushes(window, column):
    item = next(
        item for item in window.measurementsPlot.listDataItems()
        if item.name() == column
    )
    return [brush.color().name() for brush in item.scatter.data['brush']]


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
