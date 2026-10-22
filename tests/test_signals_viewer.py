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

from cellacdc.signals_viewer import SignalsViewerWin


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def viewer(app, monkeypatch):
    window = SignalsViewerWin()
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
    window.plotSignals()
    yield window
    window.deleteLater()
    app.processEvents()


def wait_navigation():
    # Navigation after clicking a point is deferred with a QTimer
    QTest.qWait(300)


def comparison(viewer):
    viewer.signalsModeCombobox.setCurrentIndex(1)
    viewer.columnsModel.item(0).setCheckState(Qt.Checked)
    viewer.columnsModel.item(1).setCheckState(Qt.Checked)
    return viewer.signalsPlot.listDataItems()[0]


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
    assert viewer.columnsModel.item(0).data(Qt.AccessibleDescriptionRole) == 'X'
    assert viewer.columnsModel.item(1).data(Qt.AccessibleDescriptionRole) == 'Y'
    assert viewer.columnsModel.item(0).isCheckable()
    assert viewer.checkedColumns() == ['area', 'intensity']
    assert viewer.signalsPlot.getAxis('bottom').labelText == 'area'
    assert viewer.signalsPlot.getAxis('left').labelText == 'intensity'


def test_switching_modes_preserves_time_course_selection(viewer):
    comparison(viewer)
    viewer.signalsModeCombobox.setCurrentIndex(0)
    item, = viewer.signalsPlot.listDataItems()
    np.testing.assert_array_equal(item.xData, [1, 2])
    np.testing.assert_array_equal(item.yData, [10, 20])
    assert viewer.checkedColumns() == ['area']
    assert viewer.currentFrameLine.isVisible()
    assert viewer.signalsPlot.getAxis('bottom').labelText == 'Frame n.'
    assert viewer.signalsPlot.getAxis('left').labelText == 'Value'
    assert not viewer.timeCourseColumnsWidget.isHidden()
    assert viewer.columnsModel.item(0).isCheckable()
    assert not viewer.columnsModel.item(0).icon().isNull()
    assert viewer.columnsModel.item(1).icon().isNull()


@pytest.mark.parametrize('mode', [0, 1])
def test_points_keep_metadata_and_hover_details(viewer, mode):
    comparison(viewer)
    viewer.signalsModeCombobox.setCurrentIndex(mode)
    viewer.normalizeCheckbox.setChecked(True)
    item, = viewer.signalsPlot.listDataItems()
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


def test_time_course_click_navigates_to_measurement(viewer, app):
    item, = viewer.signalsPlot.listDataItems()
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


def test_mode_switching_without_data(app):
    window = SignalsViewerWin()
    window.signalsModeCombobox.setCurrentIndex(1)
    assert not window.signalsPlot.listDataItems()
    assert window.signalsPlot.plotItem.titleLabel.text == 'No data loaded'
    window.signalsModeCombobox.setCurrentIndex(0)
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
    viewer.signalsModeCombobox.setCurrentIndex(mode)
    viewer.show()
    app.processEvents()
    viewer.signalsPlot.autoRange()
    app.processEvents()
    item, = viewer.signalsPlot.listDataItems()
    point = item.scatter.points()[-1]
    scenePos = viewer.signalsPlot.plotItem.vb.mapViewToScene(
        QPointF(point.pos())
    )
    viewportPos = viewer.signalsPlot.mapFromScene(scenePos)
    QApplication.sendEvent(
        viewer.signalsPlot.viewport(),
        QMouseEvent(
            QEvent.MouseMove, QPointF(viewportPos),
            Qt.NoButton, Qt.NoButton, Qt.NoModifier
        )
    )
    QTest.qWait(150)
    app.processEvents()
    tip = viewer.signalsPlot.plotItem.vb.toolTip()
    assert f'Cell ID: {point.data()["Cell_ID"]}' in tip
    assert f'Frame: {point.data()["frame_i"]+1}' in tip
    # Release the test's references so the replot really deletes the item
    del item, point
    QTest.mouseClick(
        viewer.signalsPlot.viewport(), Qt.LeftButton, pos=viewportPos
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
    viewer.plotSignals()
    assert not viewer.signalsPlot.listDataItems()
    assert viewer.signalsPlot.plotItem.titleLabel.text == (
        'No finite paired measurements'
    )


def test_axes_refresh_preserves_choices_and_clears_missing_columns(viewer):
    comparison(viewer)
    viewer.columnsModel.item(0).setCheckState(Qt.Unchecked)
    viewer.columnsModel.item(1).setCheckState(Qt.Unchecked)
    viewer.columnsModel.item(1).setCheckState(Qt.Checked)
    viewer.columnsModel.item(0).setCheckState(Qt.Checked)
    viewer.updateColumnsList()
    assert viewer.signalsAxes == {'X': 'intensity', 'Y': 'area'}
    viewer.acdc_df = viewer.acdc_df.drop(columns='area')
    viewer.updateColumnsList()
    viewer.plotSignals()
    assert viewer.signalsAxes == {'X': 'intensity', 'Y': ''}
    assert viewer.columnsModel.item(0).data(Qt.AccessibleDescriptionRole) == 'X'
    assert viewer.checkedColumns() == ['intensity']
    assert not viewer.signalsPlot.listDataItems()
    viewer.acdc_df = None
    viewer.updateColumnsList()
    viewer.plotSignals()
    assert viewer.signalsAxes == {'X': '', 'Y': ''}
    assert not viewer.signalsPlot.listDataItems()


def test_filtered_checkbox_toggles_axis_and_preserves_time_course(viewer, app):
    comparison(viewer)
    viewer.show()
    viewer.columnsSearchLineEdit.setText('intensity')
    app.processEvents()
    index = viewer.columnsSearchProxy.index(0, 0)
    assert index.data() == 'intensity'
    rect = viewer.columnsListWidget.visualRect(index)
    pos = rect.center()
    pos.setX(rect.left()+8)

    def click():
        QTest.mouseClick(viewer.columnsListWidget.viewport(), Qt.LeftButton, pos=pos)
        app.processEvents()

    click()
    assert viewer.signalsAxes == {'X': 'area', 'Y': ''}
    assert viewer.checkedColumns() == ['area']
    assert not viewer.signalsPlot.listDataItems()
    click()
    assert viewer.signalsAxes == {'X': 'area', 'Y': 'intensity'}
    assert viewer.checkedColumns() == ['area', 'intensity']
    viewer.signalsModeCombobox.setCurrentIndex(0)
    item, = viewer.signalsPlot.listDataItems()
    np.testing.assert_array_equal(item.yData, [10, 20])


def test_comparison_context_menu_toggles_check(viewer, app):
    comparison(viewer)
    viewer.show()
    viewer.columnsSearchLineEdit.setText('intensity')
    app.processEvents()
    index = viewer.columnsSearchProxy.index(0, 0)
    pos = viewer.columnsListWidget.visualRect(index).center()
    viewer.showColumnsContextMenu(pos)
    menu = QApplication.activePopupWidget()
    assert menu is not None
    actions = menu.actions()
    assert [action.text() for action in actions] == ['Plot']
    assert actions[0].isChecked()
    actions[0].trigger()
    menu.hide()
    assert viewer.signalsAxes == {'X': 'area', 'Y': ''}
    assert viewer.columnsModel.item(1).icon().isNull()
    assert viewer.checkedColumns() == ['area']


def test_modes_have_independent_checks(viewer):
    viewer.signalsModeCombobox.setCurrentIndex(1)
    assert viewer.checkedColumns() == []
    assert viewer.signalsAxes == {'X': '', 'Y': ''}
    assert all(
        viewer.columnsModel.item(i).icon().isNull()
        for i in range(viewer.columnsModel.rowCount())
    )
    viewer.columnsModel.item(1).setCheckState(Qt.Checked)
    assert viewer.signalsAxes == {'X': 'intensity', 'Y': ''}
    viewer.signalsModeCombobox.setCurrentIndex(0)
    assert viewer.checkedColumns() == ['area']
    viewer.signalsModeCombobox.setCurrentIndex(1)
    assert viewer.checkedColumns() == ['intensity']
    assert viewer.columnsModel.item(1).data(Qt.AccessibleDescriptionRole) == 'X'


def test_mode_change_repaints_indicators(viewer, monkeypatch):
    viewer.signalsModeCombobox.setCurrentIndex(1)
    viewer.columnsModel.item(1).setCheckState(Qt.Checked)
    viewport = viewer.columnsListWidget.viewport()
    repaint = Mock()
    monkeypatch.setattr(viewport, 'update', repaint)
    itemChanged = Mock()
    viewer.columnsModel.itemChanged.connect(itemChanged)
    for mode, icons in ((0, {'area'}), (1, {'intensity'})):
        repaint.reset_mock()
        viewer.signalsModeCombobox.setCurrentIndex(mode)
        repaint.assert_called()
        assert {
            viewer.columnsModel.item(i).text()
            for i in range(viewer.columnsModel.rowCount())
            if not viewer.columnsModel.item(i).icon().isNull()
        } == icons
    itemChanged.assert_not_called()
    assert viewer.signalsAxes == {'X': 'intensity', 'Y': ''}


def test_comparison_requires_unchecking_before_replacing_axis(viewer):
    viewer.acdc_df['third'] = [1, 2, 3, 4]
    viewer.updateColumnsList()
    comparison(viewer)
    viewer.columnsModel.item(2).setCheckState(Qt.Checked)
    assert viewer.columnsModel.item(2).checkState() == Qt.Unchecked
    assert 'Two signals are already selected' in viewer.statusBar().currentMessage()
    viewer.columnsModel.item(0).setCheckState(Qt.Unchecked)
    viewer.columnsModel.item(2).setCheckState(Qt.Checked)
    assert viewer.signalsAxes == {'X': 'third', 'Y': 'intensity'}
    assert viewer.columnsModel.item(2).data(Qt.AccessibleDescriptionRole) == 'X'


@pytest.mark.parametrize('mode', [0, 1])
@pytest.mark.parametrize('filtered', [False, True])
def test_indicator_replaces_tick_without_shifting_text(viewer, app, mode, filtered):
    viewer.signalsModeCombobox.setCurrentIndex(mode)
    viewer.show()
    if filtered:
        viewer.columnsSearchLineEdit.setText('intensity')
    app.processEvents()
    row = 0 if filtered else 1
    index = viewer.columnsSearchProxy.index(row, 0)
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

    def click_name(row):
        index = viewer.columnsSearchProxy.index(row, 0)
        rect = viewer.columnsListWidget.visualRect(index)
        pos = rect.center()
        # Well to the right of the fixed-width indicator, on the name
        pos.setX(rect.left() + viewer.columnsHighlightDelegate.indicatorWidth + 20)
        QTest.mouseClick(viewer.columnsListWidget.viewport(), Qt.LeftButton, pos=pos)
        app.processEvents()

    # Time course: add/remove plotted columns
    click_name(1)
    assert viewer.checkedColumns() == ['area', 'intensity']
    assert len(viewer.signalsPlot.listDataItems()) == 2
    click_name(0)
    assert viewer.checkedColumns() == ['intensity']

    # Comparison: set X then Y, clicking again frees the axis
    viewer.signalsModeCombobox.setCurrentIndex(1)
    click_name(1)
    assert viewer.signalsAxes == {'X': 'intensity', 'Y': ''}
    click_name(0)
    assert viewer.signalsAxes == {'X': 'intensity', 'Y': 'area'}
    assert len(viewer.signalsPlot.listDataItems()) == 1
    click_name(1)
    assert viewer.signalsAxes == {'X': '', 'Y': 'area'}

    viewer.signalsModeCombobox.setCurrentIndex(0)
    assert viewer.checkedColumns() == ['intensity']


def test_column_names_cannot_be_edited(viewer, app):
    viewer.show()
    app.processEvents()
    assert viewer.columnsListWidget.editTriggers() == QAbstractItemView.NoEditTriggers
    for i in range(viewer.columnsModel.rowCount()):
        assert not viewer.columnsModel.item(i).isEditable()
    index = viewer.columnsSearchProxy.index(0, 0)
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
def test_normalization_handles_nonfinite_and_constant_signals(values, expected):
    np.testing.assert_allclose(
        SignalsViewerWin._normalize(np.array(values, dtype=float)),
        expected, equal_nan=True
    )
