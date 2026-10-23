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
from cellacdc import measurements_viewer


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def viewer(app, monkeypatch):
    monkeypatch.setattr(MeasurementsViewerWin, 'askLoadConfiguration', Mock())
    window = MeasurementsViewerWin()
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


def wait_navigation():
    # Navigation after clicking a point is deferred with a QTimer
    QTest.qWait(300)


def comparison(viewer):
    viewer.measurementsModeCombobox.setCurrentIndex(1)
    viewer.columnsModel.item(0).setCheckState(Qt.Checked)
    viewer.columnsModel.item(1).setCheckState(Qt.Checked)
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
    assert viewer.columnsModel.item(0).data(Qt.AccessibleDescriptionRole) == 'X'
    assert viewer.columnsModel.item(1).data(Qt.AccessibleDescriptionRole) == 'Y'
    assert viewer.columnsModel.item(0).isCheckable()
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
    assert viewer.columnsModel.item(0).isCheckable()
    assert not viewer.columnsModel.item(0).icon().isNull()
    assert viewer.columnsModel.item(1).icon().isNull()


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
    viewer.columnsModel.item(0).setCheckState(Qt.Unchecked)
    viewer.columnsModel.item(1).setCheckState(Qt.Unchecked)
    viewer.columnsModel.item(1).setCheckState(Qt.Checked)
    viewer.columnsModel.item(0).setCheckState(Qt.Checked)
    viewer.updateColumnsList()
    assert viewer.measurementsAxes == {'X': 'intensity', 'Y': 'area'}
    viewer.acdc_df = viewer.acdc_df.drop(columns='area')
    viewer.updateColumnsList()
    viewer.plotMeasurements()
    assert viewer.measurementsAxes == {'X': 'intensity', 'Y': ''}
    assert viewer.columnsModel.item(0).data(Qt.AccessibleDescriptionRole) == 'X'
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
    index = viewer.columnsSearchProxy.index(0, 0)
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
    assert viewer.measurementsAxes == {'X': 'area', 'Y': ''}
    assert viewer.columnsModel.item(1).icon().isNull()
    assert viewer.checkedColumns() == ['area']


def test_modes_have_independent_checks(viewer):
    viewer.measurementsModeCombobox.setCurrentIndex(1)
    assert viewer.checkedColumns() == []
    assert viewer.measurementsAxes == {'X': '', 'Y': ''}
    assert all(
        viewer.columnsModel.item(i).icon().isNull()
        for i in range(viewer.columnsModel.rowCount())
    )
    viewer.columnsModel.item(1).setCheckState(Qt.Checked)
    assert viewer.measurementsAxes == {'X': 'intensity', 'Y': ''}
    viewer.measurementsModeCombobox.setCurrentIndex(0)
    assert viewer.checkedColumns() == ['area']
    viewer.measurementsModeCombobox.setCurrentIndex(1)
    assert viewer.checkedColumns() == ['intensity']
    assert viewer.columnsModel.item(1).data(Qt.AccessibleDescriptionRole) == 'X'


def test_mode_change_repaints_indicators(viewer, monkeypatch):
    viewer.measurementsModeCombobox.setCurrentIndex(1)
    viewer.columnsModel.item(1).setCheckState(Qt.Checked)
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
            viewer.columnsModel.item(i).text()
            for i in range(viewer.columnsModel.rowCount())
            if not viewer.columnsModel.item(i).icon().isNull()
        } == icons
    itemChanged.assert_not_called()
    assert viewer.measurementsAxes == {'X': 'intensity', 'Y': ''}


def test_comparison_requires_unchecking_before_replacing_axis(viewer):
    viewer.acdc_df['third'] = [1, 2, 3, 4]
    viewer.updateColumnsList()
    comparison(viewer)
    viewer.columnsModel.item(2).setCheckState(Qt.Checked)
    assert viewer.columnsModel.item(2).checkState() == Qt.Unchecked
    assert 'Two measurements are already selected' in viewer.statusBar().currentMessage()
    viewer.columnsModel.item(0).setCheckState(Qt.Unchecked)
    viewer.columnsModel.item(2).setCheckState(Qt.Checked)
    assert viewer.measurementsAxes == {'X': 'third', 'Y': 'intensity'}
    assert viewer.columnsModel.item(2).data(Qt.AccessibleDescriptionRole) == 'X'


@pytest.mark.parametrize('mode', [0, 1])
@pytest.mark.parametrize('filtered', [False, True])
def test_indicator_replaces_tick_without_shifting_text(viewer, app, mode, filtered):
    viewer.measurementsModeCombobox.setCurrentIndex(mode)
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
    assert len(viewer.measurementsPlot.listDataItems()) == 2
    click_name(0)
    assert viewer.checkedColumns() == ['intensity']

    # Comparison: set X then Y, clicking again frees the axis
    viewer.measurementsModeCombobox.setCurrentIndex(1)
    click_name(1)
    assert viewer.measurementsAxes == {'X': 'intensity', 'Y': ''}
    click_name(0)
    assert viewer.measurementsAxes == {'X': 'intensity', 'Y': 'area'}
    assert len(viewer.measurementsPlot.listDataItems()) == 1
    click_name(1)
    assert viewer.measurementsAxes == {'X': '', 'Y': 'area'}

    viewer.measurementsModeCombobox.setCurrentIndex(0)
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


@pytest.mark.parametrize('mode', [0, 1])
def test_configuration_roundtrip(viewer, tmp_path, monkeypatch, mode):
    images = configuration_data(viewer, tmp_path)
    comparison(viewer)
    # Reverse X/Y to prove order is restored, not inferred from column order.
    viewer.measurementsAxes = {'X': 'intensity', 'Y': 'area'}
    viewer.measurementsModeCombobox.setCurrentIndex(mode)
    viewer.normalizeCheckbox.setChecked(True)
    filepath = str(tmp_path / 'state.json')
    monkeypatch.setattr(
        measurements_viewer.QFileDialog, 'getSaveFileName',
        Mock(return_value=(filepath, 'JSON (*.json)'))
    )
    assert viewer.saveConfiguration()
    saved = json.loads((tmp_path / 'state.json').read_text())
    assert saved['images_paths'] == [os.path.abspath(images)]
    assert saved['channel'] == 'phase'
    assert saved['segm_endname'] == 'segm_custom'
    assert saved['time_course_columns'] == ['area']
    assert saved['measurements_axes'] == {'X': 'intensity', 'Y': 'area'}

    viewer.measurementsModeCombobox.setCurrentIndex(0)
    viewer.columnsModel.item(0).setCheckState(Qt.Unchecked)
    viewer.columnsModel.item(1).setCheckState(Qt.Checked)
    viewer.measurementsAxes = {'X': '', 'Y': ''}
    viewer.normalizeCheckbox.setChecked(False)
    load_position = Mock(return_value=viewer.data[0])
    monkeypatch.setattr(measurements_viewer, 'load_position', load_position)
    monkeypatch.setattr(
        measurements_viewer, 'get_channels_and_segm_endnames',
        Mock(return_value=(['phase'], ['segm_custom']))
    )
    monkeypatch.setattr(viewer, 'setData', Mock())
    monkeypatch.setattr(viewer, 'setOverlayItemsOpacities', Mock())
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
    dialog = Mock(side_effect=[(str(first), ''), (str(second), '')])
    monkeypatch.setattr(measurements_viewer.QFileDialog, 'getSaveFileName', dialog)
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
    path = tmp_path / 'invalid.json'
    path.write_text(json.dumps(configuration))
    viewer.configurationFilepath = 'previous.json'
    assert not viewer.loadConfiguration(filepath=str(path))
    error.assert_called_once()
    set_data.assert_not_called()
    assert viewer.configurationFilepath == 'previous.json'


def test_save_cancel_and_write_failure(viewer, tmp_path, monkeypatch):
    configuration_data(viewer, tmp_path)
    monkeypatch.setattr(
        measurements_viewer.QFileDialog, 'getSaveFileName',
        Mock(return_value=('', ''))
    )
    assert not viewer.saveConfiguration()
    assert viewer.configurationFilepath is None
    viewer.configurationFilepath = str(tmp_path / 'missing' / 'state.json')
    error = Mock()
    monkeypatch.setattr(viewer, 'configurationError', error)
    assert not viewer.saveConfiguration()
    error.assert_called_once()


def test_startup_prompt_only_once(app, monkeypatch):
    prompt = Mock()
    monkeypatch.setattr(MeasurementsViewerWin, 'askLoadConfiguration', prompt)
    window = MeasurementsViewerWin()
    window.show()
    app.processEvents()
    window.hide()
    window.show()
    app.processEvents()
    prompt.assert_called_once()
    window.deleteLater()
    app.processEvents()


def test_startup_opens_configuration_picker_directly(app, monkeypatch):
    window = MeasurementsViewerWin()
    load = Mock()
    monkeypatch.setattr(window, 'loadConfiguration', load)
    window.askLoadConfiguration()
    load.assert_called_once_with()
    window.deleteLater()
    app.processEvents()


@pytest.mark.parametrize('has_configuration', [False, True])
@pytest.mark.parametrize('choice', ['skip', 'select', 'browse'])
def test_configuration_picker_buttons(
        viewer, tmp_path, monkeypatch, has_configuration, choice):
    from cellacdc import apps
    from qtpy.QtWidgets import QPushButton

    viewer.configurationsPath = str(tmp_path)
    path = tmp_path / 'saved.json'
    if has_configuration:
        path.write_text('{}')
    external = str(tmp_path / 'external.json')
    monkeypatch.setattr(
        measurements_viewer.QFileDialog, 'getOpenFileName',
        Mock(return_value=(external, ''))
    )
    original_dialog = apps.QTreeDialog

    def create_dialog(*args, **kwargs):
        dialog = original_dialog(*args, **kwargs)
        buttons = dialog.findChildren(QPushButton)
        skip, = [button for button in buttons if button.text() == 'Skip loading']
        browse, = [
            button for button in buttons if button.text() == 'Browse configuration...'
        ]
        ok, = [button for button in buttons if button.text().lower().strip('&') == 'ok']

        def interact():
            assert ok.isEnabled() == has_configuration
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


@pytest.mark.parametrize('choice', ['cancel', 'dismiss', 'discard', 'save'])
@pytest.mark.parametrize('saved', [False, True])
def test_close_prompts_and_respects_cancel(viewer, monkeypatch, choice, saved):
    viewer.isDataLoaded = True
    buttons = [object(), object(), object()]
    msg = Mock()
    msg.question.return_value = buttons
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
    monkeypatch.setattr(MeasurementsViewerWin, 'askLoadConfiguration', Mock())
    window = MeasurementsViewerWin()
    errors = []
    monkeypatch.setattr(window, 'configurationError', lambda *args: errors.append(args))
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
        assert window.configuration() == configuration
        assert window.saveConfiguration()
        assert json.loads(path.read_text()) == configuration
        assert len(window.measurementsPlot.listDataItems()) == 1
    finally:
        window.deleteLater()
        app.processEvents()
