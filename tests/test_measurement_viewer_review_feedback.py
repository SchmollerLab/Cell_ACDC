"""Tests of the measurements viewer changes requested in the review of the
Measurements GUI pull request (blinking controls, stable colorblind-friendly
plot colors, button icons, tooltips, terminal logs, loading in a worker
thread)."""
import json
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest

from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QColor, QKeySequence
from qtpy.QtWidgets import QApplication, QWidget

from cellacdc import measurements_viewer, qutils, widgets
from cellacdc import measurements_viewer_utils
from cellacdc.measurements_viewer import (
    MeasurementsViewerWin, SIGNAL_COLORS, signal_color
)


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def viewer(app, monkeypatch):
    """Viewer with a small synthetic table (as in test_measurement_viewer)
    and 'area' plotted."""
    window = MeasurementsViewerWin()
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
    assert window.checkedColumns() == ['area']
    yield window
    window.deleteLater()
    app.processEvents()


def column(viewer, name):
    item, = [
        item for item in viewer.columnsModel.measurementItems()
        if item.text() == name
    ]
    return item


def set_plotted(viewer, name, plotted):
    column(viewer, name).setCheckState(
        Qt.Checked if plotted else Qt.Unchecked
    )


def pen_colors(viewer):
    """{column: pen color name} of the plotted time courses."""
    return {
        item.name(): pg.mkPen(item.opts['pen']).color().name()
        for item in viewer.measurementsPlot.listDataItems()
        if item.name()
    }


def default_color(index):
    return pg.mkColor(SIGNAL_COLORS[index]).name()


class BlinkRecorder:
    """Replaces `qutils.QControlBlink` and records the blinked targets."""
    calls = []

    def __init__(self, widget_to_blink, duration_ms=2000, qparent=None):
        targets = (
            widget_to_blink if isinstance(widget_to_blink, (list, tuple))
            else [widget_to_blink]
        )
        BlinkRecorder.calls.append(list(targets))

    def start(self):
        pass


@pytest.fixture
def blinks(monkeypatch):
    BlinkRecorder.calls = []
    monkeypatch.setattr(qutils, 'QControlBlink', BlinkRecorder)
    return BlinkRecorder.calls


def test_signal_colors_are_colorblind_friendly_and_not_red_first():
    assert signal_color(0).name() == '#56b4e9'
    # Vermillion (closest to the red current frame line) comes last
    assert SIGNAL_COLORS[-1] == '#D55E00'
    names = [signal_color(i).name() for i in range(len(SIGNAL_COLORS) + 3)]
    assert len(set(names)) == len(names)
    assert '#000000' not in names


def is_reddish(color):
    """True for saturated colors with a hue within 15 degrees of red (the
    current frame line)."""
    hue, saturation, value, _ = pg.mkColor(color).getHsvF()
    hue = hue * 360
    return saturation > 0.5 and value > 0.3 and (hue < 15 or hue > 345)


@pytest.mark.parametrize('light_background', [False, True])
def test_no_default_signal_color_is_red(light_background):
    colors = [
        signal_color(i, light_background=light_background)
        for i in range(30)
    ]
    assert not [color.name() for color in colors if is_reddish(color)]
    # Distinct for many more columns than the palette has
    assert len({color.name() for color in colors[:24]}) == 24


def test_light_background_replaces_yellow_with_black():
    yellow = SIGNAL_COLORS.index('#F0E442')
    assert signal_color(yellow).name() == '#f0e442'
    assert signal_color(yellow, light_background=True).name() == '#000000'
    others = [i for i in range(len(SIGNAL_COLORS)) if i != yellow]
    assert all(
        signal_color(i).name()
        == signal_color(i, light_background=True).name()
        for i in others
    )


def test_inverting_background_recolors_default_colors(viewer):
    yellow = SIGNAL_COLORS.index('#F0E442')
    viewer._defaultSignalColorIndices['area'] = yellow
    viewer.plotMeasurements()
    assert pen_colors(viewer) == {'area': '#f0e442'}
    viewer.isDataLoaded = True
    viewer.invertBwAction.blockSignals(True)
    viewer.invertBwAction.setChecked(True)
    viewer.invertBwAction.blockSignals(False)
    viewer.recolorSignalsForBackground()
    assert pen_colors(viewer) == {'area': '#000000'}


def test_default_phase_colors_are_okabe_ito_and_not_red():
    from cellacdc import phase_translator
    names = [f'phase{i}' for i in range(25)]
    colors = phase_translator.phase_colors(names)
    assert colors['phase0'] == '#56b4e9'
    assert not [name for name, color in colors.items() if is_reddish(color)]
    assert len(set(colors.values())) == len(names)


def test_conditional_formatting_default_colors_are_not_red():
    from cellacdc import apps_measurements_viewer
    colors = apps_measurements_viewer.ConditionalFormattingDialog.DEFAULT_COLORS
    assert not [color for color in colors if is_reddish(color)]
    assert '#d55e00' not in colors


def test_adding_a_column_keeps_the_plotted_color(viewer):
    assert pen_colors(viewer) == {'area': default_color(0)}
    set_plotted(viewer, 'intensity', True)
    assert pen_colors(viewer) == {
        'area': default_color(0), 'intensity': default_color(1)
    }


def test_replotted_column_gets_its_color_back_if_free(viewer):
    set_plotted(viewer, 'intensity', True)
    set_plotted(viewer, 'area', False)
    assert pen_colors(viewer) == {'intensity': default_color(1)}
    set_plotted(viewer, 'area', True)
    assert pen_colors(viewer) == {
        'area': default_color(0), 'intensity': default_color(1)
    }


def test_replotted_column_takes_a_free_color_if_its_color_was_taken(viewer):
    set_plotted(viewer, 'area', False)
    set_plotted(viewer, 'intensity', True)
    # 'area' is not plotted, so its color is free
    assert pen_colors(viewer) == {'intensity': default_color(0)}
    set_plotted(viewer, 'area', True)
    assert pen_colors(viewer) == {
        'area': default_color(1), 'intensity': default_color(0)
    }


def test_custom_color_wins_and_keeps_default_colors(viewer):
    viewer.measurementsColors['area'] = QColor('#123456')
    viewer.plotMeasurements()
    set_plotted(viewer, 'intensity', True)
    assert pen_colors(viewer) == {
        'area': '#123456', 'intensity': default_color(1)
    }
    viewer.resetSignalColor('area')
    assert pen_colors(viewer) == {
        'area': default_color(0), 'intensity': default_color(1)
    }


def test_comparison_mode_does_not_reassign_time_course_colors(viewer):
    set_plotted(viewer, 'area', False)
    set_plotted(viewer, 'intensity', True)
    assert pen_colors(viewer) == {'intensity': default_color(0)}
    viewer.measurementsModeCombobox.setCurrentIndex(1)
    set_plotted(viewer, 'area', True)
    set_plotted(viewer, 'intensity', True)
    assert viewer.measurementsAxes == {'X': 'area', 'Y': 'intensity'}
    scatter, = viewer.measurementsPlot.listDataItems()
    brush = pg.mkBrush(scatter.opts['symbolBrush'])
    assert brush.color().name() == default_color(0)
    viewer.measurementsModeCombobox.setCurrentIndex(0)
    assert pen_colors(viewer) == {'intensity': default_color(0)}


def test_renamed_column_keeps_its_color(viewer):
    set_plotted(viewer, 'intensity', True)
    set_plotted(viewer, 'area', False)
    viewer.acdc_df = viewer.acdc_df.rename(columns={'intensity': 'signal'})
    viewer.renameMeasurementColumnReferences('intensity', 'signal')
    viewer.plotMeasurements()
    assert pen_colors(viewer) == {'signal': default_color(1)}


def view_range(viewer):
    (xMin, xMax), (yMin, yMax) = viewer.measurementsPlot.viewRange()
    return (xMin, xMax), (yMin, yMax)


def flat_view_range(viewer):
    (xMin, xMax), (yMin, yMax) = view_range(viewer)
    return [xMin, xMax, yMin, yMax]


def autoscaled_view_range(viewer):
    """View range after pyqtgraph applied the pending auto range (done on
    the next paint in the app)."""
    viewer.measurementsPlot.getViewBox().updateAutoRange()
    return view_range(viewer)


def test_plotting_a_column_autoscales_x_and_y_like_the_auto_button(viewer):
    viewer.rescalePlotCheckbox.setChecked(False)
    vb = viewer.measurementsPlot.getViewBox()
    viewer.measurementsPlot.setRange(
        xRange=(1.2, 1.4), yRange=(10, 11), padding=0
    )
    assert vb.autoRangeEnabled() == [False, False]

    # Cell 1: area 10, 20 and intensity 100, 250 at frames 1, 2
    set_plotted(viewer, 'intensity', True)
    assert vb.autoRangeEnabled() == [True, True]
    (xMin, xMax), (yMin, yMax) = autoscaled_view_range(viewer)
    assert xMin <= 1 and 2 <= xMax < 2.5
    assert yMin <= 10 and yMax >= 250

    # The same range as clicking the "A" button
    autoscaled = flat_view_range(viewer)
    viewer.measurementsPlot.setRange(xRange=(0, 1), yRange=(0, 1), padding=0)
    viewer.measurementsPlot.getPlotItem().autoBtnClicked()
    vb.updateAutoRange()
    assert flat_view_range(viewer) == pytest.approx(autoscaled)

    # Changing cell ('Rescale plot' unchecked) follows the data on both
    # axes. Cell 2: area 30, 40 and intensity 150, 300 at frames 2, 3
    viewer.setSelectedID(2)
    (xMin, xMax), (yMin, yMax) = autoscaled_view_range(viewer)
    assert 1.5 < xMin <= 2 and xMax >= 3
    assert yMin <= 30 and 300 <= yMax < 400

    # A manual zoom is kept when changing cell
    vb.scaleBy((0.5, 0.5))
    zoomed = flat_view_range(viewer)
    assert vb.autoRangeEnabled() == [False, False]
    viewer.setSelectedID(1)
    assert flat_view_range(viewer) == pytest.approx(zoomed)
    vb.updateAutoRange()
    assert flat_view_range(viewer) == pytest.approx(zoomed)

    # Removing a column autoscales x and y again
    set_plotted(viewer, 'intensity', False)
    assert vb.autoRangeEnabled() == [True, True]
    (xMin, xMax), (yMin, yMax) = autoscaled_view_range(viewer)
    assert xMin <= 1 and 2 <= xMax < 2.5
    assert yMin <= 10 and 20 <= yMax < 100


def test_rescale_plot_autoscales_after_manual_zoom(viewer):
    viewer.rescalePlotCheckbox.setChecked(True)
    vb = viewer.measurementsPlot.getViewBox()
    viewer.measurementsPlot.setRange(xRange=(0, 1), yRange=(0, 1), padding=0)
    assert vb.autoRangeEnabled() == [False, False]
    # Cell 2: area 30, 40 at frames 2, 3
    viewer.idScrollBar.setValue(viewer.IDs.index(2))
    assert viewer.selectedID == 2
    assert vb.autoRangeEnabled() == [True, True]
    (xMin, xMax), (yMin, yMax) = autoscaled_view_range(viewer)
    assert 1.5 < xMin <= 2 and xMax >= 3
    assert yMin <= 30 and 40 <= yMax < 50


# pyqtgraph computes the bounds of the all-NaN scatter
@pytest.mark.filterwarnings('ignore:All-NaN slice:RuntimeWarning')
def test_autoscale_waits_for_columns_with_values(viewer):
    viewer.acdc_df['empty'] = np.nan
    viewer.updateColumnsList()
    set_plotted(viewer, 'area', False)
    vb = viewer.measurementsPlot.getViewBox()
    viewer.measurementsPlot.setRange(
        xRange=(0, 1), yRange=(500, 501), padding=0
    )
    set_plotted(viewer, 'empty', True)
    assert vb.autoRangeEnabled() == [False, False]
    assert flat_view_range(viewer) == pytest.approx([0, 1, 500, 501])
    # The autoscale is pending until there is data to fit
    set_plotted(viewer, 'area', True)
    assert vb.autoRangeEnabled() == [True, True]
    (xMin, xMax), (yMin, yMax) = autoscaled_view_range(viewer)
    assert xMin <= 1 and 2 <= xMax < 2.5
    assert yMin <= 10 and 20 <= yMax < 500


def test_phase_band_follows_the_autoscaled_time_course(viewer, tmp_path):
    # The x-linked phase band (bars from frame - 0.5 to frame + 0.5) must
    # not take over the x range: the time course fits its data points
    viewer.settings_csv_path = str(tmp_path / 'settings.csv')
    viewer.acdc_df['phase'] = ['G1', 'S', 'G1', 'S']
    viewer.showPhasesCheckbox.setChecked(True)
    viewer.measurementsPlot.setRange(xRange=(0, 1), yRange=(0, 1), padding=0)
    viewer.normalizeCheckbox.setChecked(True)
    assert not viewer.phaseBandPlot.isHidden()
    vb = viewer.measurementsPlot.getViewBox()
    band = viewer.phaseBandPlot.getViewBox()
    for cell, (first, last) in ((1, (1, 2)), (2, (2, 3))):
        viewer.setSelectedID(cell)
        band.updateAutoRange()
        (xMin, xMax), _ = autoscaled_view_range(viewer)
        assert vb.autoRangeEnabled() == [True, True]
        assert first - 0.1 < xMin <= first and last <= xMax < last + 0.1
        assert band.viewRange()[0] == pytest.approx([xMin, xMax])


def test_toggling_normalization_autoscales_x_and_y(viewer):
    viewer.measurementsPlot.setRange(
        xRange=(1.2, 1.4), yRange=(10, 11), padding=0
    )
    viewer.normalizeCheckbox.setChecked(True)
    (xMin, xMax), (yMin, yMax) = autoscaled_view_range(viewer)
    assert xMin <= 1 and xMax >= 2
    assert yMin <= 0 and 1 <= yMax < 10


def test_comparison_autoranges_when_axes_change(viewer):
    viewer.measurementsModeCombobox.setCurrentIndex(1)
    set_plotted(viewer, 'area', True)
    set_plotted(viewer, 'intensity', True)
    assert viewer.measurementsAxes == {'X': 'area', 'Y': 'intensity'}
    viewer.measurementsPlot.setRange(
        xRange=(0, 1), yRange=(0, 1), padding=0
    )
    # Change Y: free it and select it again
    set_plotted(viewer, 'intensity', False)
    set_plotted(viewer, 'intensity', True)
    vb = viewer.measurementsPlot.getViewBox()
    assert vb.autoRangeEnabled() == [True, True]
    vb.updateAutoRange()
    (xMin, xMax), (yMin, yMax) = view_range(viewer)
    assert xMin <= 10 and xMax >= 40
    assert yMin <= 100 and yMax >= 300


def test_ctrl_f_blinks_search_bar(viewer, blinks):
    assert viewer.focusOnSearchAction.shortcut() == QKeySequence('Ctrl+F')
    # As `setData` does when data is loaded
    viewer.setEnabledFileToolbar(True)
    viewer.focusOnSearchAction.trigger()
    assert blinks == [[viewer.searchWidget]]


def test_loading_data_blinks_cell_id_controls(app, monkeypatch, tmp_path, blinks):
    from tests.test_measurement_viewer import make_timelapse_position
    errors = []
    monkeypatch.setattr(
        sys, 'excepthook', lambda *exc_info: errors.append(exc_info)
    )
    window = MeasurementsViewerWin()
    window._startupConfigurationPrompted = True
    try:
        # No data: nothing to blink and no error
        window.setData([])
        app.processEvents()
        assert blinks == []
        images_path = make_timelapse_position(tmp_path)
        window.setData(
            measurements_viewer.load_positions([images_path], 'phase')
        )
        # Not inside `setData`: the synchronous rest of a load (e.g. the
        # overlay channels of a configuration) would delay the blink timers
        # until the blink is over
        assert blinks == []
        app.processEvents()
        assert blinks == [
            [window.idLabel, window.idSpinBox, window.idScrollBar]
        ]
        assert 'Cell ID' in window.statusBar().currentMessage()
        assert not errors
    finally:
        window.deleteLater()
        app.processEvents()


def test_navigate_box_aligns_labels_spinboxes_and_sliders(viewer, app):
    """Labels, spin boxes and sliders of the Navigate box are each in their
    own left-aligned grid column, also the overlay alpha sliders (added by
    the viewer's `addAlphaScrollbar` override); the options are below."""
    layout = viewer.bottomLeftLayout
    alphaScrollBar = viewer.addAlphaScrollbar('GFP', SimpleNamespace())
    alphaScrollBar.show()
    alphaScrollBar.label.show()

    def cell(widget):
        row, col, _, _ = layout.getItemPosition(layout.indexOf(widget))
        return row, col

    rows = [
        (viewer.t_label, viewer.navSpinBox, viewer.navigateScrollBar),
        (viewer.zSliceCheckbox, viewer.zSliceSpinbox, viewer.zSliceScrollBar),
        (viewer.idLabel, viewer.idSpinBox, viewer.idScrollBar),
    ]
    for label, spinbox, slider in rows:
        assert cell(label)[1] == viewer.NAV_LABEL_COL
        assert cell(spinbox)[1] == viewer.NAV_SPINBOX_COL
        assert cell(slider)[1] == viewer.NAV_SLIDER_COL
    for label in (alphaScrollBar.label, viewer.overlay_z_label):
        assert cell(label)[1] == viewer.NAV_LABEL_COL
    for slider in (alphaScrollBar, viewer.zSliceOverlay_SB):
        assert cell(slider)[1] == viewer.NAV_SLIDER_COL
    # Alpha sliders follow the Cell ID slider; the options are not in the grid
    assert cell(alphaScrollBar)[0] == cell(viewer.idScrollBar)[0] + 1
    assert layout.indexOf(viewer.followIDcheckbox) == -1
    for col in range(viewer.NAV_NUM_COLS):
        stretch = layout.columnStretch(col)
        assert stretch == (1 if col == viewer.NAV_SLIDER_COL else 0)

    viewer.show()
    try:
        app.processEvents()
        visibleRows = [
            (viewer.t_label, viewer.navSpinBox, viewer.navigateScrollBar),
            (viewer.idLabel, viewer.idSpinBox, viewer.idScrollBar),
            (alphaScrollBar.label, None, alphaScrollBar),
        ]
        for widgetsInColumn in zip(*visibleRows):
            xs = {w.geometry().x() for w in widgetsInColumn if w is not None}
            assert len(xs) == 1
        assert (
            viewer.followIDcheckbox.mapTo(viewer, viewer.followIDcheckbox.rect().topLeft()).y()
            > alphaScrollBar.mapTo(viewer, alphaScrollBar.rect().topLeft()).y()
        )
    finally:
        viewer.hide()


def test_restarting_blink_restores_original_style(app):
    widget = QWidget()
    widget.setStyleSheet('color: blue')
    first = qutils.QControlBlink(widget)
    first.start()
    first.timerCallback()
    first.timerCallback()
    assert 'orange' in widget.styleSheet()
    # A second blink while the first is running (e.g. Ctrl+F pressed twice)
    second = qutils.QControlBlink(widget)
    assert widget.styleSheet() == 'color: blue'
    second.start()
    second.timerCallback()
    second.timerCallback()
    second.stop()
    assert widget.styleSheet() == 'color: blue'
    # Stopping twice (e.g. stop timer after a manual stop) is harmless
    second.stop()
    first.stop()
    assert widget.styleSheet() == 'color: blue'
    assert widget not in qutils.QControlBlink._active
    widget.deleteLater()


@pytest.mark.parametrize('text, expected', [
    ("Don't save", widgets.noPushButton),
    ('Do not save', widgets.noPushButton),
    ('Save tables', widgets.savePushButton),
    ('Save both as...', widgets.savePushButton),
    ('Save configuration...', widgets.savePushButton),
    ('Cancel', widgets.cancelPushButton),
])
def test_save_changes_buttons_have_icons(app, text, expected):
    button, _ = widgets.getPushButton(text)
    assert type(button) is expected
    button.deleteLater()


def test_process_measurements_button_has_tooltip(viewer):
    tooltip = viewer.processMeasurementsButton.toolTip()
    assert 'recipe' in tooltip
    assert 'Export measurements' in tooltip


def test_viewer_log_messages_reach_stdout(viewer, capsys):
    viewer.logger.info('Info message of the viewer')
    viewer.logger.warning('Warning message of the viewer')
    out = capsys.readouterr().out
    assert 'Info message of the viewer\n' in out
    assert '[INFO]' not in out
    assert '[WARNING]: Warning message of the viewer' in out


def test_log_level_prefix_is_not_doubled(viewer, capsys):
    # E.g. messages of the mixins shared with the main GUI
    viewer.logger.warning('[WARNING] rp was requested before being set')
    viewer.logger.error('[ERROR]: Something failed')
    out = capsys.readouterr().out
    assert '[WARNING] rp was requested before being set\n' in out
    assert '[ERROR]: Something failed\n' in out
    assert '[WARNING]: [WARNING]' not in out
    assert '[ERROR]: [ERROR]' not in out


def test_viewer_log_handler_is_attached_once(viewer):
    logger = measurements_viewer.get_logger()
    measurements_viewer.get_logger()
    assert logger is viewer.logger
    handlers = [
        handler for handler in logger.handlers
        if getattr(handler, '_acdcMeasurementsViewerStdout', False)
    ]
    assert len(handlers) == 1
    assert not logger.propagate


def test_plotting_and_removing_a_column_logs_its_name(viewer, capsys):
    capsys.readouterr()
    set_plotted(viewer, 'intensity', True)
    assert 'Plotting "intensity"' in capsys.readouterr().out
    set_plotted(viewer, 'intensity', False)
    assert 'Removing "intensity" from the plot' in capsys.readouterr().out


def test_changing_and_resetting_a_color_is_logged(viewer, capsys, monkeypatch):
    monkeypatch.setattr(
        measurements_viewer.QColorDialog, 'getColor',
        lambda *args, **kwargs: QColor('#123456')
    )
    capsys.readouterr()
    viewer.changeSignalColor('area')
    viewer.resetSignalColor('area')
    out = capsys.readouterr().out
    assert 'Changing color of "area" to #123456' in out
    assert 'Resetting color of "area" to the default' in out


def test_save_changes_prompt_logs_the_chosen_button(viewer, capsys, monkeypatch):
    class FakeSaveBox:
        def __init__(self, *args, **kwargs):
            self.cancel = False

        def warning(self, parent, title, txt, detailsText=None, buttonsTexts=()):
            self.clickedButton = "Don't save"
            return list(buttonsTexts)

    monkeypatch.setattr(measurements_viewer.widgets, 'myMessageBox', FakeSaveBox)
    monkeypatch.setattr(viewer, 'unsavedAcdcDfPositions', lambda: [])
    monkeypatch.setattr(viewer, 'hasUnsavedConfiguration', lambda: True)
    capsys.readouterr()
    assert viewer.askSaveChanges('loading other data')
    assert (
        'Save changes before loading other data: "Don\'t save"'
        in capsys.readouterr().out
    )


# --- Loading in a worker thread (review points 10/11) ---

@pytest.fixture
def loading_window(app, monkeypatch):
    """Viewer without data that fails on exceptions raised in Qt slots."""
    errors = []
    monkeypatch.setattr(
        sys, 'excepthook', lambda *exc_info: errors.append(exc_info)
    )
    window = MeasurementsViewerWin()
    window._startupConfigurationPrompted = True
    window.slotErrors = errors
    yield window
    window._isLoadingData = False
    window.deleteLater()
    app.processEvents()


def record_load_threads(monkeypatch):
    """Wrap `load_position` and `core.count_objects_and_init_rps` to record
    the thread they run in."""
    threads = {'load_position': [], 'regionprops': []}
    load_position = measurements_viewer.load_position
    count_objects = measurements_viewer.core.count_objects_and_init_rps

    def recorded_load_position(*args, **kwargs):
        threads['load_position'].append(threading.current_thread())
        return load_position(*args, **kwargs)

    def recorded_count_objects(*args, **kwargs):
        threads['regionprops'].append(threading.current_thread())
        return count_objects(*args, **kwargs)

    monkeypatch.setattr(
        measurements_viewer_utils, 'load_position', recorded_load_position
    )
    monkeypatch.setattr(
        measurements_viewer.core, 'count_objects_and_init_rps',
        recorded_count_objects
    )
    return threads


def test_run_in_thread_keeps_event_loop_running(loading_window, capsys):
    timerFired = threading.Event()
    states = []

    def task(value, progress=None):
        progress('Halfway...')
        # Set by a timer on the GUI thread: only if its event loop runs
        return value, timerFired.wait(5), threading.current_thread()

    def onTimer():
        states.append(loading_window.isEnabled())
        timerFired.set()

    QTimer.singleShot(0, onTimer)
    value, fired, thread = loading_window.runInThread(
        task, 3, text='Testing task...', reportProgress=True
    )
    assert (value, fired) == (3, True)
    assert thread is not threading.main_thread()
    # Input disabled while the task runs, enabled again afterwards
    assert states == [False]
    assert loading_window.isEnabled()
    out = capsys.readouterr().out
    assert 'Testing task...' in out
    assert 'Halfway...' in out
    assert 'Testing task done in' in out


def test_run_in_thread_reraises_and_reenables(loading_window, capsys):
    def task():
        raise OSError('Cannot read')

    with pytest.raises(OSError, match='Cannot read'):
        loading_window.runInThread(task, text='Failing task...')
    assert loading_window.isEnabled()
    assert 'Failing task failed after' in capsys.readouterr().out


def test_load_images_paths_loads_in_worker_thread(
        loading_window, tmp_path, monkeypatch):
    from tests.test_measurement_viewer import make_timelapse_position
    images_path = make_timelapse_position(tmp_path, extra_cells=[(1, 2)])
    threads = record_load_threads(monkeypatch)
    loading_window.loadImagesPaths([images_path], channel='phase')
    assert not loading_window.slotErrors
    assert len(loading_window.data) == 1
    posData = loading_window.data[0]
    main = threading.main_thread()
    assert len(threads['load_position']) == 1
    assert threads['load_position'][0] is not main
    # The regionprops of the displayed position were computed once, in the
    # worker thread, and used by `initPosAnnotData`
    assert len(threads['regionprops']) == 1
    assert threads['regionprops'][0] is not main
    assert not hasattr(posData, '_viewerPrecomputedRps')
    assert posData.allIDs == {1, 2}
    assert len(posData.allData_li) == posData.SizeT
    assert loading_window.isEnabled()
    assert not loading_window._isLoadingData


def test_load_configuration_loads_in_worker_thread_and_reports_errors(
        loading_window, tmp_path, monkeypatch):
    from tests.test_measurement_viewer import make_timelapse_position
    images_path = make_timelapse_position(tmp_path)
    loading_window.loadImagesPaths([images_path], channel='phase')
    filepath = tmp_path / 'configuration.json'
    filepath.write_text(json.dumps(loading_window.configuration()))
    monkeypatch.setattr(
        loading_window, 'selectConfigurationInput',
        lambda config: loading_window.configurationInputOptions(config)[0]
    )
    monkeypatch.setattr(
        loading_window, 'askSaveBeforeLoading', Mock(return_value=True)
    )
    threads = record_load_threads(monkeypatch)
    assert loading_window.loadConfiguration(filepath=str(filepath))
    assert not loading_window.slotErrors
    assert threads['load_position']
    assert threading.main_thread() not in threads['load_position']
    assert loading_window.isEnabled()

    # A failure in the worker thread gives the same warning as before
    loaded = loading_window.data
    monkeypatch.setattr(
        measurements_viewer_utils, 'load_position',
        Mock(side_effect=OSError('Cannot read'))
    )
    error = Mock()
    monkeypatch.setattr(loading_window, 'configurationError', error)
    assert not loading_window.loadConfiguration(filepath=str(filepath))
    error.assert_called_once()
    operation, err = error.call_args.args
    assert operation == 'load'
    assert isinstance(err, OSError) and str(err) == 'Cannot read'
    assert loading_window.data is loaded
    assert loading_window.isEnabled()
    assert not loading_window._isLoadingData


def test_reentrant_load_and_close_are_refused(
        loading_window, tmp_path, monkeypatch):
    """The dialogs of a load (e.g. "Save changes?") do not block the
    window: a second load or closing started from them is refused."""
    from tests.test_measurement_viewer import make_timelapse_position
    images_path = make_timelapse_position(tmp_path)
    load_position = Mock(wraps=measurements_viewer.load_position)
    monkeypatch.setattr(measurements_viewer_utils, 'load_position', load_position)
    nested = {}

    def askSaveBeforeLoading():
        # As if the user acted while the prompt is open
        nested['images'] = loading_window.loadImagesPaths(
            [images_path], channel='phase'
        )
        nested['configuration'] = loading_window.loadConfiguration(
            filepath=str(tmp_path / 'missing.json')
        )
        nested['folder'] = loading_window.openFolder(folderpath=images_path)
        nested['closed'] = loading_window.close()
        nested['message'] = loading_window.statusBar().currentMessage()
        return True

    monkeypatch.setattr(
        loading_window, 'askSaveBeforeLoading', askSaveBeforeLoading
    )
    loading_window.loadImagesPaths([images_path], channel='phase')
    assert nested['images'] is None
    assert nested['configuration'] is False
    assert nested['folder'] is None
    assert nested['closed'] is False
    assert 'Wait for the data being loaded' in nested['message']
    # Only the outer load ran
    assert load_position.call_count == 1
    assert len(loading_window.data) == 1
    assert not loading_window._isLoadingData
    assert loading_window.isEnabled()


def test_run_in_thread_disables_the_other_windows(loading_window, app):
    """A dialog opened from another window would nest its event loop in the
    wait and keep the viewer disabled until it closes."""
    other = QWidget()
    other.show()
    states = []
    QTimer.singleShot(0, lambda: states.append(other.isEnabled()))
    try:
        loading_window.runInThread(lambda: None, text='Testing task...')
        assert states == [False]
        assert other.isEnabled()
    finally:
        other.close()
        other.deleteLater()


def test_load_is_refused_while_the_closing_prompt_is_open(
        loading_window, tmp_path, monkeypatch):
    from tests.test_measurement_viewer import make_timelapse_position
    images_path = make_timelapse_position(tmp_path)
    loading_window.loadImagesPaths([images_path], channel='phase')
    loaded = loading_window.data
    nested = {}

    def askSaveBeforeClosing():
        # As if the user reopened data while the prompt is open
        nested['images'] = loading_window.loadImagesPaths(
            [images_path], channel='phase'
        )
        nested['message'] = loading_window.statusBar().currentMessage()
        return False

    monkeypatch.setattr(
        loading_window, 'askSaveBeforeClosing', askSaveBeforeClosing
    )
    assert not loading_window.close()
    assert nested['images'] is None
    assert 'Save changes before closing' in nested['message']
    assert loading_window.data is loaded
    assert not loading_window._isClosingPrompt


def test_load_is_refused_while_measurement_processing_runs(
        loading_window, tmp_path, monkeypatch):
    """Processing results must not be merged into (and recorded for) the
    data being replaced."""
    from tests.test_measurement_viewer import make_timelapse_position
    images_path = make_timelapse_position(tmp_path)
    load_position = Mock(wraps=measurements_viewer.load_position)
    monkeypatch.setattr(measurements_viewer_utils, 'load_position', load_position)
    error = Mock()
    monkeypatch.setattr(loading_window, 'configurationError', error)

    loading_window.measurementProcessingThread = object()
    try:
        loading_window.loadImagesPaths([images_path], channel='phase')
        assert load_position.call_count == 0
        assert error.call_count == 1

        # Processing started while the "Save changes?" prompt was open
        loading_window.measurementProcessingThread = None

        def askSaveBeforeLoading():
            loading_window.measurementProcessingThread = object()
            return True

        monkeypatch.setattr(
            loading_window, 'askSaveBeforeLoading', askSaveBeforeLoading
        )
        loading_window.loadImagesPaths([images_path], channel='phase')
        assert load_position.call_count == 0
        assert error.call_count == 2
    finally:
        loading_window.measurementProcessingThread = None
    assert not loading_window.data


def test_processing_is_refused_while_loading(viewer, monkeypatch):
    dialog = Mock(side_effect=AssertionError('dialog must not open'))
    monkeypatch.setattr(
        measurements_viewer.apps_measurements_viewer,
        'MeasurementProcessingRecipeDialog', dialog
    )
    viewer._isLoadingData = True
    try:
        viewer.processMeasurements()
    finally:
        viewer._isLoadingData = False
    dialog.assert_not_called()
    assert 'Wait for the data being loaded' in (
        viewer.statusBar().currentMessage()
    )
    assert viewer.measurementProcessingThread is None


def test_configuration_overlays_are_read_in_the_worker_thread(
        loading_window, tmp_path, monkeypatch):
    import tifffile
    from cellacdc import load
    from tests.test_measurement_viewer import make_timelapse_position
    images_path = make_timelapse_position(tmp_path)
    overlay = np.full((3, 30, 30), 7, dtype=np.uint16)
    tifffile.imwrite(f'{images_path}/test_gfp.tif', overlay)
    loading_window.loadImagesPaths([images_path], channel='phase')
    filepath = tmp_path / 'configuration.json'
    filepath.write_text(json.dumps(loading_window.configuration()))

    def selectInput(config):
        option = loading_window.configurationInputOptions(config)[0]
        return dict(option, overlay_channels=['gfp'])

    monkeypatch.setattr(loading_window, 'selectConfigurationInput', selectInput)
    monkeypatch.setattr(
        loading_window, 'askSaveBeforeLoading', Mock(return_value=True)
    )
    threads = []
    load_image = load.load_image_and_bkgr_data

    def recorded_load_image(*args, **kwargs):
        threads.append(threading.current_thread())
        return load_image(*args, **kwargs)

    monkeypatch.setattr(load, 'load_image_and_bkgr_data', recorded_load_image)
    assert loading_window.loadConfiguration(filepath=str(filepath))
    assert not loading_window.slotErrors
    assert threads and threading.main_thread() not in threads
    posData, = loading_window.data
    assert 'gfp' in posData.loadedFluoChannels
    olData, = posData.ol_data.values()
    assert (olData == 7).all()
    assert not hasattr(posData, '_viewerPreloadedFluo')


@pytest.fixture
def props_dock_viewer(app, monkeypatch, tmp_path):
    """Viewer with real data (cell 1 in frames 1-3, cell 2 in frames 2-3)
    and the "Cell-ACDC objects" dock visible."""
    from tests.test_measurement_viewer import make_timelapse_position
    errors = []
    monkeypatch.setattr(
        sys, 'excepthook', lambda *exc_info: errors.append(exc_info)
    )
    window = MeasurementsViewerWin()
    window._startupConfigurationPrompted = True
    window.slotErrors = errors
    images_path = make_timelapse_position(
        tmp_path, SizeT=3, extra_cells=((1, 2), (2, 2))
    )
    window.setData(measurements_viewer.load_positions([images_path], 'phase'))
    window.restrictFramesCheckbox.setChecked(False)
    monkeypatch.setattr(window.propsDockWidget, 'isVisible', lambda: True)
    yield window
    window.deleteLater()
    app.processEvents()


def test_selected_cell_is_shown_in_objects_dock(props_dock_viewer):
    window = props_dock_viewer
    propsQGBox = window.guiTabControl.propsQGBox
    window.setFrame(1)
    window.goToObjectID(2)
    assert window.selectedID == 2
    assert propsQGBox.idSB.value() == 2
    # Cell 2 is 4x5 pixels, cell 1 is 3x4
    assert propsQGBox.cellAreaPxlSB.value() == 20
    assert propsQGBox.notExistingIDLabel.text() == ''
    # Cell 2 is not in the first frame: the dock says so, without the
    # values of the cell shown before
    window.setFrame(0)
    assert propsQGBox.idSB.value() == 2
    assert 'does not exist' in propsQGBox.notExistingIDLabel.text()
    assert propsQGBox.cellAreaPxlSB.text() == ''
    assert window.guiTabControl.intensMeasurQGBox.meanDSB.text() == ''
    window.goToObjectID(1)
    assert propsQGBox.idSB.value() == 1
    assert propsQGBox.cellAreaPxlSB.value() == 12
    assert propsQGBox.notExistingIDLabel.text() == ''
    assert not window.slotErrors


def test_editing_objects_dock_id_selects_the_cell(
        props_dock_viewer, monkeypatch
    ):
    window = props_dock_viewer
    propsQGBox = window.guiTabControl.propsQGBox
    window.setFrame(1)
    assert window.selectedID == 1
    plotMeasurements = Mock(wraps=window.plotMeasurements)
    monkeypatch.setattr(window, 'plotMeasurements', plotMeasurements)
    # As typing in the dock (emits `valueChanged`)
    propsQGBox.idSB.setValue(2)
    assert window.selectedID == 2
    assert window.idSpinBox.value() == 2
    assert window.IDs[window.idScrollBar.value()] == 2
    assert propsQGBox.cellAreaPxlSB.value() == 20
    assert plotMeasurements.call_count == 1
    # IDs that are not in the table do not change the selection
    propsQGBox.idSB.setValue(7)
    assert window.selectedID == 2
    assert 'does not exist' in propsQGBox.notExistingIDLabel.text()
    assert not window.slotErrors


def test_objects_dock_with_3D_cell_outside_the_z_slice(
        app, monkeypatch, tmp_path):
    from tests.test_measurement_viewer import make_timelapse_position
    errors = []
    monkeypatch.setattr(
        sys, 'excepthook', lambda *exc_info: errors.append(exc_info)
    )
    images_path = make_timelapse_position(tmp_path, SizeT=2, SizeZ=10)
    labels = np.zeros((2, 10, 30, 30), dtype=np.uint16)
    # Cell 1 in the z-slices 2-4
    labels[:, 2:5, 2:5, 3:7] = 1
    np.savez_compressed(f'{images_path}/test_segm.npz', labels)
    window = MeasurementsViewerWin()
    window._startupConfigurationPrompted = True
    try:
        window.setData(
            measurements_viewer.load_positions([images_path], 'phase')
        )
        assert window.isSegm3D
        window.followIDcheckbox.setChecked(False)
        monkeypatch.setattr(window.propsDockWidget, 'isVisible', lambda: True)
        window.zProjComboBox.setCurrentText('single z-slice')
        propsQGBox = window.guiTabControl.propsQGBox
        window.zSliceScrollBar.setSliderPosition(3)
        window.updateImages()
        assert propsQGBox.cellAreaPxlSB.value() == 12
        for z in (8, 0):
            window.zSliceScrollBar.setSliderPosition(z)
            window.updateImages()
            assert 'not in this z-slice' in (
                propsQGBox.notExistingIDLabel.text()
            )
            assert propsQGBox.cellAreaPxlSB.text() == ''
        assert not errors
    finally:
        window.deleteLater()
        app.processEvents()


def test_hover_updates_objects_dock_without_selecting(props_dock_viewer):
    window = props_dock_viewer
    propsQGBox = window.guiTabControl.propsQGBox
    window.setFrame(1)
    window.guiTabControl.highlightCheckbox.setChecked(True)
    assert window.selectedID == 1
    window.updatePropsWidget(2, fromHover=True)
    assert propsQGBox.idSB.value() == 2
    assert window.selectedID == 1
    # The next image update shows the selected cell again
    window.updateImages()
    assert propsQGBox.idSB.value() == 1
    assert window.selectedID == 1
    assert not window.slotErrors


# ---------------------------------------------------------------------------
# Documentation (Help menu, info buttons) and phase translator toolbutton
# ---------------------------------------------------------------------------
DOCS_RST_PATH = (
    __import__('pathlib').Path(measurements_viewer.__file__).parent
    / 'docs' / 'source' / 'measurements-viewer.rst'
)


class _RecordingMessageBox:
    """Stand-in for `widgets.myMessageBox` recording `information` calls."""
    calls = []

    def __init__(self, *args, **kwargs):
        pass

    def information(self, parent, title, message, **kwargs):
        type(self).calls.append(
            {'parent': parent, 'title': title, 'message': message, **kwargs}
        )


@pytest.fixture
def recorded_messages(monkeypatch):
    _RecordingMessageBox.calls = []
    monkeypatch.setattr(widgets, 'myMessageBox', _RecordingMessageBox)
    return _RecordingMessageBox.calls


@pytest.fixture
def bare_viewer(app):
    window = MeasurementsViewerWin()
    window._startupConfigurationPrompted = True
    yield window
    window.deleteLater()
    app.processEvents()


def test_phase_translator_action_has_icon_and_long_tooltip(bare_viewer):
    action = bare_viewer.phaseTranslatorAction
    assert not action.icon().isNull()
    tooltip = action.toolTip()
    assert len(tooltip) > 500
    for text in (
            'Add time point identifier', 'Apply phase evaluation',
            'phase_<name>', 'band'
        ):
        assert text in tooltip
    assert action.statusTip()
    toolbar = bare_viewer.phaseTranslatorToolbar
    assert toolbar.windowTitle() == 'Phases'
    button = toolbar.widgetForAction(action)
    assert button.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonIconOnly
    # Also reachable from a menu, hence from the functionality search
    assert action in bare_viewer.annotationsMenu.actions()
    assert 'time point to phase translator' in str(
        bare_viewer.searchWidget._search_records
    ).casefold()


def test_phase_translator_icon_is_colorful_but_not_red():
    for color in measurements_viewer.PHASE_TRANSLATOR_ICON_COLORS:
        hue = QColor(color).hue()
        assert 30 <= hue <= 330
    image = measurements_viewer.phase_translator_icon(32).pixmap(32, 32)
    image = image.toImage()
    colors = {
        image.pixelColor(x, y).name()
        for x in range(image.width()) for y in range(image.height())
        if image.pixelColor(x, y).alpha() == 255
    }
    for color in measurements_viewer.PHASE_TRANSLATOR_ICON_COLORS:
        assert color.lower() in colors


def test_help_menu_opens_user_guide(bare_viewer, monkeypatch):
    opened = []
    monkeypatch.setattr(measurements_viewer.myutils, 'browse_url', opened.append)
    helpActions = bare_viewer.helpMenu.actions()
    assert bare_viewer.focusOnSearchAction in helpActions
    userGuideAction, = [
        action for action in helpActions
        if action.text() == 'Measurements viewer user guide...'
    ]
    userGuideAction.trigger()
    assert opened == [measurements_viewer.urls.measurements_viewer_docs_url]
    bare_viewer.acdcDocsAction.trigger()
    assert opened[-1] == measurements_viewer.urls.docs_homepage


def test_quick_start_links_to_user_guide(viewer, recorded_messages):
    urls = measurements_viewer.urls
    viewer.measurementsHelpButton.click()
    viewer.quickStartAction.trigger()
    assert len(recorded_messages) == 2
    for call in recorded_messages:
        assert call['url_to_open'] == urls.measurements_viewer_docs_url
        assert 'Columns to plot' in call['message']
        assert 'Ctrl+F' in call['message']


def test_phase_translator_dialog_help_links_to_its_section(
        app, recorded_messages
    ):
    from cellacdc import apps_measurements_viewer, urls
    dialog = apps_measurements_viewer.PhaseTranslatorDialog([], [], [], {})
    try:
        dialog.helpButton.click()
    finally:
        dialog.deleteLater()
    call, = recorded_messages
    assert call['url_to_open'] == (
        urls.measurements_viewer_phase_translator_docs_url
    )
    assert call['url_to_open'].startswith(urls.measurements_viewer_docs_url)
    assert 'phase_&lt;name&gt;' in call['message']


def test_docs_page_has_the_linked_sections():
    from cellacdc import urls
    assert urls.measurements_viewer_docs_url.endswith(
        f'/{DOCS_RST_PATH.stem}.html'
    )
    lines = DOCS_RST_PATH.read_text(encoding='utf-8').splitlines()
    titles = {
        title.strip().casefold().replace(' ', '-')
        for title, underline in zip(lines, lines[1:])
        if underline and set(underline) <= set('=-~^*')
        and len(underline) >= len(title.strip()) > 0
    }
    anchor = urls.measurements_viewer_phase_translator_docs_url.split('#')[1]
    assert anchor in titles


def test_help_texts_build_real_message_boxes(app):
    from cellacdc import apps_measurements_viewer, html_utils
    for text in (
            measurements_viewer.MEASUREMENTS_VIEWER_QUICK_START,
            apps_measurements_viewer.PHASE_TRANSLATOR_HELP
        ):
        msg = widgets.myMessageBox()
        msg.information(
            None, 'Help', html_utils.paragraph(text),
            url_to_open=measurements_viewer.urls.measurements_viewer_docs_url,
            open_url_button_text='Open user guide', showDialog=False
        )
        assert msg.openUrlButton.text() == 'Open user guide'
        msg.deleteLater()


# Evaluate phases on displayed cell (toolbar button + shortcut)

def test_apply_phases_cell_action_has_icon_shortcut_and_toolbar(bare_viewer):
    action = bare_viewer.applyPhasesCellAction
    assert not action.icon().isNull()
    assert action.shortcut() == QKeySequence(
        measurements_viewer.APPLY_PHASES_CELL_SHORTCUT
    )
    assert 'Shortcut: "' in action.toolTip()
    assert action in bare_viewer.phaseTranslatorToolbar.actions()
    assert action in bare_viewer.annotationsMenu.actions()
    # Editable in the shortcuts editor
    assert bare_viewer.widgetsWithShortcut[
        'Evaluate phases on displayed cell'
    ] is action
    # Not a single key: it must not block time point/annotation shortcuts
    assert QKeySequence('P').toString() not in bare_viewer.reservedShortcuts()


def test_apply_phases_cell_action_evaluates_only_displayed_cell(viewer):
    from cellacdc.timepoint_predictors import TIMEPOINT_PREFIX
    column = f'{TIMEPOINT_PREFIX}Start'
    viewer.acdc_df[column] = [False, True, True, False]
    viewer.isDataLoaded = True
    viewer.phaseTranslatorRules = [{
        'name': 'Rule', 'predictor': {'type': 'none'},
        'timepoints': [column], 'annotations': [],
        'phases': ['Before', 'After'],
    }]
    assert viewer.selectedID == 1
    viewer.applyPhasesCellAction.trigger()
    phases = viewer.acdc_df['phase']
    assert phases.loc[(0, 1)] == 'Before'
    assert phases.loc[(1, 1)] == 'After'
    # Cell 2 (not displayed) is not evaluated
    assert not phases.loc[(1, 2)] and not phases.loc[(2, 2)]


def test_apply_phases_cell_action_without_rules_shows_hint(
        viewer, monkeypatch):
    viewer.isDataLoaded = True
    viewer.phaseTranslatorRules = []
    apply = Mock()
    monkeypatch.setattr(viewer, 'applyPhaseTranslator', apply)
    viewer.applyPhasesCellAction.trigger()
    apply.assert_not_called()
    assert 'No phase rules' in viewer.statusBar().currentMessage()


# Custom annotations are shown also when their button is not active

def test_custom_annotations_of_inactive_buttons_are_shown(app, tmp_path):
    from tests.test_measurement_viewer import make_timelapse_position
    window = MeasurementsViewerWin()
    window._startupConfigurationPrompted = True
    try:
        images_path = make_timelapse_position(
            tmp_path, SizeT=2, extra_cells=((0, 2), (1, 2))
        )
        window.setData(
            measurements_viewer.load_positions([images_path], 'phase')
        )
        # As the main GUI: the eye button "Show all custom annotations"
        assert window.viewAllCustomAnnotAction.isChecked()
        window.setCustomAnnotations([{
            'type': 'Single time-point', 'name': 'Big',
            'symbol': "'o' circle", 'shortcut': 'A', 'description': '',
            'keepActive': True, 'isHideChecked': True,
            'symbolColor': [255, 0, 0, 255],
        }])
        button = window.customAnnotButtonByName('Big')
        assert not button.isChecked() and button.isHideChecked
        assert window.toggleCustomAnnotation(button, 2)
        window.doCustomAnnotation(0)
        scatter = window.customAnnotDict[button]['scatterPlotItem']
        assert len(scatter.getData()[0]) == 1

        # Without "Show all", only the active annotation is shown
        window.viewAllCustomAnnotAction.setChecked(False)
        assert len(scatter.getData()[0]) == 0
        button.setChecked(True)
        assert len(scatter.getData()[0]) == 1
    finally:
        window.deleteLater()
        app.processEvents()
