"""Regression tests for GUI mixin methods shared by `gui.guiWin` and the
measurements viewer, driven on minimal host objects."""
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest
from qtpy.QtCore import QEvent, QPointF, Qt, QTimer
from qtpy.QtGui import QKeyEvent, QMouseEvent
from qtpy.QtTest import QTest
from qtpy.QtWidgets import (
    QAbstractSlider, QAction, QApplication, QGridLayout, QGroupBox,
    QMainWindow, QPushButton, QScrollBar, QStyle, QStyleOptionSlider, QWidget
)

from cellacdc.gui.main_window import (
    constants as gui_main_constants,
    window as gui_window,
)
from cellacdc.gui.mixins import (
    dialogs as gui_mixin_dialogs,
    widgets as gui_mixin_widgets,
)
from cellacdc.shared.widgets import (
    item_views as widgets_item_views,
    message_box as widgets_message_box,
    numeric_inputs as widgets_numeric_inputs,
    toolbars as widgets_toolbars,
)
from cellacdc.measurements_viewer.main_window.window import MeasurementsViewerWin
from cellacdc.gui.mixins.custom_annotations import CustomAnnotationsGuiElements
from cellacdc.gui.mixins.overlay import OverlayChannelsGuiElements
from cellacdc.gui.mixins.object_info import (
    CcaTableGuiElements,
    SearchIDGuiElements,
)
from cellacdc.gui.mixins.window_core import DefaultSettingsGuiElements
from cellacdc.gui.mixins.navigation import LutGuiElements

pytestmark = pytest.mark.gui


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def _sliders_host():
    groupbox = QGroupBox()
    layout = QGridLayout()
    groupbox.setLayout(layout)
    host = MagicMock()
    host.img1BottomGroupbox = groupbox
    host.bottomLeftLayout = layout
    host.annotOptionsCheckboxes = {
        0: {'layout': QGridLayout()}, 1: {'layout': QGridLayout()}
    }
    host.overlayLayersItems = {}
    host.h = 20
    host.checkBoxesHeight = 14
    host.fontPixelSize = 11
    host.alphaScrollbarRow = 0
    host.setOpacityOverlayLayersItems = lambda *args, **kwargs: None
    host._resizeSlidersArea = lambda: None
    return host


def test_alpha_scrollbar_label_follows_bottom_layout_zoom(app, monkeypatch):
    # resizeSlidersArea reassigns gui.py's module-level font: restore it
    monkeypatch.setattr(gui_main_constants, '_font', gui_main_constants._font)
    host = _sliders_host()
    gui_window.guiWin.resizeSlidersArea(host, fontSizeFactor=1.5, heightFactor=1.5)
    alphaScrollBar = OverlayChannelsGuiElements.addAlphaScrollbar(
        host, 'GFP', pg.ImageItem()
    )
    assert alphaScrollBar.label.font().pixelSize() == 16


def test_alpha_scrollbar_label_default_font_without_host_font(app):
    host = SimpleNamespace(
        bottomLeftLayout=QGridLayout(), h=20, alphaScrollbarRow=0,
        setOpacityOverlayLayersItems=lambda *args, **kwargs: None
    )
    alphaScrollBar = OverlayChannelsGuiElements.addAlphaScrollbar(
        host, 'GFP', pg.ImageItem()
    )
    assert alphaScrollBar.label.font().pixelSize() == 11


def _overlay_z_label_host(tmp_path, z, SizeZ=10):
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QComboBox, QLabel
    host = SimpleNamespace(
        editIDToolbar=MagicMock(), SizeZlabel=QLabel(),
        df_settings=pd.DataFrame({'value': []}, dtype=object),
        settings_csv_path=str(tmp_path / 'settings.csv'),
        overlay_z_label=MagicMock(), pos_i=0,
        updateOverlayZslice=lambda *args: None,
        data=[SimpleNamespace(
            SizeZ=SizeZ, frame_i=0,
            ol_data={'GFP': np.zeros((1, SizeZ, 2, 2))}
        )]
    )
    host.overlay_z_label.isVisible.return_value = True
    host.zSliceSpinbox = widgets_numeric_inputs.SpinBox(disableKeyPress=True)
    host.zSliceSpinbox.setMinimum(1)
    host.zSliceSpinbox.setMaximum(SizeZ)
    host.zSliceSpinbox.setValue(z+1)
    for name in ('zSliceScrollBar', 'zSliceOverlay_SB'):
        scrollbar = widgets_numeric_inputs.ScrollBar(Qt.Horizontal)
        scrollbar.setMaximum(SizeZ-1)
        scrollbar.setSliderPosition(z)
        setattr(host, name, scrollbar)
    for name in ('zProjOverlay_CB', 'zProjComboBox'):
        combobox = QComboBox()
        combobox.addItem('single z-slice')
        setattr(host, name, combobox)
    return host


@pytest.mark.parametrize('z', [0, 4])
@pytest.mark.parametrize('checked', [True, False])
def test_overlay_z_label_matches_z_slice_spinbox(app, tmp_path, z, checked):
    host = _overlay_z_label_host(tmp_path, z)
    gui_window.guiWin.showZsliceIndexToggled(host, checked)
    shown = host.zSliceSpinbox.text()
    expected = f'Overlay z-slice {int(shown):02}/{10 - int(checked)}'
    # Label set when toggling "Show z-slice index"...
    assert host.overlay_z_label.setText.call_args.args[0] == expected
    # ... and kept when the overlay is redrawn
    OverlayChannelsGuiElements.getOlImg(host, 'GFP')
    assert host.overlay_z_label.setText.call_args.args[0] == expected


class _ZSliceScrollBarHost:
    isZsliceScrollBarActive = gui_window.guiWin.isZsliceScrollBarActive
    zSliceScrollBarActionTriggered = gui_window.guiWin.zSliceScrollBarActionTriggered
    zSliceScrollBarReleased = gui_window.guiWin.zSliceScrollBarReleased

    def __init__(self, SizeZ):
        self.data = [SimpleNamespace(SizeZ=SizeZ, segmInfo_df=None)]
        self.pos_i = 0
        self.zSliceScrollBar = QScrollBar(Qt.Orientation.Horizontal)
        self.zSliceScrollBar.setMaximum(max(SizeZ - 1, 0))
        self.update_z_slice = MagicMock()
        self.clearTempBrushImage = MagicMock()


@pytest.mark.parametrize('SizeZ', [1, 3])
def test_z_slice_scrollbar_keys_with_2d_data(app, SizeZ):
    # A z-stack loaded before in the same GUI leaves the z-slice scrollbar
    # connected (`connectScrollbars`): Up/Down on 2D data then reached
    # `update_z_slice`, which crashed on `segmInfo_df` (None)
    host = _ZSliceScrollBarHost(SizeZ)
    host.zSliceScrollBar.actionTriggered.connect(
        host.zSliceScrollBarActionTriggered
    )
    host.zSliceScrollBar.triggerAction(
        QAbstractSlider.SliderAction.SliderSingleStepAdd
    )
    host.zSliceScrollBarReleased()
    assert host.update_z_slice.call_count == (0 if SizeZ == 1 else 2)


class _Qt5LikeAction(QAction):
    """`trigger()` emits `triggered` also when the action is disabled, as
    Qt5 does (Qt6 ignores it), so that the tests catch a disabled action
    being triggered with either binding."""
    def trigger(self):
        self.triggered.emit(False)


class _PersistentShortcutsHost(gui_window.guiWin):
    # The real `guiWin.eventFilter` without building the GUI: its fall-through
    # calls the zero-argument `super().eventFilter`, so the host must be a
    # `guiWin`. Never shown (`guiWin.showEvent` needs the launcher).
    def __init__(self):
        QMainWindow.__init__(self)
        self.isDataLoaded = True
        self.triggered = []
        self.widgetsWithShortcut = {}
        self.widgetsPersistentShortcut = {}
        keys = {'Next': Qt.Key.Key_Right, 'Previous': Qt.Key.Key_Left}
        for name, key in keys.items():
            action = _Qt5LikeAction(name, self)
            action.triggered.connect(
                lambda checked=False, name=name: self.triggered.append(name)
            )
            self.widgetsWithShortcut[name] = action
            self.widgetsPersistentShortcut[name] = key


def test_gui_persistent_shortcuts_only_for_own_windows(app):
    # The main GUI's filter is installed on the app: Left/Right step its
    # frames only for key presses of its own windows (not, e.g., of the
    # measurements viewer), including its 3D viewer, a child of the launcher
    host = _PersistentShortcutsHost()
    ownButton = QPushButton('own', host)
    launcher = QMainWindow()
    renderer = QMainWindow(launcher)
    rendererButton = QPushButton('3D', renderer)
    renderer.setCentralWidget(rendererButton)
    other = QWidget()
    otherButton = QPushButton('other', other)

    def press(receiver, key=Qt.Key.Key_Right):
        host.triggered.clear()
        event = QKeyEvent(
            QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier
        )
        handled = host.eventFilter(receiver, event)
        return bool(handled), list(host.triggered)

    try:
        assert press(otherButton) == (False, [])
        assert press(ownButton) == (True, ['Next'])
        assert press(ownButton, Qt.Key.Key_Left) == (True, ['Previous'])
        # 3D viewer never opened (no attribute) or closed (`_launch3dViewer`
        # resets it to None)
        assert press(rendererButton) == (False, [])
        host._volume_renderer = None
        assert press(rendererButton) == (False, [])
        host._volume_renderer = renderer
        assert press(rendererButton) == (True, ['Next'])
        assert press(otherButton) == (False, [])
        # Seen on the QWindow first: the focus widget tells the target
        for window, button, expected in (
                (renderer, rendererButton, ['Next']), (other, otherButton, [])
            ):
            window.show()
            window.activateWindow()
            QTest.qWaitForWindowActive(window, 1000)
            button.setFocus()
            app.processEvents()
            assert QApplication.focusWidget() is button
            handle = window.windowHandle()
            assert press(handle) == (bool(expected), expected)
        # Never while the GUI is disabled (busy, or by the measurements
        # viewer's `runInThread`), not even for its enabled 3D viewer: the
        # filter sees the key presses that Qt ignores for disabled widgets
        host.setEnabled(False)
        assert press(ownButton) == (False, [])
        assert press(host) == (False, [])
        assert press(rendererButton) == (False, [])
        host.setEnabled(True)
        assert press(ownButton) == (True, ['Next'])
        # Not while the action itself is disabled (lost cells prompt open,
        # frame navigation disabled): the key goes on to its target (e.g.,
        # to move the focus between the prompt's buttons)
        host.widgetsWithShortcut['Next'].setEnabled(False)
        assert press(ownButton) == (False, [])
        assert press(ownButton, Qt.Key.Key_Left) == (True, ['Previous'])
        host.widgetsWithShortcut['Next'].setEnabled(True)
        assert press(ownButton) == (True, ['Next'])
        # Nor for a disabled 3D viewer (its focus widget is cleared): seen
        # on its QWindow, then on the window itself
        renderer.activateWindow()
        QTest.qWaitForWindowActive(renderer, 1000)
        rendererButton.setFocus()
        renderer.setEnabled(False)
        app.processEvents()
        assert QApplication.activeWindow() is renderer
        assert QApplication.focusWidget() is None
        assert press(renderer.windowHandle()) == (False, [])
        assert press(renderer) == (False, [])
        renderer.setEnabled(True)
        rendererButton.setFocus()
        app.processEvents()
        assert press(renderer.windowHandle()) == (True, ['Next'])
    finally:
        for widget in (host, launcher, other):
            widget.hide()
            widget.deleteLater()
        app.processEvents()


class _StopAfterPersistentShortcuts(Exception):
    pass


def test_gui_key_press_event_skips_disabled_persistent_shortcuts(app):
    # A key that the app filter lets go on because its action is disabled
    # can reach `guiWin.keyPressEvent` (e.g., with the image focused:
    # pyqtgraph's view ignores the arrow keys), whose persistent shortcuts
    # must skip the disabled action too
    host = _PersistentShortcutsHost()
    host.editIDbutton = QPushButton(host)
    host.separateBudButton = QPushButton(host)

    def stop(ev):
        raise _StopAfterPersistentShortcuts

    # Called right after the persistent shortcuts (the rest of the method
    # needs the whole GUI)
    host.checkSetDelObjActionActive = stop
    # Without `exception_handler` (it would show the error)
    keyPressEvent = gui_window.guiWin.keyPressEvent.__wrapped__

    def press(key=Qt.Key.Key_Right):
        host.triggered.clear()
        event = QKeyEvent(
            QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier
        )
        with pytest.raises(_StopAfterPersistentShortcuts):
            keyPressEvent(host, event)
        return list(host.triggered)

    try:
        assert press() == ['Next']
        host.widgetsWithShortcut['Next'].setEnabled(False)
        assert press() == []
        assert press(Qt.Key.Key_Left) == ['Previous']
    finally:
        host.deleteLater()
        app.processEvents()


def _qmainwindow_method(name):
    """`QMainWindow`'s method `name`, to replace the `guiWin` one."""
    method = getattr(QMainWindow, name)
    return lambda self, *args: method(self, *args)


class _ShownPersistentShortcutsHost(_PersistentShortcutsHost):
    # Shown with the event handlers of QMainWindow (`guiWin`'s need the
    # whole GUI and the launcher)
    show = _qmainwindow_method('show')
    showEvent = _qmainwindow_method('showEvent')
    closeEvent = _qmainwindow_method('closeEvent')
    resizeEvent = _qmainwindow_method('resizeEvent')
    changeEvent = _qmainwindow_method('changeEvent')
    enterEvent = _qmainwindow_method('enterEvent')
    leaveEvent = _qmainwindow_method('leaveEvent')
    keyPressEvent = _qmainwindow_method('keyPressEvent')
    keyReleaseEvent = _qmainwindow_method('keyReleaseEvent')
    mousePressEvent = _qmainwindow_method('mousePressEvent')


@pytest.mark.slow
def test_gui_persistent_shortcuts_wait_for_the_viewer_load(app):
    # The measurements viewer's `runInThread` disables the other windows
    # (here the main GUI) while its worker runs. A disabled window can still
    # be activated (title bar click): Left/Right must not run `next_cb` /
    # `prev_cb` there, a dialog that they open would nest its event loop in
    # the wait (and keep the viewer disabled until it closes)
    host = _ShownPersistentShortcutsHost()
    host.setCentralWidget(QPushButton('own'))
    viewer = MeasurementsViewerWin()
    viewer._startupConfigurationPrompted = True
    recorded = {}

    def pressArrowKeys():
        recorded['enabled'] = host.isEnabled()
        host.activateWindow()
        QTest.qWaitForWindowActive(host, 1000)
        recorded['active'] = QApplication.activeWindow() is host
        for key in (Qt.Key.Key_Right, Qt.Key.Key_Left):
            QTest.keyClick(host.windowHandle(), key)
        recorded['triggered'] = list(host.triggered)

    app.installEventFilter(host)
    try:
        host.show()
        QTest.qWaitForWindowExposed(host, 1000)
        QTimer.singleShot(0, pressArrowKeys)
        viewer.runInThread(time.sleep, 0.2, text='Waiting...')
        assert recorded == {'enabled': False, 'active': True, 'triggered': []}
        # Enabled again: the keys step the frames
        assert host.isEnabled()
        host.activateWindow()
        QTest.qWaitForWindowActive(host, 1000)
        QTest.keyClick(host.windowHandle(), Qt.Key.Key_Right)
        assert host.triggered == ['Next']
    finally:
        app.removeEventFilter(host)
        host.hide()
        host.deleteLater()
        viewer.deleteLater()
        app.processEvents()


def test_gui_frames_scroll_bar_click_needs_an_enabled_scroll_bar(app):
    # With a single frame (range 0) the app filter makes a click on the
    # frames scroll bar's "add line" arrow step to the next frame (`next_cb`
    # through `actionTriggered`). The filter also sees the clicks that Qt
    # ignores for a disabled scroll bar (GUI busy, lost cells prompt open)
    host = _PersistentShortcutsHost()
    scrollBar = gui_mixin_widgets.navigateQScrollBar(Qt.Orientation.Horizontal, host)
    scrollBar.setRange(1, 1)
    scrollBar.resize(200, 20)
    host.navigateScrollBar = scrollBar
    actions = []
    scrollBar.actionTriggered.connect(actions.append)
    option = QStyleOptionSlider()
    scrollBar.initStyleOption(option)
    addLine = scrollBar.style().subControlRect(
        QStyle.ComplexControl.CC_ScrollBar, option,
        QStyle.SubControl.SC_ScrollBarAddLine, scrollBar
    ).center()

    def click():
        actions.clear()
        event = QMouseEvent(
            QEvent.Type.MouseButtonPress, QPointF(addLine),
            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier
        )
        host.eventFilter(scrollBar, event)
        return list(actions)

    try:
        # `actionTriggered` emits an int
        assert click() == [gui_main_constants.SliderSingleStepAdd]
        scrollBar.setEnabled(False)
        assert click() == []
        scrollBar.setEnabled(True)
        host.setEnabled(False)
        assert click() == []
    finally:
        host.deleteLater()
        app.processEvents()


def test_view_cca_table_without_annotations_warns(app, monkeypatch):
    warnings = []
    monkeypatch.setattr(
        widgets_message_box.myMessageBox, 'warning',
        lambda self, *args, **kwargs: warnings.append(args)
    )
    posData = SimpleNamespace(
        frame_i=1, cca_df=None,
        allData_li=[{'acdc_df': None}, {'acdc_df': None}]
    )
    host = SimpleNamespace(
        data=[posData], pos_i=0, ccaTableWin=None,
        getZoomIDs=lambda: [1, 2], logger=MagicMock()
    )
    CcaTableGuiElements.viewCcaTable(host)
    assert len(warnings) == 1
    assert host.ccaTableWin is None


def test_set_all_ids_skips_positions_without_stored_data():
    # Viewer positions that were never displayed have no allData_li yet
    notShownPos = SimpleNamespace(
        segm_data=[None, None], allData_li=None, allIDs={7}
    )
    shownPos = SimpleNamespace(
        segm_data=['lab0', 'lab1'], allIDs=set(),
        allData_li=[
            {'labels': 'lab0', 'regionprops': SimpleNamespace(IDs=[1, 2])},
            {'labels': None, 'regionprops': None},
        ]
    )
    host = SimpleNamespace(
        data=[notShownPos, shownPos],
        _acdcRegionProps=lambda lab, precache_centroids=True: (
            SimpleNamespace(IDs=[3])
        )
    )
    SearchIDGuiElements.setAllIDs(host)
    assert notShownPos.allIDs == {7}
    assert shownPos.allIDs == {1, 2, 3}
    assert shownPos.allData_li[1]['regionprops'].IDs == [3]


def test_restore_default_settings_with_inverted_bw(app, tmp_path):
    from qtpy.QtWidgets import QAction
    settings = {
        'contLineWeight': 2, 'mothBudLineSize': 2, 'mothBudLineColor': '',
        'contLineColor': '', 'overlaySegmMasksAlpha': 0.5,
        'img_cmap': 'viridis', 'labels_bkgrColor': '',
        'is_bw_inverted': 'Yes'
    }
    df_settings = pd.DataFrame(
        {'value': list(settings.values())}, index=list(settings.keys()),
        dtype=object
    )
    df_settings.index.name = 'setting'
    invertCalls = []

    class Host(DefaultSettingsGuiElements, LutGuiElements):
        def invertBw(self, checked, update=True):
            invertCalls.append((checked, update))
            self.df_settings.at['is_bw_inverted', 'value'] = (
                'Yes' if checked else 'No'
            )

    host = Host()
    host.df_settings = df_settings
    host.settings_csv_path = str(tmp_path / 'settings.csv')
    for name in (
            '_updateContColour', '_updateMothBudLineColour',
            '_updateMothBudLineSize', '_updateContLineThickness',
            'imgGrad', 'labelsGrad', 'updateAllImages'
        ):
        setattr(host, name, MagicMock())
    host.overlayLayersItems = {}
    # As `updateLabelsCmap` (sigGradientChangeFinished), which rebinds it
    def loadPreset(name):
        host.df_settings = host.df_settings.copy()
    host.labelsGrad.item.loadPreset.side_effect = loadPreset
    host.invertBwAction = QAction('Invert black/white')
    host.invertBwAction.setCheckable(True)
    host.invertBwAction.setChecked(True)
    toggledCalls = []
    host.invertBwAction.toggled.connect(toggledCalls.append)

    host.restoreDefaultSettings()

    assert invertCalls == [(False, False)]
    # The window's menu action is synced without re-triggering invertBw
    assert not host.invertBwAction.isChecked()
    assert toggledCalls == []
    # The LUTs are restored from the current (not inverted) settings
    restored = host.imgGrad.restoreState.call_args.args[0]
    assert restored.at['is_bw_inverted', 'value'] == 'No'
    saved = pd.read_csv(host.settings_csv_path, index_col='setting')
    assert saved.at['is_bw_inverted', 'value'] == 'No'
    assert saved.at['labels_bkgrColor', 'value'] == '(25, 25, 25)'


def test_annotate_obj_track_settings_dialog_3d_toggle(app):
    settings = {
        'length': 5, 'against_prev': False, 'color': (255, 100, 0),
        'n_fade': 15, 'max_width': 4, 'min_width': 1, 'max_alpha_perc': 100,
        'min_alpha_perc': 10, 'against_prev_line_color': (255, 165, 0),
        'against_prev_line_width': 2,
        'against_prev_contour_color': (245, 184, 0),
        'against_prev_contour_width': 2, '3D_show_only_visible': False
    }
    win = gui_mixin_dialogs.AnnotateObjTrackSettingsDialog(settings=settings)
    emitted = []
    win.sigValuesChanged.connect(emitted.append)

    win.showOnlyVisible3DCheckBox.setChecked(True)
    assert len(emitted) == 1
    assert emitted[-1]['3D_show_only_visible'] is True

    # setValues applies the value (and does not stack connections)
    win.setValues(settings)
    assert not win.showOnlyVisible3DCheckBox.isChecked()
    assert len(emitted) == 2
    win.ok_cb()
    assert win.settings['3D_show_only_visible'] is False
    win.close()


class _CustomAnnotHost(QWidget):
    def __init__(self, canDelete):
        super().__init__()
        self.canDelete = canDelete

    def readSavedCustomAnnot(self):
        self.savedCustomAnnot = {'Big': {}}

    def canDeleteSavedCustomAnnot(self):
        return self.canDelete

    def deleteSavedAnnotation(self):
        pass


@pytest.mark.parametrize('canDelete', [True, False])
def test_saved_custom_annotation_delete_button_follows_host(
        app, monkeypatch, canDelete
    ):
    monkeypatch.setattr(widgets_item_views.QDialogListbox, 'exec_', lambda self: None)
    host = _CustomAnnotHost(canDelete)
    CustomAnnotationsGuiElements.loadCustomAnnotations(host)
    hasDeleteButton = bool(
        getattr(host.selectAnnotWin, '_additionalButtons', None)
    )
    assert hasDeleteButton is canDelete

    dialog = gui_mixin_dialogs.customAnnotationDialog(
        {'Big': {}}, allowDeleteSaved=canDelete
    )
    dialog.loadSavedAnnot()
    hasDeleteButton = bool(
        getattr(dialog.selectAnnotWin, '_additionalButtons', None)
    )
    assert hasDeleteButton is canDelete
    dialog.close()
    host.close()


def test_measurements_viewer_cannot_delete_saved_custom_annotations():
    # The viewer's saveCustomAnnot does not write the global file
    from cellacdc.measurements_viewer.main_window.window import MeasurementsViewerWin
    assert MeasurementsViewerWin.canDeleteSavedCustomAnnot(None) is False
    assert CustomAnnotationsGuiElements.canDeleteSavedCustomAnnot(None)


class _CustomAnnotModifyHost(QMainWindow, CustomAnnotationsGuiElements):
    """Main GUI-like host (default `onCustomAnnotationChanged` hook)."""


@pytest.mark.parametrize('hide', [True, False])
def test_modify_dialog_hide_option_updates_drawn_symbols(
        app, monkeypatch, hide
    ):
    """Switching "Hide annotations" in the Modify dialog of an inactive
    button clears or redraws its symbols, as the context menu option does
    (regression: the main GUI kept the old display until the next frame)."""
    host = _CustomAnnotModifyHost()
    host.customAnnotDict = {}
    host.data = []
    host.savedCustomAnnot = {}
    host.logger = MagicMock()
    host.annotateToolbar = widgets_toolbars.ToolBar('Custom annotations', host)
    host.viewAllCustomAnnotAction = QAction(host)
    host.viewAllCustomAnnotAction.setCheckable(True)
    host.ax1 = pg.PlotItem()
    settings = {}
    host._get_setting_value = (
        lambda key, default, cast=None: settings.get(key, default)
    )
    host._set_setting_value = settings.__setitem__
    host.saveCustomAnnot = MagicMock()
    host.doCustomAnnotation = MagicMock()
    color = pg.mkColor((0, 114, 178))
    state = {
        'type': 'Single time-point', 'name': 'Dead', 'symbol': "'o' circle",
        'shortcut': '', 'description': '', 'keepActive': True,
        'isHideChecked': not hide, 'symbolColor': color,
    }
    button, action = host.addCustomAnnotationButton(
        'o', color, '', 'Dead', 'Dead', True, not hide
    )
    host.customAnnotDict[button] = {
        'action': action, 'state': state, 'annotatedIDs': [],
    }
    host.addCustomAnnnotScatterPlot(color, 'o', button)
    item = host.customAnnotDict[button]['scatterPlotItem']
    if hide:
        # Not hidden yet: the symbols of the inactive button are drawn
        item.setData([5, 6], [5, 6])
    dialog = SimpleNamespace(
        cancel=False, state={**state, 'isHideChecked': hide}, symbol='o',
        exec_=lambda: None, sigDeleteSelecAnnot=MagicMock(),
        deleteLater=lambda: None
    )
    monkeypatch.setattr(
        gui_mixin_dialogs, 'customAnnotationDialog', lambda *args, **kwargs: dialog
    )
    try:
        host.customAnnotModify(button)
        assert button.isHideChecked is hide
        assert settings['Dead_annotation_hidden'] is hide
        if hide:
            assert len(item.getData()[0]) == 0
            host.doCustomAnnotation.assert_not_called()
        else:
            host.doCustomAnnotation.assert_called_once_with(0)
    finally:
        host.close()
        host.deleteLater()
