"""Regression tests for GUI mixin methods shared by `gui.guiWin` and the
measurements viewer, driven on minimal host objects."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractSlider, QAction, QApplication, QGridLayout, QGroupBox,
    QMainWindow, QScrollBar, QWidget
)

from cellacdc import apps, gui, widgets
from cellacdc.gui_custom_annotations import CustomAnnotationsGuiElements
from cellacdc.gui_overlay import OverlayChannelsGuiElements
from cellacdc.gui_viewer_tools import (
    CcaTableGuiElements, DefaultSettingsGuiElements, SearchIDGuiElements
)


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
    monkeypatch.setattr(gui, '_font', gui._font)
    host = _sliders_host()
    gui.guiWin.resizeSlidersArea(host, fontSizeFactor=1.5, heightFactor=1.5)
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
    host.zSliceSpinbox = widgets.SpinBox(disableKeyPress=True)
    host.zSliceSpinbox.setMinimum(1)
    host.zSliceSpinbox.setMaximum(SizeZ)
    host.zSliceSpinbox.setValue(z+1)
    for name in ('zSliceScrollBar', 'zSliceOverlay_SB'):
        scrollbar = widgets.ScrollBar(Qt.Horizontal)
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
    gui.guiWin.showZsliceIndexToggled(host, checked)
    shown = host.zSliceSpinbox.text()
    expected = f'Overlay z-slice {int(shown):02}/{10 - int(checked)}'
    # Label set when toggling "Show z-slice index"...
    assert host.overlay_z_label.setText.call_args.args[0] == expected
    # ... and kept when the overlay is redrawn
    OverlayChannelsGuiElements.getOlImg(host, 'GFP')
    assert host.overlay_z_label.setText.call_args.args[0] == expected


class _ZSliceScrollBarHost:
    isZsliceScrollBarActive = gui.guiWin.isZsliceScrollBarActive
    zSliceScrollBarActionTriggered = gui.guiWin.zSliceScrollBarActionTriggered
    zSliceScrollBarReleased = gui.guiWin.zSliceScrollBarReleased

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


def test_view_cca_table_without_annotations_warns(app, monkeypatch):
    warnings = []
    monkeypatch.setattr(
        widgets.myMessageBox, 'warning',
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

    class Host(DefaultSettingsGuiElements):
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
    win = apps.AnnotateObjTrackSettingsDialog(settings=settings)
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
    monkeypatch.setattr(widgets.QDialogListbox, 'exec_', lambda self: None)
    host = _CustomAnnotHost(canDelete)
    CustomAnnotationsGuiElements.loadCustomAnnotations(host)
    hasDeleteButton = bool(
        getattr(host.selectAnnotWin, '_additionalButtons', None)
    )
    assert hasDeleteButton is canDelete

    dialog = apps.customAnnotationDialog(
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
    from cellacdc.measurements_viewer import MeasurementsViewerWin
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
    host.annotateToolbar = widgets.ToolBar('Custom annotations', host)
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
        apps, 'customAnnotationDialog', lambda *args, **kwargs: dialog
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
