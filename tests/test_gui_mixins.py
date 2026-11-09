"""Regression tests for GUI mixin methods shared by `gui.guiWin` and the
measurements viewer, driven on minimal host objects."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pyqtgraph as pg
import pytest
from qtpy.QtWidgets import QApplication, QGridLayout, QGroupBox, QWidget

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
    saved = pd.read_csv(host.settings_csv_path, index_col='setting')
    assert saved.at['is_bw_inverted', 'value'] == 'No'


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
