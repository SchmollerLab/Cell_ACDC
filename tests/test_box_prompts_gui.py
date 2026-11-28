"""GUI pieces of the box prompts of promptable models (e.g., micro-sam)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from qtpy.QtCore import QPointF, QRect, QRectF, Qt
from qtpy.QtWidgets import QApplication, QMenu

import pyqtgraph as pg

from cellacdc.gui.mixins import widgets as gui_mixin_widgets
from cellacdc.gui.widgets import points_layer_widgets as gui_widgets_points_layer_widgets
from cellacdc.gui.workers import segmentation_tracking as gui_workers_segmentation_tracking
from cellacdc.shared.widgets import plot_items as widgets_plot_items
from cellacdc.gui.main_window.window import guiWin
from cellacdc.service.client import ModelClient

pytestmark = pytest.mark.gui


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def popped(monkeypatch):
    menus = []
    monkeypatch.setattr(QMenu, 'popup', lambda self, *args: menus.append(self))
    return menus


def _box(app, editable=True):
    return gui_widgets_points_layer_widgets.PromptBoxROI(
        (10, 20), (30, 40), 7, (255, 0, 0),
        maxBounds=QRectF(QRect(0, 0, 100, 100)), editable=editable,
        boxData={'id': 7}
    )


def test_prompt_box_roi(app, popped):
    roi = _box(app)
    assert roi.bbox() == (20, 10, 60, 40)
    assert roi.labelItem.toPlainText() == '7'
    assert len(roi.getHandles()) == 4
    # Resized only with the handles (pyqtgraph Shift+drag ignores maxBounds)
    assert roi.mouseDragHandler.scaleModifier is None and not roi.resizable

    # Only a band centered on the border catches the mouse (no view: band
    # width is 10% of the smallest side)
    shape = roi.shape()
    assert shape.contains(QPointF(-1, 20)) and shape.contains(QPointF(1, 20))
    assert not shape.contains(QPointF(15, 20))
    assert roi.boundingRect().contains(QPointF(-1, 20))

    # Context menu with only the box action
    roi.raiseContextMenu(SimpleNamespace(screenPos=lambda: QPointF(5, 5)))
    assert [a.text() for a in popped[-1].actions()] == [
        'Remove box prompt (ID 7)'
    ]

    # Right-click while moving only cancels the move
    class ClickEvent:
        accepted = False
        def button(self): return Qt.MouseButton.RightButton
        def accept(self): self.accepted = True

    roi.isMoving = True
    roi.preMoveState = roi.getState()
    n_menus = len(popped)
    event = ClickEvent()
    roi.mouseClickEvent(event)
    assert event.accepted and not roi.isMoving and len(popped) == n_menus

    # Alt+drag on a corner handle is left to the view (pan)
    class DragEvent:
        ignored = False
        def isStart(self): return True
        def modifiers(self): return Qt.KeyboardModifier.AltModifier
        def button(self): return Qt.MouseButton.LeftButton
        def ignore(self): self.ignored = True

    state = roi.getState()
    for handle in roi.getHandles():
        assert isinstance(handle, gui_widgets_points_layer_widgets._PromptBoxHandle)
        event = DragEvent()
        handle.mouseDragEvent(event)
        assert event.ignored
    assert roi.getState() == state

    # Not editable: no handles, no menu, not movable, no mouse area
    roi.setEditable(False)
    assert not roi.translatable and not roi.contextMenuEnabled()
    assert not any(handle.isVisible() for handle in roi.getHandles())
    assert roi.shape().isEmpty()
    # pyqtgraph selects the ROI at the start of every left drag
    roi.setSelected(True)
    assert not any(handle.isVisible() for handle in roi.getHandles())
    roi.raiseContextMenu(SimpleNamespace(screenPos=lambda: QPointF(5, 5)))
    assert len(popped) == n_menus


def test_supported_prompt_types_keep_the_current_type(app):
    toolbar = gui_widgets_points_layer_widgets.MagicPromptsToolbar(None)
    toolbar.setSupportedPromptTypes(('point', 'box'))
    assert toolbar.promptType() == 'point'
    toolbar.setPromptType('box')
    # Re-initializing a model with points and boxes keeps the boxes
    toolbar.setSupportedPromptTypes(('point', 'box'))
    assert toolbar.promptType() == 'box'
    # Models without boxes --> first supported type
    toolbar.setSupportedPromptTypes(('point',))
    assert toolbar.promptType() == 'point'
    toolbar.setSupportedPromptTypes(('brush_stroke',))
    assert toolbar.promptType() == 'brush_stroke'
    toolbar.setSupportedPromptTypes(('point', 'box'))
    assert toolbar.promptType() == 'point'


def test_frame_change_keeps_box_prompts(app):
    """Changing frame removes the deletion ROIs of the frame, not the other
    ROIs (box prompts, box being drawn, label ROI)."""
    gui = SimpleNamespace(
        pos_i=0, labelRoiItem=gui_mixin_widgets.ROI((0, 0), (0, 0)),
        data=[SimpleNamespace(
            frame_i=0, allData_li=[{'delROIs_info': {'rois': []}}]
        )],
        ax1=widgets_plot_items.MainPlotItem(), ax2=widgets_plot_items.MainPlotItem(),
    )
    delRoi = pg.ROI((1, 1), (5, 5))
    gui.ax1.addDelRoiItem(delRoi, 'del1')
    delRoi2 = pg.ROI((1, 1), (5, 5))
    gui.ax2.addDelRoiItem(delRoi2, 'del2')
    boxRoi = _box(app)
    gui.ax1.addItem(boxRoi, ignoreBounds=True)
    drawingRoi = gui_mixin_widgets.ROI((20, 20), (3, 3))
    gui.ax1.addItem(drawingRoi)
    gui.ax1.addItem(gui.labelRoiItem)

    guiWin.removeAlldelROIsCurrentFrame(gui)

    assert delRoi not in gui.ax1.items and delRoi2 not in gui.ax2.items
    assert boxRoi in gui.ax1.items and drawingRoi in gui.ax1.items
    assert gui.labelRoiItem in gui.ax1.items


def test_clip_boxes_to_zoom():
    df_boxes = pd.DataFrame(
        [(0, 4, None, 10, 20, 50, 60), (0, 5, None, 80, 90, 95, 99)],
        columns=['frame_i', 'id', 'z', 'ymin', 'xmin', 'ymax', 'xmax']
    )
    zoomed = guiWin.clipMagicPromptsBoxesToZoom(df_boxes, 30, 40, 100, 120)
    assert zoomed[['id', 'ymin', 'xmin', 'ymax', 'xmax']].values.tolist() == [
        [4, 0, 0, 20, 20], [5, 50, 50, 65, 59]
    ]
    # Boxes outside the zoomed area are dropped
    assert guiWin.clipMagicPromptsBoxesToZoom(
        df_boxes, 0, 0, 10, 20
    ) is None
    # The input is not modified
    assert df_boxes['ymin'].tolist() == [10, 80]


class _Model:
    prompt_types = ('point', 'box')

    def __init__(self):
        self.prompts = []

    def add_prompt(self, **kwargs):
        self.prompts.append(kwargs)

    def segment(self, image, lab=None, **kwargs):
        return np.zeros(lab.shape, np.uint32)


class _Client(ModelClient):
    """Model in its isolated worker process (no server)"""
    prompt_types = ('point', 'box')

    def __init__(self, model_name):
        self.model_name = model_name
        self.prompts = []

    add_prompt = _Model.add_prompt
    segment = _Model.segment


@pytest.mark.parametrize('model, zoom, image_omitted', [
    (_Model(), False, False),  # in-process (e.g., custom) model
    (_Client('micro-sam_v1'), False, True),
    (_Client('micro-sam_v1'), True, False),  # prompts on a crop
    (_Client('micro-sam_v2'), False, True),
    (_Client('sam2'), False, True),
    (_Client('SAM2'), False, False),  # custom model, case-sensitive
    (_Client('my_custom_model'), False, False),
])
def test_worker_sends_the_image_with_the_prompts_only_if_needed(
        app, model, zoom, image_omitted
    ):
    img_data = np.zeros((1, 20, 30), np.uint16)
    posData = SimpleNamespace(lab=np.zeros((20, 30), np.uint32))
    image = img_data[0]
    origin = (0, 0, 0)
    if zoom:
        image = img_data[0][5:15, 10:20]
        origin = (0, 5, 10)
    df_points = pd.DataFrame(
        [(0, 0, np.nan, 2.0, 3.0, 1)],
        columns=['frame_i', 'Cell_ID', 'z', 'y', 'x', 'id']
    )
    df_boxes = pd.DataFrame(
        [(0, 2, np.nan, 1, 1, 8, 8)],
        columns=['frame_i', 'id', 'z', 'ymin', 'xmin', 'ymax', 'xmax']
    )
    worker = gui_workers_segmentation_tracking.MagicPromptsWorker(
        posData, image, df_points, model, {}, image_origin=origin,
        global_image=img_data[0], df_boxes=df_boxes
    )
    critical = []
    worker.signals.critical.connect(critical.append)
    worker.run()
    assert not critical, critical

    point, box = model.prompts
    assert point['prompt_type'] == 'point' and box['prompt_type'] == 'box'
    assert box['prompt'] == (None, 1.0, 1.0, 8.0, 8.0)
    assert box['prompt_id'] == 2 and box['image_origin'] == origin
    for prompt in (point, box):
        if image_omitted:
            assert prompt['image'] is None
        else:
            assert prompt['image'] is image
