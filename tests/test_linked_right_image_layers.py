"""Tests of the left image layers that are mirrored on the right image when
"Show duplicated left image" is on (`widgets.ParentImageItem` with a linked
right item): the "Keep IDs" layer (activation and rendering) and the
brush/eraser temp layer (activation only).

The `guiWin` methods are tested on fake `self` objects (no main window).
"""
import ast
import inspect
import os
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

pytest.importorskip('qtpy')

import pandas as pd
import pyqtgraph as pg
from qtpy.QtWidgets import QAction, QApplication

from cellacdc import gui, widgets


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def _lab():
    lab = np.zeros((6, 8), dtype=np.uint32)
    lab[0:2, 0:2] = 1
    lab[0:2, 4:6] = 2
    lab[4:6, 5:8] = 3
    return lab


LUT = np.array(
    [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8
)


def _guiWithPlotItems():
    """Fake `guiWin` with the items created by `gui_createPlotItems`"""
    labelsGrad = SimpleNamespace(
        showRightImgAction=QAction('Show duplicated left image'),
        showNextFrameAction=QAction('Show next frame'),
    )
    labelsGrad.showRightImgAction.setCheckable(True)
    labelsGrad.showNextFrameAction.setCheckable(True)
    fake = SimpleNamespace(
        df_settings=pd.DataFrame(columns=['value']),
        gui_createTextAnnotColors=Mock(),
        gui_createContourPens=Mock(),
        gui_createMothBudLinePens=Mock(),
        NewBudMoth_Pen=pg.mkPen('r'),
        redDashLinePen=pg.mkPen('r'),
        rightImageFramesScrollbar=None,
        imgGrad=Mock(),
        imgGradRight=Mock(),
        ax1=Mock(),
        ax2=Mock(),
        labelsGrad=labelsGrad,
        rescaleIntensitiesLut=Mock(),
    )
    MethodType(gui.guiWin.gui_createPlotItems, fake)()
    return fake


def _keepIDsGui(keptIDs):
    fake = _guiWithPlotItems()
    lab = _lab()
    fake.lut = LUT
    fake.keepIDsButton = SimpleNamespace(isChecked=lambda: True)
    fake.keptObjectsIDs = list(keptIDs)
    fake.getObjectsLab2D = lambda IDs: np.where(
        np.isin(lab, list(IDs)), lab, 0
    ).astype(lab.dtype)
    fake.initKeepObjLabelsLayers = MethodType(
        gui.guiWin.initKeepObjLabelsLayers, fake
    )
    fake.updateTempLayerKeepIDs = MethodType(
        gui.guiWin.updateTempLayerKeepIDs, fake
    )
    return fake


def _rgba(item, x, y):
    item.render()
    color = item.qimage.pixelColor(x, y)
    return color.red(), color.green(), color.blue(), color.alpha()


def _parentImageItemCalls():
    tree = ast.parse(inspect.getsource(gui))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == 'ParentImageItem':
            yield node


def test_parent_image_item_keywords_are_not_swallowed():
    # Unknown keywords end up in `**kargs` and are silently ignored by
    # `pg.ImageItem` (e.g., the `activatingAction` typo)
    params = inspect.signature(widgets.ParentImageItem.__init__).parameters
    named = {
        name for name, param in params.items()
        if param.kind is not inspect.Parameter.VAR_KEYWORD
    }
    calls = list(_parentImageItemCalls())
    assert len(calls) >= 3
    for call in calls:
        for keyword in call.keywords:
            assert keyword.arg in named, (
                f'gui.py:{call.lineno}: unknown keyword `{keyword.arg}` '
                'passed to ParentImageItem'
            )


@pytest.mark.parametrize('name', ['keepIDsTempLayerLeft', 'tempLayerImg1'])
def test_linked_layer_follows_show_right_image(app, name):
    fake = _guiWithPlotItems()
    item = getattr(fake, name)
    action = fake.labelsGrad.showRightImgAction

    action.setChecked(False)
    assert not item.isLinkedImageItemActive()

    action.setChecked(True)
    assert item.isLinkedImageItemActive()

    # The next frame on the right is a different frame: do not mirror
    action.setChecked(False)
    fake.labelsGrad.showNextFrameAction.setChecked(True)
    assert not item.isLinkedImageItemActive()


def test_keep_ids_not_drawn_on_right_image_when_hidden(app):
    fake = _keepIDsGui([1, 3])
    fake.labelsGrad.showRightImgAction.setChecked(False)
    fake.initKeepObjLabelsLayers()
    fake.updateTempLayerKeepIDs()

    assert fake.keepIDsTempLayerLeft.image is not None
    assert fake.keepIDsTempLayerRight.image is None


def test_keep_ids_drawn_on_right_image_like_left(app):
    fake = _keepIDsGui([1, 3])
    fake.labelsGrad.showRightImgAction.setChecked(True)
    fake.initKeepObjLabelsLayers()
    fake.updateTempLayerKeepIDs()

    left = fake.keepIDsTempLayerLeft
    right = fake.keepIDsTempLayerRight
    assert right.image is not None
    assert np.array_equal(right.image, left.image)
    assert np.array_equal(right.lut, left.lut)
    assert np.allclose(right.levels, [0, len(LUT)])
    # Kept objects in their labels colors, transparent background
    assert _rgba(right, 0, 0) == _rgba(left, 0, 0) == (255, 0, 0, 255)
    assert _rgba(right, 6, 5) == _rgba(left, 6, 5) == (0, 0, 255, 255)
    assert _rgba(right, 4, 0) == (0, 0, 0, 0)  # ID 2 not kept
    assert _rgba(right, 3, 3) == (0, 0, 0, 0)  # background

    # Levels do not follow the max kept ID (colors must not change)
    fake.keptObjectsIDs = [1]
    fake.updateTempLayerKeepIDs()
    assert np.allclose(right.levels, [0, len(LUT)])
    assert _rgba(right, 0, 0) == _rgba(left, 0, 0) == (255, 0, 0, 255)
    assert _rgba(right, 3, 3) == (0, 0, 0, 0)


def test_keep_ids_right_image_shown_after_selection(app):
    # Right image activated after "Keep IDs" was initialised
    fake = _keepIDsGui([3])
    fake.labelsGrad.showRightImgAction.setChecked(False)
    fake.initKeepObjLabelsLayers()
    fake.updateTempLayerKeepIDs()
    assert fake.keepIDsTempLayerRight.image is None

    fake.labelsGrad.showRightImgAction.setChecked(True)
    # `updateAllImages` clears the layers (`clearAx1Items`) and redraws
    fake.keepIDsTempLayerLeft.clear()
    fake.keepIDsTempLayerRight.clear()
    fake.updateTempLayerKeepIDs()

    right = fake.keepIDsTempLayerRight
    assert _rgba(right, 6, 5) == (0, 0, 255, 255)
    assert _rgba(right, 3, 3) == (0, 0, 0, 0)

    fake.keepIDsTempLayerLeft.clear()
    assert right.image is None


def test_init_keep_obj_layers_other_item_does_not_touch_right_layer(app):
    # Used by the "select cells" of the brush strokes prompts
    fake = _keepIDsGui([1])
    item = pg.ImageItem()
    fake.initKeepObjLabelsLayers(imageItem=item)

    assert item.lut is not None
    assert fake.keepIDsTempLayerRight.lut is None
    assert fake.keepIDsTempLayerLeft.lut is None
