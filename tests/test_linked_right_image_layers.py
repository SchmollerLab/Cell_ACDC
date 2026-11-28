"""Tests of the left image layers that are mirrored on the right image
(`widgets.ParentImageItem` with a linked right item): the "Keep IDs" layer
(when "Show duplicated left image" is on) and the brush/wand temp layer
(always drawn on the right too, because painting on the right image is
forwarded to the left image), activation and rendering.

The `guiWin` methods are tested on fake `self` objects (no main window).
"""
import ast
import inspect
import os
import pathlib
import warnings
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

pytestmark = pytest.mark.gui

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

pytest.importorskip('qtpy')

import pandas as pd
import pyqtgraph as pg
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QAction, QApplication

import cellacdc
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
    """`(path, call)` of the `ParentImageItem(...)` calls in the modules of
    cellacdc (`path`: posix path relative to the package folder)."""
    package = os.path.dirname(cellacdc.__file__)
    for dirpath, dirnames, filenames in os.walk(package):
        dirnames[:] = sorted(
            name for name in dirnames if name != '__pycache__'
        )
        for filename in sorted(filenames):
            if not filename.endswith('.py') or filename.startswith('qrc_'):
                continue
            path = os.path.join(dirpath, filename)
            with open(path, encoding='utf-8') as file:
                source = file.read()
            if 'ParentImageItem(' not in source:
                continue
            with warnings.catch_warnings():
                # Invalid escape sequences of old code
                warnings.simplefilter('ignore', SyntaxWarning)
                tree = ast.parse(source)
            relpath = pathlib.Path(os.path.relpath(path, package)).as_posix()
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = getattr(func, 'attr', getattr(func, 'id', None))
                if name == 'ParentImageItem':
                    yield relpath, node


@pytest.mark.slow
def test_parent_image_item_keywords_are_not_swallowed():
    # Unknown keywords end up in `**kargs` and are silently ignored by
    # `pg.ImageItem` (e.g., the `activatingAction` typo)
    params = inspect.signature(widgets.ParentImageItem.__init__).parameters
    named = {
        name for name, param in params.items()
        if param.kind is not inspect.Parameter.VAR_KEYWORD
    }
    calls = list(_parentImageItemCalls())
    # The main GUI's layers (also the viewer's image)
    assert len(calls) >= 3
    for path, call in calls:
        for keyword in call.keywords:
            if keyword.arg is None:
                # **kwargs passed on
                continue
            assert keyword.arg in named, (
                f'{path}:{call.lineno}: unknown keyword `{keyword.arg}` '
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


# Brush/wand temp layer (`tempLayerImg1` --> `tempLayerRightImage`)
BRUSH_LUT = np.array(
    [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255],
     [200, 100, 50]], dtype=np.uint8
)
BRUSH_ID = 5
BRUSH_RGBA = (200, 100, 50, 255)
CONTOURS_RGBA = (255, 255, 0, 255)
TRANSPARENT = (0, 0, 0, 0)


def _brushGui(rightImage='duplicated', contours=False, shape=(20, 30)):
    fake = _guiWithPlotItems()
    if rightImage == 'duplicated':
        fake.labelsGrad.showRightImgAction.setChecked(True)
    elif rightImage == 'next_frame':
        fake.labelsGrad.showNextFrameAction.setChecked(True)
    # rightImage == 'other' --> e.g., labels image or right image hidden
    lab = np.zeros(shape, dtype=np.uint32)
    lab[2:6, 20:26] = 2
    fake.data = [SimpleNamespace(brushID=BRUSH_ID, lab=lab)]
    fake.pos_i = 0
    fake.currentLab2D = lab
    fake.lut = BRUSH_LUT
    fake.img1.image = np.zeros(shape, dtype=np.uint16)
    fake.hideItemsHoverBrush = Mock()
    fake.contours = contours
    fake.isContoursChecked = lambda ax: fake.contours
    fake.get2DRP = lambda: SimpleNamespace(
        get_obj_from_ID=lambda ID, warn=False: None
    )
    fake.imgGrad = SimpleNamespace(
        labelsAlphaSlider=SimpleNamespace(value=lambda: 0.3),
        contoursColorButton=SimpleNamespace(
            color=lambda: QColor(*CONTOURS_RGBA)
        ),
    )
    for name in (
            'initTempLayerBrush', 'setTempImg1Brush', 'getTempLayerBrushLevels',
            'setTempBrushMaskFromWand', 'clearTempBrushImage'
        ):
        setattr(fake, name, MethodType(getattr(gui.guiWin, name), fake))
    return fake


def _brushMask(shape=(20, 30)):
    mask = np.zeros(shape, dtype=bool)
    mask[8:13, 8:13] = True
    return mask


def test_brush_drawn_on_right_image_with_brush_color(app):
    fake = _brushGui('duplicated')
    fake.setTempImg1Brush(True, _brushMask(), BRUSH_ID)
    fake.setTempImg1Brush(False, np.roll(_brushMask(), 3, axis=1), BRUSH_ID)

    left, right = fake.tempLayerImg1, fake.tempLayerRightImage
    assert np.array_equal(right.image, left.image)
    assert np.array_equal(right.lut, left.lut)
    assert np.allclose(right.levels, left.levels)
    assert right.opacity() == pytest.approx(left.opacity())
    # Brush color on the stroke, transparent background (not white on an
    # opaque black layer)
    assert _rgba(right, 10, 10) == _rgba(left, 10, 10) == BRUSH_RGBA
    assert _rgba(right, 14, 10) == _rgba(left, 14, 10) == BRUSH_RGBA
    assert _rgba(right, 1, 15) == _rgba(left, 1, 15) == TRANSPARENT


@pytest.mark.parametrize('rightImage', ['next_frame', 'other'])
def test_brush_drawn_on_right_image_not_duplicated(app, rightImage):
    # Painting on the right image is forwarded to the left image
    # (`gui_mousePressRightImage`) --> the preview is drawn on the right too
    fake = _brushGui(rightImage)
    fake.setTempImg1Brush(True, _brushMask(), BRUSH_ID)

    right = fake.tempLayerRightImage
    assert _rgba(right, 10, 10) == BRUSH_RGBA
    assert _rgba(right, 1, 15) == TRANSPARENT


def test_wand_drawn_on_right_image_with_brush_color(app):
    fake = _brushGui('duplicated')
    flood = np.zeros((20, 30), dtype=bool)
    flood[10:15, 2:7] = True
    fake.setTempBrushMaskFromWand(flood, init=True)

    right = fake.tempLayerRightImage
    assert _rgba(right, 4, 12) == BRUSH_RGBA
    assert _rgba(right, 1, 15) == TRANSPARENT


def test_brush_color_on_right_image_follows_brush_id(app):
    # E.g., `changeBrushID` re-initialises the layer with a new ID
    fake = _brushGui('duplicated')
    fake.setTempImg1Brush(True, _brushMask(), BRUSH_ID)
    fake.data[0].brushID = 3
    fake.setTempImg1Brush(True, _brushMask(), 3)

    left, right = fake.tempLayerImg1, fake.tempLayerRightImage
    assert _rgba(right, 10, 10) == _rgba(left, 10, 10) == (0, 0, 255, 255)


def test_brush_contours_drawn_on_right_image(app):
    # Contours mode after an overlay masks stroke: RGBA image (lut ignored)
    fake = _brushGui('duplicated')
    fake.setTempImg1Brush(True, _brushMask(), BRUSH_ID)
    fake.contours = True
    fake.setTempImg1Brush(True, _brushMask(), BRUSH_ID)

    left, right = fake.tempLayerImg1, fake.tempLayerRightImage
    assert right.image.shape == (20, 30, 4)
    assert _rgba(right, 8, 10) == _rgba(left, 8, 10) == CONTOURS_RGBA
    assert _rgba(right, 10, 10) == _rgba(left, 10, 10) == TRANSPARENT
    assert _rgba(right, 1, 15) == TRANSPARENT


def test_small_brush_stroke_visible_on_large_image(app):
    # Auto-levels are computed on a subsample of the pixels and can miss a
    # small stroke (it would be mapped to the transparent color)
    shape = (2048, 2048)
    fake = _brushGui('duplicated', shape=shape)
    mask = np.zeros(shape, dtype=bool)
    mask[1002:1007, 1002:1007] = True
    fake.setTempImg1Brush(True, mask, BRUSH_ID)

    left, right = fake.tempLayerImg1, fake.tempLayerRightImage
    assert np.allclose(left.levels, [0, BRUSH_ID])
    assert np.allclose(right.levels, [0, BRUSH_ID])
    assert _rgba(left, 1004, 1004) == _rgba(right, 1004, 1004) == BRUSH_RGBA


@pytest.mark.parametrize('contours', [False, True])
def test_cleared_brush_is_transparent(app, contours):
    fake = _brushGui('duplicated', contours=contours)
    fake.setTempImg1Brush(True, _brushMask(), BRUSH_ID)
    fake.clearTempBrushImage()

    left, right = fake.tempLayerImg1, fake.tempLayerRightImage
    assert right.image.shape == (2, 2)
    for x, y in ((0, 0), (1, 1)):
        assert _rgba(left, x, y) == _rgba(right, x, y) == TRANSPARENT


def test_cleared_before_any_brush_stroke_is_transparent(app):
    # E.g., `onEscape` calls `clearTempBrushImage`
    fake = _brushGui('duplicated')
    fake.clearTempBrushImage()

    left, right = fake.tempLayerImg1, fake.tempLayerRightImage
    assert _rgba(left, 0, 0) == _rgba(right, 0, 0) == TRANSPARENT


def test_right_brush_cleared_after_switching_right_image(app):
    # Stroke mirrored, then the right image switched to the next frame:
    # clearing still reaches the right layer (no stale strokes)
    fake = _brushGui('duplicated')
    fake.setTempImg1Brush(True, _brushMask(), BRUSH_ID)
    fake.labelsGrad.showRightImgAction.setChecked(False)
    fake.labelsGrad.showNextFrameAction.setChecked(True)
    fake.clearTempBrushImage()

    right = fake.tempLayerRightImage
    assert right.image.shape == (2, 2)
    assert _rgba(right, 0, 0) == TRANSPARENT
