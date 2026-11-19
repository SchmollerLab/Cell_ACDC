"""Tests of the overlay layer shared by the "Keep IDs" and the "whitelist"
tools (`keepIDsTempLayerLeft`): a full redraw must not clear the Keep IDs
selection, and turning the whitelist off must still clear its own overlay.

The `guiWin` methods are tested on fake `self` objects (no main window).
"""
import os
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

pytest.importorskip('qtpy')

import pyqtgraph as pg
import skimage.measure
from qtpy.QtWidgets import QApplication, QToolButton

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


def _only(lab, IDs):
    return np.where(np.isin(lab, list(IDs)), lab, 0)


def _shownIDs(imageItem):
    if imageItem.image is None:
        return set()
    return {int(ID) for ID in np.unique(imageItem.image) if ID > 0}


def _bind(window, *names):
    for name in names:
        func = getattr(gui.guiWin, name)
        func = getattr(func, '__wrapped__', func)
        setattr(window, name, MethodType(func, window))


# Methods called by `guiWin.updateAllImages` that are not under test
_UPDATE_ALL_IMAGES_MOCKS = (
    'rescaleIntensitiesLut', 'setImageImg1', 'setImageImg2',
    'setOverlayImages', 'setOverlayLabelsItems', 'setOverlaySegmMasks',
    'setAllTextAnnotations', 'setAllContoursImages', 'drawAllMothBudLines',
    'highlightLostNew', 'doCustomAnnotation', 'annotate_rip_and_bin_IDs',
    'drawPointsLayers', 'setBrushStrokesLayerImage',
    'setManualBackgroundImage', 'annotateAssignedObjsAcdcTrackerSecondStep',
    'highlightSearchedID', 'updateTimestampFrame', 'annotateAllObjectTracks',
)


def _make_window(keepIDsON=False, whitelistON=False, keptIDs=()):
    lab = _lab()
    keepIDsButton = QToolButton()
    keepIDsButton.setCheckable(True)
    keepIDsButton.setChecked(keepIDsON)
    whitelistIDsButton = QToolButton()
    whitelistIDsButton.setCheckable(True)
    whitelistIDsButton.setChecked(whitelistON)

    keepIDsTempLayerRight = pg.ImageItem()
    keepIDsTempLayerLeft = widgets.ParentImageItem(
        linkedImageItem=keepIDsTempLayerRight
    )

    def clearAllItems():
        # Part of `guiWin.clearAx1Items` that matters here
        keepIDsTempLayerLeft.clear()

    posData = SimpleNamespace(
        frame_i=0, rp=skimage.measure.regionprops(lab), whitelist=None
    )
    window = SimpleNamespace(
        data=[posData],
        pos_i=0,
        currentLab2D=lab,
        isSegm3D=False,
        z_lab=Mock(return_value=None),
        keepIDsButton=keepIDsButton,
        whitelistIDsButton=whitelistIDsButton,
        keepIDsTempLayerLeft=keepIDsTempLayerLeft,
        keepIDsTempLayerRight=keepIDsTempLayerRight,
        keptObjectsIDs=list(keptIDs),
        tempWhitelistIDs=set(),
        slideshowWin=None,
        ccaTableWin=None,
        highlightedID=0,
        isLineageTreeLinesModeActive=Mock(return_value=False),
        clearAllItems=Mock(side_effect=clearAllItems),
    )
    for name in _UPDATE_ALL_IMAGES_MOCKS:
        setattr(window, name, Mock(name=name))

    _bind(
        window, 'updateAllImages', 'updateTempLayerKeepIDs',
        'whitelistUpdateTempLayer', 'getObjectsLab2D', 'isObjInCurrSlice',
        'getObjSlice', 'getObjImage',
    )
    return window, lab


def test_whitelist_off_does_not_clear_keep_ids_selection(app):
    window, lab = _make_window(keepIDsON=True, keptIDs=[1, 3])
    window.updateTempLayerKeepIDs()
    assert _shownIDs(window.keepIDsTempLayerLeft) == {1, 3}

    window.whitelistUpdateTempLayer()

    assert _shownIDs(window.keepIDsTempLayerLeft) == {1, 3}


def test_redraw_keeps_keep_ids_selection(app):
    # E.g., frame change, z-slice change, undo/redo
    window, lab = _make_window(keepIDsON=True, keptIDs=[1, 3])
    window.updateTempLayerKeepIDs()

    window.updateAllImages()

    window.clearAllItems.assert_called_once()
    image = window.keepIDsTempLayerLeft.image
    assert image is not None
    assert np.array_equal(image, _only(lab, [1, 3]))


def test_redraw_draws_whitelist_when_whitelist_on(app):
    window, lab = _make_window(whitelistON=True)
    window.tempWhitelistIDs = {2}

    window.updateAllImages()

    assert np.array_equal(window.keepIDsTempLayerLeft.image, _only(lab, [2]))


def test_toggling_whitelist_off_clears_its_overlay(app):
    window, lab = _make_window(whitelistON=True)
    window.tempWhitelistIDs = {2}
    window.whitelistUpdateTempLayer()
    assert _shownIDs(window.keepIDsTempLayerLeft) == {2}

    window.whitelistIDsToolbar = Mock()
    window.whitelistHighlightIDs = Mock()
    window.whitelistIDsUpdateText = Mock()
    window.updateAllImages = Mock()
    _bind(window, 'whitelistIDs_cb')
    window.whitelistIDsButton.setChecked(False)

    window.whitelistIDs_cb(False)

    # Cleared by the whitelist itself, not only by the redraw
    assert window.keepIDsTempLayerLeft.image is None
    assert window.keepIDsTempLayerRight.image is None
    window.updateAllImages.assert_called_once()


def test_switching_between_keep_ids_and_whitelist(app):
    """Keep IDs and whitelist are mutually exclusive: activating one
    deactivates the other (real `toggled` signals) and only the overlay of
    the active tool is shown."""
    window, lab = _make_window()
    window.tempWhitelistIDs = {2}
    for name in (
            # keepIDs_cb
            'isCellCycleInfoChecked', 'annotOptionsCheckboxes',
            'initKeepObjLabelsLayers', 'setAllIDs', 'clearTempBrushImage',
            'imgGrad', 'labelsLayerImg1', 'labelsLayerRightImg',
            'ax1_contoursImageItem', 'ax2_contoursImageItem',
            'ax1_lostObjImageItem', 'ax2_lostObjImageItem',
            'ax1_lostTrackedObjImageItem', 'ax2_lostTrackedObjImageItem',
            'keepIDsToolbar', 'keptIDsLineEdit', 'keepIDsConfirmAction',
            # whitelistIDs_cb
            'disconnectLeftClickButtons', 'connectLeftClickButtons',
            'whitelistIDsToolbar', 'whitelistHighlightIDs',
            'whitelistIDsUpdateText',
            # uncheckLeftClickButtons
            'magicPromptsToolbar', 'labelRoiButton', 'labelRoi_cb',
            'secondLevelToolbar', 'enableSizeSpinbox',
        ):
        setattr(window, name, Mock(name=name))
    window.isCellCycleInfoChecked.return_value = False
    window.magicPromptsToolButton = Mock(isChecked=Mock(return_value=False))
    window.promptPointsLayerClickButtons = Mock(return_value=[])
    window.leftClickButtons = [window.whitelistIDsButton]
    window.controlToolBars = []
    _bind(
        window, 'keepIDs_cb', 'whitelistIDs_cb', 'uncheckLeftClickButtons'
    )
    # PyQt cannot connect methods bound to a non-QObject --> lambda
    window.keepIDsButton.toggled.connect(
        lambda checked: window.keepIDs_cb(checked)
    )
    window.whitelistIDsButton.toggled.connect(
        lambda checked: window.whitelistIDs_cb(checked)
    )

    window.whitelistIDsButton.setChecked(True)
    assert _shownIDs(window.keepIDsTempLayerLeft) == {2}

    # Activating Keep IDs deactivates the whitelist and hides its overlay
    window.keepIDsButton.setChecked(True)
    assert not window.whitelistIDsButton.isChecked()
    assert _shownIDs(window.keepIDsTempLayerLeft) == set()

    window.keptObjectsIDs.append(1)
    window.updateTempLayerKeepIDs()
    assert _shownIDs(window.keepIDsTempLayerLeft) == {1}

    # Activating the whitelist deactivates Keep IDs and shows the whitelist
    window.whitelistIDsButton.setChecked(True)
    assert not window.keepIDsButton.isChecked()
    assert _shownIDs(window.keepIDsTempLayerLeft) == {2}

    # Turning the whitelist off clears its overlay
    window.whitelistIDsButton.setChecked(False)
    assert _shownIDs(window.keepIDsTempLayerLeft) == set()
