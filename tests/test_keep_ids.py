"""Tests of the "Keep IDs" selection (`widgets.KeptObjectIDsList` bound to
the IDs line edit and the confirm action): undo/redo of the selection,
editing of the IDs in the line edit, and validation of the IDs against the
IDs existing in any frame (`guiWin.setAllIDs`).

The `guiWin` methods are tested on fake `self` objects (no main window).
"""
import os
import sys
from collections import defaultdict
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

pytestmark = pytest.mark.gui

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

pytest.importorskip('qtpy')

from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QKeyEvent
from qtpy.QtTest import QTest
from qtpy.QtWidgets import (
    QApplication, QLabel, QLineEdit, QMainWindow, QMenu, QStyle,
    QStyleOptionSpinBox, QWidget
)

from cellacdc import gui, widgets
from cellacdc.regionprops import acdcRegionprops

INSTRUCTIONS = ' (Separate IDs by comma)'
WARNING = '  Some of the IDs are not existing --> they will be IGNORED'


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def slot_errors(monkeypatch):
    """Exceptions raised in Qt slots (PyQt only prints them)"""
    errors = []
    monkeypatch.setattr(
        sys, 'excepthook', lambda *exc_info: errors.append(exc_info[1])
    )
    return errors


class _FakeGui(SimpleNamespace):
    """Fake `guiWin` (weak-referenceable to connect Qt signals to it)"""


def _bind(window, *names):
    for name in names:
        func = getattr(gui.guiWin, name)
        func = getattr(func, '__wrapped__', func)
        setattr(window, name, MethodType(func, window))


def _lab():
    lab = np.zeros((6, 8), dtype=np.uint32)
    lab[0:2, 0:2] = 1
    lab[0:2, 4:6] = 2
    lab[4:6, 5:8] = 3
    lab[4:6, 0:2] = 4
    return lab


def _make_window(segm_data=None):
    if segm_data is None:
        segm_data = _lab()[np.newaxis]
    lab = segm_data[0].copy()
    rp = acdcRegionprops(lab, precache_centroids=False)
    # Regionprops of the frames cached at loading (see `setAllIDs`)
    allData_li = [{'labels': lab.copy(), 'regionprops': rp}]
    allData_li.extend(
        {'labels': None, 'regionprops': None} for _ in segm_data[1:]
    )
    posData = SimpleNamespace(
        lab=lab, rp=rp, IDs=rp.IDs, frame_i=0, cca_df=None, editID_info=[],
        binnedIDs=set(), ripIDs=set(), UndoRedoStates=defaultdict(list),
        segm_data=segm_data.copy(), allData_li=allData_li, allIDs=set(),
    )

    def update_rp(*args, **kwargs):
        # As `guiWin.update_rp` on the current frame (in-place update)
        posData.rp.update_regionprops(posData.lab)
        posData.IDs = posData.rp.IDs

    def store_data(*args, **kwargs):
        # As `guiWin.store_data`
        frameData = posData.allData_li[posData.frame_i]
        frameData['regionprops'] = posData.rp
        frameData['labels'] = posData.lab.copy()

    instructionsLabel = QLabel(INSTRUCTIONS)
    lineEdit = widgets.KeepIDsLineEdit(instructionsLabel)
    confirmAction = gui.QAction()
    window = _FakeGui(
        data=[posData],
        pos_i=0,
        UndoCount=0,
        isSnapshot=True,
        _acdcRegionProps=acdcRegionprops,
        keptIDsLineEdit=lineEdit,
        keptIDsInstructionsLabel=instructionsLabel,
        keepIDsConfirmAction=confirmAction,
        keepIDsButton=Mock(isChecked=Mock(return_value=True)),
        whitelistIDsButton=Mock(isChecked=Mock(return_value=False)),
        isBrushStrokesPromptActive=Mock(return_value=False),
        isBoxPromptActive=Mock(return_value=False),
        buttonAddPointsByClickingActive=Mock(return_value=None),
        undoAction=Mock(),
        redoAction=Mock(),
        update_rp=Mock(side_effect=update_rp),
        updateAllImages=Mock(),
        store_data=Mock(side_effect=store_data),
        clearHighlightedText=Mock(),
        highlightLabelID=Mock(),
        updateTempLayerKeepIDs=Mock(),
        # `applyKeepObjects` (snapshot)
        highlightHoverIDsKeptObj=Mock(),
        tracking=Mock(),
        fixCcaDfAfterEdit=Mock(),
        logger=Mock(),
    )
    # As in `guiWin.__init__` and `guiWin.gui_createActions`
    window.keptObjectsIDs = widgets.KeptObjectIDsList(
        lineEdit, confirmAction
    )
    _bind(
        window, 'addCurrentState', 'getCurrentState', 'undo', 'redo',
        'updateKeepIDs', 'setAllIDs', '_keepObjects', 'storeUndoRedoStates',
        'applyKeepObjects', 'askKeepIDsDeleteAllObjectsCurrentFrame',
    )
    lineEdit.sigIDsChanged.connect(window.updateKeepIDs)
    # As at the end of loading the data
    window.setAllIDs()
    return window, posData


def _click_keep_IDs(window, *IDs):
    # As in `guiWin.gui_mousePressEventImg1` with "Keep IDs" active
    for ID in IDs:
        window.keptObjectsIDs.append(ID)


def _edit(window, posData, func):
    # Edit done before activating "Keep IDs" (undo state stored first)
    window.storeUndoRedoStates(False)
    func(posData.lab)
    window.update_rp()
    window.store_data()


def _activate_keep_IDs(window):
    # What `keepIDs_cb(True)` does with the existing IDs and the selection
    window.setAllIDs()
    window.keptObjectsIDs = widgets.KeptObjectIDsList(
        window.keptIDsLineEdit, window.keepIDsConfirmAction
    )


def _IDs_left_by_keep(window, posData):
    lab = window._keepObjects(lab=posData.lab.copy())
    return set(np.unique(lab)) - {0}


def _delete_4(lab):
    lab[lab == 4] = 0


def _draw_5(lab):
    lab[2:4, 2:4] = 5


def test_store_undo_state_does_not_touch_keep_ids_selection(app):
    window, posData = _make_window()
    _click_keep_IDs(window, 1, 2)
    keptObjectsIDs = window.keptObjectsIDs

    window.addCurrentState()

    stored = posData.UndoRedoStates[0][0]['keptObejctsIDs']
    assert stored == [1, 2]
    assert stored is not keptObjectsIDs
    assert window.keptObjectsIDs is keptObjectsIDs
    assert window.keptIDsLineEdit.text() == '1-2'
    assert window.keepIDsConfirmAction.isEnabled()


def test_undo_apply_keep_ids_restores_selection(app, slot_errors):
    window, posData = _make_window()
    _click_keep_IDs(window, 1, 2)
    # `applyKeepObjects` stores the undo state, deletes the non-kept
    # objects and resets the selection
    window.addCurrentState()
    posData.lab[np.isin(posData.lab, [3, 4])] = 0
    window.keptObjectsIDs = widgets.KeptObjectIDsList(
        window.keptIDsLineEdit, window.keepIDsConfirmAction
    )

    window.undo()

    assert (posData.lab == _lab()).all()
    kept = window.keptObjectsIDs
    assert isinstance(kept, widgets.KeptObjectIDsList)
    assert list(kept) == [1, 2]
    assert window.keptIDsLineEdit.text() == '1-2'
    assert window.keepIDsConfirmAction.isEnabled()
    assert (
        window.keptIDsInstructionsLabel.text() == ' (Separate IDs by comma)'
    )

    # Clicking a cell updates the line edit
    _click_keep_IDs(window, 3)
    assert window.keptIDsLineEdit.text() == '1-3'
    # The restored selection is not the list stored in the undo state
    assert posData.UndoRedoStates[0][1]['keptObejctsIDs'] == [1, 2]

    # Typing in the line edit updates the selection
    window.keptIDsLineEdit.setText('1-3, 4')
    assert list(window.keptObjectsIDs) == [1, 2, 3, 4]
    assert slot_errors == []

    # Redo restores the (empty) selection after applying "Keep IDs"
    window.redo()
    assert isinstance(window.keptObjectsIDs, widgets.KeptObjectIDsList)
    assert list(window.keptObjectsIDs) == []
    assert window.keptIDsLineEdit.text() == ''
    assert not window.keepIDsConfirmAction.isEnabled()
    assert slot_errors == []


def test_undo_resets_keep_ids_line_edit_and_confirm_action(
        app, slot_errors
    ):
    window, posData = _make_window()
    # Undo state stored by an edit done before selecting the IDs to keep
    window.addCurrentState()
    _click_keep_IDs(window, 1, 2)

    window.undo()

    # The line edit and the confirm action must not keep the previous
    # selection (applying it would delete all the objects)
    assert isinstance(window.keptObjectsIDs, widgets.KeptObjectIDsList)
    assert list(window.keptObjectsIDs) == []
    assert window.keptIDsLineEdit.text() == ''
    assert not window.keepIDsConfirmAction.isEnabled()

    window.redo()
    assert list(window.keptObjectsIDs) == [1, 2]
    assert window.keptIDsLineEdit.text() == '1-2'
    assert window.keepIDsConfirmAction.isEnabled()
    assert slot_errors == []


@pytest.mark.parametrize(
    'text, expected', [('', []), ('4', [4]), ('1', [1]), ('2-3', [2, 3])]
)
def test_editing_line_edit_removes_all_unlisted_kept_ids(
        app, slot_errors, text, expected
    ):
    window, _ = _make_window()
    _click_keep_IDs(window, 1, 2, 3)
    assert window.keptIDsLineEdit.text() == '1-3'

    # E.g., select all the text of the line edit and type `text`
    window.keptIDsLineEdit.setText(text)

    assert sorted(window.keptObjectsIDs) == expected
    assert window.keepIDsConfirmAction.isEnabled() == bool(expected)
    assert slot_errors == []


def test_escape_clears_keep_ids_overlay(app, slot_errors):
    window, _ = _make_window()
    # Record the selection drawn on the "Keep IDs" overlay
    drawn = []
    window.updateTempLayerKeepIDs = Mock(
        side_effect=lambda: drawn.append(sorted(window.keptObjectsIDs))
    )
    window.highlightHoverIDsKeptObj = Mock()
    window.autoRange = Mock()
    _bind(window, 'onEscape')
    _click_keep_IDs(window, 1, 2, 3, 4)
    assert drawn[-1] == [1, 2, 3, 4]

    window.onEscape()

    assert list(window.keptObjectsIDs) == []
    assert window.keptIDsLineEdit.text() == ''
    assert not window.keepIDsConfirmAction.isEnabled()
    # No object is left on the overlay after resetting the selection
    assert drawn[-1] == []
    assert slot_errors == []


# The IDs that can be kept are validated against the IDs existing in any
# frame (`posData.allIDs`), which must be up to date while "Keep IDs" is
# active (undo/redo, applying the selection, visiting new frames)
@pytest.mark.parametrize('how', ['type', 'click'])
def test_object_restored_by_undo_can_be_kept(app, slot_errors, how):
    window, posData = _make_window()
    _edit(window, posData, _delete_4)
    _activate_keep_IDs(window)

    window.undo()
    if how == 'type':
        window.keptIDsLineEdit.setText('1, 4')
    else:
        _click_keep_IDs(window, 1, 4)

    assert sorted(window.keptObjectsIDs) == [1, 4]
    assert window.keptIDsInstructionsLabel.text() == INSTRUCTIONS
    assert _IDs_left_by_keep(window, posData) == {1, 4}

    # Removing an ID from the line edit removes it from the selection
    window.keptIDsLineEdit.setText('1')
    assert list(window.keptObjectsIDs) == [1]
    assert _IDs_left_by_keep(window, posData) == {1}
    assert slot_errors == []


def test_object_restored_by_redo_can_be_kept(app, slot_errors):
    window, posData = _make_window()
    _edit(window, posData, _draw_5)
    window.undo()
    _activate_keep_IDs(window)

    window.redo()
    window.keptIDsLineEdit.setText('5')

    assert list(window.keptObjectsIDs) == [5]
    assert window.keepIDsConfirmAction.isEnabled()
    assert window.keptIDsInstructionsLabel.text() == INSTRUCTIONS
    assert slot_errors == []


def test_undo_apply_after_reactivating_keep_IDs(app, slot_errors):
    window, posData = _make_window()
    _activate_keep_IDs(window)
    _click_keep_IDs(window, 1, 2)
    window.applyKeepObjects()
    _activate_keep_IDs(window)  # Toggled off and on again

    window.undo()
    _click_keep_IDs(window, 3)

    assert sorted(window.keptObjectsIDs) == [1, 2, 3]
    assert window.keptIDsInstructionsLabel.text() == INSTRUCTIONS
    assert slot_errors == []


@pytest.mark.parametrize('undo_redo', [False, True])
def test_object_deleted_by_apply_cannot_be_kept(app, slot_errors, undo_redo):
    window, posData = _make_window()
    _activate_keep_IDs(window)
    _click_keep_IDs(window, 1, 2)
    window.applyKeepObjects()
    if undo_redo:
        window.undo()
        window.redo()

    window.keptIDsLineEdit.setText('3')

    # Applying a selection of deleted IDs would delete all the objects
    assert list(window.keptObjectsIDs) == []
    assert not window.keepIDsConfirmAction.isEnabled()
    assert window.keptIDsInstructionsLabel.text() == WARNING
    assert slot_errors == []


@pytest.mark.parametrize(
    'text, expected, warn', [
        ('1, 9', [1], True),
        ('9', [], True),
        ('1-2', [1, 2], False),
        ('', [], False),
    ]
)
def test_kept_IDs_are_the_existing_typed_IDs(
        app, slot_errors, text, expected, warn
    ):
    window, posData = _make_window()
    _activate_keep_IDs(window)
    _click_keep_IDs(window, 1, 3)

    window.keptIDsLineEdit.setText(text)

    assert sorted(window.keptObjectsIDs) == expected
    assert window.keepIDsConfirmAction.isEnabled() == bool(expected)
    label = WARNING if warn else INSTRUCTIONS
    assert window.keptIDsInstructionsLabel.text() == label
    assert slot_errors == []


def test_restored_non_existing_kept_ID_is_not_kept(app, slot_errors):
    # Selection restored by undo with an ID that does not exist anymore
    # (e.g., deleted in another frame) --> not kept once re-validated
    window, posData = _make_window()
    _activate_keep_IDs(window)
    window.addCurrentState()
    posData.UndoRedoStates[0][0]['keptObejctsIDs'] = [1, 9]

    window.undo()
    assert sorted(window.keptObjectsIDs) == [1, 9]
    _click_keep_IDs(window, 2)

    assert sorted(window.keptObjectsIDs) == [1, 2]
    assert window.keptIDsInstructionsLabel.text() == WARNING
    assert _IDs_left_by_keep(window, posData) == {1, 2}
    assert slot_errors == []


def test_clicking_non_existing_ID_does_not_enable_confirm(app, slot_errors):
    # `KeptObjectIDsList.append` triggers `updateKeepIDs` (via the line
    # edit) which removes the ID again --> the confirm action must not be
    # enabled with an empty selection
    window, posData = _make_window()
    _activate_keep_IDs(window)

    _click_keep_IDs(window, 9)

    assert list(window.keptObjectsIDs) == []
    assert not window.keepIDsConfirmAction.isEnabled()
    assert slot_errors == []


def test_ID_existing_only_in_other_frame_can_be_kept(app, slot_errors):
    # Issue #218: the IDs are checked in the whole video
    frame1 = _lab()
    frame1[2:4, 2:4] = 9
    window, posData = _make_window(np.array([_lab(), frame1]))
    _activate_keep_IDs(window)

    window.keptIDsLineEdit.setText('1, 9')

    assert sorted(window.keptObjectsIDs) == [1, 9]
    assert window.keptIDsInstructionsLabel.text() == INSTRUCTIONS
    assert slot_errors == []


def _visit_frame_1_relabelled(window, posData):
    # Visit the unvisited frame 1 (regionprops not cached, see
    # `guiWin._get_data_unvisited`), where tracking relabels 9 --> 10
    window.store_data()
    posData.frame_i = 1
    posData.lab = posData.segm_data[1].copy()
    posData.rp = acdcRegionprops(posData.lab, precache_centroids=False)
    posData.lab[posData.lab == 9] = 10
    window.update_rp()


@pytest.mark.parametrize('activateBeforeVisiting', [False, True])
def test_IDs_of_displayed_frame_are_up_to_date(
        app, slot_errors, activateBeforeVisiting
    ):
    frame1 = _lab()
    frame1[2:4, 2:4] = 9
    window, posData = _make_window(np.array([_lab(), frame1]))
    if activateBeforeVisiting:
        _activate_keep_IDs(window)
        _visit_frame_1_relabelled(window, posData)
    else:
        posData.allData_li[1] = {'labels': None, 'regionprops': None}
        _visit_frame_1_relabelled(window, posData)
        _activate_keep_IDs(window)

    _click_keep_IDs(window, 10)
    assert list(window.keptObjectsIDs) == [10]
    assert window.keptIDsInstructionsLabel.text() == INSTRUCTIONS
    assert window.keepIDsConfirmAction.isEnabled()

    window.keptIDsLineEdit.setText('9')
    assert list(window.keptObjectsIDs) == []
    assert window.keptIDsInstructionsLabel.text() == WARNING
    assert not window.keepIDsConfirmAction.isEnabled()
    assert slot_errors == []


def test_setAllIDs_pos_i_refreshes_only_that_position(app):
    window, posData = _make_window()
    other = SimpleNamespace(
        segm_data=np.zeros((1, 2, 2), np.uint32), allIDs={'untouched'},
        allData_li=[{'labels': None, 'regionprops': None}], frame_i=0,
    )
    window.data.append(other)
    posData.allIDs = set()

    window.setAllIDs(pos_i=0)

    assert posData.allIDs == {1, 2, 3, 4}
    assert other.allIDs == {'untouched'}
    assert other.allData_li[0]['regionprops'] is None


def test_setAllIDs_assigns_a_new_complete_set(app):
    # Readers in other threads (e.g., `trackingWorker`) must never get a
    # partially filled set
    window, posData = _make_window()
    old = posData.allIDs
    oldIDs = set(old)
    _draw_5(posData.lab)
    window.update_rp()

    window.setAllIDs(pos_i=0)

    assert posData.allIDs is not old
    assert old == oldIDs
    assert posData.allIDs == {1, 2, 3, 4, 5}


def test_setAllIDs_live_rp_only_for_displayed_frame(app):
    frame1 = _lab()
    frame1[2:4, 2:4] = 9
    window, posData = _make_window(np.array([_lab(), frame1]))
    # Other position whose `rp` differs from its cached regionprops: it is
    # not the displayed position --> the cached regionprops are used
    otherLab = _lab()
    other = SimpleNamespace(
        segm_data=otherLab[np.newaxis], frame_i=0, allIDs=set(),
        rp=SimpleNamespace(IDs=[99]),
        allData_li=[{
            'labels': otherLab,
            'regionprops': acdcRegionprops(otherLab, precache_centroids=False)
        }],
    )
    window.data.append(other)
    # Displayed frame edited but not stored yet (stale cached regionprops)
    _draw_5(posData.lab)
    window.update_rp()
    posData.allData_li[0]['regionprops'] = acdcRegionprops(
        _lab(), precache_centroids=False
    )

    window.setAllIDs()

    assert posData.allIDs == {1, 2, 3, 4, 5, 9}
    assert other.allIDs == {1, 2, 3, 4}


def test_setAllIDs_only_visited_includes_displayed_frame(app):
    # The displayed frame is visited for the first time (not stored yet)
    frames = np.array([_lab(), _lab(), _lab()])
    frames[1][2:4, 2:4] = 9
    frames[2][2:4, 2:4] = 11
    window, posData = _make_window(frames)
    _visit_frame_1_relabelled(window, posData)

    window.setAllIDs(onlyVisited=True)

    # Frame 2 is not visited yet
    assert posData.allIDs == {1, 2, 3, 4, 10}


def _visit_frames_past_segm_data(window, posData):
    # Segmentation file with less frames than the images: the frames past
    # its end are visited with empty masks (see `guiWin.get_labels_array`)
    posData.allData_li.extend(
        {'labels': None, 'regionprops': None} for _ in range(3)
    )
    # Draw object 7 in frame 1 and object 8 in (displayed) frame 2
    for frame_i, ID in ((1, 7), (2, 8)):
        window.store_data()
        posData.frame_i = frame_i
        posData.lab = np.zeros_like(posData.lab)
        posData.rp = acdcRegionprops(posData.lab, precache_centroids=False)
        posData.lab[2:4, 2:4] = ID
        window.update_rp()


@pytest.mark.parametrize('onlyVisited', [False, True])
def test_setAllIDs_frames_past_segm_data(app, onlyVisited):
    window, posData = _make_window()
    _visit_frames_past_segm_data(window, posData)

    window.setAllIDs(onlyVisited=onlyVisited)

    assert len(posData.segm_data) == 1
    assert posData.allIDs == {1, 2, 3, 4, 7, 8}
    # Frame 3 is not segmented and not visited
    assert posData.allData_li[3]['regionprops'] is None


@pytest.mark.parametrize('how', ['type', 'click'])
def test_IDs_of_frames_past_segm_data_can_be_kept(app, slot_errors, how):
    window, posData = _make_window()
    _visit_frames_past_segm_data(window, posData)
    _activate_keep_IDs(window)

    if how == 'type':
        window.keptIDsLineEdit.setText('7, 8')
    else:
        _click_keep_IDs(window, 8)
        window.keptIDsLineEdit.setText('7-8')

    assert sorted(window.keptObjectsIDs) == [7, 8]
    assert window.keptIDsInstructionsLabel.text() == INSTRUCTIONS
    assert window.keepIDsConfirmAction.isEnabled()
    assert _IDs_left_by_keep(window, posData) == {8}
    assert slot_errors == []


# Applying "Keep IDs" (`guiWin.applyKeepObjects`) must never delete all the
# objects because nothing is selected, and Enter must apply it only once
NO_IDS_WARNING = '  No existing IDs selected --> nothing to apply'


class _Qt5LikeAction(gui.QAction):
    """QAction emitting `triggered` also when disabled (as in Qt5)"""
    def trigger(self):
        self.triggered.emit(False)


class _Host(QMainWindow):
    """Main window delegating the key presses to the fake `guiWin`"""
    def __init__(self, window):
        super().__init__()
        self._window = window

    def keyPressEvent(self, event):
        self._window.keyPressEvent(event)


def _unchecked():
    return Mock(isChecked=Mock(return_value=False))


def _make_apply_window(segm_data=None, isSnapshot=True, qt5Action=False):
    window, posData = _make_window(segm_data)
    window.isSnapshot = isSnapshot
    posData.SizeT = len(posData.segm_data)
    if qt5Action:
        window.keepIDsConfirmAction = _Qt5LikeAction()
        window.keptObjectsIDs = widgets.KeptObjectIDsList(
            window.keptIDsLineEdit, window.keepIDsConfirmAction
        )
    # Time-lapse path of `applyKeepObjects`
    posData.UndoFutFrames_keepID = False
    posData.applyFutFrames_keepID = False
    posData.doNotShowAgain_keepID = False
    posData.includeUnvisitedInfo = {'Keep ID': False}
    window.warnEditingWithCca_df = Mock(return_value=True)
    window.askHowFutureFramesActions = {'Keep ID': _unchecked()}
    window.propagateChange = Mock(return_value=(False, False, None, False))
    # Attributes read by `guiWin.keyPressEvent` before the Enter branch
    window.__dict__.update(
        isDataLoaded=True, debug=False, isSegm3D=False,
        applyEditIDKeyModifiers=Mock(return_value=False),
        applySplitIDsKeyModifiers=Mock(return_value=False),
        widgetsPersistentShortcut={}, checkSetDelObjActionActive=Mock(),
        overlayButton=_unchecked(), brushButton=_unchecked(),
        eraserButton=_unchecked(), manualTrackingButton=_unchecked(),
        manualBackgroundButton=_unchecked(), interpZButton=_unchecked(),
        expandLabelToolButton=_unchecked(), wandToolButton=_unchecked(),
        labelRoiButton=_unchecked(),
        checkTriggerKeyPressShortcuts=Mock(return_value=(False, False)),
        isOverlaySegmMaskChecked=Mock(return_value=False),
        isBrushStrokesSelectCellsActive=Mock(return_value=False),
    )
    _bind(
        window, 'keyPressEvent', 'getFocusedInputWidget',
        'isEnterConfirmingTypedValue', 'triggerPersistentShortcut'
    )
    # As in `guiWin.gui_createActions`
    window.keptIDsLineEdit.sigEnterPressed.connect(
        lambda: window.applyKeepObjects()
    )
    window.keepIDsConfirmAction.triggered.connect(
        lambda checked=False: window.applyKeepObjects()
    )
    # Line edit in the "Keep IDs" toolbar of the main window (the key
    # presses not accepted by the line edit go to `guiWin.keyPressEvent`)
    host = _Host(window)
    toolbar = widgets.ToolBar('Keep IDs controls', host)
    toolbar.addAction(window.keepIDsConfirmAction)
    toolbar.addWidget(window.keptIDsLineEdit)
    host.addToolBar(toolbar)
    # Image (it takes the focus when clicked, and it does not handle Enter)
    window._view = QWidget()
    window._view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    host.setCentralWidget(window._view)
    host.show()
    window.keptIDsLineEdit.setFocus()
    QApplication.processEvents()
    window._host = host
    return window, posData


def _press_enter_in_line_edit(window, text=None):
    if text is not None:
        window.keptIDsLineEdit.setText(text)
    QTest.keyClick(window.keptIDsLineEdit, Qt.Key.Key_Return)
    QApplication.processEvents()


def _press_enter_in_main_window(window):
    # E.g., after clicking on the image (no input box has the focus)
    window._view.setFocus()
    QApplication.processEvents()
    assert QApplication.focusWidget() is window._view
    event = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier
    )
    window.keyPressEvent(event)


def _num_undo_states(posData):
    return len(posData.UndoRedoStates[posData.frame_i])


@pytest.mark.parametrize('text', ['', '99', ', -', '0'])
@pytest.mark.parametrize('qt5Action', [False, True])
def test_apply_keep_ids_without_selection_is_refused(
        app, slot_errors, text, qt5Action
    ):
    window, posData = _make_apply_window(qt5Action=qt5Action)

    _press_enter_in_line_edit(window, text)
    _press_enter_in_main_window(window)
    window.applyKeepObjects()  # E.g., the toolbar button

    assert (posData.lab == _lab()).all()
    assert _num_undo_states(posData) == 0
    assert list(window.keptObjectsIDs) == []
    assert window.keptIDsInstructionsLabel.text() == NO_IDS_WARNING
    assert window.keptIDsInstructionsLabel.styleSheet() == 'color: red'
    # Main window Enter: disabled action not triggered (also on Qt5)
    assert window.logger.warning.call_count == 2
    assert slot_errors == []

    # The warning is reset by editing the IDs
    window.keptIDsLineEdit.setText('1')
    assert list(window.keptObjectsIDs) == [1]
    assert window.keptIDsInstructionsLabel.text() == INSTRUCTIONS
    assert window.keptIDsInstructionsLabel.styleSheet() == ''
    assert window.keepIDsConfirmAction.isEnabled()


def test_no_ids_warning_reset_when_reactivating_keep_ids(app, slot_errors):
    window, posData = _make_apply_window()
    _press_enter_in_line_edit(window)
    assert window.keptIDsInstructionsLabel.text() == NO_IDS_WARNING

    window.__dict__.update(
        currentLab2D=posData.lab,
        isCellCycleInfoChecked=Mock(return_value=False),
        uncheckLeftClickButtons=Mock(), initKeepObjLabelsLayers=Mock(),
        clearTempBrushImage=Mock(), imgGrad=Mock(), keepIDsToolbar=Mock(),
        labelsLayerImg1=Mock(), labelsLayerRightImg=Mock(),
        ax1_contoursImageItem=Mock(), ax2_contoursImageItem=Mock(),
        ax1_lostObjImageItem=Mock(), ax2_lostObjImageItem=Mock(),
        ax1_lostTrackedObjImageItem=Mock(),
        ax2_lostTrackedObjImageItem=Mock(),
    )
    _bind(window, 'keepIDs_cb')
    window.keepIDs_cb(False)
    window.keepIDs_cb(True)

    assert window.keptIDsInstructionsLabel.text() == INSTRUCTIONS
    assert window.keptIDsInstructionsLabel.styleSheet() == ''
    assert slot_errors == []


@pytest.mark.parametrize('qt5Action', [False, True])
def test_enter_in_main_window_applies_only_enabled_action(
        app, slot_errors, qt5Action
    ):
    window, posData = _make_apply_window(qt5Action=qt5Action)
    triggered = Mock()
    window.keepIDsConfirmAction.triggered.connect(
        lambda checked=False: triggered()
    )

    _press_enter_in_main_window(window)
    assert triggered.call_count == 0

    _click_keep_IDs(window, 1)
    _press_enter_in_main_window(window)
    assert triggered.call_count == 1
    assert set(np.unique(posData.lab)) == {0, 1}
    assert slot_errors == []


@pytest.mark.parametrize('qt5Action', [False, True])
def test_enter_in_line_edit_applies_keep_ids_once(
        app, slot_errors, qt5Action
    ):
    window, posData = _make_apply_window(qt5Action=qt5Action)

    QTest.keyClicks(window.keptIDsLineEdit, '1,2')
    _press_enter_in_line_edit(window)

    # On Qt5, Enter was also propagated to `guiWin.keyPressEvent` that
    # applied again the (reset) empty selection --> all objects deleted
    assert set(np.unique(posData.lab)) == {0, 1, 2}
    assert _num_undo_states(posData) == 1
    window.fixCcaDfAfterEdit.assert_called_once()
    assert list(window.keptObjectsIDs) == []
    assert not window.keepIDsConfirmAction.isEnabled()
    assert slot_errors == []


@pytest.mark.parametrize('qt5Action', [False, True])
def test_enter_in_line_edit_cancelled_apply_not_repeated(
        app, slot_errors, qt5Action
    ):
    # Time-lapse: the user cancels the dialog about the annotations, which
    # took the focus from the line edit (the selection is not reset)
    window, posData = _make_apply_window(
        np.array([_lab(), _lab(), _lab()]), isSnapshot=False,
        qt5Action=qt5Action
    )
    window.warnEditingWithCca_df = Mock(
        side_effect=lambda *args, **kwargs: (
            window.keptIDsLineEdit.clearFocus(), False
        )[1]
    )

    QTest.keyClicks(window.keptIDsLineEdit, '1,2')
    _press_enter_in_line_edit(window)

    window.warnEditingWithCca_df.assert_called_once()
    assert _num_undo_states(posData) == 1
    assert sorted(window.keptObjectsIDs) == [1, 2]
    assert slot_errors == []


def test_apply_keep_ids_timelapse_propagates_selection(app, slot_errors):
    window, posData = _make_apply_window(
        np.array([_lab(), _lab(), _lab()]), isSnapshot=False
    )
    propagated = []

    def propagateChange(IDs, key, *args, **kwargs):
        # Copy because the selection is reset (emptied) afterwards
        propagated.append((list(IDs), key))
        return False, False, None, False

    window.propagateChange = Mock(side_effect=propagateChange)

    _press_enter_in_line_edit(window, '2')

    assert set(np.unique(posData.lab)) == {0, 2}
    window.warnEditingWithCca_df.assert_called_once()
    assert propagated == [([2], 'Keep ID')]
    assert list(window.keptObjectsIDs) == []
    assert slot_errors == []


class _FakeMessageBox:
    """Fake `widgets.myMessageBox` answering `answer` (the dialog takes the
    focus)"""
    asked = []
    answer = 'cancel'

    def __init__(self, *args, **kwargs):
        self.cancel = False

    def warning(self, parent, title, txt, buttonsTexts=None, **kwargs):
        _FakeMessageBox.asked.append((title, buttonsTexts))
        QApplication.focusWidget() and QApplication.focusWidget().clearFocus()
        self.cancel = _FakeMessageBox.answer == 'cancel'


def _frames_with_ID_7_in_frame_1():
    frame1 = _lab()
    frame1[2:4, 2:4] = 7
    return np.array([_lab(), frame1, _lab()])


@pytest.mark.parametrize('answer', ['cancel', 'yes'])
def test_apply_keep_ids_none_in_frame_asks_confirmation(
        app, slot_errors, monkeypatch, answer
    ):
    monkeypatch.setattr(_FakeMessageBox, 'asked', [])
    monkeypatch.setattr(_FakeMessageBox, 'answer', answer)
    monkeypatch.setattr(widgets, 'myMessageBox', _FakeMessageBox)
    window, posData = _make_apply_window(
        _frames_with_ID_7_in_frame_1(), isSnapshot=False
    )

    # ID 7 exists only in frame 1 --> all the objects of frame 0 deleted
    _press_enter_in_line_edit(window, '7')

    assert _FakeMessageBox.asked == [(
        'Delete all objects of the current frame?',
        ('Cancel', 'Yes, delete all objects')
    )]
    if answer == 'cancel':
        assert (posData.lab == _lab()).all()
        assert _num_undo_states(posData) == 0
        assert list(window.keptObjectsIDs) == [7]
        assert window.keepIDsConfirmAction.isEnabled()
    else:
        assert not posData.lab.any()
        assert _num_undo_states(posData) == 1
        assert list(window.keptObjectsIDs) == []
    assert slot_errors == []


@pytest.mark.parametrize('emptyFrame', [False, True])
def test_apply_keep_ids_does_not_ask_if_not_deleting_all(
        app, slot_errors, monkeypatch, emptyFrame
    ):
    monkeypatch.setattr(_FakeMessageBox, 'asked', [])
    monkeypatch.setattr(widgets, 'myMessageBox', _FakeMessageBox)
    window, posData = _make_apply_window(
        _frames_with_ID_7_in_frame_1(), isSnapshot=False
    )
    if emptyFrame:
        # Nothing to delete in the current frame
        posData.lab[:] = 0
        window.update_rp()
        text = '7'
    else:
        text = '1, 7'

    _press_enter_in_line_edit(window, text)

    assert _FakeMessageBox.asked == []
    assert _num_undo_states(posData) == 1
    expected = {0} if emptyFrame else {0, 1}
    assert set(np.unique(posData.lab)) == expected
    assert slot_errors == []


@pytest.mark.parametrize('cls', ['KeepIDsLineEdit', 'IDsLineEdit'])
def test_ids_line_edit_accepts_enter(app, cls):
    # The line edits of the "Keep IDs", whitelist, merge and select cells
    # toolbars handle Enter (it must not reach `guiWin.keyPressEvent`)
    pressed = []

    class Host(QMainWindow):
        def keyPressEvent(self, event):
            pressed.append(event.key())

    host = Host()
    lineEdit = getattr(widgets, cls)(QLabel(INSTRUCTIONS))
    enterPressed = Mock()
    lineEdit.sigEnterPressed.connect(lambda: enterPressed())
    toolbar = widgets.ToolBar('IDs', host)
    toolbar.addWidget(lineEdit)
    host.addToolBar(toolbar)
    host.show()

    QTest.keyClick(lineEdit, Qt.Key.Key_Return)
    QTest.keyClick(lineEdit, Qt.Key.Key_Enter)
    QTest.keyClick(lineEdit, Qt.Key.Key_PageDown)

    assert enterPressed.call_count == 2
    # Other keys are still propagated
    assert pressed == [Qt.Key.Key_PageDown]
    host.close()


# Enter in the input boxes of the main window (e.g., frame number) only
# confirms the typed value: it must not apply the active tool (e.g., "Keep
# IDs" with the IDs selected so far)
INPUT_BOXES = ['frame number spin box', 'spin box', 'line edit']


def _add_input_box(window, kind):
    if kind == 'frame number spin box':
        # As the frame and z-slice numbers (and the toolbar spin boxes)
        box = widgets.SpinBox(disableKeyPress=True)
    elif kind == 'spin box':
        # E.g., the "Zoom to ID" spin box
        box = widgets.SpinBox()
    else:
        box = QLineEdit()
    _add_toolbar(window._host, box)
    box.setFocus()
    QApplication.processEvents()
    assert QApplication.focusWidget() is box
    return box


def _add_toolbar(host, widget):
    """Add a toolbar with `widget` to the shown `host` and wait until Qt shows
    it, so that `widget` can take the focus.

    The toolbar hides `widget` until its (queued) layout shows it, and Qt
    only remembers the focus given to a hidden widget (it gives it when the
    widget is shown). Qt forgets it if the focus moves meanwhile, e.g., when
    the new toolbar does not fit next to the "Keep IDs" toolbar (larger
    fonts, as on most CI machines): that toolbar then hides its line edit,
    which has the focus, in its extension menu, and Qt moves the focus to
    the next widget
    """
    toolbar = widgets.ToolBar('Input', host)
    toolbar.addWidget(widget)
    host.addToolBar(toolbar)
    QApplication.processEvents()
    assert widget.isVisible()


def _type_and_press_enter(box, text):
    box.selectAll()
    QTest.keyClicks(box, text)
    QTest.keyClick(box, Qt.Key.Key_Return)
    QApplication.processEvents()


@pytest.mark.parametrize('kind', INPUT_BOXES)
def test_enter_in_input_box_does_not_apply_keep_ids(app, slot_errors, kind):
    window, posData = _make_apply_window()
    triggered = Mock()
    window.keepIDsConfirmAction.triggered.connect(
        lambda checked=False: triggered()
    )
    _click_keep_IDs(window, 1)
    box = _add_input_box(window, kind)

    _type_and_press_enter(box, '12')

    assert triggered.call_count == 0
    assert (posData.lab == _lab()).all()
    assert _num_undo_states(posData) == 0
    assert list(window.keptObjectsIDs) == [1]
    if kind == 'line edit':
        assert box.text() == '12'
    else:
        # The typed value is confirmed
        assert box.value() == 12
    # The spin boxes with `disableKeyPress` leave the focus (as with the
    # other keys that they do not handle), the others keep it
    assert box.hasFocus() == (kind != 'frame number spin box')

    # Enter in the main window (e.g., on the image) applies the selection
    _press_enter_in_main_window(window)
    assert triggered.call_count == 1
    assert set(np.unique(posData.lab)) == {0, 1}
    assert slot_errors == []


@pytest.mark.parametrize('tool', ['interpolate Z', 'select cells'])
@pytest.mark.parametrize('kind', INPUT_BOXES)
def test_enter_in_input_box_does_not_apply_other_tools(
        app, slot_errors, kind, tool
    ):
    window, posData = _make_apply_window()
    window.keepIDsButton = _unchecked()
    if tool == 'interpolate Z':
        window.interpZButton = Mock(isChecked=Mock(return_value=True))
        window.interpZConfirmAction = Mock()
        applyTool = window.interpZConfirmAction.trigger
    else:
        window.isBrushStrokesSelectCellsActive = Mock(return_value=True)
        toolbar = window.brushStrokesSelectCellsToolbar = Mock()
        # Its IDs line edit does not have the focus
        toolbar.selectedIDsLineEdit.hasFocus.return_value = False
        applyTool = toolbar.emitSigAddToTrainingData
    box = _add_input_box(window, kind)

    _type_and_press_enter(box, '3')
    assert applyTool.call_count == 0

    _press_enter_in_main_window(window)
    assert applyTool.call_count == 1
    assert slot_errors == []


def _click_spin_box_up_arrow(spinBox):
    opt = QStyleOptionSpinBox()
    spinBox.initStyleOption(opt)
    rect = spinBox.style().subControlRect(
        QStyle.ComplexControl.CC_SpinBox, opt, QStyle.SubControl.SC_SpinBoxUp,
        spinBox
    )
    QTest.mouseClick(
        spinBox, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
        rect.center()
    )
    QApplication.processEvents()
    # Clicking the arrows gives the focus to the spin box
    assert QApplication.focusWidget() is spinBox


def _send_enter_to_window(window, autoRepeat=False):
    # As the platform: to the widget with the focus (or to the main window)
    event = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Return,
        Qt.KeyboardModifier.NoModifier, '\r', autoRepeat
    )
    QApplication.sendEvent(window._host.windowHandle(), event)
    QApplication.processEvents()


@pytest.mark.parametrize('kind', ['frame number spin box', 'spin box'])
def test_enter_after_clicking_spin_box_arrows_applies_keep_ids(
        app, slot_errors, kind
    ):
    # E.g., the user selects the IDs, checks the next frame or z-slice with
    # the arrows, and applies "Keep IDs" with Enter (nothing typed to confirm)
    window, posData = _make_apply_window()
    triggered = Mock()
    window.keepIDsConfirmAction.triggered.connect(
        lambda checked=False: triggered()
    )
    _click_keep_IDs(window, 1)
    box = _add_input_box(window, kind)
    window._view.setFocus()
    _click_spin_box_up_arrow(box)
    assert box.value() == 1

    _send_enter_to_window(window)

    assert triggered.call_count == 1
    assert set(np.unique(posData.lab)) == {0, 1}
    assert slot_errors == []


def test_held_enter_after_typing_frame_number_does_not_apply_keep_ids(
        app, slot_errors
    ):
    window, posData = _make_apply_window()
    triggered = Mock()
    window.keepIDsConfirmAction.triggered.connect(
        lambda checked=False: triggered()
    )
    _click_keep_IDs(window, 1)
    box = _add_input_box(window, 'frame number spin box')
    box.selectAll()
    QTest.keyClicks(box, '12')

    _send_enter_to_window(window)
    assert box.value() == 12
    assert QApplication.focusWidget() is None
    # Enter held down: the repeats go to the main window (no focus widget)
    _send_enter_to_window(window, autoRepeat=True)
    _send_enter_to_window(window, autoRepeat=True)

    assert triggered.call_count == 0
    assert (posData.lab == _lab()).all()
    assert list(window.keptObjectsIDs) == [1]

    # Pressing Enter again applies the selection
    _send_enter_to_window(window)
    assert triggered.call_count == 1
    assert set(np.unique(posData.lab)) == {0, 1}
    assert slot_errors == []


class _KeyHost(QMainWindow):
    """Main window recording the key presses passed on to it"""
    def __init__(self):
        super().__init__()
        self.pressed = []

    def keyPressEvent(self, event):
        self.pressed.append(event.key())


def _show_spin_boxes(host, *spinBoxes):
    toolbar = widgets.ToolBar('Spin boxes', host)
    for spinBox in spinBoxes:
        toolbar.addWidget(spinBox)
    host.addToolBar(toolbar)
    host.show()
    spinBoxes[0].setFocus()
    QApplication.processEvents()
    assert spinBoxes[0].hasFocus()


@pytest.mark.parametrize('typed', [True, False])
@pytest.mark.parametrize('key', [Qt.Key.Key_Return, Qt.Key.Key_Enter])
@pytest.mark.parametrize('keyboardTracking', [True, False])
@pytest.mark.parametrize('cls', ['SpinBox', 'DoubleSpinBox'])
def test_spin_box_with_disable_key_press_enter(
        app, slot_errors, cls, keyboardTracking, key, typed
    ):
    host = _KeyHost()
    spinBox = getattr(widgets, cls)(disableKeyPress=True)
    spinBox.setKeyboardTracking(keyboardTracking)
    spinBox.setValue(3)
    editingFinished = Mock()
    spinBox.editingFinished.connect(lambda: editingFinished())
    _show_spin_boxes(host, spinBox)

    if typed:
        spinBox.selectAll()
        QTest.keyClicks(spinBox, '7')
    else:
        _click_spin_box_up_arrow(spinBox)
    assert spinBox.isValueTyped() == typed

    QTest.keyClick(spinBox, key)
    QApplication.processEvents()

    # The value is confirmed and the spin box leaves the focus
    assert spinBox.value() == (7 if typed else 4)
    assert not spinBox.hasFocus()
    assert editingFinished.call_count == 1
    if typed:
        # Enter only confirms the typed value: not passed on to the main
        # window (it would apply the active tool, e.g., "Keep IDs")
        assert host.pressed == []
    else:
        # Nothing typed to confirm (e.g., only the arrows clicked): passed
        # on to the main window (it applies the active tool)
        assert host.pressed == [key]

    # Getting the focus again resets the typed value: clicking the arrows
    # and pressing Enter passes Enter on
    host.pressed.clear()
    _click_spin_box_up_arrow(spinBox)
    assert not spinBox.isValueTyped()
    QTest.keyClick(spinBox, key)
    assert host.pressed == [key]
    assert slot_errors == []
    host.close()


@pytest.mark.parametrize('cls', ['SpinBox', 'DoubleSpinBox'])
def test_spin_box_keys_passed_on(app, slot_errors, cls):
    host = _KeyHost()
    spinBox = getattr(widgets, cls)(disableKeyPress=True)
    # Spin box without `disableKeyPress` (e.g., in a dialog)
    otherSpinBox = getattr(widgets, cls)()
    _show_spin_boxes(host, spinBox, otherSpinBox)

    # The keys that the spin boxes with `disableKeyPress` do not handle are
    # passed on (shortcuts)
    spinBox.selectAll()
    QTest.keyClicks(spinBox, '7')
    QTest.keyClick(spinBox, Qt.Key.Key_PageDown)
    assert host.pressed == [Qt.Key.Key_PageDown]
    assert not spinBox.hasFocus()

    # Without `disableKeyPress` Enter is passed on as usual also after
    # typing (e.g., to the default button of a dialog)
    otherSpinBox.setFocus()
    QApplication.processEvents()
    otherSpinBox.selectAll()
    QTest.keyClicks(otherSpinBox, '5')
    assert otherSpinBox.isValueTyped()
    QTest.keyClick(otherSpinBox, Qt.Key.Key_Return)
    assert otherSpinBox.value() == 5
    assert host.pressed == [Qt.Key.Key_PageDown, Qt.Key.Key_Return]
    assert slot_errors == []
    host.close()


def _focus_elsewhere_and_back(box, how):
    """Take the focus away from the input box and give it back"""
    host = box.window()
    if how == 'switch window':
        # E.g., Alt+Tab, or the mouse leaving the main window while the
        # secondary viewer is open (see `guiWin.leaveEvent`)
        other = QMainWindow()
        other.show()
        other.activateWindow()
        QApplication.processEvents()
        assert QApplication.activeWindow() is other
        other.close()
        host.activateWindow()
    elif how == 'popup':
        # E.g., the context menu of the spin box
        menu = QMenu(box)
        menu.addAction('Copy')
        menu.popup(box.mapToGlobal(box.rect().center()))
        QApplication.processEvents()
        assert QApplication.activePopupWidget() is menu
        menu.close()
        QApplication.processEvents()
        if QApplication.activeWindow() is None:
            # The offscreen platform activates the menu too, and Qt 6 for
            # macOS follows it (`QApplicationPrivate::
            # notifyActiveWindowChange`; Qt 5 ignores activations while a
            # popup is open), so closing it leaves no active window there.
            # A desktop does not activate context menus: activate the
            # window again (the focus returns to the box with
            # `ActiveWindowFocusReason`)
            host.activateWindow()
    else:
        # The user clicks another widget, and then the box again (or Qt
        # gives it the focus when activating the window)
        other = QLineEdit()
        _add_toolbar(host, other)
        other.setFocus(Qt.FocusReason.MouseFocusReason)
        QApplication.processEvents()
        assert QApplication.focusWidget() is other
        if how == 'other widget':
            box.setFocus(Qt.FocusReason.MouseFocusReason)
        else:
            box.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
    QApplication.processEvents()
    assert QApplication.activeWindow() is host
    assert QApplication.focusWidget() is box


@pytest.mark.parametrize('how', ['switch window', 'popup'])
@pytest.mark.parametrize('kind', ['frame number spin box', 'spin box'])
def test_enter_after_typing_and_focus_return_does_not_apply_keep_ids(
        app, slot_errors, kind, how
    ):
    window, posData = _make_apply_window()
    triggered = Mock()
    window.keepIDsConfirmAction.triggered.connect(
        lambda checked=False: triggered()
    )
    _click_keep_IDs(window, 1)
    box = _add_input_box(window, kind)
    box.selectAll()
    QTest.keyClicks(box, '12')

    _focus_elsewhere_and_back(box, how)
    _send_enter_to_window(window)

    # Enter confirms the typed value
    assert box.value() == 12
    assert triggered.call_count == 0
    assert (posData.lab == _lab()).all()
    assert list(window.keptObjectsIDs) == [1]
    assert slot_errors == []


@pytest.mark.parametrize(
    'how, isStillTyped',
    [
        ('switch window', True), ('popup', True), ('other widget', False),
        ('other widget, back by window activation', False)
    ]
)
@pytest.mark.parametrize('cls', ['SpinBox', 'DoubleSpinBox'])
def test_spin_box_typed_value_when_focus_returns(
        app, slot_errors, cls, how, isStillTyped
    ):
    host = _KeyHost()
    spinBox = getattr(widgets, cls)(disableKeyPress=True)
    _show_spin_boxes(host, spinBox)
    spinBox.selectAll()
    QTest.keyClicks(spinBox, '7')
    assert spinBox.isValueTyped()

    _focus_elsewhere_and_back(spinBox, how)

    # Switching window or opening a popup does not start a new edit
    assert spinBox.isValueTyped() == isStillTyped
    QTest.keyClick(spinBox, Qt.Key.Key_Return)
    assert spinBox.value() == 7
    assert not spinBox.hasFocus()
    assert host.pressed == ([] if isStillTyped else [Qt.Key.Key_Return])
    assert slot_errors == []
    host.close()
