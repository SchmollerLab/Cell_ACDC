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

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

pytest.importorskip('qtpy')

from qtpy.QtWidgets import QApplication, QLabel

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
    )
    # As in `guiWin.__init__` and `guiWin.gui_createActions`
    window.keptObjectsIDs = widgets.KeptObjectIDsList(
        lineEdit, confirmAction
    )
    _bind(
        window, 'addCurrentState', 'getCurrentState', 'undo', 'redo',
        'updateKeepIDs', 'setAllIDs', '_keepObjects', 'storeUndoRedoStates',
        'applyKeepObjects',
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
