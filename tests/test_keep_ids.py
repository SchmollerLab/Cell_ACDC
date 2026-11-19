"""Tests of the "Keep IDs" selection (`widgets.KeptObjectIDsList` bound to
the IDs line edit and the confirm action): undo/redo of the selection and
editing of the IDs in the line edit.

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


def _make_window():
    posData = SimpleNamespace(
        lab=_lab(), frame_i=0, cca_df=None, editID_info=[],
        binnedIDs=set(), ripIDs=set(), UndoRedoStates=defaultdict(list),
        allIDs={1, 2, 3, 4},
    )
    instructionsLabel = QLabel(' (Separate IDs by comma)')
    lineEdit = widgets.KeepIDsLineEdit(instructionsLabel)
    confirmAction = gui.QAction()
    window = _FakeGui(
        data=[posData],
        pos_i=0,
        UndoCount=0,
        keptIDsLineEdit=lineEdit,
        keptIDsInstructionsLabel=instructionsLabel,
        keepIDsConfirmAction=confirmAction,
        keepIDsButton=Mock(isChecked=Mock(return_value=True)),
        whitelistIDsButton=Mock(isChecked=Mock(return_value=False)),
        isBrushStrokesPromptActive=Mock(return_value=False),
        buttonAddPointsByClickingActive=Mock(return_value=None),
        undoAction=Mock(),
        redoAction=Mock(),
        update_rp=Mock(),
        updateAllImages=Mock(),
        store_data=Mock(),
        clearHighlightedText=Mock(),
        highlightLabelID=Mock(),
        updateTempLayerKeepIDs=Mock(),
    )
    # As in `guiWin.__init__` and `guiWin.gui_createActions`
    window.keptObjectsIDs = widgets.KeptObjectIDsList(
        lineEdit, confirmAction
    )
    _bind(
        window, 'addCurrentState', 'getCurrentState', 'undo', 'redo',
        'updateKeepIDs',
    )
    lineEdit.sigIDsChanged.connect(window.updateKeepIDs)
    return window, posData


def _click_keep_IDs(window, *IDs):
    # As in `guiWin.gui_mousePressEventImg1` with "Keep IDs" active
    for ID in IDs:
        window.keptObjectsIDs.append(ID)


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
