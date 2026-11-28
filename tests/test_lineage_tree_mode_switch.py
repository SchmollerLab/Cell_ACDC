"""Regression tests for the lineage tree edit backup of the main GUI
(`guiWin.repeat_click_and_backup` / `askLineageTreeChanges`), for
entering and leaving the "Normal division: Lineage tree" mode
(`changeMode`, `initLinTree`, `initMissingFramesLinTree`) and for
annotating frames again after their lineage tree annotations were reset in
that mode (`resetLin_tree_future`, `autoLinTree_df`).

The real `gui.guiWin` methods run on a stub host (a real `guiWin` with
loaded data hangs offscreen): the mode combo box is a real
`widgets.ComboBox` wired to the real `changeMode`, as in the main GUI, and
only drawing/widget methods are stubbed. Frame changes mirror the table
steps of `guiWin.next_frame` / `prev_frame`.

Synthetic data: frames 0-1 hold cells 1 and 2, at frame 2 cell 1 divides
into 3 and 4 (frames 2-4: 2, 3, 4).
"""
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from qtpy.QtCore import QPointF, Qt
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QAction, QApplication, QComboBox

from cellacdc.gui.dialogs import segmentation as gui_dialogs_segmentation
from cellacdc.gui.main_window import window as gui_window
from cellacdc.myutils import table_utils
from cellacdc.shared.widgets import (
    combo_boxes as widgets_combo_boxes,
    message_box as widgets_message_box,
)
from cellacdc import base_cca_dict, cca_df_colnames, lineage_tree_cols
from cellacdc.regionprops import acdcRegionprops

pytestmark = pytest.mark.gui

G = gui_window.guiWin
VIEWER = 'Viewer'
SEGM = 'Segmentation and Tracking'
CCA = 'Cell cycle analysis'
LIN = 'Normal division: Lineage tree'

SHAPE = (40, 60)
FRAMES_BOXES = [
    {1: (0, 20, 0, 20), 2: (25, 35, 40, 55)},
    {1: (0, 20, 0, 20), 2: (25, 35, 40, 55)},
    {3: (0, 10, 0, 20), 4: (10, 20, 0, 20), 2: (25, 35, 40, 55)},
    {3: (0, 10, 0, 20), 4: (10, 20, 0, 20), 2: (25, 35, 40, 55)},
    {3: (0, 10, 0, 20), 4: (10, 20, 0, 20), 2: (25, 35, 40, 55)},
]
SIZE_T = len(FRAMES_BOXES)
# {ID: (generation_num_tree, parent_ID_tree)} of the annotated frames
EXPECTED_LINEAGE = [
    {1: (1, -1), 2: (1, -1)},
    {1: (1, -1), 2: (1, -1)},
    {2: (1, -1), 3: (2, 1), 4: (2, 1)},
    {2: (1, -1), 3: (2, 1), 4: (2, 1)},
    {2: (1, -1), 3: (2, 1), 4: (2, 1)},
]
GOTO_TITLE = 'Go to last annotated frame?'
CHANGES_TITLE = 'Changes in lineage tree'
EDIT_TITLE = 'Edited segmentation with annotations!'


def _segm_data():
    labs = np.zeros((SIZE_T, *SHAPE), np.uint32)
    for frame_i, boxes in enumerate(FRAMES_BOXES):
        for ID, (r0, r1, c0, c1) in boxes.items():
            labs[frame_i, r0:r1, c0:c1] = ID
    return labs


class FakeMsg:
    """`widgets.myMessageBox` that records the prompt titles and answers
    from the class attributes `goto_answers` ('yes', 'stay' or 'cancel',
    one per "Go to last annotated frame?" prompt, then 'cancel'),
    `changes_answer` ('Propagate', 'Discard' or 'Cancel') and
    `edit_answer` (start of the "Edited segmentation with annotations!"
    button text: 'Remove', 'Do not remove' or 'Cancel')."""
    titles = []
    goto_answers = []
    changes_answer = 'Propagate'
    edit_answer = 'Remove'

    def __init__(self, *args, **kwargs):
        self.cancel = False
        self.clickedButton = None

    def _show(self, parent, title, txt='', buttonsTexts=None, **kwargs):
        if buttonsTexts is None:
            buttonsTexts = ('Ok',)
        buttons = [SimpleNamespace(text=text) for text in buttonsTexts]
        FakeMsg.titles.append(title)
        if title == GOTO_TITLE:
            answer = (
                FakeMsg.goto_answers.pop(0) if FakeMsg.goto_answers
                else 'cancel'
            )
            prefix = {'yes': 'Yes', 'stay': 'No', 'cancel': 'Cancel'}[answer]
            answer_text = next(
                text for text in buttonsTexts if text.startswith(prefix)
            )
        elif title == CHANGES_TITLE:
            answer_text = FakeMsg.changes_answer
        elif title == EDIT_TITLE:
            answer_text = next(
                text for text in buttonsTexts
                if text.startswith(FakeMsg.edit_answer)
            )
        else:
            answer_text = buttonsTexts[0]
        self.clickedButton = buttons[buttonsTexts.index(answer_text)]
        self.cancel = answer_text == 'Cancel'
        return tuple(buttons)

    warning = critical = question = information = _show

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class Bar:
    def __init__(self):
        self._max = 1

    def setMaximum(self, value):
        self._max = value

    def maximum(self):
        return self._max


# Real guiWin methods run by the tested flows
REAL_METHODS = [
    'changeMode', 'cancelModeInit', 'askLineageTreeChanges',
    'get_difference_table', 'clearLineageTreeBackup',
    'lineageTreeBackupMismatch', 'clearStaleLineageTreeBackup',
    'repeat_click_and_backup', 'annotate_unknown_lineage_action',
    'initMissingFramesLinTree', 'autoLinTree_df', 'attempt_auto_cca',
    'askInitLinTreeFirstFrame', 'get_last_tracked_i',
    'updateLastVisitedFrame', 'setNavigateScrollBarMaximum', 'store_data',
    'store_cca_df', 'store_cca_df_checker', 'ccaCheckerStopChecking',
    'getConcatCcaDf', 'storeFromConcatCcaDf', 'get_cca_df',
    'getBaseCca_df', 'addMissingIDs_cca_df', 'update_cca_df_deletedIDs',
    '_get_data_visited', '_get_data_unvisited', 'get_labels_array',
    'apply_manual_edits_to_lab_if_needed', '_get_editID_info',
    'update_rp_metadata', 'warnFrameNeverVisitedSegmMode',
    'annotateLinTreeFramesBefore', 'warnEditingWithCca_df',
    '_applyEditingWithAnnotAnswer', 'resetLin_tree_future',
    '_blankLinTreeLoadedAcdcDf', '_removeLinTreeFrames',
    'propagateLinTreeAction', 'closePostProcessSegmWin',
    'postProcessSegmWinClosed',
]
# Decorators that show errors in a dialog: unwrapped to raise them
UNWRAPPED_METHODS = ['get_data', 'initLinTree']
# Widgets and drawing methods
STUBS = [
    'reconnectUndoRedo', 'updateModeMenuAction', 'clearCustomAnnot',
    'annotateToolbar', 'copyLostObjButton', 'stopCcaIntegrityCheckerWorker',
    'setAutoSaveSegmentationEnabled', 'setAutoSaveAnnotationsEnabled',
    'editLin_TreeBar', 'uncheckAllButtonsFromButtonGroup',
    'editLin_TreeGroup', 'setEnabledCcaToolbar',
    'setSwitchViewedPlaneDisabled', 'trackingMenu', 'modeToolBar',
    'initSegmTrackMode', 'setEnabledEditToolbarButton', 'addExistingDelROIs',
    'isFirstTimeOnNextFrame', 'realTimeTrackingToggle',
    'restorePrevAnnotOptions', 'whitelistViewOGIDs',
    'startCcaIntegrityCheckerWorker', 'initCca', 'editToolBar',
    'removeAlldelROIsCurrentFrame', 'setAnnotOptionsCcaMode', 'clearGhost',
    'autoSaveTimer', 'setStatusBarLabel', 'doCustomAnnotation',
    'setEnabledSnapshotMode', 'setAnnotOptionsLin_treeMode',
    'setNonFunctionalButtonsDisabled', 'drawAllLineageTreeLines',
    'setAllTextAnnotations', 'updateAllImages', 'setManualBackgroundLab',
    'pointsLayerDfsToData', 'pointsLayerDataToDf', 'enqAutosave',
    'enqCcaIntegrityChecker', 'setLookingGoodTitle', 'updateScrollbars',
    'updateLastCheckedFrameWidgets', 'setDisabled', 'activateWindow',
    'storeUndoRedoCca', 'lastTrackedFrameLabel', 'titleLabel', 'undoAction',
    'redoAction', 'manualAnnotPastButton',
    'annotateSingleMotherBudPairButton',
]


class Host:
    """Stub main GUI holding one position of the synthetic data."""

    def __init__(self, mode=SEGM):
        posData = SimpleNamespace()
        posData.allData_li = [
            table_utils.get_empty_stored_data_dict() for _ in range(SIZE_T)
        ]
        posData.frame_i = 0
        posData.SizeT = SIZE_T
        posData.segmSizeT = SIZE_T
        posData.SizeY, posData.SizeX = SHAPE
        posData.segm_data = _segm_data()
        posData.acdc_df = None
        posData.cca_df = None
        posData.lab = None
        posData.rp = None
        posData.IDs = []
        posData.new_IDs = []
        posData.binnedIDs = set()
        posData.ripIDs = set()
        posData.editID_info = []
        posData.manualBackgroundLab = None
        posData.UndoRedoStates = [[] for _ in range(SIZE_T)]
        posData.UndoRedoCcaStates = [[] for _ in range(SIZE_T)]
        posData.last_tracked_i = 0
        posData.acdc_output_csv_path = 'acdc_output.csv'
        posData.segm_npz_path = 'segm.npz'
        self.posData = posData
        self.data = [posData]
        self.pos_i = 0
        for name in STUBS:
            setattr(self, name, MagicMock(name=name))
        self.manualAnnotPastButton.isChecked.return_value = False
        self.annotateSingleMotherBudPairButton.isChecked.return_value = False
        self.isSnapshot = False
        self.isSegm3D = False
        self.isCancellingModeInit = False
        self.lineage_tree = None
        self.original_df_lin_tree = None
        self.original_df_lin_tree_i = None
        self.right_click_ID = None
        self.right_click_i = 0
        self.ccaTableWin = None
        self.postProcessSegmWin = None
        self.ccaCheckerRunning = False
        self.cca_df_colnames = cca_df_colnames
        self.cca_df_default_values = list(base_cca_dict.values())
        self.cca_df_int_cols = [
            col for col in cca_df_colnames
            if type(base_cca_dict[col]) == int
        ]
        self._acdcRegionProps = acdcRegionprops
        self.navigateScrollBar = Bar()
        self.navSpinBox = Bar()
        self.titleColor = 'w'
        self.logger = logging.getLogger('test_lineage_tree_mode_switch')
        self.last_cca_frame_i = 0
        self.last_lin_tree_frame_i = 0
        self.warnEditingWithAnnotActions = {}
        combo = widgets_combo_boxes.ComboBox()
        combo.addItems([VIEWER, SEGM, CCA, LIN])
        combo.blockSignals(True)
        combo.setCurrentText(mode)
        combo.blockSignals(False)
        self.modeComboBox = combo
        # As in `guiWin.gui_createModeToolbar`
        combo.sigTextChanged.connect(self.changeMode)

    def get_2Dlab(self, lab, force_z=True):
        return lab


for _name in REAL_METHODS:
    setattr(Host, _name, getattr(G, _name))
for _name in UNWRAPPED_METHODS:
    setattr(Host, _name, getattr(G, _name).__wrapped__)


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def fake_msg(monkeypatch):
    FakeMsg.titles = []
    FakeMsg.goto_answers = []
    FakeMsg.changes_answer = 'Propagate'
    FakeMsg.edit_answer = 'Remove'
    monkeypatch.setattr(widgets_message_box, 'myMessageBox', FakeMsg)
    return FakeMsg


@pytest.fixture
def host_logs(caplog):
    caplog.set_level(logging.INFO, logger='test_lineage_tree_mode_switch')
    return caplog


def switch_mode(host, mode):
    """Select `mode` in the combo box as the user does: the click stores
    the previous text, the popup changes the index without the
    `ComboBox.setCurrentText` override. The button is released again:
    Qt 6 QTest keeps pressed buttons for the whole process (a later
    `QTest.mouseMove` would drag)."""
    combo = host.modeComboBox
    QTest.mousePress(combo, Qt.MouseButton.LeftButton)
    try:
        combo.hidePopup()
        QComboBox.setCurrentIndex(combo, combo.findText(mode))
    finally:
        QTest.mouseRelease(combo, Qt.MouseButton.LeftButton)
    assert combo.currentText() == mode


def next_frame(host):
    """Table steps of `guiWin.next_frame` (tracking skipped: the
    synthetic data is tracked)."""
    posData = host.posData
    host.askInitLinTreeFirstFrame()
    mode = host.modeComboBox.currentText()
    if posData.frame_i >= posData.SizeT - 1:
        return False
    if mode != VIEWER:
        host.store_data(debug=False)
    host.askLineageTreeChanges()
    posData.frame_i += 1
    proceed, _ = host.get_data()
    if not proceed:
        posData.frame_i -= 1
        host.get_data()
        return False
    host.attempt_auto_cca()
    host.updateLastVisitedFrame(last_visited_frame_i=posData.frame_i - 1)
    host.setNavigateScrollBarMaximum()
    return True


def go_to_frame(host, frame_i, ask_changes=True):
    """Frame change of `guiWin.prev_frame` (`ask_changes=True`) or of the
    frames scroll bar (which does not check the lineage tree edits)."""
    posData = host.posData
    if host.modeComboBox.currentText() != VIEWER:
        host.store_data(debug=False)
    if ask_changes:
        host.askLineageTreeChanges()
    posData.frame_i = frame_i
    host.get_data()


def lineage(host, frame_i):
    df = host.posData.allData_li[frame_i]['acdc_df']
    if 'parent_ID_tree' not in df.columns:
        return None
    return {
        int(ID): (int(row.generation_num_tree), int(row.parent_ID_tree))
        for ID, row in df.iterrows()
    }


def visit_frames_segm_mode(host):
    host.posData.frame_i = 0
    host.get_data()
    while next_frame(host):
        pass


def annotated_host():
    """Host in lineage tree mode with every frame annotated, on frame 2."""
    host = Host(mode=SEGM)
    visit_frames_segm_mode(host)
    switch_mode(host, VIEWER)
    host.posData.frame_i = 0
    host.get_data()
    switch_mode(host, LIN)
    while next_frame(host):
        pass
    assert [lineage(host, i) for i in range(SIZE_T)] == EXPECTED_LINEAGE
    go_to_frame(host, 2)
    FakeMsg.titles = []
    return host


def right_click_unknown_lineage(host, ID):
    """Right-click on `ID` with the "unknown lineage" tool (the first
    right-click backs up the frame's table)."""
    posData = host.posData
    posData.new_IDs = [3, 4] if posData.frame_i == 2 else []
    r0, r1, c0, c1 = FRAMES_BOXES[posData.frame_i][ID]
    y, x = (r0 + r1) // 2, (c0 + c1) // 2
    event = SimpleNamespace(
        pos=lambda: QPointF(x, y),
        modifiers=lambda: Qt.KeyboardModifier.NoModifier,
    )
    host.annotate_unknown_lineage_action(posData, event, y, x)


def _is_frame_loaded(host, frame_i):
    return np.array_equal(
        host.posData.lab, host.posData.allData_li[frame_i]['labels']
    )


def post_processing_edit(host):
    """What the post-processing dialog's `apply` runs on the current frame
    before it sets the new masks (parameter changed, or a frame change
    with the dialog open, `guiWin.postProcessing`)."""
    host.warnEditingWithCca_df(
        'post-processing segmentation mask', update_images=False
    )


@pytest.mark.parametrize('answer', ['Propagate', 'Discard', 'Cancel'])
def test_leaving_lineage_mode_checks_pending_edits(app, host_logs, answer):
    host = annotated_host()
    right_click_unknown_lineage(host, 3)
    assert host.original_df_lin_tree_i == 2
    assert lineage(host, 2)[3] == (1, -1)

    FakeMsg.changes_answer = answer
    switch_mode(host, SEGM)

    assert CHANGES_TITLE in FakeMsg.titles
    assert host.modeComboBox.currentText() == SEGM
    assert host.lineage_tree is None
    assert host.original_df_lin_tree is None
    assert host.original_df_lin_tree_i is None
    assert host.posData.frame_i == 2
    if answer == 'Propagate':
        assert lineage(host, 2)[3] == (1, -1)
        assert lineage(host, 3)[3] == (1, -1)
        assert 'Lineage tree propagated.' in host_logs.messages
    elif answer == 'Discard':
        assert lineage(host, 2) == EXPECTED_LINEAGE[2]
        assert 'Lineage tree changes discarded.' in host_logs.messages
    else:
        # Kept without propagating, the mode switch is not cancelled
        assert 'Changes kept but not propagated!' in FakeMsg.titles
        assert lineage(host, 2)[3] == (1, -1)
        assert lineage(host, 3) == EXPECTED_LINEAGE[3]
        assert (
            'Lineage tree changes kept but not propagated.'
            in host_logs.messages
        )
        assert 'Lineage tree changes discarded.' not in host_logs.messages


def test_leaving_lineage_mode_without_edits_clears_the_backup(app):
    host = annotated_host()
    # A right-click on the background also backs up the frame
    host.repeat_click_and_backup(
        host.posData, SimpleNamespace(pos=lambda: QPointF(59, 39)), 39, 59
    )
    assert host.original_df_lin_tree_i == 2

    switch_mode(host, VIEWER)

    assert CHANGES_TITLE not in FakeMsg.titles
    assert host.lineage_tree is None
    assert host.original_df_lin_tree is None
    assert host.original_df_lin_tree_i is None


def test_backup_does_not_survive_reset_of_lineage_annotations(app):
    """Edit in lineage tree mode, leave the mode, reset the lineage tree
    annotations (segmentation edit) and come back: navigation and saving
    in lineage tree mode used to raise a KeyError on every attempt."""
    host = annotated_host()
    right_click_unknown_lineage(host, 3)
    FakeMsg.changes_answer = 'Cancel'
    switch_mode(host, SEGM)
    go_to_frame(host, 1)
    # What "Remove annotations from future frames" leaves
    for i in range(1, SIZE_T):
        df = host.posData.allData_li[i]['acdc_df']
        host.posData.allData_li[i]['acdc_df'] = df.drop(
            columns=lineage_tree_cols, errors='ignore'
        )

    FakeMsg.goto_answers = ['yes']
    switch_mode(host, LIN)
    assert host.posData.frame_i == 0
    FakeMsg.titles = []
    assert next_frame(host)

    assert host.posData.frame_i == 1
    assert _is_frame_loaded(host, 1)
    assert CHANGES_TITLE not in FakeMsg.titles
    assert lineage(host, 1) == EXPECTED_LINEAGE[1]


@pytest.mark.parametrize(
    'stale_change', ['lineage columns dropped', 'lineage values NaN',
                     'object deleted']
)
def test_stale_backup_is_cleared_without_error(app, stale_change):
    host = annotated_host()
    right_click_unknown_lineage(host, 3)
    go_to_frame(host, 1, ask_changes=False)
    # A backup that its frame does not match anymore, e.g. left over from
    # a lineage tree session ended in a way that did not clear it
    df = host.posData.allData_li[2]['acdc_df']
    if stale_change == 'lineage columns dropped':
        df = df.drop(columns=lineage_tree_cols, errors='ignore')
    elif stale_change == 'lineage values NaN':
        df = df.copy()
        df[['generation_num_tree', 'parent_ID_tree']] = np.nan
    else:
        df = df.drop(index=4)
    host.posData.allData_li[2]['acdc_df'] = df

    assert next_frame(host)

    assert host.original_df_lin_tree is None
    assert host.original_df_lin_tree_i is None
    assert CHANGES_TITLE not in FakeMsg.titles
    assert host.posData.frame_i == 2
    assert _is_frame_loaded(host, 2)
    # The frame is not overwritten with the backup
    df = host.posData.allData_li[2]['acdc_df']
    if stale_change == 'lineage columns dropped':
        assert 'parent_ID_tree' not in df.columns
    elif stale_change == 'lineage values NaN':
        assert df['parent_ID_tree'].isna().all()
    else:
        assert 4 not in df.index


def test_edits_of_other_frame_are_checked_on_that_frame(app):
    """Frame changed with the scroll bar after an edit: the check jumps to
    the edited frame and back."""
    host = annotated_host()
    right_click_unknown_lineage(host, 3)
    go_to_frame(host, 0, ask_changes=False)

    FakeMsg.changes_answer = 'Discard'
    assert next_frame(host)

    assert CHANGES_TITLE in FakeMsg.titles
    assert lineage(host, 2) == EXPECTED_LINEAGE[2]
    assert host.original_df_lin_tree_i is None
    assert host.posData.frame_i == 1
    assert _is_frame_loaded(host, 1)


def test_frame_is_restored_when_checking_other_frame_fails(app):
    """The check of an edit on another frame loads that frame: an error
    must not leave it loaded instead of the current frame."""
    host = annotated_host()
    right_click_unknown_lineage(host, 3)
    go_to_frame(host, 3, ask_changes=False)

    def fail(*args, **kwargs):
        raise RuntimeError('check failed')

    host.get_difference_table = fail
    with pytest.raises(RuntimeError):
        host.askLineageTreeChanges()

    assert host.posData.frame_i == 3
    assert _is_frame_loaded(host, 3)
    # The next check does not fail again
    assert host.original_df_lin_tree is None
    del host.get_difference_table
    host.askLineageTreeChanges()
    assert host.posData.frame_i == 3


@pytest.mark.parametrize('answer', ['Propagate', 'Discard'])
def test_right_click_on_other_frame_checks_edits_of_backed_up_frame(
        app, answer
    ):
    """Edit, frame changed with the scroll bar, right-click on the new
    frame: the backup of the edited frame used to be replaced without
    checking its edits, which were never propagated."""
    host = annotated_host()
    right_click_unknown_lineage(host, 3)
    go_to_frame(host, 3, ask_changes=False)

    FakeMsg.changes_answer = answer
    # Refused (cell 3 is not a new ID on frame 3), the check runs anyway
    right_click_unknown_lineage(host, 3)

    assert FakeMsg.titles.count(CHANGES_TITLE) == 1
    assert host.posData.frame_i == 3
    assert _is_frame_loaded(host, 3)
    if answer == 'Propagate':
        for frame_i in (2, 3, 4):
            assert lineage(host, frame_i)[3] == (1, -1), frame_i
    else:
        for frame_i in (2, 3, 4):
            assert lineage(host, frame_i) == EXPECTED_LINEAGE[frame_i]
    # The new backup is the current frame after the check
    assert host.original_df_lin_tree_i == 3
    assert host.original_df_lin_tree.equals(
        host.posData.allData_li[3]['acdc_df']
    )
    assert next_frame(host)
    assert FakeMsg.titles.count(CHANGES_TITLE) == 1


@pytest.mark.parametrize('how', ['dialog edit', 'previous frame'])
def test_frame_reset_in_lineage_mode_is_annotated_again_on_next(app, how):
    """A segmentation edit warning answered with "Remove" in lineage tree
    mode (post-processing dialog left open) resets the current frame: the
    next frame used to be annotated from it, every cell a new root."""
    host = annotated_host()
    if how == 'dialog edit':
        post_processing_edit(host)
    else:
        # `guiWin.prev_frame` from frame 3 with the dialog open
        go_to_frame(host, 3)
        go_to_frame(host, 2)
        post_processing_edit(host)
        host.get_data()
    assert FakeMsg.titles == [EDIT_TITLE]
    assert lineage(host, 2) is None
    assert sorted(host.lineage_tree.frames_for_dfs) == [0, 1]
    scroll_bar_max_after_reset = host.navigateScrollBar.maximum()

    assert next_frame(host)

    assert [lineage(host, i) for i in range(4)] == EXPECTED_LINEAGE[:4]
    assert sorted(host.lineage_tree.frames_for_dfs) == [0, 1, 2, 3]
    # The frames scroll bar reaches the current frame
    assert scroll_bar_max_after_reset == 3
    assert host.navigateScrollBar.maximum() == 4


def test_next_annotates_every_frame_missing_before_it_in_order(app):
    host = annotated_host()
    go_to_frame(host, 3)
    host.resetLin_tree_future(from_frame_i=1)
    assert sorted(host.lineage_tree.frames_for_dfs) == [0]

    assert next_frame(host)

    assert [lineage(host, i) for i in range(SIZE_T)] == EXPECTED_LINEAGE
    assert sorted(host.lineage_tree.frames_for_dfs) == list(range(SIZE_T))


def test_reset_of_frame_0_in_lineage_mode_starts_the_tree_again(app):
    host = annotated_host()
    go_to_frame(host, 0)
    FakeMsg.titles = []

    post_processing_edit(host)

    assert FakeMsg.titles == [EDIT_TITLE]
    # The tree used to be dropped: Propagate raised AttributeError on None
    assert host.lineage_tree is not None
    assert sorted(host.lineage_tree.frames_for_dfs) == [0]
    assert lineage(host, 0) == EXPECTED_LINEAGE[0]
    assert all(lineage(host, i) is None for i in range(1, SIZE_T))
    assert host.navigateScrollBar.maximum() == 1
    host.propagateLinTreeAction()

    while next_frame(host):
        pass
    assert [lineage(host, i) for i in range(SIZE_T)] == EXPECTED_LINEAGE


def test_propagate_does_nothing_without_annotations_of_the_frame(
        app, host_logs
    ):
    host = annotated_host()
    post_processing_edit(host)
    assert lineage(host, 2) is None

    # Used to raise a KeyError on the reset frame
    host.propagateLinTreeAction()

    assert lineage(host, 2) is None
    assert host_logs.messages[-1] == (
        'Frame n. 3 has no lineage tree annotations: nothing to propagate.'
    )

    host.lineage_tree = None
    host.propagateLinTreeAction()


def _open_post_processing_dialog(host):
    """Post-processing dialog open on the current frame, as
    `guiWin.postProcessSegm` opens it (the stub host cannot be the
    dialog's parent), showing a parameter change not applied yet."""
    posData = host.posData
    posData.SizeZ = 1
    posData.isSegm3D = False
    posData.user_ch_name = 'channel'
    host.storeUndoRedoStates = MagicMock(name='storeUndoRedoStates')
    host.postProcessSegm = MagicMock(name='postProcessSegm')
    action = QAction()
    action.setCheckable(True)
    action.setChecked(True)
    action.toggled.connect(host.postProcessSegm)
    host.postProcessSegmAction = action
    win = gui_dialogs_segmentation.PostProcessSegmDialog(posData)
    win.mainWin = host
    win.setPosData()
    win.sigClosed.connect(host.postProcessSegmWinClosed)
    host.postProcessSegmWin = win
    win.show()
    # Preview of a parameter change (`postProcessSegmValueChanged`)
    preview_lab = posData.lab.copy()
    preview_lab[preview_lab == 2] = 0
    posData.lab = preview_lab
    return win, action, preview_lab


@pytest.mark.parametrize('mode', [VIEWER, LIN])
def test_leaving_segm_mode_closes_post_processing_dialog(app, mode):
    """Left open, the dialog warned about (and removed) the annotations of
    every frame visited in the other modes. Closing it keeps the shown
    masks, restores nothing and asks nothing; a preview not applied yet
    is stored with matching regionprops and table."""
    host = Host(mode=SEGM)
    visit_frames_segm_mode(host)
    go_to_frame(host, 0)
    win, action, preview_lab = _open_post_processing_dialog(host)
    host.warnEditingWithCca_df = MagicMock(name='warnEditingWithCca_df')

    def update_rp(*args, **kwargs):
        host.posData.rp = acdcRegionprops(host.posData.lab)
        host.update_rp_metadata()
    host.update_rp = MagicMock(name='update_rp', side_effect=update_rp)

    switch_mode(host, mode)

    assert host.postProcessSegmWin is None
    assert not win.isVisible()
    assert not action.isChecked()
    host.postProcessSegm.assert_not_called()
    host.warnEditingWithCca_df.assert_not_called()
    host.update_rp.assert_called_once()
    assert FakeMsg.titles == []
    assert np.array_equal(host.posData.lab, preview_lab)
    assert host.modeComboBox.currentText() == mode
    stored = host.posData.allData_li[0]
    shown_IDs = set(np.unique(preview_lab)) - {0}
    assert 2 not in shown_IDs
    assert set(np.unique(stored['labels'])) - {0} == shown_IDs
    assert set(stored['regionprops'].IDs) == shown_IDs
    assert set(stored['acdc_df'].index) == shown_IDs


def test_stay_on_current_frame_initialises_the_lineage_tree_once(app):
    """"No, stay on current frame" annotates the missing frames up to the
    current one (it used to ask again until Yes or Cancel)."""
    host = Host(mode=SEGM)
    visit_frames_segm_mode(host)
    go_to_frame(host, 3)

    FakeMsg.goto_answers = ['stay']
    switch_mode(host, LIN)

    assert FakeMsg.titles.count(GOTO_TITLE) == 1
    assert host.modeComboBox.currentText() == LIN
    assert host.lineage_tree is not None
    assert host.posData.frame_i == 3
    assert [lineage(host, i) for i in range(4)] == EXPECTED_LINEAGE[:4]
    assert sorted(host.lineage_tree.frames_for_dfs) == [0, 1, 2, 3]

    assert next_frame(host)
    assert lineage(host, 4) == EXPECTED_LINEAGE[4]
    assert FakeMsg.titles.count(GOTO_TITLE) == 1


def test_loading_data_clears_the_backup():
    """`reInitGui` (run by every Open folder/file) on a mock host."""
    host = MagicMock()
    host.checkAskSavePointsLayers.return_value = False
    host.isDataLoaded = False
    host.lineage_tree = object()
    host.original_df_lin_tree = object()
    host.original_df_lin_tree_i = 2
    host.clearLineageTreeBackup = lambda: G.clearLineageTreeBackup(host)

    G.reInitGui(host)

    assert host.lineage_tree is None
    assert host.original_df_lin_tree is None
    assert host.original_df_lin_tree_i is None
