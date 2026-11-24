"""Regression tests for removing lineage tree annotations after editing the
segmentation of an annotated frame in the main GUI
(`guiWin.warnEditingWithCca_df` -> "Remove annotations from future frames",
`guiWin.resetLin_tree_future`), for Repeat segmentation / Repeat tracking on
the video and for the Merge IDs back-propagation to past frames.

The tests run the real `gui.guiWin` methods on a minimal host holding
synthetic per-frame tables (6 frames, ID 1 divides symmetrically into 3 and
4 at frame 2, new cell 5 at frame 4); the widgets are stubs and the message
boxes are answered by the tests.
"""
import logging
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from qtpy.QtWidgets import QAction, QApplication

from cellacdc import apps, gui, load, myutils, widgets, workers
from cellacdc import base_cca_dict, cca_df_colnames
from cellacdc.regionprops import acdcRegionprops
from cellacdc.trackers.CellACDC_normal_division.CellACDC_normal_division_tracker import (
    LINEAGE_TREE_SHARED_CCA_COLS, has_lineage_tree_values, lineage_tree_columns,
    normal_division_lineage_tree
)

G = gui.guiWin
SEGM = 'Segmentation and Tracking'
LIN = 'Normal division: Lineage tree'

EDIT_TITLE = 'Edited segmentation with annotations!'
REMOVE, KEEP = 1, 2
REPEAT_TRACKING_TITLE = 'Repating tracking with annotations!'
NO, YES = 0, 1

SHAPE = (24, 40)
SIZE_T = 6
_BOX_1 = (0, 10, 0, 10)
_BOX_2 = (12, 22, 20, 30)
_BOX_3 = (0, 5, 0, 10)
_BOX_4 = (5, 10, 0, 10)
_BOX_5 = (12, 22, 0, 10)
FRAMES_BOXES = (
    {1: _BOX_1, 2: _BOX_2},
    {1: _BOX_1, 2: _BOX_2},
    {3: _BOX_3, 4: _BOX_4, 2: _BOX_2},
    {3: _BOX_3, 4: _BOX_4, 2: _BOX_2},
    {3: _BOX_3, 4: _BOX_4, 2: _BOX_2, 5: _BOX_5},
    {3: _BOX_3, 4: _BOX_4, 2: _BOX_2, 5: _BOX_5},
)
# ID: (generation, parent, root, sister, emerged at frame, history known)
LINEAGE = {
    1: (1, -1, 1, -1, 0, False),
    2: (1, -1, 2, -1, 0, False),
    3: (2, 1, 1, 4, 2, True),
    4: (2, 1, 1, 3, 2, True),
    5: (1, -1, 5, -1, 4, False),
}

REAL_METHODS = (
    'warnEditingWithCca_df', '_applyEditingWithAnnotAnswer',
    'warnEditingWithAnnotActionTriggered',
    'resetLin_tree_future', '_blankLinTreeLoadedAcdcDf',
    '_removeLinTreeFrames', 'clearLineageTreeBackup',
    'resetCcaFuture', 'resetFutureCcaColCurrentFrame',
    'removeCcaAnnotationsCurrentFrame', 'resetWillDivideInfo',
    'getConcatCcaDf', 'storeFromConcatCcaDf', 'ccaCheckerStopChecking',
    'setNavigateScrollBarMaximum', 'get_cca_df', 'store_cca_df',
    'store_cca_df_checker', 'addMissingIDs_cca_df', 'getBaseCca_df',
    'update_cca_df_deletedIDs', 'updateCcaDfDeletedIDsTimelapse',
    'update_rp_metadata', 'isFrameCcaAnnotated', 'getLastAnnotatedFrames',
    'get_last_cca_frame_i',
    'get_last_lin_tree_frame_i', 'askRemoveAnnotationsRepeatTrackingVideo',
    'warnRepeatTrackingVideoWithAnnotations', 'segmWorkerFinished',
    'mergeSelectedIDs',
)
# Decorated with `exception_handler`: unwrap so that errors surface
UNWRAPPED_METHODS = ('store_data', 'repeatTrackingVideo')


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


class _MessageBox:
    """Replaces `widgets.myMessageBox`: records every message and clicks
    the button with index `answers[title]` (default 0, None cancels)."""
    answers = {}
    remember = False
    log = []

    def __init__(self, *args, **kwargs):
        self.cancel = False
        self.clickedButton = None

    def _show(
            self, parent, title, txt='', buttonsTexts=None, widgets=None,
            **kwargs
        ):
        buttons = tuple(
            SimpleNamespace(text=text) for text in (buttonsTexts or ('Ok',))
        )
        type(self).log.append(SimpleNamespace(title=title, txt=txt))
        if widgets is not None and type(self).remember:
            widgets.setChecked(True)
        index = type(self).answers.get(title, 0)
        if index is None:
            self.cancel = True
        else:
            self.clickedButton = buttons[index]
        return buttons

    warning = critical = question = information = _show

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


@pytest.fixture
def prompts(monkeypatch):
    class Prompts(_MessageBox):
        answers = {}
        remember = False
        log = []

    monkeypatch.setattr(widgets, 'myMessageBox', Prompts)
    return Prompts


def _edit_prompts(prompts):
    return [p for p in prompts.log if p.title == EDIT_TITLE]


class _Mode:
    def __init__(self, text):
        self.text = text

    def currentText(self):
        return self.text


class _Bar:
    def __init__(self):
        self.max = None

    def setMaximum(self, value):
        self.max = value

    def maximum(self):
        return self.max


class _Unchecked:
    def isChecked(self):
        return False


def _lab(boxes):
    lab = np.zeros(SHAPE, np.uint32)
    for ID, (r0, r1, c0, c1) in boxes.items():
        lab[r0:r1, c0:c1] = ID
    return lab


def _frame_acdc_df(boxes, with_lineage=True, with_cca=False):
    IDs = list(boxes)
    df = pd.DataFrame({
        'Cell_ID': IDs,
        'is_cell_dead': False,
        'is_cell_excluded': False,
        'x_centroid': 0,
        'y_centroid': 0,
        'was_manually_edited': 0,
    }).set_index('Cell_ID')
    if with_cca:
        for col, value in base_cca_dict.items():
            df[col] = value
    if with_lineage:
        values = [LINEAGE[ID] for ID in IDs]
        df['Cell_ID_tree'] = IDs
        df['generation_num_tree'] = [v[0] for v in values]
        df['parent_ID_tree'] = [v[1] for v in values]
        df['root_ID_tree'] = [v[2] for v in values]
        df['sister_ID_tree'] = [v[3] for v in values]
        # Extra sister column of the lineage tree (`max_daughters_added`)
        df['sister_ID_tree_0'] = [v[3] for v in values]
        df['emerg_frame_i'] = [v[4] for v in values]
        df['division_frame_i'] = [v[4] for v in values]
        df['is_history_known'] = [v[5] for v in values]
    return df


class _Host:
    """Main-GUI host running the real `guiWin` methods of `REAL_METHODS`
    on synthetic data, every frame already visited (`allData_li`) and the
    table loaded from file in `posData.acdc_df`."""

    def __init__(self, with_lineage=True, with_cca=False, mode=SEGM):
        posData = SimpleNamespace()
        posData.allData_li = []
        frames_dfs = []
        for boxes in FRAMES_BOXES:
            data_dict = myutils.get_empty_stored_data_dict()
            lab = _lab(boxes)
            data_dict['labels'] = lab
            data_dict['regionprops'] = acdcRegionprops(
                lab, precache_centroids=False
            )
            df = _frame_acdc_df(
                boxes, with_lineage=with_lineage, with_cca=with_cca
            )
            data_dict['acdc_df'] = df.copy()
            posData.allData_li.append(data_dict)
            frames_dfs.append(df)
        posData.acdc_df = pd.concat(
            frames_dfs, keys=range(SIZE_T), names=['frame_i']
        )
        posData.segm_data = np.array([d['labels'] for d in posData.allData_li])
        posData.frame_i = 0
        posData.SizeT = SIZE_T
        posData.segmSizeT = SIZE_T
        posData.SizeZ = 1
        posData.last_tracked_i = SIZE_T - 1
        posData.cca_df = None
        posData.lab = None
        posData.rp = None
        posData.IDs = []
        posData.binnedIDs = set()
        posData.ripIDs = set()
        posData.editID_info = []
        posData.manualBackgroundLab = None
        posData.segmInfo_df = None
        self.data = [posData]
        self.pos_i = 0
        self.isSnapshot = False
        self.isSegm3D = False
        self.lineage_tree = None
        self.original_df_lin_tree = None
        self.original_df_lin_tree_i = None
        self.modeComboBox = _Mode(mode)
        self.manualAnnotPastButton = _Unchecked()
        self.annotateSingleMotherBudPairButton = _Unchecked()
        self.ccaTableWin = None
        self.ccaCheckerRunning = False
        self.cca_df_colnames = cca_df_colnames
        self.cca_df_default_values = list(base_cca_dict.values())
        self.navigateScrollBar = _Bar()
        self.navSpinBox = _Bar()
        self.lastTrackedFrameLabel = MagicMock()
        self.titleLabel = MagicMock()
        self.titleColor = 'w'
        self.last_cca_frame_i = SIZE_T - 1 if with_cca else 0
        self.logger = logging.getLogger('test_lineage_tree_reset')
        self.app = None
        self.warnEditingWithAnnotActions = {}
        for key in ('Delete ID', 'Merge IDs'):
            # As in `guiWin.gui_populateToolSettingsMenu` (Settings > Warnings)
            action = QAction()
            action.setCheckable(True)
            action.setChecked(True)
            action.removeAnnot = False
            action.triggered.connect(
                partial(self.warnEditingWithAnnotActionTriggered, action)
            )
            self.warnEditingWithAnnotActions[key] = action
        self.calls = []

    @property
    def posData(self):
        return self.data[0]

    # Widget stubs
    def updateAllImages(self, *args, **kwargs):
        self.calls.append('updateAllImages')

    def pointsLayerDataToDf(self, *args, **kwargs):
        pass

    def enqAutosave(self, *args, **kwargs):
        pass

    def enqCcaIntegrityChecker(self, *args, **kwargs):
        pass

    def updateLastCheckedFrameWidgets(self, *args, **kwargs):
        pass

    def storeUndoRedoCca(self, *args, **kwargs):
        pass

    def setDisabled(self, *args, **kwargs):
        pass

    def activateWindow(self, *args, **kwargs):
        pass


for _name in REAL_METHODS:
    setattr(_Host, _name, getattr(G, _name))
for _name in UNWRAPPED_METHODS:
    setattr(_Host, _name, getattr(G, _name).__wrapped__)


def _go_to(host, frame_i):
    """Show a visited frame, like `get_data` (`_get_data_visited`)."""
    posData = host.posData
    posData.frame_i = frame_i
    data_dict = posData.allData_li[frame_i]
    posData.lab = data_dict['labels'].copy()
    posData.rp = data_dict['regionprops']
    posData.IDs = posData.rp.IDs
    host.update_rp_metadata(draw=False)
    host.get_cca_df()


def _delete_ID(host, ID):
    """Delete ID like the Delete ID tool: edit, store, then warn."""
    posData = host.posData
    posData.lab[posData.lab == ID] = 0
    posData.rp = acdcRegionprops(posData.lab, precache_centroids=False)
    posData.IDs = posData.rp.IDs
    host.update_rp_metadata(draw=False)
    host.store_data(autosave=False)
    return host.warnEditingWithCca_df('Delete ID', update_images=False)


def _frame_df(host, frame_i):
    return host.posData.allData_li[frame_i]['acdc_df']


def _has_cca(host, frame_i):
    return 'cell_cycle_stage' in _frame_df(host, frame_i).columns


def _lineage_and_shared_cols(df):
    return lineage_tree_columns(df) + [
        col for col in LINEAGE_TREE_SHARED_CCA_COLS if col in df.columns
    ]


def _assert_lineage_removed_from(host, from_frame_i, original):
    for frame_i in range(SIZE_T):
        df = _frame_df(host, frame_i)
        if frame_i < from_frame_i:
            pd.testing.assert_frame_equal(df, original[frame_i])
            continue
        assert lineage_tree_columns(df) == [], frame_i
        if not _has_cca(host, frame_i):
            assert not set(LINEAGE_TREE_SHARED_CCA_COLS) & set(df.columns)


def _visited_dfs(host):
    return [_frame_df(host, i).copy() for i in range(SIZE_T)]


def test_remove_clears_lineage_from_edited_frame_everywhere(app, prompts):
    host = _Host()
    original = _visited_dfs(host)
    loaded = host.posData.acdc_df.copy()
    prompts.answers[EDIT_TITLE] = REMOVE
    _go_to(host, 3)

    assert _delete_ID(host, 2) is True

    edit_prompts = _edit_prompts(prompts)
    assert len(edit_prompts) == 1
    assert 'lineage tree annotations' in edit_prompts[0].txt
    assert 'cell cycle' not in edit_prompts[0].txt
    # Visited frames: lineage tree, extra sister and shared columns gone
    _assert_lineage_removed_from(host, 3, original)
    assert not any(_has_cca(host, i) for i in range(SIZE_T))
    # Table loaded from file (read for unvisited frames): values blanked,
    # rows and other columns kept
    acdc_df = host.posData.acdc_df
    assert acdc_df.index.equals(loaded.index)
    cols = _lineage_and_shared_cols(loaded)
    assert 'sister_ID_tree_0' in cols
    assert acdc_df.loc[3:, cols].isna().all(axis=None)
    pd.testing.assert_frame_equal(
        acdc_df.loc[:2, cols].astype(float), loaded.loc[:2, cols].astype(float)
    )
    pd.testing.assert_frame_equal(
        acdc_df.drop(columns=cols), loaded.drop(columns=cols)
    )


def test_remove_on_mixed_table_clears_cell_cycle_and_lineage(app, prompts):
    host = _Host(with_cca=True)
    prompts.answers[EDIT_TITLE] = REMOVE
    _go_to(host, 3)

    _delete_ID(host, 2)

    txt = _edit_prompts(prompts)[0].txt
    assert 'cell cycle and lineage tree annotations' in txt
    for frame_i in range(3):
        assert _has_cca(host, frame_i)
        assert 'parent_ID_tree' in _frame_df(host, frame_i).columns
    for frame_i in range(3, SIZE_T):
        df = _frame_df(host, frame_i)
        assert not set(cca_df_colnames) & set(df.columns), frame_i
        assert lineage_tree_columns(df) == [], frame_i
    lineage_cols = lineage_tree_columns(host.posData.acdc_df)
    assert host.posData.acdc_df.loc[3:, lineage_cols].isna().all(axis=None)


def test_do_not_remove_adds_no_cell_cycle_table_then_remove_works(
        app, prompts
    ):
    host = _Host()
    original = _visited_dfs(host)
    _go_to(host, 3)

    prompts.answers[EDIT_TITLE] = KEEP
    _delete_ID(host, 2)

    assert not _has_cca(host, 3)
    assert host.posData.cca_df is None
    kept = _frame_df(host, 3)
    pd.testing.assert_frame_equal(
        kept[_lineage_and_shared_cols(kept)],
        original[3].loc[[3, 4], _lineage_and_shared_cols(kept)]
    )

    prompts.answers[EDIT_TITLE] = REMOVE
    _delete_ID(host, 4)

    second_prompt = _edit_prompts(prompts)[1]
    assert 'lineage tree annotations' in second_prompt.txt
    assert 'cell cycle' not in second_prompt.txt
    _assert_lineage_removed_from(host, 3, original)


def test_remembered_remove_resets_without_asking(app, prompts):
    host = _Host()
    original = _visited_dfs(host)
    prompts.answers[EDIT_TITLE] = REMOVE
    prompts.remember = True
    _go_to(host, 4)
    _delete_ID(host, 5)
    action = host.warnEditingWithAnnotActions['Delete ID']
    assert not action.isChecked() and action.removeAnnot

    _go_to(host, 2)
    _delete_ID(host, 3)

    assert len(_edit_prompts(prompts)) == 1
    _assert_lineage_removed_from(host, 2, original)


def test_remembered_do_not_remove_runs_the_keep_branch(app, prompts):
    host = _Host(with_lineage=False, with_cca=True)
    prompts.answers[EDIT_TITLE] = KEEP
    prompts.remember = True
    _go_to(host, 4)
    _delete_ID(host, 5)
    action = host.warnEditingWithAnnotActions['Delete ID']
    assert not action.isChecked() and not action.removeAnnot

    _go_to(host, 2)
    assert 3 in host.posData.cca_df.index
    _delete_ID(host, 3)

    assert len(_edit_prompts(prompts)) == 1
    # Kept cell cycle annotations, the deleted ID dropped ("drop removed
    # IDs" of the keep branch)
    assert 3 not in host.posData.cca_df.index
    assert all(_has_cca(host, i) for i in range(SIZE_T))


def test_remove_without_remember_is_not_replayed(app, prompts):
    """A Remove answered without "Remember my choice", then the warning
    turned off in Settings > Warnings: later edits keep the annotations
    (turning a warning off means "do not ask", as before)."""
    host = _Host(with_lineage=True, with_cca=True)
    prompts.answers[EDIT_TITLE] = REMOVE
    _go_to(host, 4)
    _delete_ID(host, 5)
    action = host.warnEditingWithAnnotActions['Delete ID']
    assert action.isChecked() and not action.removeAnnot

    action.trigger()  # the user unchecks the menu item
    assert not action.isChecked()
    before = _visited_dfs(host)
    _go_to(host, 2)
    _delete_ID(host, 3)

    assert len(_edit_prompts(prompts)) == 1
    assert all(_has_cca(host, i) for i in range(4))
    for frame_i in (2, 3):
        df = _frame_df(host, frame_i)
        assert lineage_tree_columns(df) == lineage_tree_columns(before[frame_i])
        assert has_lineage_tree_values(df)


def test_toggling_the_warning_forgets_the_remembered_remove(app, prompts):
    host = _Host()
    prompts.answers[EDIT_TITLE] = REMOVE
    prompts.remember = True
    _go_to(host, 4)
    _delete_ID(host, 5)
    action = host.warnEditingWithAnnotActions['Delete ID']
    assert not action.isChecked() and action.removeAnnot

    action.trigger()  # warning on again ...
    action.trigger()  # ... and off again from the menu
    assert not action.isChecked() and not action.removeAnnot
    _go_to(host, 2)
    _delete_ID(host, 3)

    assert len(_edit_prompts(prompts)) == 1
    assert has_lineage_tree_values(_frame_df(host, 2))
    assert has_lineage_tree_values(_frame_df(host, 3))


@pytest.mark.parametrize('unannotated_value', [np.nan, 0])
def test_no_prompt_on_frames_without_lineage_values(
        app, prompts, unannotated_value
    ):
    # Reset (NaN) or 0-filled frames reloaded from file keep the columns
    host = _Host()
    for frame_i in range(3, SIZE_T):
        df = _frame_df(host, frame_i)
        df[lineage_tree_columns(df)] = unannotated_value
    prompts.answers[EDIT_TITLE] = REMOVE
    _go_to(host, 3)

    assert _delete_ID(host, 2) is True

    assert _edit_prompts(prompts) == []
    assert not _has_cca(host, 3)


def test_live_lineage_tree_forgets_reset_frames(app, prompts):
    host = _Host(mode=LIN)
    tree = normal_division_lineage_tree(gui=host)
    host.lineage_tree = tree
    assert tree.frames_for_dfs == set(range(SIZE_T))
    tree.first_frame_i_for_ID.update({1: 0, 3: 2, 5: 4})
    tree.ID_frame_i_lookup.update({2: 1, 4: 3, 5: 5})
    host.original_df_lin_tree = _frame_df(host, 4).copy()
    host.original_df_lin_tree_i = 4
    prompts.answers[EDIT_TITLE] = REMOVE
    _go_to(host, 3)

    _delete_ID(host, 2)

    assert host.lineage_tree is tree
    assert tree.frames_for_dfs == {0, 1, 2}
    assert tree.first_frame_i_for_ID == {1: 0, 3: 2}
    assert tree.ID_frame_i_lookup == {2: 1}
    # Frames scrollbar ends at the current frame (n. 4), annotated again
    # when the next frame is entered (`autoLinTree_df`)
    assert host.navigateScrollBar.maximum() == 4
    assert host.original_df_lin_tree is None
    assert host.original_df_lin_tree_i is None


def test_reset_keeps_backup_of_earlier_frame_and_restarts_tree_from_frame_0(
        app
    ):
    host = _Host(mode=LIN)
    host.lineage_tree = normal_division_lineage_tree(gui=host)
    backup = _frame_df(host, 1).copy()
    host.original_df_lin_tree = backup
    host.original_df_lin_tree_i = 1
    host.posData.frame_i = 2

    host.resetLin_tree_future()

    assert host.lineage_tree.frames_for_dfs == {0, 1}
    assert host.original_df_lin_tree is backup

    old_tree = host.lineage_tree
    host.posData.frame_i = 0
    host.resetLin_tree_future(from_frame_i=0)

    # Lineage tree mode needs a tree (Propagate raised on None): started
    # again from frame 0, every object a root
    assert host.lineage_tree is not None
    assert host.lineage_tree is not old_tree
    assert host.lineage_tree.frames_for_dfs == {0}
    first_df = _frame_df(host, 0)
    assert (first_df['generation_num_tree'] == 1).all()
    assert (first_df['parent_ID_tree'] == -1).all()
    assert (first_df['root_ID_tree'] == first_df.index).all()
    assert host.navigateScrollBar.maximum() == 1
    assert host.original_df_lin_tree is None
    assert all(
        lineage_tree_columns(_frame_df(host, i)) == []
        for i in range(1, SIZE_T)
    )

    # Outside lineage tree mode the tree is dropped
    host.modeComboBox = _Mode(SEGM)
    host.resetLin_tree_future(from_frame_i=0)

    assert host.lineage_tree is None
    assert all(
        lineage_tree_columns(_frame_df(host, i)) == [] for i in range(SIZE_T)
    )


def test_return_answer_asks_only_about_cell_cycle_and_removes_nothing(
        app, prompts
    ):
    host = _Host(with_cca=True)
    original = _visited_dfs(host)
    prompts.answers[EDIT_TITLE] = REMOVE
    _go_to(host, 0)

    answer = host.warnEditingWithCca_df(
        'Re-initialize cell cyle annotations first frame', return_answer=True
    )

    assert answer is True
    txt = _edit_prompts(prompts)[0].txt
    assert 'cell cycle annotations' in txt
    assert 'lineage' not in txt
    for frame_i in range(SIZE_T):
        pd.testing.assert_frame_equal(_frame_df(host, frame_i), original[frame_i])


def _segm_worker_host():
    host = _Host()
    for name in (
            'activateAnnotations', 'update_rp', 'tracking', 'checkIfAutoSegm',
            'resizeGui'
        ):
        setattr(host, name, MagicMock())
    host.checkHandleTooManyNewItems = lambda *args, **kwargs: True
    return host


def test_repeat_segmentation_prompts_once(app, prompts):
    host = _segm_worker_host()
    prompts.answers[EDIT_TITLE] = KEEP
    _go_to(host, 3)

    host.segmWorkerFinished(host.posData.lab.copy(), 0.1)

    assert len(_edit_prompts(prompts)) == 1
    assert not _has_cca(host, 3)


@pytest.fixture
def repeat_tracking(monkeypatch):
    """Run the real `repeatTrackingVideo` from frame n. `start_n` with the
    tracker selection, the tracker and the worker stubbed."""
    def run(host, start_n):
        class TrackerSelection:
            cancel = False
            selectedItemsText = ['CellACDC']
            startFrame = start_n
            stopFrame = SIZE_T

            def __init__(self, *args, **kwargs):
                pass

            def exec_(self):
                pass

        monkeypatch.setattr(widgets, 'selectTrackerGUI', TrackerSelection)
        monkeypatch.setattr(
            myutils, 'init_tracker', lambda *args, **kwargs: (None, {}, {})
        )
        monkeypatch.setattr(
            myutils, 'validate_tracker_input', lambda *args, **kwargs: None
        )
        monkeypatch.setattr(apps, 'QDialogWorkerProgress', MagicMock())
        host.startTrackingWorker = MagicMock()
        host.repeatTrackingVideo()
        return host.startTrackingWorker.called

    return run


def _repeat_tracking_prompts(prompts):
    return [p for p in prompts.log if p.title == REPEAT_TRACKING_TITLE]


def test_repeat_tracking_video_removes_lineage_after_first_tracked_frame(
        app, prompts, repeat_tracking
    ):
    # The first re-tracked frame (n. 3, index 2) gets its IDs back
    # (`trackingWorker._relabel_first_frame_labels`): IDs can change from
    # index 3 on
    host = _Host()
    original = _visited_dfs(host)
    prompts.answers[REPEAT_TRACKING_TITLE] = YES
    _go_to(host, 5)

    assert repeat_tracking(host, start_n=3)

    warnings = _repeat_tracking_prompts(prompts)
    assert len(warnings) == 1
    assert 'lineage tree annotations' in warnings[0].txt
    assert 'cell cycle' not in warnings[0].txt
    assert 'annotations from frame 4 to frame' in warnings[0].txt
    _assert_lineage_removed_from(host, 3, original)
    lineage_cols = lineage_tree_columns(host.posData.acdc_df)
    assert host.posData.acdc_df.loc[3:, lineage_cols].isna().all(axis=None)
    assert host.posData.acdc_df.loc[:2, lineage_cols].notna().all(axis=None)


@pytest.mark.parametrize('with_cca', [True, False])
def test_repeat_tracking_video_from_first_frame_keeps_first_frame(
        app, prompts, repeat_tracking, with_cca
    ):
    host = _Host(with_lineage=not with_cca, with_cca=with_cca)
    original = _visited_dfs(host)
    prompts.answers[REPEAT_TRACKING_TITLE] = YES
    _go_to(host, 5)

    assert repeat_tracking(host, start_n=1)

    assert len(_repeat_tracking_prompts(prompts)) == 1
    if with_cca:
        assert [_has_cca(host, i) for i in range(SIZE_T)] == (
            [True] + [False]*(SIZE_T-1)
        )
    else:
        _assert_lineage_removed_from(host, 1, original)


def test_repeat_tracking_video_of_annotated_frames_only_does_not_warn(
        app, prompts, repeat_tracking
    ):
    # Annotations up to index 2 = the first re-tracked frame of start_n 3,
    # which keeps its IDs
    host = _Host()
    host.resetLin_tree_future(from_frame_i=3)
    _go_to(host, 5)

    assert repeat_tracking(host, start_n=3)

    assert _repeat_tracking_prompts(prompts) == []


def test_repeat_tracking_video_stopped_keeps_lineage(
        app, prompts, repeat_tracking
    ):
    host = _Host()
    original = _visited_dfs(host)
    prompts.answers[REPEAT_TRACKING_TITLE] = NO
    _go_to(host, 5)

    assert not repeat_tracking(host, start_n=3)

    assert len(_repeat_tracking_prompts(prompts)) == 1
    for frame_i in range(SIZE_T):
        pd.testing.assert_frame_equal(_frame_df(host, frame_i), original[frame_i])


def test_repeat_tracking_video_without_annotations_does_not_warn(
        app, prompts, repeat_tracking
    ):
    host = _Host(with_lineage=False)
    _go_to(host, 5)

    assert repeat_tracking(host, start_n=2)

    assert _repeat_tracking_prompts(prompts) == []


def test_repeat_tracking_video_with_cell_cycle_annotations_still_warns(
        app, prompts, repeat_tracking
    ):
    host = _Host(with_lineage=False, with_cca=True)
    prompts.answers[REPEAT_TRACKING_TITLE] = YES
    _go_to(host, 5)

    assert repeat_tracking(host, start_n=3)

    warnings = _repeat_tracking_prompts(prompts)
    assert len(warnings) == 1
    assert 'cell cycle annotations' in warnings[0].txt
    assert 'lineage' not in warnings[0].txt
    assert [_has_cca(host, i) for i in range(SIZE_T)] == (
        [True]*3 + [False]*(SIZE_T-3)
    )


def _merge_host(with_lineage):
    host = _Host(with_lineage=with_lineage)
    host.zProjComboBox = _Mode('single z-slice')
    host.mergeIDsToolbar = MagicMock()
    for name in (
            'storeUndoRedoStates', 'update_rp_get_bbox', 'update_rp',
            'tracking', 'propagateMergeObjsPast', 'whitelistPropagateIDs'
        ):
        setattr(host, name, MagicMock())
    host.askPropagateChangePast = MagicMock(return_value=False)
    host.warnEditingWithCca_df = MagicMock(return_value=True)
    return host


@pytest.mark.parametrize('with_lineage', [True, False])
def test_merge_back_propagation_is_not_offered_on_lineage_frames(
        app, with_lineage
    ):
    host = _merge_host(with_lineage)
    _go_to(host, 3)

    host.mergeSelectedIDs([3, 4])

    assert host.askPropagateChangePast.called is not with_lineage
    host.warnEditingWithCca_df.assert_called_once_with('Merge IDs')


def test_merge_back_propagation_is_not_offered_after_lineage_reset(app):
    # Frame 3 was reset: the past frames are still annotated
    host = _merge_host(with_lineage=True)
    host.resetLin_tree_future(from_frame_i=3)
    _go_to(host, 3)

    host.mergeSelectedIDs([3, 4])

    host.askPropagateChangePast.assert_not_called()


def test_merge_back_propagation_stops_before_annotated_frames(app):
    # Lineage annotated up to frame 2: from frame 5 the merge can still be
    # propagated to frames 4 and 3, never to the annotated ones
    host = _merge_host(with_lineage=True)
    host.resetLin_tree_future(from_frame_i=3)
    host.askPropagateChangePast = MagicMock(return_value=True)
    _go_to(host, 5)

    host.mergeSelectedIDs([3, 4])

    host.askPropagateChangePast.assert_called_once()
    host.propagateMergeObjsPast.assert_called_once_with(
        [3, 4], stop_frame_i=2
    )


def test_propagate_merge_to_past_frames_stops_at_stop_frame(app):
    host = _Host()
    host.firstID = 3
    stored_frames = []
    host.store_data = lambda *args, **kwargs: stored_frames.append(
        host.posData.frame_i
    )
    host.get_data = MagicMock()
    host.update_rp_get_bbox = MagicMock()
    host.update_rp = MagicMock()
    _go_to(host, 5)

    G.propagateMergeObjsPast(host, [3, 4], stop_frame_i=2)

    # The current frame is stored first, then frames 4 and 3 only
    assert stored_frames == [5, 4, 3]
    assert host.posData.frame_i == 5


def test_autosave_does_not_rejoin_removed_annotations(app, tmp_path):
    """The autosave recovery table takes columns missing in memory (e.g.
    measurements) from the saved table, but not removed annotations (reset
    from the first frame), else 'Recover unsaved data' restores them."""
    saved = pd.DataFrame({
        'frame_i': [0, 0, 1, 1],
        'Cell_ID': [1, 2, 1, 2],
        'is_cell_dead': [0, 0, 0, 0],
        'cell_area_pxl': [10.0, 11.0, 12.0, 13.0],
        'generation_num_tree': [1, 1, 1, 1],
        'parent_ID_tree': [-1, -1, -1, -1],
        'root_ID_tree': [1, 2, 1, 2],
        'sister_ID_tree': [-1, -1, -1, -1],
        'sister_ID_tree_0': [-1, -1, -1, -1],
        'emerg_frame_i': [0, 0, 0, 0],
        'division_frame_i': [0, 0, 0, 0],
        'is_history_known': [False] * 4,
    })
    csv_path = tmp_path / 'acdc_output.csv'
    saved.to_csv(csv_path, index=False)
    recovery_folderpath = tmp_path / 'recovery'
    recovery_folderpath.mkdir()
    posData = SimpleNamespace(
        acdc_output_csv_path=str(csv_path),
        recoveryFolderpath=lambda: str(recovery_folderpath)
    )
    # In memory after `resetLin_tree_future(from_frame_i=0)`
    frame_dfs = [
        pd.DataFrame({'Cell_ID': [1, 2], 'is_cell_dead': [0, 1]})
        .set_index('Cell_ID') for _ in range(2)
    ]
    recovery_acdc_df = pd.concat(
        frame_dfs, keys=[(0, 0.0), (1, 1.0)],
        names=['frame_i', 'time_seconds', 'Cell_ID']
    )
    worker = SimpleNamespace(logger=SimpleNamespace(log=print))

    workers.AutoSaveWorker._save_acdc_df(worker, recovery_acdc_df, posData)

    recovered = pd.read_csv(
        load.get_last_stored_unsaved_acdc_df_filepath(
            str(recovery_folderpath)
        )
    )
    assert recovered['cell_area_pxl'].tolist() == [10.0, 11.0, 12.0, 13.0]
    removed = [
        col for col in saved.columns
        if col in lineage_tree_columns(saved)
        or col in LINEAGE_TREE_SHARED_CCA_COLS
    ]
    assert removed and not set(removed) & set(recovered.columns)
