"""Regression tests for the lineage tree edit backup of `gui.guiWin`
(`original_df_lin_tree`), compared with the current frame table by
`get_difference_table` when saving, changing frame or leaving the mode."""
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from cellacdc import lineage_tree_cols
from cellacdc.gui.main_window import window as gui_window
from cellacdc.shared.widgets import message_box as widgets_message_box

pytestmark = pytest.mark.gui

LIN_TREE_MODE = 'Normal division: Lineage tree'


def _frame_df(parents):
    """Lineage tree table of one frame, `parents` = {Cell_ID: parent ID}."""
    IDs = list(parents)
    return pd.DataFrame({
        'Cell_ID': IDs,
        'generation_num_tree': [1 if p == -1 else 2 for p in parents.values()],
        'parent_ID_tree': list(parents.values()),
        'root_ID_tree': [ID if p == -1 else p for ID, p in parents.items()],
        'sister_ID_tree': -1,
        'is_history_known': [p != -1 for p in parents.values()],
    }).set_index('Cell_ID')


# Real guiWin methods of the backup check, bound to the stub host
REAL_METHODS = (
    'get_difference_table', 'clearLineageTreeBackup',
    'lineageTreeBackupMismatch', 'clearStaleLineageTreeBackup',
)


def _host(frames_dfs, frame_i, backup_frame_i=None):
    posData = SimpleNamespace(
        # 'labels' only marks the frames as loaded
        # (`lineageTreeBackupMismatch`)
        allData_li=[{'acdc_df': df, 'labels': True} for df in frames_dfs],
        frame_i=frame_i, SizeT=len(frames_dfs), acdc_df=None
    )
    host = MagicMock()
    host.data = [posData]
    host.pos_i = 0
    host.modeComboBox.currentText.return_value = LIN_TREE_MODE
    host.lineage_tree = SimpleNamespace(
        frames_for_dfs=set(range(len(frames_dfs))), propagate=MagicMock()
    )
    host.original_df_lin_tree = None
    host.original_df_lin_tree_i = None
    if backup_frame_i is not None:
        host.original_df_lin_tree = frames_dfs[backup_frame_i].copy()
        host.original_df_lin_tree_i = backup_frame_i
    for name in REAL_METHODS:
        setattr(host, name, partial(getattr(gui_window.guiWin, name), host))
    return host


@pytest.fixture
def no_message_box(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError('No question expected')
    monkeypatch.setattr(widgets_message_box, 'myMessageBox', fail)


def test_save_after_lineage_annotations_were_removed(no_message_box):
    # Edited frame 1 in lineage tree mode (backup), then its lineage tree
    # columns were dropped ("Remove annotations from future frames"):
    # saving raised KeyError "['generation_num_tree', ...] not in index"
    frames = [_frame_df({1: -1, 2: -1}), _frame_df({1: -1, 3: 2})]
    host = _host(frames, frame_i=1, backup_frame_i=1)
    posData = host.data[0]
    posData.allData_li[1]['acdc_df'] = frames[1].drop(
        columns=lineage_tree_cols, errors='ignore'
    )

    gui_window.guiWin.askLineageTreeChanges(host)

    assert host.original_df_lin_tree is None
    assert host.original_df_lin_tree_i is None
    host.lineage_tree.propagate.assert_not_called()


@pytest.mark.parametrize('reset_frame_i, backup_kept', [(1, False), (2, True)])
def test_resetting_future_lineage_drops_backup_of_reset_frames(
        reset_frame_i, backup_kept
    ):
    frames = [
        _frame_df({1: -1, 2: -1}), _frame_df({1: -1, 3: 2}),
        _frame_df({1: -1, 3: 2})
    ]
    host = _host(frames, frame_i=reset_frame_i, backup_frame_i=1)

    gui_window.guiWin.resetLin_tree_future(host)

    allData_li = host.data[0].allData_li
    assert 'parent_ID_tree' in allData_li[reset_frame_i-1]['acdc_df']
    for i in range(reset_frame_i, len(frames)):
        assert 'parent_ID_tree' not in allData_li[i]['acdc_df']
    assert (host.original_df_lin_tree is not None) == backup_kept
    assert (host.original_df_lin_tree_i == 1) == backup_kept


def test_difference_lists_changed_parents():
    backup = _frame_df({1: -1, 2: -1, 3: -1})
    host = _host([_frame_df({1: -1, 2: -1}), backup], 1, backup_frame_i=1)
    # Parent of ID 3 changed after the backup
    host.data[0].allData_li[1]['acdc_df'] = _frame_df({1: -1, 2: -1, 3: 2})

    _, _, differences = gui_window.guiWin.get_difference_table(
        host, return_css_separated=True, return_differece=True
    )

    assert differences['Cell_ID'].tolist() == [3]
    assert differences['self'].tolist() == [-1]
    assert differences['other'].tolist() == [2]


def test_backup_dropped_when_IDs_added_or_removed_after_backup():
    # Discard would restore the rows of the old objects: the backup is
    # stale (`lineageTreeBackupMismatch`) and cleared without comparing
    backup = _frame_df({1: -1, 2: -1, 3: -1})
    host = _host([_frame_df({1: -1, 2: -1}), backup], 1, backup_frame_i=1)
    # ID 2 deleted, ID 4 added and parent of ID 3 changed after the backup
    host.data[0].allData_li[1]['acdc_df'] = _frame_df({1: -1, 3: 2, 4: -1})

    result = gui_window.guiWin.get_difference_table(
        host, return_css_separated=True, return_differece=True
    )

    assert result is None
    assert host.original_df_lin_tree is None
    assert host.original_df_lin_tree_i is None


def test_no_difference_without_parent_changes():
    backup = _frame_df({1: -1, 3: 2})
    host = _host([_frame_df({1: -1, 2: -1}), backup], 1, backup_frame_i=1)
    changed = backup.copy()
    changed['is_history_known'] = False

    host.data[0].allData_li[1]['acdc_df'] = changed

    assert gui_window.guiWin.get_difference_table(host) is None
