"""Regression tests for the GUI mode of the normal division lineage tree
(`normal_division_lineage_tree` with a fake GUI holding `allData_li`)."""
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cellacdc.regionprops import acdcRegionprops
from cellacdc import base_cca_dict, load
from cellacdc.trackers.CellACDC_normal_division.CellACDC_normal_division_tracker import (
    normal_division_lineage_tree, is_lineage_tree_annotated,
    init_lineage_tree_cols_of_added_IDs, has_lineage_tree_values,
    lineage_tree_columns
)

SHAPE = (80, 120)


def _lab(boxes):
    lab = np.zeros(SHAPE, np.uint16)
    for ID, (r0, r1, c0, c1) in boxes.items():
        lab[r0:r1, c0:c1] = ID
    return lab


def _frame_data(boxes):
    lab = _lab(boxes)
    rp = acdcRegionprops(lab, precache_centroids=False)
    acdc_df = pd.DataFrame(index=pd.Index(rp.IDs, name='Cell_ID'))
    return {'labels': lab, 'regionprops': rp, 'acdc_df': acdc_df}


def _annotate(frames_boxes):
    """Annotate all frames with `real_time` like the GUI does."""
    allData_li = [_frame_data(boxes) for boxes in frames_boxes]
    gui = SimpleNamespace(
        data=[SimpleNamespace(allData_li=allData_li)], pos_i=0
    )
    tree = normal_division_lineage_tree(gui=gui)
    for frame_i in range(1, len(allData_li)):
        _real_time(tree, allData_li, frame_i)
    return tree, allData_li


def _real_time(tree, allData_li, frame_i):
    tree.real_time(
        frame_i, allData_li[frame_i]['labels'],
        allData_li[frame_i-1]['labels'],
        rp=allData_li[frame_i]['regionprops'],
        prev_rp=allData_li[frame_i-1]['regionprops']
    )


def _lineage(df, ID):
    return tuple(
        int(df.at[ID, col])
        for col in ('generation_num_tree', 'parent_ID_tree', 'root_ID_tree')
    )


def _families_frames(mothers):
    """mothers: list of (mother, (d1, d2), (g1, g2), lost_ID, col0).

    Frame 0: the lost cells (later picked as mothers by the user).
    Frames 1-2: the mothers appear as orphans. Frame 3: all mothers divide
    in the same frame. Frames 5-6: d1 divides into g1, g2.
    """
    frames = [dict() for _ in range(7)]
    for mother, (d1, d2), (g1, g2), lost_ID, c0 in mothers:
        frames[0][lost_ID] = (60, 75, c0, c0 + 20)
        frames[1][mother] = frames[2][mother] = (0, 20, c0, c0 + 20)
        for frame_i in (3, 4):
            frames[frame_i][d1] = (0, 10, c0, c0 + 20)
            frames[frame_i][d2] = (10, 20, c0, c0 + 20)
        for frame_i in (5, 6):
            frames[frame_i][g1] = (0, 5, c0, c0 + 20)
            frames[frame_i][g2] = (5, 10, c0, c0 + 20)
            frames[frame_i][d2] = (10, 20, c0, c0 + 20)
    return frames


def _set_mothers_parents(allData_li, mothers):
    """What find_mother_action does: the orphan mothers get a parent."""
    df = allData_li[1]['acdc_df']
    for mother, _, _, lost_ID, _ in mothers:
        df.at[mother, 'parent_ID_tree'] = lost_ID
        df.at[mother, 'generation_num_tree'] = 2


def _wrong_lineage(allData_li, mothers):
    """(frame_i, ID, found, expected) of the cells with a wrong lineage
    after propagating `_set_mothers_parents`."""
    expected = {}
    for mother, (d1, d2), (g1, g2), lost_ID, _ in mothers:
        expected[mother] = (2, lost_ID, lost_ID)
        expected[d1] = expected[d2] = (3, mother, lost_ID)
        expected[g1] = expected[g2] = (4, d1, lost_ID)

    return [
        (frame_i, ID, _lineage(data['acdc_df'], ID), values)
        for frame_i, data in enumerate(allData_li)
        for ID, values in expected.items()
        if ID in data['acdc_df'].index
        and _lineage(data['acdc_df'], ID) != values
    ]


@pytest.mark.parametrize('relevant_cells', [True, False])
@pytest.mark.parametrize('mothers', [
    [(1, (3, 4), (7, 8), 10, 0), (2, (5, 6), (9, 11), 20, 40)],
    [
        (2, (3, 4), (7, 8), 10, 0), (1, (5, 6), (9, 11), 20, 40),
        (12, (13, 14), (15, 16), 30, 80)
    ],
])
def test_propagate_updates_descendants_of_every_mother(mothers, relevant_cells):
    tree, allData_li = _annotate(_families_frames(mothers))
    _set_mothers_parents(allData_li, mothers)

    if relevant_cells:
        tree.propagate(1, relevant_cells=[m[0] for m in mothers])
    else:
        tree.propagate(1)

    assert not _wrong_lineage(allData_li, mothers)


@pytest.mark.parametrize('relevant_cells', [True, False])
def test_propagate_computes_children_once(monkeypatch, relevant_cells):
    """The children of a parent are computed once, not again in every frame
    they appear in (that made Propagate several times slower on long
    videos)."""
    mothers = [(1, (3, 4), (7, 8), 10, 0), (2, (5, 6), (9, 11), 20, 40)]
    frames = _families_frames(mothers)
    tree, allData_li = _annotate(frames + [frames[-1]] * 13)
    _set_mothers_parents(allData_li, mothers)

    # Each computation of a parent's children reads the frame of the sisters
    sisters_frames = []
    get_df_from_frame_i = tree._get_df_from_frame_i
    update_consistency = tree._update_consistency

    def spy_get_df_from_frame_i(frame_i):
        sisters_frames.append(frame_i)
        return get_df_from_frame_i(frame_i)

    def spy_update_consistency(*args, **kwargs):
        monkeypatch.setattr(
            tree, '_get_df_from_frame_i', spy_get_df_from_frame_i
        )
        try:
            return update_consistency(*args, **kwargs)
        finally:
            monkeypatch.setattr(
                tree, '_get_df_from_frame_i', get_df_from_frame_i
            )

    monkeypatch.setattr(tree, '_update_consistency', spy_update_consistency)
    if relevant_cells:
        # IDs missing in the fixed frame (10: frame 0 only) are ignored
        tree.propagate(1, relevant_cells=[1, 2, 10])
    else:
        tree.propagate(1)

    # Daughters of 1 and 2 (born in frame 3), of 3 and 5 (frame 5)
    assert sorted(sisters_frames) == [3, 3, 5, 5]
    assert not _wrong_lineage(allData_li, mothers)


def _store_data_reindex(allData_li, frame_i, boxes):
    """The acdc_df part of `guiWin.store_data` after a segmentation edit."""
    data = allData_li[frame_i]
    data.update({
        k: v for k, v in _frame_data(boxes).items() if k != 'acdc_df'
    })
    IDs = data['regionprops'].IDs
    acdc_df = data['acdc_df']
    added_IDs = [ID for ID in IDs if ID not in acdc_df.index]
    is_annotated = is_lineage_tree_annotated(acdc_df)
    acdc_df = acdc_df.reindex(IDs, fill_value=0)
    prev_acdc_df = allData_li[frame_i-1]['acdc_df'] if frame_i > 0 else None
    data['acdc_df'] = init_lineage_tree_cols_of_added_IDs(
        acdc_df, added_IDs, frame_i, is_annotated, prev_acdc_df=prev_acdc_df
    )
    return added_IDs


def test_object_added_to_annotated_frame_gets_valid_lineage():
    A, B, C = (0, 20, 0, 20), (0, 20, 40, 60), (40, 60, 0, 20)
    tree, allData_li = _annotate([{1: A, 2: B}] * 3 + [{1: A, 2: B, 99: C}])
    frame_2 = allData_li[2]['acdc_df']

    # Delete 2 (its row is dropped at the next store), then undo it and
    # draw the new object 99 in the already annotated frame 2
    allData_li[2]['acdc_df'] = frame_2.drop(index=2)
    added_IDs = _store_data_reindex(allData_li, 2, {1: A, 2: B, 99: C})
    assert added_IDs == [2, 99]

    df = allData_li[2]['acdc_df']
    prev_df = allData_li[1]['acdc_df']
    lineage_cols = [
        'generation_num_tree', 'parent_ID_tree', 'root_ID_tree',
        'sister_ID_tree', 'emerg_frame_i', 'division_frame_i',
        'is_history_known'
    ]
    # The restored cell continues its lineage from the previous frame
    assert (
        df.loc[2, lineage_cols].tolist() == prev_df.loc[2, lineage_cols].tolist()
    )
    # The new object is a new cell with unknown history
    assert _lineage(df, 99) == (1, -1, 99)
    assert df.at[99, 'sister_ID_tree'] == -1
    assert df.at[99, 'emerg_frame_i'] == 2
    assert df.at[99, 'division_frame_i'] == 2
    assert not df.at[99, 'is_history_known']

    # Used to raise "ID 0 not found in any frame"
    tree.propagate(2)
    tree.propagate(2, relevant_cells=[99])

    # The next frame copies valid values, not zeros
    _real_time(tree, allData_li, 3)
    assert _lineage(allData_li[3]['acdc_df'], 99) == (1, -1, 99)


def test_object_added_to_unannotated_frame_stays_unannotated():
    acdc_df = pd.DataFrame(
        {'generation_num_tree': [np.nan], 'parent_ID_tree': [np.nan]},
        index=pd.Index([1], name='Cell_ID')
    )
    is_annotated = is_lineage_tree_annotated(acdc_df)
    acdc_df = acdc_df.reindex([1, 5], fill_value=0)
    acdc_df = init_lineage_tree_cols_of_added_IDs(acdc_df, [5], 3, is_annotated)

    assert not is_annotated
    assert acdc_df.loc[5].isna().all()
    assert not is_lineage_tree_annotated(acdc_df)


def test_added_IDs_without_lineage_columns_are_unchanged():
    acdc_df = pd.DataFrame(
        {'is_cell_dead': [False]}, index=pd.Index([1], name='Cell_ID')
    )
    acdc_df = acdc_df.reindex([1, 5], fill_value=0)
    expected = acdc_df.copy()
    init_lineage_tree_cols_of_added_IDs(acdc_df, [5], 3, False)
    pd.testing.assert_frame_equal(acdc_df, expected)


def test_added_IDs_into_integer_and_bool_columns_do_not_warn():
    acdc_df = pd.DataFrame(
        {
            'generation_num_tree': [2], 'parent_ID_tree': [4],
            'root_ID_tree': [4], 'sister_ID_tree': [-1],
            'sister_ID_tree_0': [-1], 'emerg_frame_i': [0],
            'division_frame_i': [0], 'is_history_known': [True],
        },
        index=pd.Index([1], name='Cell_ID')
    )
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        annotated = acdc_df.reindex([1, 5], fill_value=0)
        init_lineage_tree_cols_of_added_IDs(annotated, [5], 3, True)
        unannotated = acdc_df.reindex([1, 5], fill_value=0)
        init_lineage_tree_cols_of_added_IDs(unannotated, [5], 3, False)

    assert _lineage(annotated, 5) == (1, -1, 5)
    assert annotated.at[5, 'sister_ID_tree_0'] == -1
    assert annotated.at[5, 'is_history_known'] == False
    assert unannotated.loc[5, 'generation_num_tree':'sister_ID_tree_0'].isna().all()


# --- What counts as lineage annotated: NaN and 0 do not -------------------


def _generations_df(generations):
    return pd.DataFrame(
        {'generation_num_tree': generations, 'parent_ID_tree': -1},
        index=pd.Index(range(1, len(generations) + 1), name='Cell_ID')
    )


@pytest.mark.parametrize('generations, has_values, annotated', [
    ([1, 2], True, True),
    ([1, np.nan], True, False),
    ([1, 0], True, False),
    ([0, 0], False, False),
    ([np.nan, np.nan], False, False),
    ([0, np.nan], False, False),
])
def test_zero_and_nan_lineage_values_are_not_annotated(
        generations, has_values, annotated
    ):
    df = _generations_df(generations)
    assert has_lineage_tree_values(df) is has_values
    assert is_lineage_tree_annotated(df) is annotated


def test_lineage_criteria_without_columns_or_rows():
    no_cols = pd.DataFrame(index=pd.Index([1], name='Cell_ID'))
    empty = _generations_df([])
    for df in (None, no_cols, empty):
        assert not has_lineage_tree_values(df)
        assert not is_lineage_tree_annotated(df)


def test_lineage_tree_columns_include_extra_sister_columns():
    df = pd.DataFrame(columns=[
        'cell_area_pxl', 'generation_num_tree', 'parent_ID_tree',
        'root_ID_tree', 'sister_ID_tree', 'sister_ID_tree_0',
        'sister_ID_tree_1', 'Cell_ID_tree', 'emerg_frame_i', 'phase_G1'
    ])
    assert lineage_tree_columns(df) == [
        'generation_num_tree', 'parent_ID_tree', 'root_ID_tree',
        'sister_ID_tree', 'sister_ID_tree_0', 'sister_ID_tree_1',
        'Cell_ID_tree'
    ]


def test_zero_filled_frames_are_not_in_frames_for_dfs():
    """Frames whose lineage was reset and saved come back from older
    versions of the loader as 0: the tree must annotate them again."""
    tree, allData_li = _annotate(
        [{1: (0, 20, 0, 20), 2: (0, 20, 40, 60)}] * 4
    )
    for frame_i in (2, 3):
        df = allData_li[frame_i]['acdc_df']
        for col in lineage_tree_columns(df):
            df[col] = 0
    gui = SimpleNamespace(
        data=[SimpleNamespace(allData_li=allData_li)], pos_i=0
    )
    reloaded_tree = normal_division_lineage_tree(gui=gui)
    assert reloaded_tree.frames_for_dfs == {0, 1}


def test_loader_keeps_nan_lineage_values_of_tables_with_cca_columns(tmp_path):
    """With every cell cycle column present the loader fills NaN with 0 in
    the other columns, but not in the lineage tree columns (reset or never
    annotated frames must stay unannotated)."""
    index = pd.MultiIndex.from_product(
        [[0, 1], [1, 2]], names=['frame_i', 'Cell_ID']
    )
    table = pd.DataFrame(
        {
            'cell_area_pxl': [10.0, np.nan, 12.0, 13.0],
            'generation_num_tree': [1, 1, np.nan, np.nan],
            'parent_ID_tree': [-1, -1, np.nan, np.nan],
            'root_ID_tree': [1, 2, np.nan, np.nan],
            'sister_ID_tree': [-1, -1, np.nan, np.nan],
            'sister_ID_tree_0': [-1, -1, np.nan, np.nan],
        },
        index=index
    )
    for column, value in base_cca_dict.items():
        table[column] = value
    path = tmp_path / 'acdc_output.csv'
    table.to_csv(path)

    loaded = load._load_acdc_df_file(path)

    assert loaded.at[(0, 2), 'cell_area_pxl'] == 0
    frame_1 = loaded.loc[1]
    for col in lineage_tree_columns(loaded):
        assert frame_1[col].isna().all(), col
    assert not has_lineage_tree_values(frame_1)
