"""Regression tests for the yes/no columns of a loaded acdc_output table
(`load.acdc_df_bool_cols`), cast by `load._load_acdc_df_file`."""
import numpy as np
import pandas as pd
import pytest

from cellacdc import load

pytestmark = pytest.mark.core

CSV = """frame_i,Cell_ID,is_cell_dead,is_cell_excluded,is_history_known,cell_cycle_stage,relationship
0,1,0,0,1,G1,mother
0,2,0,1,0,G1,mother
1,1,1,0,1,G1,mother
1,2,0,0,,,
"""


def _load(tmp_path):
    csv_path = tmp_path / 'acdc_output.csv'
    csv_path.write_text(CSV)
    return load._load_acdc_df_file(str(csv_path))


def test_bool_cols_are_plain_bool_with_missing_values_false(tmp_path):
    acdc_df = _load(tmp_path)

    for col in load.acdc_df_bool_cols:
        assert acdc_df[col].dtype == np.dtype(bool), col
    assert acdc_df['is_history_known'].tolist() == [True, False, True, False]
    assert acdc_df['is_cell_excluded'].tolist() == [False, True, False, False]


def test_frames_with_added_IDs_can_be_concatenated(tmp_path):
    # The main GUI's `store_data` reindexes a frame to its current objects
    # with fill_value=0. On nullable 'boolean' columns this gave arrays that
    # made the save fail in `cli._concat_and_save_acdc_df` with "values
    # should be boolean numpy array"
    acdc_df = _load(tmp_path)
    frames = [acdc_df.loc[0], acdc_df.loc[1].reindex([1, 2, 3], fill_value=0)]

    all_frames = pd.concat(
        frames, keys=[0, 1], names=['frame_i', 'Cell_ID']
    )

    assert all_frames.loc[(1, 3), 'is_history_known'] == 0
    assert len(all_frames) == 5
