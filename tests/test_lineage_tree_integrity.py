"""Tests of ``cellacdc.lineage_tree_integrity`` (no Qt)."""
import pandas as pd
import pytest

from cellacdc import lineage_tree_integrity as lti

# Lineage: 1 divides into 3 + 4 (frame 3), 3 divides into 5 + 6 (frame 6).
# 2 has unknown history for the whole movie, 4 disappears at frame 7.
N_FRAMES = 10


def _cell(ID, first, last, parent=-1, gen=1, root=None, sisters=()):
    return dict(
        ID=ID, first=first, last=last, parent=parent, gen=gen,
        root=ID if root is None else root, sisters=list(sisters)
    )


def _cells():
    return [
        _cell(1, 0, 2),
        _cell(2, 0, N_FRAMES - 1),
        _cell(3, 3, 5, parent=1, gen=2, root=1, sisters=[4]),
        _cell(4, 3, 6, parent=1, gen=2, root=1, sisters=[3]),
        _cell(5, 6, N_FRAMES - 1, parent=3, gen=3, root=1, sisters=[6]),
        _cell(6, 6, N_FRAMES - 1, parent=3, gen=3, root=1, sisters=[5]),
    ]


def make_table(cells=None, extra_sister_col=True):
    """Pure lineage tree table (as saved by the normal division tracker)."""
    rows = []
    for cell in cells or _cells():
        known = cell['parent'] > 0
        emerg = cell['first'] if known or cell['first'] > 0 else -1
        sisters = cell['sisters'] or [-1]
        for frame_i in range(cell['first'], cell['last'] + 1):
            row = {
                'frame_i': frame_i, 'Cell_ID': cell['ID'],
                'generation_num': cell['gen'],
                'emerg_frame_i': emerg, 'division_frame_i': emerg,
                'is_history_known': known,
                'generation_num_tree': cell['gen'],
                'parent_ID_tree': cell['parent'],
                'root_ID_tree': cell['root'],
                'sister_ID_tree': sisters[0],
                'cell_area_pxl': 100.0,
            }
            if extra_sister_col:
                row['sister_ID_tree_0'] = sisters[0]
            rows.append(row)
    return pd.DataFrame(rows).sort_values(['frame_i', 'Cell_ID'])


def checks(report, severity=('error', 'warning')):
    return sorted({issue.check for issue in report.filter(severity=severity)})


def issues_of(report, check):
    return report.filter(check=check)


def set_value(df, ID, col, value, frames=None):
    mask = df['Cell_ID'] == ID
    if frames is not None:
        mask &= df['frame_i'].isin(frames)
    if df[col].dtype != object and isinstance(value, str):
        df[col] = df[col].astype(object)
    df.loc[mask, col] = value


def test_consistent_tree_has_no_issues():
    df = make_table()
    report = lti.check_lineage_tree(df)
    assert report.issues == []
    assert report.ok and not report.has_errors
    assert 'No inconsistencies found.' in report.to_text()
    # Same result with (frame_i, Cell_ID) as index, input not modified
    indexed = df.set_index(['frame_i', 'Cell_ID'])
    before = indexed.copy()
    assert lti.check_lineage_tree(indexed).issues == []
    pd.testing.assert_frame_equal(indexed, before)


def test_sister_cleared_after_it_disappeared_is_accepted():
    df = make_table()
    # 4 disappears after frame 6: 3 is gone from frame 6, 4 lists no sister
    set_value(df, 4, 'sister_ID_tree', -1, frames=[6])
    set_value(df, 4, 'sister_ID_tree_0', -1, frames=[6])
    assert lti.check_lineage_tree(df).issues == []


def test_stringified_series_from_old_bug(tmp_path):
    df = make_table()
    corrupted = (
        'Cell_ID\n5    3.0\n5    4.0\nName: generation_num_tree, '
        'dtype: float64'
    )
    set_value(df, 5, 'generation_num_tree', corrupted, frames=[8, 9])
    path = tmp_path / 'pos_acdc_output.csv'
    df.to_csv(path, index=False)

    report = lti.check_lineage_tree_file(path)
    invalid = issues_of(report, 'invalid_values')
    assert len(invalid) == 1
    issue = invalid[0]
    assert (issue.Cell_ID, issue.column) == (5, 'generation_num_tree')
    assert (issue.first_frame_i, issue.last_frame_i, issue.n_rows) == (8, 9, 2)
    assert 'Name: generation_num_tree' in issue.value
    assert '\n' not in issue.message
    assert report.has_errors
    assert report.source == str(path)
    # The invalid rows are left out of the structural checks
    assert checks(report) == ['invalid_values']


def test_self_parent_and_value_changes():
    df = make_table()
    # 5 kept the ID of its mother after dividing again at frame 8
    set_value(df, 5, 'parent_ID_tree', 5, frames=[8, 9])
    report = lti.check_lineage_tree(df)
    self_parent = issues_of(report, 'self_parent')
    assert [(i.Cell_ID, i.first_frame_i, i.last_frame_i) for i in self_parent] == [
        (5, 8, 9)
    ]
    changes = issues_of(report, 'value_changes_over_time')
    assert [(i.Cell_ID, i.column, i.value) for i in changes] == [
        (5, 'parent_ID_tree', '3 -> 5')
    ]
    assert 'frame n. 9-10' in self_parent[0].message


def test_issues_are_not_split_by_unrelated_changes():
    df = make_table()
    set_value(df, 5, 'parent_ID_tree', 5)
    # emerg_frame_i changes in the middle: still one self_parent issue
    set_value(df, 5, 'emerg_frame_i', 8, frames=[8, 9])
    set_value(df, 5, 'division_frame_i', 8, frames=[8, 9])
    report = lti.check_lineage_tree(df)
    self_parent = issues_of(report, 'self_parent')
    assert [(i.first_frame_i, i.last_frame_i, i.n_rows) for i in self_parent] == [
        (6, 9, 4)
    ]


def test_generation_and_root_mismatch():
    df = make_table()
    set_value(df, 6, 'generation_num_tree', 4)
    set_value(df, 4, 'root_ID_tree', 2)
    set_value(df, 2, 'generation_num_tree', 2)
    report = lti.check_lineage_tree(df)
    gen = {(i.Cell_ID, i.value, i.expected) for i in issues_of(
        report, 'generation_num_tree_mismatch'
    )}
    assert gen == {(6, '4', '3'), (2, '2', '1')}
    root = [(i.Cell_ID, i.value, i.expected) for i in issues_of(
        report, 'root_ID_tree_mismatch'
    )]
    assert root == [(4, '2', '1')]


def test_parent_timing_checks():
    cells = _cells()
    cells[5]['parent'] = 9  # 6: parent that never exists
    cells.append(_cell(7, 4, 9, parent=2, gen=2, root=2))  # 2 still exists
    cells.append(_cell(8, 8, 9, parent=4, gen=3, root=1))  # 4 gone at 6
    cells.append(_cell(9, 9, 9, parent=-1, gen=1, root=9))
    df = make_table(cells)
    # 5's sister now has another parent; not the subject of this test
    set_value(df, 5, 'sister_ID_tree', -1)
    set_value(df, 5, 'sister_ID_tree_0', -1)
    set_value(df, 6, 'sister_ID_tree', -1)
    set_value(df, 6, 'sister_ID_tree_0', -1)
    report = lti.check_lineage_tree(df)
    found = {(i.check, i.Cell_ID) for i in report.issues}
    assert ('parent_after_daughter', 6) in found  # 9 appears at frame 9
    assert ('parent_present_after_division', 7) in found
    assert ('parent_gap', 8) in found
    assert ('single_daughter', 2) in found
    assert ('single_daughter', 4) in found


def test_parent_not_found():
    cells = _cells()
    cells[4]['parent'] = 99
    df = make_table(cells)
    report = lti.check_lineage_tree(df)
    not_found = issues_of(report, 'parent_not_found')
    assert [(i.Cell_ID, i.value) for i in not_found] == [(5, '99')]
    # 5 and 6 list each other as sisters, but have different parents
    invalid = issues_of(report, 'invalid_sister')
    assert {(i.Cell_ID, i.value) for i in invalid} == {(5, '6'), (6, '5')}


def test_sister_checks():
    df = make_table()
    set_value(df, 5, 'sister_ID_tree', 5)  # itself
    set_value(df, 5, 'sister_ID_tree_0', 5)
    set_value(df, 6, 'sister_ID_tree', 77)  # does not exist
    set_value(df, 6, 'sister_ID_tree_0', 77)
    set_value(df, 2, 'sister_ID_tree', 1)  # unknown history
    set_value(df, 2, 'sister_ID_tree_0', 1)
    report = lti.check_lineage_tree(df)
    invalid = {(i.Cell_ID, i.message.split(': ', 1)[1]) for i in issues_of(
        report, 'invalid_sister'
    )}
    both = 'sister_ID_tree, sister_ID_tree_0'
    assert invalid == {
        (5, f'{both} lists 5, which is the cell itself'),
        (6, f'{both} lists 77, which does not exist in any frame'),
        (2, f'sisters {{1}} listed in {both}, but the parent is unknown'),
    }
    missing = {(i.Cell_ID, i.expected) for i in issues_of(
        report, 'missing_sister'
    )}
    assert missing == {(5, '{6}'), (6, '{5}')}


def test_sister_messages_name_column_and_parent():
    df = make_table()
    # Stale value only in the extra column: 3 lists its own parent 1
    set_value(df, 3, 'sister_ID_tree_0', 1)
    # 6 lists 2 (unknown history) instead of 5
    set_value(df, 6, 'sister_ID_tree', 2)
    report = lti.check_lineage_tree(df)
    invalid = {(i.Cell_ID, i.column, i.message.split(': ', 1)[1]) for i in
               issues_of(report, 'invalid_sister')}
    assert invalid == {
        (3, 'sister_ID_tree_0',
         'sister_ID_tree_0 lists 1, which is the parent of the cell'),
        (6, 'sister_ID_tree',
         'sister_ID_tree lists 2, which has parent unknown, not 3'),
    }


def test_root_minus_one_means_own_root():
    """Convention of base_cca_tree_dict / myutils.df_ctc_to_acdc_df."""
    df = make_table()
    set_value(df, 1, 'root_ID_tree', -1)
    set_value(df, 2, 'root_ID_tree', -1)
    assert lti.check_lineage_tree(df).issues == []
    set_value(df, 4, 'root_ID_tree', 2)
    report = lti.check_lineage_tree(df)
    assert [(i.Cell_ID, i.value, i.expected) for i in issues_of(
        report, 'root_ID_tree_mismatch'
    )] == [(4, '2', '1')]


def test_gui_fill_value_rows():
    tree_cols = [
        'generation_num_tree', 'parent_ID_tree', 'root_ID_tree',
        'sister_ID_tree', 'sister_ID_tree_0', 'emerg_frame_i',
        'division_frame_i'
    ]
    cells = _cells() + [_cell(7, 4, 9)]
    df = make_table(cells)
    # 7 added in frames 4-5 after annotating, frames >= 8 not annotated
    df.loc[(df['Cell_ID'] == 7) & (df['frame_i'] <= 5), tree_cols] = 0
    late = df['frame_i'] >= 8
    df.loc[late, tree_cols] = float('nan')
    df.loc[late & (df['Cell_ID'] == 7), tree_cols] = 0
    report = lti.check_lineage_tree(df)
    found = [(i.check, i.Cell_ID, i.first_frame_i, i.last_frame_i)
             for i in report.issues]
    assert found == [
        ('fill_value_rows', 7, 4, 5),
        ('unannotated_frames', None, 8, 9),
    ]


@pytest.mark.parametrize('dtype', [object, 'string'])
def test_invalid_values_in_text_columns(dtype):
    df = make_table()
    df['root_ID_tree'] = df['root_ID_tree'].astype(str).astype(dtype)
    df.loc[df['Cell_ID'] == 5, 'root_ID_tree'] = 'Cell_ID\n5    1.0'
    report = lti.check_lineage_tree(df)
    assert [(i.check, i.Cell_ID) for i in report.issues] == [
        ('invalid_values', 5)
    ]


def test_missing_values_do_not_cause_value_changes():
    df = make_table()
    df['emerg_frame_i'] = float('nan')
    df['division_frame_i'] = float('nan')
    set_value(df, 6, 'parent_ID_tree', 6, frames=[9])
    report = lti.check_lineage_tree(df)
    assert [(i.check, i.Cell_ID, i.column) for i in report.issues] == [
        ('self_parent', 6, 'parent_ID_tree'),
        ('value_changes_over_time', 6, 'parent_ID_tree'),
    ]


def test_corrupted_values_do_not_hide_self_parent():
    df = make_table()
    set_value(df, 5, 'parent_ID_tree', 5, frames=[7, 8, 9])
    set_value(df, 5, 'root_ID_tree', 'Cell_ID\n5    1.0', frames=[8, 9])
    report = lti.check_lineage_tree(df)
    self_parent = issues_of(report, 'self_parent')
    assert [(i.first_frame_i, i.last_frame_i) for i in self_parent] == [
        (7, 9)
    ]


def test_lineage_restarted_after_absence_is_explained():
    cells = _cells()
    cells[5]['last'] = 7
    # 6 is missing at frame 8 and comes back with unknown history
    cells.append(_cell(6, 9, 9))
    df = make_table(cells)
    report = lti.check_lineage_tree(df)
    change = [i for i in issues_of(report, 'value_changes_over_time')
              if i.column == 'parent_ID_tree']
    assert len(change) == 1
    assert (
        '3 (frame n. 7-8), [cell absent in 1 frame(s) between frame n. 8 '
        'and 10], -1 (frame n. 10)'
    ) in change[0].message


def test_zero_in_sister_columns_is_info():
    df = make_table()
    set_value(df, 5, 'sister_ID_tree_0', 0, frames=[7, 9])
    report = lti.check_lineage_tree(df)
    assert report.ok
    zero = issues_of(report, 'zero_sister_IDs')
    assert [(i.Cell_ID, i.first_frame_i, i.last_frame_i, i.n_rows) for i in zero] == [
        (5, 7, 9, 2)
    ]
    assert '(2 frames)' in zero[0].message


def test_is_history_known_mismatch_is_info_per_cell():
    df = make_table()
    set_value(df, 5, 'is_history_known', 'False', frames=[7, 9])
    report = lti.check_lineage_tree(df)
    assert report.ok
    mismatch = issues_of(report, 'is_history_known_mismatch')
    assert [(i.Cell_ID, i.severity, i.n_rows) for i in mismatch] == [
        (5, 'info', 2)
    ]


def test_emerg_and_division_frame_checks():
    df = make_table()
    set_value(df, 5, 'emerg_frame_i', 7)
    set_value(df, 5, 'division_frame_i', 7)
    set_value(df, 6, 'division_frame_i', 5)
    report = lti.check_lineage_tree(df)
    assert [(i.Cell_ID, i.value, i.expected) for i in issues_of(
        report, 'emerg_frame_i_mismatch'
    )] == [(5, '7', '6')]
    assert [(i.Cell_ID, i.value, i.expected) for i in issues_of(
        report, 'division_frame_i_mismatch'
    )] == [(6, '5', '6')]


def test_budding_cca_columns_skip_pure_tree_checks():
    df = make_table()
    df['cell_cycle_stage'] = 'G1'
    set_value(df, 5, 'emerg_frame_i', 7)
    set_value(df, 5, 'is_history_known', False)
    assert lti.check_lineage_tree(df).issues == []


def test_lineage_cycle():
    cells = [
        _cell(1, 0, 9, parent=2, gen=2, root=1),
        _cell(2, 0, 9, parent=1, gen=2, root=1),
    ]
    report = lti.check_lineage_tree(make_table(cells))
    cycles = issues_of(report, 'lineage_cycle')
    assert [(i.Cell_ID, i.value) for i in cycles] == [(1, '{1, 2}')]


def test_daughter_count_and_birth_frames():
    cells = _cells()
    cells.append(_cell(7, 4, 9, parent=1, gen=2, root=1))
    df = make_table(cells)
    report = lti.check_lineage_tree(df)
    found = {(i.check, i.Cell_ID) for i in report.issues}
    assert ('too_many_daughters', 1) in found
    assert ('daughters_born_in_different_frames', 1) in found
    assert ('missing_sister', 3) in found
    assert ('parent_gap', 7) in found


def test_duplicated_rows_and_missing_values():
    df = make_table()
    df = pd.concat([df, df[(df['Cell_ID'] == 2) & (df['frame_i'] < 2)]])
    set_value(df, 5, 'root_ID_tree', float('nan'), frames=[9])
    report = lti.check_lineage_tree(df)
    dup = issues_of(report, 'duplicated_rows')
    assert [(i.Cell_ID, i.first_frame_i, i.last_frame_i) for i in dup] == [
        (2, 0, 1)
    ]
    missing = issues_of(report, 'missing_values')
    assert [(i.Cell_ID, i.column, i.first_frame_i) for i in missing] == [
        (5, 'root_ID_tree', 9)
    ]


def test_unannotated_frames_are_info():
    df = make_table()
    tree_cols = [
        'generation_num_tree', 'parent_ID_tree', 'root_ID_tree',
        'sister_ID_tree', 'sister_ID_tree_0'
    ]
    df.loc[df['frame_i'] >= 7, tree_cols] = float('nan')
    report = lti.check_lineage_tree(df)
    assert checks(report, severity=None) == ['unannotated_frames']
    issue = issues_of(report, 'unannotated_frames')[0]
    assert (issue.first_frame_i, issue.last_frame_i) == (7, 9)
    assert 'frame n. 8-10' in issue.message


def test_budding_lineage_tree_and_missing_columns():
    df = make_table()
    df['Cell_ID_tree'] = df['Cell_ID']
    df.loc[df['frame_i'] > 5, 'Cell_ID_tree'] = 100
    report = lti.check_lineage_tree(df)
    assert checks(report, severity=None) == ['budding_lineage_tree']

    no_tree = make_table().drop(columns=[
        'generation_num_tree', 'parent_ID_tree', 'root_ID_tree',
        'sister_ID_tree', 'sister_ID_tree_0'
    ])
    assert lti.check_lineage_tree(no_tree).issues == []
    with pytest.raises(ValueError):
        lti.check_lineage_tree(no_tree.drop(columns='Cell_ID'))


def test_report_outputs(tmp_path):
    df = make_table()
    set_value(df, 5, 'parent_ID_tree', 5)
    report = lti.check_lineage_tree(df, source='table.csv')
    text = report.to_text()
    assert text.startswith('Lineage tree check: table.csv')
    assert 'ERRORS' in text and 'self_parent' in text
    # 3 is left with one daughter (6), and 6 lists 5 as sister
    assert report.cell_IDs() == [3, 5, 6]

    issues_df = report.to_dataframe()
    assert len(issues_df) == len(report)
    assert issues_df['source'].eq('table.csv').all()

    csv_path = tmp_path / 'report.csv'
    report.save(csv_path)
    assert len(pd.read_csv(csv_path)) == len(report)
    txt_path = tmp_path / 'report.txt'
    report.save(txt_path)
    assert txt_path.read_text(encoding='utf-8').startswith('Lineage tree')


def test_files_and_command_line(tmp_path, capsys):
    good = tmp_path / 'Position_1' / 'Images'
    bad = tmp_path / 'Position_2' / 'Images'
    good.mkdir(parents=True)
    bad.mkdir(parents=True)
    make_table().to_csv(good / 'exp_acdc_output.csv', index=False)
    corrupted = make_table()
    set_value(corrupted, 4, 'parent_ID_tree', 4)
    corrupted.to_csv(bad / 'exp_acdc_output.csv', index=False)
    # Tables without lineage columns and other CSV files are skipped
    pd.DataFrame({'frame_i': [0], 'Cell_ID': [1]}).to_csv(
        good / 'other_acdc_output_auto.csv', index=False
    )
    make_table().to_csv(good / 'exp_metadata.csv', index=False)
    # Raw tracker tables are skipped in folders by default
    corrupted.to_csv(good / 'exp_acdc_output_auto.csv', index=False)

    reports = lti.check_lineage_tree_files(tmp_path)
    assert [report.has_errors for report in reports] == [False, True]
    with_auto = lti.check_lineage_tree_files(tmp_path, include_auto=True)
    assert [report.has_errors for report in with_auto] == [False, True, True]
    explicit = lti.check_lineage_tree_files(good / 'exp_acdc_output_auto.csv')
    assert [report.has_errors for report in explicit] == [True]

    out = tmp_path / 'report.csv'
    assert lti.main([str(tmp_path), '--out', str(out)]) == 1
    printed = capsys.readouterr().out
    assert 'Checked 2 file(s): 1 with errors or warnings.' in printed
    assert pd.read_csv(out)['check'].eq('self_parent').any()
    assert lti.main([str(good)]) == 0
    assert lti.main([str(tmp_path / 'missing')]) == 2
    assert 'Path not found' in capsys.readouterr().err


def test_unreadable_files_do_not_stop_the_scan(tmp_path):
    (tmp_path / 'a_acdc_output.csv').write_text('')
    no_ID = make_table().drop(columns='Cell_ID')
    no_ID.to_csv(tmp_path / 'b_acdc_output.csv', index=False)
    make_table().to_csv(tmp_path / 'c_acdc_output.csv', index=False)
    reports = lti.check_lineage_tree_files(tmp_path)
    assert [checks(report) for report in reports] == [
        ['unreadable_file'], ['unreadable_file'], []
    ]
