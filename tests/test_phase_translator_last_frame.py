"""Phase evaluation restricted to the frames up to the last analyzed frame
(``phase_translator.evaluate(..., last_frame=...)``)."""
import numpy as np
import pandas as pd
import pytest

from cellacdc import phase_translator
from cellacdc.custom_annotation_predictors import CustomAnnotationPredictor
from cellacdc.timepoint_predictors import TIMEPOINT_PREFIX

START = f'{TIMEPOINT_PREFIX}Start'
END = f'{TIMEPOINT_PREFIX}End'
FRAMES = 6


def make_table(cells):
    """``cells``: {ID: (frames, start frames, end frames, flag frames)}."""
    rows = []
    for ID, (frames, starts, ends, flags) in cells.items():
        for frame in frames:
            rows.append({
                'frame_i': frame, 'Cell_ID': ID, 'area': 10. * frame + ID,
                START: frame in starts, END: frame in ends,
                'Flag': int(frame in flags),
            })
    return pd.DataFrame(rows).set_index(['frame_i', 'Cell_ID'])


@pytest.fixture
def table():
    every = range(FRAMES)
    return make_table({
        # Both boundaries before the last analyzed frame (4).
        1: (every, [1], [3], [0]),
        # End marked only after the last analyzed frame.
        2: (every, [1], [5], [0]),
        # Duplicate start, the second one after the last analyzed frame.
        3: (every, [1, 5], [3], [0]),
        # Required annotation only after the last analyzed frame.
        4: (every, [1], [3], [5]),
    })


def rule(**kwargs):
    return {
        'name': 'Cycle', 'predictor': {'type': 'none'},
        'timepoints': [START, END], 'annotations': ['Flag'],
        'phases': ['G1', 'S', 'G2'], **kwargs
    }


def fallback(**kwargs):
    return rule(**{
        'name': 'Fallback', 'timepoints': [], 'annotations': [],
        'phases': ['Other'], **kwargs
    })


def frame_values(df):
    if 'frame_i' in df.index.names:
        return df.index.get_level_values('frame_i').to_numpy()
    return df['frame_i'].to_numpy()


def phases(result, ID):
    return result.table.xs(ID, level='Cell_ID')['phase'].tolist()


def assert_evaluations_equal(left, right):
    pd.testing.assert_frame_equal(left.table, right.table)
    assert left.columns == right.columns
    assert left.warnings == right.warnings


def test_boundaries_annotations_and_last_interval_stop_at_last_frame(table):
    result = phase_translator.evaluate(
        table, [rule(), fallback()], last_frame=4
    )
    # The last interval stops at the last analyzed frame.
    assert phases(result, 1) == ['G1', 'S', 'S', 'G2', 'G2', '']
    # A boundary marked only after last_frame counts as missing.
    assert phases(result, 2) == ['Other'] * 5 + ['']
    # A duplicate boundary after last_frame is ignored (no warning either).
    assert phases(result, 3) == ['G1', 'S', 'S', 'G2', 'G2', '']
    # A required annotation marked only after last_frame is missing.
    assert phases(result, 4) == ['Other'] * 5 + ['']
    assert result.warnings == []
    # The row set never changes.
    assert result.table.index.equals(table.index)
    after = result.table.index.get_level_values('frame_i') > 4
    for column in result.columns - {'phase'}:
        assert result.table[column].dtype == bool
        assert not result.table.loc[after, column].any()


def test_cell_without_matching_rule_stays_unassigned(table):
    result = phase_translator.evaluate(table, [rule()], last_frame=4)
    assert phases(result, 2) == [''] * FRAMES
    assert phases(result, 4) == [''] * FRAMES
    # Without the mask the duplicate start makes cell 3 unmatched.
    unmasked = phase_translator.evaluate(table, [rule()])
    assert phases(unmasked, 3) == [''] * FRAMES
    assert '3' in unmasked.warnings[0]
    assert phases(unmasked, 2) == ['G1', 'S', 'S', 'S', 'S', 'G2']


@pytest.mark.parametrize('IDs', [None, [5], [1, 5]])
def test_cell_existing_only_after_last_frame(table, IDs, monkeypatch):
    late = make_table({5: ([5], [], [], [5])})
    df = pd.concat([table, late])
    predicted_ids = []
    predict = CustomAnnotationPredictor.predict

    def recording_predict(acdc_df, predictor, SizeT=None, IDs=None):
        predicted_ids.extend(IDs)
        return predict(acdc_df, predictor, SizeT=SizeT, IDs=IDs)

    monkeypatch.setattr(
        CustomAnnotationPredictor, 'predict', staticmethod(recording_predict)
    )
    rules = [
        rule(),
        fallback(predictor={'type': 'equation', 'equation': 'area >= 0'}),
        fallback(name='Always', phases=['Always']),
    ]
    result = phase_translator.evaluate(df, rules, IDs=IDs, last_frame=4)
    assert phases(result, 5) == ['']
    assert 5 not in predicted_ids
    assert result.table.index.equals(df.index)
    if IDs is None or 1 in IDs:
        assert phases(result, 1) == ['G1', 'S', 'S', 'G2', 'G2', '']
    # Missing cells are still checked on the whole table.
    with pytest.raises(ValueError, match='No measurement rows'):
        phase_translator.evaluate(df, rules, IDs=[6], last_frame=4)


def test_no_analyzed_rows_at_all(table):
    result = phase_translator.evaluate(
        table, [rule(), fallback()], last_frame=-1
    )
    assert (result.table['phase'] == '').all()
    assert not result.table[['phase_G1', 'phase_Other']].any().any()


@pytest.mark.parametrize('SizeT, expected', [(None, True), (10, True), (3, False)])
def test_formula_gets_analyzed_rows_and_clipped_size_t(
        table, SizeT, expected
    ):
    # SizeT is min(SizeT, last_frame + 1); with SizeT=3 frames from 3 on are
    # outside the video, and SizeT == 5 is False.
    formula = fallback(predictor={'type': 'equation', 'equation': 'SizeT == 5'})
    result = phase_translator.evaluate(
        table, [formula], IDs=[1], SizeT=SizeT, last_frame=4
    )
    assert phases(result, 1) == (['Other'] * 5 if expected else [''] * 5) + ['']


@pytest.mark.parametrize('equation, masked, unmasked', [
    ('LAST(frame_i) == 4', True, False),
    ('MAX(area) < 50', True, False),
    ('MAX(area_other) < 50', True, False),
])
def test_formula_ignores_rows_after_last_frame(
        table, equation, masked, unmasked
    ):
    formula = fallback(predictor={
        'type': 'equation', 'equation': equation, 'other_cells': {}
    })
    result = phase_translator.evaluate(table, [formula], IDs=[1], last_frame=4)
    assert phases(result, 1) == (['Other'] * 5 if masked else [''] * 5) + ['']
    result = phase_translator.evaluate(table, [formula], IDs=[1])
    assert phases(result, 1) == (
        ['Other'] * FRAMES if unmasked else [''] * FRAMES
    )


def test_predict_receives_only_analyzed_rows(table, monkeypatch):
    calls = []
    predict = CustomAnnotationPredictor.predict

    def recording_predict(acdc_df, predictor, SizeT=None, IDs=None):
        calls.append((acdc_df.copy(), SizeT, sorted(IDs)))
        return predict(acdc_df, predictor, SizeT=SizeT, IDs=IDs)

    monkeypatch.setattr(
        CustomAnnotationPredictor, 'predict', staticmethod(recording_predict)
    )
    formula = fallback(predictor={'type': 'equation', 'equation': 'area > 0'})
    flat = table.reset_index().sample(frac=1, random_state=0)
    for df in (table, flat):
        calls.clear()
        phase_translator.evaluate(df, [formula], last_frame=2)
        (received, SizeT, IDs), = calls
        assert SizeT == 3
        assert IDs == [1, 2, 3, 4]
        assert sorted(set(frame_values(received))) == [0, 1, 2]
        pd.testing.assert_frame_equal(
            received, df.iloc[np.flatnonzero(frame_values(df) <= 2)]
        )


def test_flat_unsorted_table_matches_indexed_table(table):
    rules = [rule(), fallback()]
    indexed = phase_translator.evaluate(table, rules, last_frame=4)
    flat = table.reset_index().sample(frac=1, random_state=1)
    result = phase_translator.evaluate(flat, rules, last_frame=4)
    assert result.table.index.equals(flat.index)
    expected = indexed.table.reset_index().set_index(['frame_i', 'Cell_ID'])
    got = result.table.set_index(['frame_i', 'Cell_ID']).loc[expected.index]
    pd.testing.assert_series_equal(got['phase'], expected['phase'])


def test_other_cells_rows_after_last_frame_are_cleared(table):
    first = phase_translator.evaluate(table, [rule(), fallback()])
    assert phases(first, 2)[-1] == 'G2'
    # Cell 2 keeps its phases up to last_frame but none after it, although
    # only cell 1 is re-evaluated.
    result = phase_translator.evaluate(
        first.table, [rule(), fallback()], IDs=[1],
        owned_columns=first.columns, last_frame=4
    )
    assert phases(result, 2) == ['G1', 'S', 'S', 'S', 'S', '']
    assert phases(result, 3) == ['Other'] * 5 + ['']
    cell2 = result.table.xs(2, level='Cell_ID')
    assert cell2['phase_G2'].tolist() == [False] * FRAMES
    assert cell2['phase_S'].tolist() == [False, True, True, True, True, False]


def test_reevaluated_cell_loses_boundary_after_last_frame(table):
    late_rule = rule(
        name='Late', annotations=[], phases=['Early', 'Mid', 'LateOnly']
    )
    one_cell = table.xs(2, level='Cell_ID', drop_level=False)
    first = phase_translator.evaluate(one_cell, [late_rule])
    assert phases(first, 2) == ['Early'] + ['Mid'] * 4 + ['LateOnly']
    result = phase_translator.evaluate(
        first.table, [late_rule], owned_columns=first.columns, last_frame=4
    )
    # Outputs of the current rules are kept, but nothing is assigned.
    assert result.columns == first.columns
    assert not result.table['phase_LateOnly'].any()
    assert phases(result, 2) == [''] * FRAMES


def test_obsolete_output_of_non_target_cell_after_last_frame(table):
    """A phase name used by a non-target cell only after last_frame (and not
    produced by the current rules) is removed with its marker column."""
    old_rules = [rule(annotations=[], phases=['G1', 'S', 'Gone'])]
    first = phase_translator.evaluate(table, old_rules)
    assert 'Gone' in phases(first, 2)
    assert phases(first, 2).index('Gone') == 5
    new_rules = [rule(), fallback()]
    # Cells 1, 3 and 4 have 'Gone' before frame 5: re-evaluate them, cell 2
    # keeps its phases except after last_frame.
    result = phase_translator.evaluate(
        first.table, new_rules, IDs=[1, 3, 4],
        owned_columns=first.columns, last_frame=4
    )
    assert 'phase_Gone' not in result.table
    assert 'phase_Gone' not in result.columns
    assert phases(result, 2) == ['G1', 'S', 'S', 'S', 'S', '']


def test_last_frame_none_and_beyond_last_frame_change_nothing(table):
    rules = [
        rule(predictor={'type': 'equation', 'equation': 'MAX(area) > 20'}),
        fallback(),
    ]
    reference = phase_translator.evaluate(table, rules, SizeT=FRAMES)
    assert phases(reference, 1) == ['G1', 'S', 'S', 'G2', 'G2', 'G2']
    assert_evaluations_equal(
        phase_translator.evaluate(table, rules, SizeT=FRAMES, last_frame=None),
        reference,
    )
    for last_frame in (FRAMES - 1, FRAMES + 3):
        assert_evaluations_equal(
            phase_translator.evaluate(
                table, rules, SizeT=FRAMES, last_frame=last_frame
            ),
            reference,
        )
    for IDs in ([2], [1, 3]):
        reference = phase_translator.evaluate(table, rules, IDs=IDs)
        assert_evaluations_equal(
            phase_translator.evaluate(
                table, rules, IDs=IDs, last_frame=FRAMES - 1
            ),
            reference,
        )


# --- Annotations overriding phases ------------------------------------------

def test_override_counts_and_assigns_only_up_to_last_frame(table):
    table['Excluded'] = 0
    # Cell 1: set in an analyzed frame; cell 2: only after last_frame
    table.loc[(2, 1), 'Excluded'] = 1
    table.loc[(5, 2), 'Excluded'] = 1
    rules = [rule(), fallback()]
    result = phase_translator.evaluate(
        table, rules, last_frame=4, annotation_overrides=['Excluded']
    )
    assert phases(result, 1) == ['Excluded'] * 5 + ['']
    # A flag only after last_frame does not override the phases
    assert phases(result, 2) == ['Other'] * 5 + ['']
    after = result.table.index.get_level_values('frame_i') > 4
    assert result.table['phase_Excluded'].tolist() == (
        result.table['phase'] == 'Excluded'
    ).tolist()
    assert not result.table.loc[after, 'phase_Excluded'].any()
    assert result.warnings == []
    assert result.table.index.equals(table.index)
    # Same phases as evaluating only the analyzed rows
    analyzed = table[~after]
    cut = phase_translator.evaluate(
        analyzed, rules, annotation_overrides=['Excluded']
    )
    pd.testing.assert_series_equal(
        result.table.loc[~after, 'phase'], cut.table['phase']
    )
    assert cut.columns == result.columns
    assert cut.warnings == result.warnings
    # Without the mask the late flag of cell 2 overrides it everywhere
    unmasked = phase_translator.evaluate(
        table, rules, annotation_overrides=['Excluded']
    )
    assert phases(unmasked, 2) == ['Excluded'] * FRAMES


def test_override_without_rules_and_analyzed_rows(table):
    table['Excluded'] = 1
    result = phase_translator.evaluate(
        table, [], last_frame=-1, annotation_overrides=['Excluded']
    )
    assert (result.table['phase'] == '').all()
    assert not result.table['phase_Excluded'].any()
    # Cell 5 exists only after the last analyzed frame
    late = make_table({5: ([5], [], [], [])})
    late['Excluded'] = 1
    df = pd.concat([table, late])
    result = phase_translator.evaluate(
        df, [], IDs=[1, 5], last_frame=4, annotation_overrides=['Excluded']
    )
    assert phases(result, 1) == ['Excluded'] * 5 + ['']
    assert phases(result, 5) == ['']
    assert phases(result, 2) == [''] * FRAMES
