"""Re-structure goldens and the golden-comparison machinery.

Every case of `scenarios.RESTRUCTURE` is run with tifffile inputs and
compared with `golden/restructure/<id>.json` (captured on main) under the
registry `expected_restructure_changes.CHANGES`. Fast; needs only tifffile
(plus Qt on main, where the core lives in `cellacdc.dataReStruct`).
"""
from __future__ import annotations

import os

import pytest

from . import expected_restructure_changes, harness_restructure, manifest
from . import scenarios


@pytest.fixture(scope='module')
def restructure_core():
    if harness_restructure.core_module_name() is None:
        pytest.skip('restructure core needs Qt on main, where it lives '
                    'in cellacdc.dataReStruct')


@pytest.mark.parametrize('case_id', scenarios.RESTRUCTURE_IDS)
def test_restructure_golden(case_id, tmp_path, restructure_core):
    golden = scenarios.load_golden('restructure', case_id)
    actual = harness_restructure.run_restructure_case(
        case_id, str(tmp_path / 'work')
    )
    actual[manifest.ENV_KEY] = manifest.env_info()
    scenarios.assert_golden(
        case_id, golden, actual, expected_restructure_changes.CHANGES
    )


def test_restructure_registry_is_consistent():
    assert scenarios.check_registry(
        expected_restructure_changes.CHANGES, scenarios.RESTRUCTURE_IDS
    ) == []


def test_every_golden_exists_and_is_json_only():
    expected = {
        scenarios.golden_path('restructure', case_id)
        for case_id in scenarios.RESTRUCTURE_IDS
    } | {
        scenarios.golden_path('module0', scenario_id)
        for scenario_id in scenarios.MODULE0_IDS
    } | {scenarios.golden_path('cascade')}
    missing = sorted(p for p in expected if not os.path.isfile(p))
    assert missing == []
    stored = []
    for dirpath, _, filenames in os.walk(scenarios.GOLDEN_DIR):
        stored.extend(os.path.join(dirpath, name) for name in filenames)
    assert [p for p in stored if not p.endswith('.json')] == []


def test_restructure_case_ids_cover_ra_to_ro_and_combinations():
    ids = scenarios.RESTRUCTURE_IDS
    assert [i for i in ids if len(i) == 2] == [
        f'R{letter}' for letter in 'ABCDEFGHIJKLMNO'
    ]
    assert len([i for i in ids if i.startswith('combo_')]) == 27
    assert len(set(ids)) == len(ids)


# ------------------------------------------------- comparison machinery
_GOLDEN = {'dst': {'a.csv': {'kind': 'csv', 'rows': [['x', '1']]}},
           'signals': [['progress', 'a']]}
_ACTUAL = {'dst': {'a.csv': {'kind': 'csv', 'rows': [['x', '2']]}},
           'signals': [['progress', 'a']]}


def _change(**kwargs):
    fields = dict(gv='GV-T', scenarios=('S*',),
                  key_glob='dst/a.csv/rows/*', description='test')
    fields.update(kwargs)
    return scenarios.ExpectedChange(**fields)


def test_assert_golden_equal_needs_no_entry():
    assert scenarios.assert_golden('S1', _GOLDEN, _GOLDEN, []) == []


def test_assert_golden_claimed_diff_passes():
    diffs = scenarios.assert_golden('S1', _GOLDEN, _ACTUAL, [_change()])
    assert [d.key for d in diffs] == ['dst/a.csv/rows/0/1']


def test_assert_golden_unclaimed_diff_fails():
    with pytest.raises(AssertionError, match='UNCLAIMED dst/a.csv'):
        scenarios.assert_golden('S1', _GOLDEN, _ACTUAL, [])
    other_scenario = _change(scenarios=('R*',))
    with pytest.raises(AssertionError, match='UNCLAIMED'):
        scenarios.assert_golden('S1', _GOLDEN, _ACTUAL, [other_scenario])
    rejecting = _change(predicate=lambda diff: diff.new == '3')
    with pytest.raises(AssertionError, match='UNCLAIMED'):
        scenarios.assert_golden('S1', _GOLDEN, _ACTUAL, [rejecting])


def test_assert_golden_unused_entry_fails():
    with pytest.raises(AssertionError, match='UNUSED GV-T'):
        scenarios.assert_golden('S1', _GOLDEN, _GOLDEN, [_change()])


def test_assert_golden_ignores_info_and_env():
    actual = dict(_GOLDEN, _info={'x': 1})
    actual[manifest.ENV_KEY] = {'python': 'other'}
    assert scenarios.assert_golden('S1', _GOLDEN, actual, []) == []


def test_assert_golden_describes_list_changes():
    actual = dict(_GOLDEN, signals=[['progress', 'a'], ['progress', 'b']])
    with pytest.raises(AssertionError) as info:
        scenarios.assert_golden('S1', _GOLDEN, actual, [])
    assert '+["progress", "b"]' in str(info.value)


def test_check_registry_reports_unknown_scenarios():
    problems = scenarios.check_registry(
        [_change(scenarios=('nope',))], ['S1']
    )
    assert problems and 'nope' in problems[0]
