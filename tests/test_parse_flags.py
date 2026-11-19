"""Tests of `timepoint_predictors.parse_flags`, the shared reader of stored
True/False columns (time points, visited, custom annotations, phase markers).
"""
import numpy as np
import pandas as pd
import pytest

from cellacdc.timepoint_predictors import parse_flags


@pytest.mark.parametrize('values, expected', [
    (['True', 'False', '0', '1', ' true ', '', 'x'],
     [True, False, False, True, True, False, False]),
    ([True, False, 0, 1, np.nan, None, 'False'],
     [True, False, False, True, False, False, False]),
    (['yes', '2', '-1', '0.0', 'nan', b'True'],
     [True, True, True, False, False, True]),
    ([1., 0., np.nan], [True, False, False]),
    ([2, 0, -3], [True, False, True]),
    (pd.array([True, None, False], dtype='boolean'), [True, False, False]),
])
def test_lenient_parses_every_column(values, expected):
    result = parse_flags(pd.Series(values))
    assert result.dtype == bool
    assert result.tolist() == expected


@pytest.mark.parametrize('values, expected', [
    ([True, False, 0, 1, np.nan], [True, False, False, True, False]),
    (['True', 'false', '0', ' 1 ', np.nan], [True, False, False, True, False]),
    (['1.0', '0.0'], [True, False]),
    (pd.array([True, None, False], dtype='boolean'), [True, False, False]),
    (['True', 'G1', 'False'], None),
    ([True, 2, False], None),
    (['yes', 'False'], None),
    (['True', ''], None),
    ([0., 1., 0.], None),  # numeric measurement such as phase_contr_mean
    ([0, 1, 1], None),
])
def test_strict_accepts_only_bool_like_columns(values, expected):
    flags = parse_flags(pd.Series(values), strict=True)
    if expected is None:
        assert flags is None
    else:
        assert flags.dtype == bool and flags.tolist() == expected


@pytest.mark.parametrize('strict', [False, True])
def test_index_and_name_are_kept(strict):
    index = pd.MultiIndex.from_tuples(
        [(0, 1), (0, 2), (1, 1)], names=['frame_i', 'Cell_ID']
    )
    series = pd.Series(['True', 'False', np.nan], index=index, name='flag')
    result = parse_flags(series, strict=strict)
    assert result.index.equals(index)
    assert result.name == 'flag'
    assert result.tolist() == [True, False, False]
