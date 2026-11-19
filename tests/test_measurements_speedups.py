"""Tests for the helpers used to speed up the measurements in
`cellacdc.measurements`.

The helpers must behave exactly like the plain pandas/numpy operations they
replace (`df.to_dict('list')`, `df.at`, one `np.quantile` call per quantile,
etc.), hence every test compares a helper with the reference operation on
randomized inputs.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
import skimage.measure

from cellacdc import core, measurements

QS = [0.25, 0.75, 0.05, 0.95]


def _same_float(a, b):
    # Exact comparison of two floats, including the sign of zero (NaN
    # matches only NaN). Raw bytes cannot be compared because extended
    # precision long doubles (e.g., 80-bit on Linux x86-64) are stored with
    # uninitialized padding bytes.
    a_nan, b_nan = bool(np.isnan(a)), bool(np.isnan(b))
    if a_nan or b_nan:
        return a_nan and b_nan
    return bool(a == b) and bool(np.signbit(a)) == bool(np.signbit(b))


def _assert_same_value(a, b):
    assert type(a) is type(b), (a, b)
    if a is None or b is None:
        assert a is b
        return
    if isinstance(a, (complex, np.complexfloating)):
        assert _same_float(a.real, b.real), (a, b)
        assert _same_float(a.imag, b.imag), (a, b)
        return
    if isinstance(a, (float, np.floating)):
        assert _same_float(a, b), (a, b)
        return
    try:
        both_nan = bool(np.isnan(a)) and bool(np.isnan(b))
    except TypeError:
        both_nan = False
    if both_nan:
        return
    assert np.array(a).tobytes() == np.array(b).tobytes(), (a, b)


def _assert_same_dict_of_lists(result, expected):
    assert list(result.keys()) == list(expected.keys())
    for key in expected:
        assert len(result[key]) == len(expected[key]), key
        for a, b in zip(result[key], expected[key]):
            _assert_same_value(a, b)


def _assert_same_df(result, expected):
    pd.testing.assert_frame_equal(result, expected, check_exact=True)
    for col in expected.columns:
        for a, b in zip(result[col].to_numpy(), expected[col].to_numpy()):
            _assert_same_value(a, b)


def _get_error(func, *args):
    # Older pandas versions raise instead of upcasting for some values
    # (e.g., bool into a float column) --> compare the errors too
    try:
        func(*args)
    except Exception as err:
        return type(err), str(err)
    return None


def _random_value(rng):
    choices = (
        lambda: float(rng.normal(10, 5)),
        lambda: np.float64(rng.normal(10, 5)),
        lambda: np.float32(rng.normal(10, 5)),
        lambda: np.float16(rng.normal(10, 5)),
        lambda: int(rng.integers(-1000, 1000)),
        lambda: 2**60 + 1,
        lambda: np.int64(rng.integers(-1000, 1000)),
        lambda: np.uint64(rng.integers(0, 1000)),
        lambda: np.uint16(rng.integers(0, 1000)),
        lambda: np.nan,
        lambda: None,
        lambda: True,
        lambda: np.bool_(False),
        lambda: 'text',
        lambda: np.longdouble(1.5),
        lambda: complex(1, 2),
        lambda: np.complex64(1 + 2j),
    )
    return choices[rng.integers(0, len(choices))]()


@pytest.mark.parametrize(
    'dtype', [np.uint8, np.uint16, np.uint32, np.uint64, np.float64]
)
def test_multi_quantile_matches_single_quantiles(dtype):
    rng = np.random.default_rng(0)
    for size in (1, 2, 3, 4, 7, 10, 101, 1000):
        for _ in range(10):
            if np.issubdtype(dtype, np.integer):
                arr = rng.integers(0, 250, size=size).astype(dtype)
            else:
                arr = rng.normal(100, 30, size=size).astype(dtype)
                if rng.random() < 0.2:
                    arr[rng.integers(0, size)] = 0.0
            for arr_input in (arr, list(arr)):
                multi = measurements._multi_quantile(arr_input, QS)
                for q in QS:
                    _assert_same_value(multi[q], np.quantile(arr_input, q=q))

    # List of rows like the dataPrep background data
    arr_rows = list(rng.integers(0, 1000, size=(13, 7)).astype(np.uint16))
    multi = measurements._multi_quantile(arr_rows, QS)
    for q in QS:
        _assert_same_value(multi[q], np.quantile(arr_rows, q=q))


def test_multi_quantile_falls_back_when_not_equivalent():
    # float32 results would be promoted to float64 --> not equivalent
    assert measurements._multi_quantile(np.ones(5, np.float32), QS) is None
    # np.quantile raises on empty input
    assert measurements._multi_quantile([], QS) is None
    # 0-d input (e.g., default value of missing background data)
    assert measurements._multi_quantile(0, QS) is None
    assert measurements._multi_quantile(np.arange(5), [0.5]) is None
    # Signed integers (overflows behave differently with scalars)
    assert measurements._multi_quantile(np.arange(5, dtype=np.int16), QS) is None
    # -0.0 and 0.0 are ties in the partition
    signed_zeros = np.array([-0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0])
    assert measurements._multi_quantile(signed_zeros, QS) is None
    # NaNs
    assert measurements._multi_quantile(np.array([1.0, np.nan, 2.0]), QS) is None
    # Positive zeros only are fine
    assert measurements._multi_quantile(np.zeros(5), QS) is not None


def test_float64_settable_checks():
    for value in (np.timedelta64(1, 's'), True, np.bool_(True), 'a', 1j):
        assert not measurements._is_float64_settable(value)
        assert not measurements._creates_float64_column(value)
    for value in (np.float32(1.5), np.float16(1.5)):
        assert measurements._is_float64_settable(value)
        # float32/float16 values create float32/float16 columns
        assert not measurements._creates_float64_column(value)
    for value in (1.5, np.float64(1.5), 3, np.int64(3), np.uint16(3)):
        assert measurements._is_float64_settable(value)
        assert measurements._creates_float64_column(value)
    assert not measurements._is_single_value_update_dtype(np.dtype('g'))
    assert not measurements._is_single_value_update_dtype(np.dtype(complex))
    assert not measurements._is_single_value_update_dtype(np.dtype(object))
    assert measurements._is_single_value_update_dtype(np.dtype(np.float32))


def test_standard_quantile_funcs_unchanged():
    metrics_func, _ = measurements.standard_metrics_func()
    arr = np.arange(17, dtype=np.uint16)
    for name, q in (('q25', 0.25), ('q75', 0.75), ('q05', 0.05), ('q95', 0.95)):
        func = metrics_func[name]
        assert measurements._get_func_quantile_q(func) == q
        _assert_same_value(func(arr), np.quantile(arr, q=q))
        assert np.isnan(func([]))


@pytest.mark.parametrize('seed', range(40))
def test_set_column_values_at_matches_df_at(seed):
    rng = np.random.default_rng(seed)
    IDs = [3, 5, 7, 11, 13]
    base = pd.DataFrame(
        {
            'existing_float': rng.normal(size=5),
            'existing_int': np.arange(5),
            'existing_obj': ['a']*5,
        },
        index=pd.Index(IDs, name='Cell_ID')
    )
    for col in ('existing_float', 'existing_int', 'existing_obj', 'new_col'):
        n_values = int(rng.integers(1, 7))
        if rng.random() < 0.5:
            # Only "standard" metric values
            values = [
                rng.choice([np.float64(rng.normal()), np.nan, None])
                for _ in range(n_values)
            ]
        else:
            values = [_random_value(rng) for _ in range(n_values)]
        IDs_to_set = [int(ID) for ID in rng.choice(IDs, size=n_values)]
        rows = [IDs.index(ID) for ID in IDs_to_set]

        expected = base.copy()
        result = base.copy()

        def set_values_at():
            for ID, value in zip(IDs_to_set, values):
                expected.at[ID, col] = value

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            expected_error = _get_error(set_values_at)
            error = _get_error(
                measurements._set_column_values_at,
                result, col, IDs_to_set, rows, values
            )
        assert error == expected_error
        _assert_same_df(result, expected)


@pytest.mark.parametrize('seed', range(30))
def test_metrics_values_cache_matches_to_dict(seed):
    rng = np.random.default_rng(seed)
    IDs = [2, 4, 6, 8, 10, 12]
    data = {
        'cell_area_pxl': rng.integers(10, 100, size=6).astype(float),
        'cell_vol_fl': rng.normal(50, 5, size=6),
        'ch_mean': rng.normal(size=6),
        'ch_int': np.arange(6),
        'ch_obj': ['G1']*6,
    }
    df = pd.DataFrame(data, index=pd.Index(IDs, name='Cell_ID'), copy=True)
    # Reference modified only with `df.at` --> the cache must not modify
    # `df` in any other way (e.g., consolidating it in place)
    df_ref = pd.DataFrame(data, index=pd.Index(IDs, name='Cell_ID'), copy=True)
    columns = [*df.columns, 'ch_CV', 'ch_CV0', 'cell_vol_vox', 'new_obj']
    cache = measurements._MetricsValuesCache(df)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for step in range(60):
            _assert_same_df(df, df_ref)
            metrics_values, volumes_areas = cache.get()
            _assert_same_dict_of_lists(metrics_values, df.to_dict('list'))
            for result, expected in zip(
                    volumes_areas, measurements.get_cell_volumes_areas(df)
                ):
                assert len(result) == len(expected)
                for a, b in zip(result, expected):
                    _assert_same_value(a, b)

            # Fast `df.at` reader must match `df.at`
            for ID in rng.choice(df.index, size=3):
                for col in rng.choice(df.columns, size=3):
                    _assert_same_value(
                        cache.df_reader.at[ID, col], df.at[ID, col]
                    )

            if rng.random() < 0.05:
                # New row
                ID = int(df.index.max()) + 1
            else:
                ID = int(rng.choice(df.index))
            col = str(rng.choice(columns))
            value = _random_value(rng)

            def set_value_at():
                df_ref.at[ID, col] = value

            expected_error = _get_error(set_value_at)
            error = _get_error(cache.set_value, ID, col, value)
            assert error == expected_error
            if error is not None:
                # The kernel never uses the cache after `df.at` raised
                cache = measurements._MetricsValuesCache(df)


def _make_df_with_longdouble_col():
    df = pd.DataFrame(
        {
            'cell_vol_vox': [10.0, 20.0, 30.0],
            'ch_mean': [0.5, 1.5, 2.5],
            'ch_int': [1, 2, 3]
        },
        index=pd.Index([1, 2, 3], name='Cell_ID')
    )
    # Assigned after creation to get a separate longdouble block
    df['ch_autoBkgr_bkgrVal_median'] = np.longdouble(2.5)
    return df


def test_metrics_values_cache_with_longdouble_column():
    # Where longdouble is 64-bit (e.g., Windows), pandas < 2 can merge
    # longdouble and float64 columns in place (e.g., in `df[[col]]` or
    # when `df.at` upcasts a column), changing the dtype of other columns
    df = _make_df_with_longdouble_col()
    df_ref = _make_df_with_longdouble_col()
    cache = measurements._MetricsValuesCache(df)
    writes = [
        (1, 'ch_new', 1.5),
        (2, 'ch_text', 'text'),
        (3, 'ch_new', np.float32(2.5)),
        (2, 'cell_vol_vox', 21.0),
        (3, 'ch_new_ld', np.longdouble(3.5)),
        (1, 'ch_mean', 0.25),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for ID, col, value in writes:
            cache.set_value(ID, col, value)
            df_ref.at[ID, col] = value
            _assert_same_df(df, df_ref)
            metrics_values, volumes_areas = cache.get()
            _assert_same_dict_of_lists(metrics_values, df_ref.to_dict('list'))
            expected_volumes_areas = measurements.get_cell_volumes_areas(df_ref)
            for result, expected in zip(volumes_areas, expected_volumes_areas):
                for a, b in zip(result, expected):
                    _assert_same_value(a, b)
            for ID_ in df_ref.index:
                for col_ in df_ref.columns:
                    _assert_same_value(
                        cache.df_reader.at[ID_, col_], df_ref.at[ID_, col_]
                    )


def test_float64_dtype_checks():
    assert measurements._is_float64_dtype(np.dtype(np.float64))
    # On Windows np.dtype('g') == np.float64 is True
    assert not measurements._is_float64_dtype(np.dtype('g'))
    assert not measurements._is_float64_dtype(np.dtype('>f8'))
    assert not measurements._is_float64_dtype(np.dtype(np.float32))
    assert not measurements._is_float64_dtype(pd.Int64Dtype())
    assert measurements._has_longdouble_dtype([np.dtype('g')])
    assert measurements._has_longdouble_dtype([np.dtype(int), np.dtype('G')])
    assert not measurements._has_longdouble_dtype(
        [np.dtype(float), np.dtype(complex), pd.Int64Dtype()]
    )
    assert measurements._multi_quantile(np.ones(5, np.longdouble), QS) is None


def test_metrics_values_cache_non_unique_index_falls_back():
    df = pd.DataFrame({'a': [1.0, 2.0, 3.0]}, index=[1, 1, 2])
    cache = measurements._MetricsValuesCache(df)
    _assert_same_dict_of_lists(cache.get()[0], df.to_dict('list'))
    cache.set_value(2, 'a', 5.0)
    _assert_same_dict_of_lists(cache.get()[0], df.to_dict('list'))


def test_batched_at_writer_matches_df_at():
    rng = np.random.default_rng(0)
    IDs = [1, 2, 3, 4]
    base = pd.DataFrame(
        {'a': np.zeros(4), 'b': np.zeros(4)}, index=pd.Index(IDs)
    )
    assignments = [
        (int(rng.choice(IDs)), str(rng.choice(['a', 'b', 'c', 'd'])), value)
        for value in rng.normal(size=30)
    ]
    assignments.append((2, 'e', np.float32(1.5)))
    assignments.append((3, 'e', 2.0))
    expected = base.copy()
    for ID, col, value in assignments:
        expected.at[ID, col] = value

    result = base.copy()
    writer = measurements._BatchedAtWriter.create(result, IDs)
    for ID, col, value in assignments:
        writer.set(ID, col, value)
    writer.flush()
    _assert_same_df(result, expected)

    # Writer cannot be used when assigning would add rows
    assert measurements._BatchedAtWriter.create(base, [1, 99]) is None


def _reference_regionprops_table(labels, props):
    # Original implementation: additional regionprops computed per prop
    rp = skimage.measure.regionprops(labels)
    if 'label' not in props:
        props = ('label', *props)
    empty_metric = [None]*len(rp)
    rp_table = {}
    error_ids = {}
    for prop in props:
        for o, obj in enumerate(rp):
            try:
                obj = measurements.calc_additional_regionprops(obj)
                metric = getattr(obj, prop)
                _type = measurements.PROPS_DTYPES[prop]
                if _type == int or _type == float:
                    if prop not in rp_table:
                        rp_table[prop] = empty_metric.copy()
                    rp_table[prop][o] = metric
                elif _type == tuple:
                    for m, val in enumerate(metric):
                        prop_1d = f'{prop}-{m}'
                        if prop_1d not in rp_table:
                            rp_table[prop_1d] = empty_metric.copy()
                        rp_table[prop_1d][o] = val
                elif _type == np.ndarray:
                    for i, val in enumerate(metric.flatten()):
                        indices = np.unravel_index(i, metric.shape)
                        s = '-'.join([str(idx) for idx in indices])
                        prop_1d = f'{prop}-{s}'
                        if prop_1d not in rp_table:
                            rp_table[prop_1d] = empty_metric.copy()
                        rp_table[prop_1d][o] = val
            except Exception as e:
                if prop not in error_ids:
                    error_ids[prop] = {'ids': [obj.label], 'error': e}
                else:
                    error_ids[prop]['ids'].append(obj.label)
    return rp_table, error_ids


@pytest.mark.parametrize('is3D', [False, True])
def test_regionprops_table_matches_reference(is3D):
    if is3D:
        lab = np.zeros((6, 30, 30), np.uint16)
        lab[1:5, 3:12, 3:12] = 1
        lab[2, 15, 15:20] = 2  # line-shaped slice
        lab[1:4, 20:26, 20:27] = 3
        lab[3, 25, 3] = 4  # single voxel
        props = measurements.get_props_names_3D()
    else:
        lab = np.zeros((30, 30), np.uint16)
        lab[3:12, 3:12] = 1
        lab[15, 15:20] = 2  # line
        lab[20:26, 20:27] = 3
        lab[25, 3] = 4  # single pixel
        props = measurements.get_props_names()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result, result_errors = measurements.regionprops_table(
            lab, props, logger_func=lambda *args: None
        )
        expected, expected_errors = _reference_regionprops_table(lab, props)

    assert list(result.keys()) == list(expected.keys())
    for key in expected:
        for a, b in zip(result[key], expected[key]):
            _assert_same_value(a, b)

    assert list(result_errors.keys()) == list(expected_errors.keys())
    for prop, error_info in expected_errors.items():
        assert result_errors[prop]['ids'] == error_info['ids']
        assert type(result_errors[prop]['error']) is type(error_info['error'])
        assert str(result_errors[prop]['error']) == str(error_info['error'])


def _reference_twoframes_velocity(prev_lab, lab, spacing=None):
    prev_rp = skimage.measure.regionprops(prev_lab)
    rp = skimage.measure.regionprops(lab)
    prev_IDs = [obj.label for obj in prev_rp]
    velocities_pxl = [0]*len(rp)
    velocities_um = [0]*len(rp)
    for i, obj in enumerate(rp):
        if obj.label not in prev_IDs:
            continue
        prev_obj = prev_rp[prev_IDs.index(obj.label)]
        diff = np.subtract(obj.centroid, prev_obj.centroid)
        velocities_pxl[i] = np.linalg.norm(diff)
        if spacing is not None:
            velocities_um[i] = np.linalg.norm(diff*spacing)
    return velocities_pxl, velocities_um


def test_compute_twoframes_velocity_matches_reference():
    rng = np.random.default_rng(1)
    prev_lab = rng.integers(0, 12, size=(40, 40)).astype(np.uint16)
    lab = rng.integers(0, 15, size=(40, 40)).astype(np.uint16)
    lab[lab == 3] = 0
    for spacing in (None, np.array([0.5, 0.25])):
        result = core.compute_twoframes_velocity(prev_lab, lab, spacing)
        expected = _reference_twoframes_velocity(prev_lab, lab, spacing)
        for res_list, exp_list in zip(result, expected):
            assert len(res_list) == len(exp_list)
            for a, b in zip(res_list, exp_list):
                _assert_same_value(a, b)
