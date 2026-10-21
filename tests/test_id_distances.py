import numpy as np
import pytest

from cellacdc import myutils
from cellacdc.regionprops import acdcRegionprops


@pytest.fixture(params=[None, [1, 8, 16], [16, 8, 1]])
def rps(request):
    lab = np.zeros((12, 12), dtype=np.uint16)
    lab[2, 7] = 1
    lab[2, 10] = 8
    lab[9, 1] = 16
    rp = acdcRegionprops(lab)
    if request.param is not None:
        rp.IDs = request.param
    return rp


def test_distances_follow_region_iteration_order(rps):
    distances = myutils.find_distances_ID(rps, point=(7.5, 2.5))

    np.testing.assert_allclose(distances, [[0, 3, np.sqrt(85)]])


@pytest.mark.parametrize('kwargs', [{'point': (7.5, 2.5)}, {'ID': 1}])
@pytest.mark.parametrize(
    'relevant_ids, expected',
    [
        (None, [1, 8, 16]),
        ([16, 8], [8, 16]),
        ([8], [8]),
        ([], []),
    ],
)
def test_sort_ids_nearest_first(rps, kwargs, relevant_ids, expected):
    assert myutils.sort_IDs_dist(
        rps, relevant_IDs=relevant_ids, **kwargs
    ) == expected


def test_id_distance_does_not_reverse_centroid(rps):
    np.testing.assert_allclose(
        myutils.find_distances_ID(rps, ID=1),
        myutils.find_distances_ID(rps, point=(7.5, 2.5)),
    )


def test_empty_distances_preserve_matrix_shape(rps):
    assert myutils.find_distances_ID(
        rps, point=(7.5, 2.5), relevant_IDs=[]
    ).shape == (1, 0)


@pytest.mark.parametrize('helper', [myutils.find_distances_ID, myutils.sort_IDs_dist])
@pytest.mark.parametrize('relevant_ids', [None, [], [8]])
@pytest.mark.parametrize('kwargs', [{}, {'point': (7, 2), 'ID': 1}, {'ID': 99}])
def test_invalid_distance_inputs_raise(rps, helper, relevant_ids, kwargs):
    with pytest.raises(ValueError):
        helper(rps, relevant_IDs=relevant_ids, **kwargs)


def test_3d_id_and_point_distances_agree():
    lab = np.zeros((5, 12, 12), dtype=np.uint16)
    lab[1, 2, 7] = 1
    lab[3, 2, 7] = 8
    rps = acdcRegionprops(lab)

    np.testing.assert_allclose(
        myutils.find_distances_ID(rps, ID=1),
        myutils.find_distances_ID(rps, point=(7.5, 2.5, 1.5)),
    )
