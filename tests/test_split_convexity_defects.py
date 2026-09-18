import numpy as np
import pytest

from cellacdc.core_split_IDs import (
    _convexity_defect_plane_3D,
    split_along_convexity_defects,
    split_along_convexity_defects_3D,
)


def _dumbbell_mask(shape=(40, 40)):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (
        ((yy - 20)**2 + (xx - 13)**2 <= 9**2)
        | ((yy - 20)**2 + (xx - 27)**2 <= 9**2)
    )


def test_split_along_convexity_defects_preserves_pixels_and_assigns_two_ids():
    lab = np.zeros((40, 40), dtype=np.uint32)
    lab[_dumbbell_mask()] = 7
    original_mask = lab > 0

    result, success, split_ids = split_along_convexity_defects(
        7, lab, max_ID=7
    )

    assert success
    assert split_ids == [7, 8]
    assert set(np.unique(result)) == {0, 7, 8}
    np.testing.assert_array_equal(result > 0, original_mask)


def test_split_along_convexity_defects_3d_propagates_split_through_volume():
    lab_2d = np.zeros((40, 40), dtype=np.uint32)
    lab_2d[_dumbbell_mask()] = 7
    lab = np.repeat(lab_2d[np.newaxis], 5, axis=0)
    original_mask = lab > 0

    result, success, split_ids = split_along_convexity_defects_3D(
        7, lab, max_ID=7
    )

    assert success
    assert split_ids == [7, 8]
    assert set(np.unique(result)) == {0, 7, 8}
    assert np.all(np.any(result == 7, axis=(1, 2)))
    assert np.all(np.any(result == 8, axis=(1, 2)))
    np.testing.assert_array_equal(result > 0, original_mask)


def test_split_along_convexity_defects_3d_splits_diagonal_touching_spheres():
    zz, yy, xx = np.ogrid[:60, :60, :60]
    sphere_1 = (
        (zz - 23)**2 + (yy - 25)**2 + (xx - 26)**2 <= 10**2
    )
    sphere_2 = (
        (zz - 37)**2 + (yy - 35)**2 + (xx - 34)**2 <= 10**2
    )
    lab = np.zeros((60, 60, 60), dtype=np.uint32)
    lab[sphere_1 | sphere_2] = 7

    result, success, split_ids = split_along_convexity_defects_3D(
        7, lab, max_ID=7
    )

    assert success
    assert split_ids == [7, 8]
    split_1 = result == split_ids[0]
    split_2 = result == split_ids[1]
    correctly_assigned = max(
        np.count_nonzero(split_1 & sphere_1)
        + np.count_nonzero(split_2 & sphere_2),
        np.count_nonzero(split_1 & sphere_2)
        + np.count_nonzero(split_2 & sphere_1),
    )
    assert correctly_assigned / np.count_nonzero(lab) > 0.99
    np.testing.assert_array_equal(result > 0, lab > 0)


def test_convexity_defect_plane_3d_passes_through_sphere_neck():
    zz, yy, xx = np.ogrid[:60, :60, :60]
    sphere_1 = (
        (zz - 23)**2 + (yy - 25)**2 + (xx - 26)**2 <= 10**2
    )
    sphere_2 = (
        (zz - 37)**2 + (yy - 35)**2 + (xx - 34)**2 <= 10**2
    )

    plane_origin, plane_normal = _convexity_defect_plane_3D(
        sphere_1 | sphere_2, spacing=np.ones(3)
    )
    expected_neck = np.array([30, 30, 30])

    assert abs((expected_neck - plane_origin) @ plane_normal) < 1.0


def test_split_along_convexity_defects_3d_labels_disconnected_components():
    lab = np.zeros((5, 20, 20), dtype=np.uint32)
    lab[1:4, 2:6, 2:6] = 4
    lab[1:4, 12:18, 12:18] = 4
    original_mask = lab > 0

    result, success, split_ids = split_along_convexity_defects_3D(
        4, lab, max_ID=10
    )

    assert success
    assert split_ids == [4, 11]
    assert set(np.unique(result)) == {0, 4, 11}
    assert np.count_nonzero(result == 4) > np.count_nonzero(result == 11)
    np.testing.assert_array_equal(result > 0, original_mask)


def test_split_along_convexity_defects_3d_rejects_2d_input():
    with pytest.raises(ValueError, match='Expected a 3D label image'):
        split_along_convexity_defects_3D(
            1, np.ones((5, 5), dtype=np.uint32), max_ID=1
        )