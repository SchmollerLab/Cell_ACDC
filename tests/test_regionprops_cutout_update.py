import numpy as np

from cellacdc.regionprops import acdcRegionprops


def test_update_regionprops_via_cutout_reuses_cutout_object_with_global_coords():
    old_lab = np.zeros((12, 12), dtype=np.uint16)
    old_lab[1:3, 1:3] = 1

    new_lab = old_lab.copy()
    new_lab[5:7, 6:9] = 2

    rp = acdcRegionprops(old_lab)
    rp.update_regionprops_via_cutout(new_lab, cutout_bbox=(4, 5, 9, 10))

    obj = rp.get_obj_from_ID(2)

    assert obj is not None
    assert obj.bbox == (5, 6, 7, 9)
    assert tuple((slc.start, slc.stop) for slc in obj.slice) == ((5, 7), (6, 9))
    assert obj.centroid == (5.5, 7.0)
    np.testing.assert_array_equal(
        obj.coords,
        np.array(
            [
                [5, 6],
                [5, 7],
                [5, 8],
                [6, 6],
                [6, 7],
                [6, 8],
            ]
        ),
    )
    assert new_lab[obj.slice].shape == obj.image.shape
    assert np.all(new_lab[obj.slice][obj.image] == 2)


def test_update_regionprops_via_cutout_refreshes_preserved_id_image():
    old_lab = np.zeros((10, 10), dtype=np.uint16)
    old_lab[2:4, 2:4] = 1

    new_lab = old_lab.copy()
    new_lab[2:5, 2:5] = 1

    rp = acdcRegionprops(old_lab)
    rp.update_regionprops_via_cutout(new_lab, cutout_bbox=(1, 1, 6, 6))

    obj = rp.get_obj_from_ID(1)

    assert obj is not None
    assert obj.bbox == (2, 2, 5, 5)
    np.testing.assert_array_equal(obj.image, new_lab[obj.slice] == 1)
    np.testing.assert_array_equal(obj.coords, np.argwhere(new_lab == 1))


def test_update_regionprops_via_cutout_batches_border_touching_ids():
    old_lab = np.zeros((14, 14), dtype=np.uint16)
    old_lab[1:3, 1:3] = 1

    new_lab = old_lab.copy()
    new_lab[4:9, 5:8] = 2
    new_lab[7:11, 9:13] = 3

    rp = acdcRegionprops(old_lab)
    rp.update_regionprops_via_cutout(new_lab, cutout_bbox=(6, 7, 10, 12))

    obj2 = rp.get_obj_from_ID(2)
    obj3 = rp.get_obj_from_ID(3)

    assert obj2 is not None
    assert obj2.bbox == (4, 5, 9, 8)
    np.testing.assert_array_equal(obj2.coords, np.argwhere(new_lab == 2))

    assert obj3 is not None
    assert obj3.bbox == (7, 9, 11, 13)
    np.testing.assert_array_equal(obj3.coords, np.argwhere(new_lab == 3))


def test_update_regionprops_via_assignments_rebinds_label_image():
    old_lab = np.zeros((8, 8), dtype=np.uint16)
    old_lab[2:5, 3:6] = 1

    new_lab = np.zeros_like(old_lab)
    new_lab[2:5, 3:6] = 7

    rp = acdcRegionprops(old_lab)
    rp.update_regionprops_via_assignments({1: 7}, new_lab)

    obj = rp.get_obj_from_ID(7)

    assert obj is not None
    assert rp.lab is new_lab
    assert obj._label_image is new_lab
    np.testing.assert_array_equal(obj.image, new_lab[obj.slice] == 7)
    np.testing.assert_array_equal(obj.coords, np.argwhere(new_lab == 7))


def test_slice_regionprops_are_lazy_and_initialized_on_access():
    lab = np.zeros((3, 6, 6), dtype=np.uint16)
    lab[1, 2:4, 1:3] = 4

    rp = acdcRegionprops(lab)

    assert rp._slice_rps['z'] == {}
    assert rp._slice_rps['y'] == {}
    assert rp._slice_rps['x'] == {}

    z1 = rp.get_slice_rp(1)
    assert z1 is not None
    assert 1 in rp._slice_rps['z']
    assert rp._slice_rps['z'][1] is z1
    assert z1.lab.ndim == 2
    assert z1.get_obj_from_ID(4) is not None


def test_projection_regionprops_are_lazy_and_initialized_on_access():
    lab = np.zeros((3, 6, 6), dtype=np.uint16)
    lab[1, 2:4, 1:3] = 4

    rp = acdcRegionprops(lab)

    assert rp._proj_rps['z'] == {}
    assert rp._proj_rps['y'] == {}
    assert rp._proj_rps['x'] == {}

    zmax = rp.get_proj_rp(kind='max', slicing='z')
    assert zmax is not None
    assert 'max' in rp._proj_rps['z']
    assert rp._proj_rps['z']['max'] is zmax
    assert zmax.lab.ndim == 2
    assert zmax.get_obj_from_ID(4) is not None


def test_projection_regionprops_support_most_common_kind():
    lab = np.array(
        [
            [[0, 1], [2, 2]],
            [[0, 1], [2, 3]],
            [[4, 1], [0, 3]],
        ],
        dtype=np.uint16,
    )

    rp = acdcRegionprops(lab)
    most_common = rp.get_proj_rp(kind='most common', slicing='z')

    expected = np.array(
        [[0, 1], [2, 3]],
        dtype=np.uint16,
    )
    np.testing.assert_array_equal(most_common.lab, expected)
    assert rp.get_obj_from_proj_rp(3, kind='most common z-projection', warn=False) is not None


def test_most_common_projection_uses_local_cutout_update(monkeypatch):
    old_lab = np.zeros((3, 6, 6), dtype=np.uint16)
    old_lab[:, 1:4, 1:4] = 1

    rp = acdcRegionprops(old_lab)
    proj_before = rp.get_proj_rp(kind='most_common', slicing='z')
    expected_before = rp._get_lab_projection(old_lab, slicing='z', kind='most_common')
    np.testing.assert_array_equal(proj_before.lab, expected_before)

    new_lab = old_lab.copy()
    new_lab[0:2, 2:5, 2:5] = 2

    original_replace_cached = rp._replace_cached_lab_projection

    def _replace_cached_should_not_run_for_most_common(slicing, kind):
        if kind == 'most_common':
            raise AssertionError(
                'most_common projection should be updated locally for cutout updates.'
            )
        return original_replace_cached(slicing, kind)

    monkeypatch.setattr(rp, '_replace_cached_lab_projection', _replace_cached_should_not_run_for_most_common)

    rp.update_regionprops_via_cutout(new_lab, cutout_bbox=(2, 2, 5, 5))

    proj_after = rp.get_proj_rp(kind='most_common', slicing='z')
    expected_after = rp._get_lab_projection(new_lab, slicing='z', kind='most_common')
    np.testing.assert_array_equal(proj_after.lab, expected_after)


def test_get_obj_from_id_for_stored_slice_and_projection_rps():
    lab = np.zeros((4, 8, 8), dtype=np.uint16)
    lab[1:3, 2:5, 3:6] = 2

    rp = acdcRegionprops(lab)

    obj_slice = rp.get_obj_from_slice_rp(2, slice_number=1, slicing='z', warn=False)
    assert obj_slice is not None

    obj_proj = rp.get_obj_from_proj_rp(2, kind='max', slicing='z', warn=False)
    assert obj_proj is not None


def test_slice_regionprops_follow_assignments_and_deletions():
    lab = np.zeros((4, 8, 8), dtype=np.uint16)
    lab[1:3, 2:5, 3:6] = 2

    rp = acdcRegionprops(lab)
    _ = rp.get_slice_rp(1, 'z')
    _ = rp.get_slice_rp(2, 'y')
    _ = rp.get_slice_rp(3, 'x')
    _ = rp.get_proj_rp('max', 'z')
    _ = rp.get_proj_rp('mean', 'y')
    _ = rp.get_proj_rp('median', 'x')

    remapped_lab = np.zeros_like(lab)
    remapped_lab[1:3, 2:5, 3:6] = 9
    rp.update_regionprops_via_assignments({2: 9}, remapped_lab)

    assert rp.get_slice_rp(1, 'z').get_obj_from_ID(9, warn=False) is not None
    assert rp.get_slice_rp(2, 'y').get_obj_from_ID(9, warn=False) is not None
    assert rp.get_slice_rp(3, 'x').get_obj_from_ID(9, warn=False) is not None
    assert rp.get_proj_rp('max', 'z').get_obj_from_ID(9, warn=False) is not None
    assert rp.get_slice_rp(1, 'z').get_obj_from_ID(2, warn=False) is None

    rp.update_regionprops_via_deletions({9})
    assert rp.get_slice_rp(1, 'z').get_obj_from_ID(9, warn=False) is None
    assert rp.get_slice_rp(2, 'y').get_obj_from_ID(9, warn=False) is None
    assert rp.get_slice_rp(3, 'x').get_obj_from_ID(9, warn=False) is None
    assert rp.get_proj_rp('max', 'z').get_obj_from_ID(9, warn=False) is None


def test_slice_regionprops_update_from_2d_cutout_on_3d():
    old_lab = np.zeros((5, 12, 12), dtype=np.uint16)
    old_lab[1:4, 2:4, 2:4] = 5

    new_lab = old_lab.copy()
    new_lab[3, 6:9, 7:10] = 6

    rp = acdcRegionprops(old_lab)
    _ = rp.get_slice_rp(3, 'z')
    _ = rp.get_slice_rp(6, 'y')
    _ = rp.get_slice_rp(7, 'x')

    # 2D cutout on y/x; implementation expands to all z for touched IDs.
    rp.update_regionprops_via_cutout(new_lab, cutout_bbox=(5, 6, 10, 11))

    assert rp.get_obj_from_ID(6, warn=False) is not None
    assert rp.get_slice_rp(3, 'z').get_obj_from_ID(6, warn=False) is not None
    assert rp.get_slice_rp(6, 'y').get_obj_from_ID(6, warn=False) is not None
    assert rp.get_slice_rp(7, 'x').get_obj_from_ID(6, warn=False) is not None