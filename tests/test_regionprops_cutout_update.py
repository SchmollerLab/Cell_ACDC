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