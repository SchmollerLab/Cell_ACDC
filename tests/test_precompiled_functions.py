"""Tests for Cython functions in cellacdc/precompiled_functions.pyx.

Each Cython function is validated against a pure-Python / skimage reference
implementation on realistic synthetic label images built from filled discs
(2-D) and filled spheres (3-D).

Run with:
    pytest tests/test_precompiled_functions.py -v
"""

import pytest
import numpy as np
from skimage.draw import disk, ellipsoid
from skimage.measure import regionprops

# ---------------------------------------------------------------------------
# Skip the whole module when the Cython extension is not compiled yet
# ---------------------------------------------------------------------------
pytest.importorskip(
    "cellacdc.precompiled.precompiled_functions",
    reason="Cython extension not compiled; run: python precompile_functions.py build_ext --inplace",
)
from cellacdc.precompiled.precompiled_functions import (
    find_all_objects_2D,
    find_all_objects_3D,
    most_common_projection_3D,
    calc_IoA_matrix_2D,
    calc_IoA_matrix_3D,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_disc_label_image(shape, specs):
    """Build a 2-D uint32 label image from a list of (label, cy, cx, radius)."""
    img = np.zeros(shape, dtype=np.uint32)
    for label, cy, cx, r in specs:
        rr, cc = disk((cy, cx), r, shape=shape)
        img[rr, cc] = label
    return img


def _make_sphere_label_image(shape, specs):
    """Build a 3-D uint32 label image from (label, cz, cy, cx, rz, ry, rx)."""
    img = np.zeros(shape, dtype=np.uint32)
    for label, cz, cy, cx, rz, ry, rx in specs:
        sph = ellipsoid(rz, ry, rx)
        sz, sy, sx = sph.shape
        z0 = cz - sz // 2
        y0 = cy - sy // 2
        x0 = cx - sx // 2
        for dz in range(sz):
            for dy in range(sy):
                for dx in range(sx):
                    if not sph[dz, dy, dx]:
                        continue
                    zi, yi, xi = z0 + dz, y0 + dy, x0 + dx
                    if 0 <= zi < shape[0] and 0 <= yi < shape[1] and 0 <= xi < shape[2]:
                        img[zi, yi, xi] = label
    return img


def _reference_bboxes_2D(label_img):
    """skimage regionprops bounding boxes as {label: (r0, r1, c0, c1)}."""
    result = {}
    for obj in regionprops(label_img.astype(np.int32)):
        r0, c0, r1, c1 = obj.bbox
        result[obj.label] = (r0, r1, c0, c1)
    return result


def _reference_bboxes_3D(label_img):
    """skimage regionprops bounding boxes as {label: (z0, z1, r0, r1, c0, c1)}."""
    result = {}
    for obj in regionprops(label_img.astype(np.int32)):
        z0, r0, c0, z1, r1, c1 = obj.bbox
        result[obj.label] = (z0, z1, r0, r1, c0, c1)
    return result


def _python_ioa_matrix(lab, prev_lab, rp, prev_rp, use_union):
    """Pure-Python IoA matrix (the original fallback in CellACDC_tracker)."""
    IDs_curr = [obj.label for obj in rp]
    IDs_prev = [obj.label for obj in prev_rp]
    IoA = np.zeros((len(IDs_curr), len(IDs_prev)), dtype=np.float64)
    rp_mapper = {obj.label: obj for obj in rp}
    idx_curr = {ID: i for i, ID in enumerate(IDs_curr)}
    for j, obj_prev in enumerate(prev_rp):
        if use_union:
            pass  # denom computed per overlap
        else:
            denom_val = obj_prev.area
        intersect_IDs, intersects = np.unique(
            lab[obj_prev.slice][obj_prev.image], return_counts=True
        )
        for intersect_ID, I in zip(intersect_IDs, intersects):
            if intersect_ID == 0 or I == 0:
                continue
            if use_union:
                if intersect_ID not in rp_mapper:
                    continue
                obj_curr = rp_mapper[intersect_ID]
                denom_val = obj_prev.area + obj_curr.area - I
                if denom_val == 0:
                    continue
            idx = idx_curr.get(intersect_ID)
            if idx is None:
                continue
            IoA[idx, j] = I / denom_val
    return IoA, IDs_curr, IDs_prev


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DISC_SPECS = [
    # (label, cy, cx, radius)
    (1,  30,  30, 20),   # large disc, centre-left
    (2,  30,  90, 15),   # medium disc, centre-right
    (3,  80,  60, 10),   # smaller disc, bottom
    (4,  80, 110, 12),   # slightly overlapping with disc 3 at boundary
]

DISC_SPECS_SHIFTED = [
    # same cells shifted a few pixels to simulate motion
    (1,  32,  32, 20),
    (2,  28,  92, 15),
    (3,  82,  62, 10),
    (4,  78, 112, 12),
]

SPHERE_SPECS = [
    # (label, cz, cy, cx, rz, ry, rx)
    (1, 10, 20, 20, 5, 8, 8),
    (2, 10, 20, 50, 4, 6, 6),
    (3, 10, 45, 35, 3, 5, 5),
]

SPHERE_SPECS_SHIFTED = [
    (1, 11, 21, 21, 5, 8, 8),
    (2,  9, 19, 52, 4, 6, 6),
    (3, 10, 46, 34, 3, 5, 5),
]


@pytest.fixture(scope="module")
def label_2d():
    return _make_disc_label_image((128, 128), DISC_SPECS)


@pytest.fixture(scope="module")
def label_2d_shifted():
    return _make_disc_label_image((128, 128), DISC_SPECS_SHIFTED)


@pytest.fixture(scope="module")
def label_3d():
    return _make_sphere_label_image((20, 64, 64), SPHERE_SPECS)


@pytest.fixture(scope="module")
def label_3d_shifted():
    return _make_sphere_label_image((20, 64, 64), SPHERE_SPECS_SHIFTED)


# ---------------------------------------------------------------------------
# find_all_objects_2D
# ---------------------------------------------------------------------------

class TestFindAllObjects2D:
    def test_labels_match(self, label_2d):
        labels, _ = find_all_objects_2D(label_2d)
        assert set(labels.tolist()) == {1, 2, 3, 4}

    def test_bbox_matches_skimage(self, label_2d):
        labels, bboxes = find_all_objects_2D(label_2d)
        ref = _reference_bboxes_2D(label_2d)
        for lbl, bbox in zip(labels.tolist(), bboxes.tolist()):
            r0, r1, c0, c1 = bbox
            assert (r0, r1, c0, c1) == ref[lbl], (
                f"Label {lbl}: got {(r0,r1,c0,c1)}, expected {ref[lbl]}"
            )

    def test_empty_image_returns_empty(self):
        empty = np.zeros((64, 64), dtype=np.uint32)
        result = find_all_objects_2D(empty)
        assert result == ([], [])

    def test_single_pixel_object(self):
        img = np.zeros((10, 10), dtype=np.uint32)
        img[5, 7] = 1
        labels, bboxes = find_all_objects_2D(img)
        assert labels[0] == 1
        assert list(bboxes[0]) == [5, 6, 7, 8]

    def test_label_above_300_triggers_growth(self):
        """Label > 300 must still produce the correct bounding box."""
        img = np.zeros((64, 64), dtype=np.uint32)
        rr, cc = disk((32, 32), 10, shape=img.shape)
        img[rr, cc] = 350
        labels, bboxes = find_all_objects_2D(img)
        ref = _reference_bboxes_2D(img)
        r0, r1, c0, c1 = bboxes[0].tolist()
        assert (r0, r1, c0, c1) == ref[350]

    def test_label_above_600_triggers_second_growth(self):
        img = np.zeros((64, 64), dtype=np.uint32)
        rr, cc = disk((32, 32), 10, shape=img.shape)
        img[rr, cc] = 650
        labels, bboxes = find_all_objects_2D(img)
        ref = _reference_bboxes_2D(img)
        r0, r1, c0, c1 = bboxes[0].tolist()
        assert (r0, r1, c0, c1) == ref[650]

    def test_multiple_labels_across_300_boundary(self):
        img = np.zeros((64, 64), dtype=np.uint32)
        rr1, cc1 = disk((20, 20), 8, shape=img.shape)
        rr2, cc2 = disk((20, 50), 8, shape=img.shape)
        img[rr1, cc1] = 1
        img[rr2, cc2] = 301
        labels, bboxes = find_all_objects_2D(img)
        ref = _reference_bboxes_2D(img)
        for lbl, bbox in zip(labels.tolist(), bboxes.tolist()):
            assert tuple(bbox) == ref[lbl]

    def test_bbox_dtype_is_uint32(self, label_2d):
        _, bboxes = find_all_objects_2D(label_2d)
        assert bboxes.dtype == np.uint32


# ---------------------------------------------------------------------------
# find_all_objects_3D
# ---------------------------------------------------------------------------

class TestFindAllObjects3D:
    def test_labels_match(self, label_3d):
        labels, _ = find_all_objects_3D(label_3d)
        assert set(labels.tolist()) == {1, 2, 3}

    def test_bbox_matches_skimage(self, label_3d):
        labels, bboxes = find_all_objects_3D(label_3d)
        ref = _reference_bboxes_3D(label_3d)
        for lbl, bbox in zip(labels.tolist(), bboxes.tolist()):
            z0, z1, r0, r1, c0, c1 = bbox
            assert (z0, z1, r0, r1, c0, c1) == ref[lbl], (
                f"Label {lbl}: got {(z0,z1,r0,r1,c0,c1)}, expected {ref[lbl]}"
            )

    def test_empty_image_returns_empty(self):
        empty = np.zeros((8, 16, 16), dtype=np.uint32)
        result = find_all_objects_3D(empty)
        assert result == ([], [])

    def test_single_voxel_object(self):
        img = np.zeros((8, 8, 8), dtype=np.uint32)
        img[3, 4, 5] = 2
        labels, bboxes = find_all_objects_3D(img)
        assert labels[0] == 2
        assert list(bboxes[0]) == [3, 4, 4, 5, 5, 6]

    def test_bbox_dtype_is_uint32(self, label_3d):
        _, bboxes = find_all_objects_3D(label_3d)
        assert bboxes.dtype == np.uint32


# ---------------------------------------------------------------------------
# most_common_projection_3D
# ---------------------------------------------------------------------------

class TestMostCommonProjection3D:
    @pytest.mark.parametrize(
        "axis, expected",
        [
            (
                0,
                np.array(
                    [
                        [1, 1],
                        [1, 2],
                        [2, 2],
                        [2, 2],
                        [1, 0],
                        [1, 0],
                    ],
                    dtype=np.uint32,
                ),
            ),
            (1, np.array([[1, 2]], dtype=np.uint32)),
            (2, np.array([[1, 1, 2, 2, 1, 1]], dtype=np.uint32)),
        ],
    )
    def test_counts_across_full_axis_not_runs(self, axis, expected):
        """A label split into multiple runs must still be counted globally."""
        lab = np.array(
            [
                [
                    [1, 1],
                    [1, 2],
                    [2, 2],
                    [2, 2],
                    [1, 0],
                    [1, 0],
                ]
            ],
            dtype=np.uint32,
        )
        out = most_common_projection_3D(lab, axis)
        assert out.shape == expected.shape
        np.testing.assert_array_equal(out, expected)

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_ignores_zero_but_returns_zero_if_no_nonzero(self, axis):
        lab = np.zeros((3, 4, 5), dtype=np.uint32)
        out = most_common_projection_3D(lab, axis)
        assert out.dtype == np.uint32
        np.testing.assert_array_equal(out, np.zeros_like(out, dtype=np.uint32))

    def test_invalid_axis_raises(self):
        lab = np.zeros((2, 2, 2), dtype=np.uint32)
        with pytest.raises(ValueError):
            most_common_projection_3D(lab, 3)


# ---------------------------------------------------------------------------
# calc_IoA_matrix_2D
# ---------------------------------------------------------------------------

def _run_cython_ioa_2d(lab, prev_lab, use_union):
    rp      = regionprops(lab.astype(np.int32))
    prev_rp = regionprops(prev_lab.astype(np.int32))
    curr_IDs_arr   = np.array([obj.label for obj in rp],      dtype=np.uint32)
    prev_IDs_arr   = np.array([obj.label for obj in prev_rp], dtype=np.uint32)
    prev_areas_arr = np.array([obj.area  for obj in prev_rp], dtype=np.uint32)
    if use_union:
        rp_mapper = {obj.label: obj for obj in rp}
        curr_areas_arr = np.array(
            [rp_mapper[ID].area for ID in curr_IDs_arr.tolist()], dtype=np.uint32
        )
    else:
        curr_areas_arr = np.empty(0, dtype=np.uint32)
    return (
        calc_IoA_matrix_2D(
            lab.astype(np.uint32), prev_lab.astype(np.uint32),
            curr_IDs_arr, prev_IDs_arr,
            prev_areas_arr, curr_areas_arr, use_union,
        ),
        rp, prev_rp,
    )


class TestCalcIoAMatrix2D:
    @pytest.mark.parametrize("use_union", [False, True])
    def test_matches_python_reference(self, label_2d, label_2d_shifted, use_union):
        mat_cy, rp, prev_rp = _run_cython_ioa_2d(label_2d_shifted, label_2d, use_union)
        mat_py, _, _ = _python_ioa_matrix(
            label_2d_shifted, label_2d, rp, prev_rp, use_union
        )
        np.testing.assert_allclose(mat_cy, mat_py, rtol=1e-9, atol=1e-12,
            err_msg=f"Mismatch for use_union={use_union}")

    def test_shape(self, label_2d, label_2d_shifted):
        mat_cy, rp, prev_rp = _run_cython_ioa_2d(label_2d_shifted, label_2d, False)
        assert mat_cy.shape == (len(rp), len(prev_rp))

    def test_values_between_0_and_1(self, label_2d, label_2d_shifted):
        mat_cy, _, _ = _run_cython_ioa_2d(label_2d_shifted, label_2d, False)
        assert np.all(mat_cy >= 0.0)
        assert np.all(mat_cy <= 1.0 + 1e-12)

    def test_no_overlap_gives_zero_matrix(self):
        """Two images with disjoint objects should produce an all-zero IoA matrix."""
        lab      = _make_disc_label_image((64, 64), [(1, 10, 10, 5)])
        prev_lab = _make_disc_label_image((64, 64), [(1, 50, 50, 5)])
        mat_cy, _, _ = _run_cython_ioa_2d(lab, prev_lab, False)
        np.testing.assert_array_equal(mat_cy, np.zeros_like(mat_cy))

    def test_identical_images_diagonal_is_one(self):
        """When lab == prev_lab and IDs match, area_prev IoA should be 1."""
        lab = _make_disc_label_image((64, 64), [
            (1, 20, 20, 8),
            (2, 20, 50, 8),
        ])
        mat_cy, _, _ = _run_cython_ioa_2d(lab, lab, False)
        np.testing.assert_allclose(np.diag(mat_cy), 1.0, rtol=1e-9)

    def test_dtype_is_float64(self, label_2d, label_2d_shifted):
        mat_cy, _, _ = _run_cython_ioa_2d(label_2d_shifted, label_2d, False)
        assert mat_cy.dtype == np.float64


# ---------------------------------------------------------------------------
# calc_IoA_matrix_3D
# ---------------------------------------------------------------------------

def _run_cython_ioa_3d(lab, prev_lab, use_union):
    rp      = regionprops(lab.astype(np.int32))
    prev_rp = regionprops(prev_lab.astype(np.int32))
    curr_IDs_arr   = np.array([obj.label for obj in rp],      dtype=np.uint32)
    prev_IDs_arr   = np.array([obj.label for obj in prev_rp], dtype=np.uint32)
    prev_areas_arr = np.array([obj.area  for obj in prev_rp], dtype=np.uint32)
    if use_union:
        rp_mapper = {obj.label: obj for obj in rp}
        curr_areas_arr = np.array(
            [rp_mapper[ID].area for ID in curr_IDs_arr.tolist()], dtype=np.uint32
        )
    else:
        curr_areas_arr = np.empty(0, dtype=np.uint32)
    return (
        calc_IoA_matrix_3D(
            lab.astype(np.uint32), prev_lab.astype(np.uint32),
            curr_IDs_arr, prev_IDs_arr,
            prev_areas_arr, curr_areas_arr, use_union,
        ),
        rp, prev_rp,
    )


class TestCalcIoAMatrix3D:
    @pytest.mark.parametrize("use_union", [False, True])
    def test_matches_python_reference(self, label_3d, label_3d_shifted, use_union):
        mat_cy, rp, prev_rp = _run_cython_ioa_3d(label_3d_shifted, label_3d, use_union)
        mat_py, _, _ = _python_ioa_matrix(
            label_3d_shifted, label_3d, rp, prev_rp, use_union
        )
        np.testing.assert_allclose(mat_cy, mat_py, rtol=1e-9, atol=1e-12,
            err_msg=f"3D mismatch for use_union={use_union}")

    def test_shape(self, label_3d, label_3d_shifted):
        mat_cy, rp, prev_rp = _run_cython_ioa_3d(label_3d_shifted, label_3d, False)
        assert mat_cy.shape == (len(rp), len(prev_rp))

    def test_values_between_0_and_1(self, label_3d, label_3d_shifted):
        mat_cy, _, _ = _run_cython_ioa_3d(label_3d_shifted, label_3d, False)
        assert np.all(mat_cy >= 0.0)
        assert np.all(mat_cy <= 1.0 + 1e-12)

    def test_no_overlap_gives_zero_matrix(self):
        lab      = _make_sphere_label_image((20, 32, 32), [(1, 5, 8, 8, 2, 4, 4)])
        prev_lab = _make_sphere_label_image((20, 32, 32), [(1, 15, 24, 24, 2, 4, 4)])
        mat_cy, _, _ = _run_cython_ioa_3d(lab, prev_lab, False)
        np.testing.assert_array_equal(mat_cy, np.zeros_like(mat_cy))

    def test_identical_images_diagonal_is_one(self):
        lab = _make_sphere_label_image((20, 32, 32), [
            (1,  8, 10, 10, 3, 4, 4),
            (2,  8, 10, 22, 3, 4, 4),
        ])
        mat_cy, _, _ = _run_cython_ioa_3d(lab, lab, False)
        np.testing.assert_allclose(np.diag(mat_cy), 1.0, rtol=1e-9)

    def test_dtype_is_float64(self, label_3d, label_3d_shifted):
        mat_cy, _, _ = _run_cython_ioa_3d(label_3d_shifted, label_3d, False)
        assert mat_cy.dtype == np.float64
