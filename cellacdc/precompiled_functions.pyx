# precompiled_functions.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True
# rand change to trigger gh actions: 1
import numpy as np
cimport numpy as np
from libc.limits cimport UINT_MAX

def find_all_objects_2D(np.uint32_t[:, :] label_img):
    cdef Py_ssize_t n_rows = label_img.shape[0]
    cdef Py_ssize_t n_cols = label_img.shape[1]
    cdef Py_ssize_t i, j
    cdef unsigned int label, max_label = 0
    cdef unsigned int capacity = 300, new_cap

    cdef np.ndarray[np.uint32_t, ndim=1] _rs = np.full(capacity, UINT_MAX, dtype=np.uint32)
    cdef np.ndarray[np.uint32_t, ndim=1] _re = np.zeros(capacity,           dtype=np.uint32)
    cdef np.ndarray[np.uint32_t, ndim=1] _cs = np.full(capacity, UINT_MAX, dtype=np.uint32)
    cdef np.ndarray[np.uint32_t, ndim=1] _ce = np.zeros(capacity,           dtype=np.uint32)

    cdef unsigned int[:] rs = _rs, re = _re, cs = _cs, ce = _ce

    # Single pass: compute bounding boxes, growing arrays in 300-label steps if needed
    for i in range(n_rows):
        for j in range(n_cols):
            label = label_img[i, j]
            if label == 0:
                continue
            if label >= capacity:
                new_cap = ((label // 300) + 1) * 300
                _rs = np.concatenate((_rs, np.full(new_cap - capacity, UINT_MAX, dtype=np.uint32)))
                _re = np.concatenate((_re, np.zeros(new_cap - capacity,           dtype=np.uint32)))
                _cs = np.concatenate((_cs, np.full(new_cap - capacity, UINT_MAX, dtype=np.uint32)))
                _ce = np.concatenate((_ce, np.zeros(new_cap - capacity,           dtype=np.uint32)))
                rs = _rs; re = _re; cs = _cs; ce = _ce
                capacity = new_cap
            if label > max_label:
                max_label = label
            if i     < rs[label]: rs[label] = <unsigned int>i
            if i + 1 > re[label]: re[label] = <unsigned int>(i + 1)
            if j     < cs[label]: cs[label] = <unsigned int>j
            if j + 1 > ce[label]: ce[label] = <unsigned int>(j + 1)

    if max_label == 0:
        return np.array([], dtype=np.uint32), np.empty((0, 4), dtype=np.uint32)
    # Collect present labels into compact numpy arrays (avoids per-label tuple allocation)
    cdef unsigned int n_labels = 0
    for lbl in range(1, max_label + 1):
        if re[lbl] != 0:
            n_labels += 1

    cdef np.ndarray[np.uint32_t, ndim=1] out_labels = np.empty(n_labels, dtype=np.uint32)
    cdef np.ndarray[np.uint32_t, ndim=2] out_bboxes = np.empty((n_labels, 4), dtype=np.uint32)
    cdef unsigned int idx = 0
    for lbl in range(1, max_label + 1):
        if re[lbl] != 0:
            out_labels[idx] = lbl
            out_bboxes[idx, 0] = rs[lbl]
            out_bboxes[idx, 1] = re[lbl]
            out_bboxes[idx, 2] = cs[lbl]
            out_bboxes[idx, 3] = ce[lbl]
            idx += 1
    return out_labels, out_bboxes

def find_all_objects_3D(np.uint32_t[:, :, :] label_img):
    cdef Py_ssize_t n_z = label_img.shape[0]
    cdef Py_ssize_t n_rows = label_img.shape[1]
    cdef Py_ssize_t n_cols = label_img.shape[2]
    cdef Py_ssize_t i, j, k
    cdef unsigned int label, max_label = 0
    cdef unsigned int capacity = 300, new_cap

    cdef np.ndarray[np.uint32_t, ndim=1] _zs = np.full(capacity, UINT_MAX, dtype=np.uint32)
    cdef np.ndarray[np.uint32_t, ndim=1] _ze = np.zeros(capacity,           dtype=np.uint32)
    cdef np.ndarray[np.uint32_t, ndim=1] _rs = np.full(capacity, UINT_MAX, dtype=np.uint32)
    cdef np.ndarray[np.uint32_t, ndim=1] _re = np.zeros(capacity,           dtype=np.uint32)
    cdef np.ndarray[np.uint32_t, ndim=1] _cs = np.full(capacity, UINT_MAX, dtype=np.uint32)
    cdef np.ndarray[np.uint32_t, ndim=1] _ce = np.zeros(capacity,           dtype=np.uint32)

    cdef unsigned int[:] zs = _zs, ze = _ze, rs = _rs, re = _re, cs = _cs, ce = _ce

    # Single pass: compute bounding boxes, growing arrays in 300-label steps if needed
    for i in range(n_z):
        for j in range(n_rows):
            for k in range(n_cols):
                label = label_img[i, j, k]
                if label == 0:
                    continue
                if label >= capacity:
                    new_cap = ((label // 300) + 1) * 300
                    _zs = np.concatenate((_zs, np.full(new_cap - capacity, UINT_MAX, dtype=np.uint32)))
                    _ze = np.concatenate((_ze, np.zeros(new_cap - capacity,           dtype=np.uint32)))
                    _rs = np.concatenate((_rs, np.full(new_cap - capacity, UINT_MAX, dtype=np.uint32)))
                    _re = np.concatenate((_re, np.zeros(new_cap - capacity,           dtype=np.uint32)))
                    _cs = np.concatenate((_cs, np.full(new_cap - capacity, UINT_MAX, dtype=np.uint32)))
                    _ce = np.concatenate((_ce, np.zeros(new_cap - capacity,           dtype=np.uint32)))
                    zs = _zs; ze = _ze; rs = _rs; re = _re; cs = _cs; ce = _ce
                    capacity = new_cap
                if label > max_label:
                    max_label = label
                if i     < zs[label]: zs[label] = <unsigned int>i
                if i + 1 > ze[label]: ze[label] = <unsigned int>(i + 1)
                if j     < rs[label]: rs[label] = <unsigned int>j
                if j + 1 > re[label]: re[label] = <unsigned int>(j + 1)
                if k     < cs[label]: cs[label] = <unsigned int>k
                if k + 1 > ce[label]: ce[label] = <unsigned int>(k + 1)

    if max_label == 0:
        return np.array([], dtype=np.uint32), np.empty((0, 6), dtype=np.uint32)

    # Collect present labels into compact numpy arrays (avoids per-label tuple allocation)
    cdef unsigned int n_labels = 0
    for lbl in range(1, max_label + 1):
        if ze[lbl] != 0:
            n_labels += 1

    cdef np.ndarray[np.uint32_t, ndim=1] out_labels = np.empty(n_labels, dtype=np.uint32)
    cdef np.ndarray[np.uint32_t, ndim=2] out_bboxes = np.empty((n_labels, 6), dtype=np.uint32)
    cdef unsigned int idx = 0
    for lbl in range(1, max_label + 1):
        if ze[lbl] != 0:
            out_labels[idx] = lbl
            out_bboxes[idx, 0] = zs[lbl]
            out_bboxes[idx, 1] = ze[lbl]
            out_bboxes[idx, 2] = rs[lbl]
            out_bboxes[idx, 3] = re[lbl]
            out_bboxes[idx, 4] = cs[lbl]
            out_bboxes[idx, 5] = ce[lbl]
            idx += 1
    return out_labels, out_bboxes

def most_common_projection_3D(np.uint32_t[:, :, :] lab, int axis):
    """Most-common-value projection for a 3-D label image along `axis`.

    Tie-break matches np.unique(..., return_counts=True) + np.argmax(counts),
    i.e. the smallest label wins when counts are equal.
    """
    if axis < 0 or axis > 2:
        raise ValueError(f'axis must be 0, 1, or 2. Got {axis}.')

    cdef Py_ssize_t z = lab.shape[0]
    cdef Py_ssize_t y = lab.shape[1]
    cdef Py_ssize_t x = lab.shape[2]
    cdef Py_ssize_t i, j, a, b, depth
    cdef unsigned int v, vv
    cdef unsigned int best_label, best_count, curr_count
    cdef bint seen
    cdef np.uint32_t[:, :] out_view

    if axis == 0:
        depth = z
        out = np.empty((y, x), dtype=np.uint32)
        out_view = out
        for i in range(y):
            for j in range(x):
                best_count = 0
                best_label = 0
                for a in range(depth):
                    v = lab[a, i, j]
                    if v == 0:
                        continue
                    seen = False
                    for b in range(a):
                        if lab[b, i, j] == v:
                            seen = True
                            break
                    if seen:
                        continue

                    # Count all remaining occurrences of this label along the full axis.
                    curr_count = 1
                    for b in range(a + 1, depth):
                        if lab[b, i, j] == v:
                            curr_count += 1

                    if curr_count > best_count or (curr_count == best_count and v < best_label):
                        best_count = curr_count
                        best_label = v

                out_view[i, j] = best_label
        return out

    if axis == 1:
        depth = y
        out = np.empty((z, x), dtype=np.uint32)
        out_view = out
        for i in range(z):
            for j in range(x):
                best_count = 0
                best_label = 0
                for a in range(depth):
                    v = lab[i, a, j]
                    if v == 0:
                        continue
                    seen = False
                    for b in range(a):
                        if lab[i, b, j] == v:
                            seen = True
                            break
                    if seen:
                        continue

                    curr_count = 1
                    for b in range(a + 1, depth):
                        if lab[i, b, j] == v:
                            curr_count += 1

                    if curr_count > best_count or (curr_count == best_count and v < best_label):
                        best_count = curr_count
                        best_label = v

                out_view[i, j] = best_label
        return out

    depth = x
    out = np.empty((z, y), dtype=np.uint32)
    out_view = out
    for i in range(z):
        for j in range(y):
            best_count = 0
            best_label = 0
            for a in range(depth):
                v = lab[i, j, a]
                if v == 0:
                    continue
                seen = False
                for b in range(a):
                    vv = lab[i, j, b]
                    if vv == v:
                        seen = True
                        break
                if seen:
                    continue

                curr_count = 1
                for b in range(a + 1, depth):
                    vv = lab[i, j, b]
                    if vv == v:
                        curr_count += 1

                if curr_count > best_count or (curr_count == best_count and v < best_label):
                    best_count = curr_count
                    best_label = v

            out_view[i, j] = best_label
    return out

def object_projections_and_size_3D(
        np.uint32_t[:, :, :] cutout,
        unsigned int obj_id,
):
    """Return binary XY/XZ/YZ projections and voxel count for specified object in a cutout."""
    cdef Py_ssize_t z = cutout.shape[0]
    cdef Py_ssize_t y = cutout.shape[1]
    cdef Py_ssize_t x = cutout.shape[2]
    cdef Py_ssize_t i, j, k
    cdef unsigned int size = 0

    cdef np.ndarray[np.uint8_t, ndim=2] proj_z = np.zeros((y, x), dtype=np.uint8)
    cdef np.ndarray[np.uint8_t, ndim=2] proj_y = np.zeros((z, x), dtype=np.uint8)
    cdef np.ndarray[np.uint8_t, ndim=2] proj_x = np.zeros((z, y), dtype=np.uint8)
    cdef np.uint8_t[:, :] proj_z_view = proj_z
    cdef np.uint8_t[:, :] proj_y_view = proj_y
    cdef np.uint8_t[:, :] proj_x_view = proj_x

    for i in range(z):
        for j in range(y):
            for k in range(x):
                if cutout[i, j, k] != obj_id:
                    continue
                size += 1
                proj_z_view[j, k] = 1
                proj_y_view[i, k] = 1
                proj_x_view[i, j] = 1

    return proj_z, proj_y, proj_x, size

def object_projection_and_size_3D(
        np.uint32_t[:, :, :] cutout,
        unsigned int obj_id,
        int axis,
):
    """Return one binary projection and voxel count for one object in a 3-D cutout.

    axis=0 -> XY projection (collapse z)
    axis=1 -> XZ projection (collapse y)
    axis=2 -> YZ projection (collapse x)
    """
    if axis < 0 or axis > 2:
        raise ValueError(f'axis must be 0, 1, or 2. Got {axis}.')

    cdef Py_ssize_t z = cutout.shape[0]
    cdef Py_ssize_t y = cutout.shape[1]
    cdef Py_ssize_t x = cutout.shape[2]
    cdef Py_ssize_t i, j, k
    cdef unsigned int size = 0

    cdef np.ndarray[np.uint8_t, ndim=2] proj
    cdef np.uint8_t[:, :] proj_view

    if axis == 0:
        proj = np.zeros((y, x), dtype=np.uint8)
        proj_view = proj
        for i in range(z):
            for j in range(y):
                for k in range(x):
                    if cutout[i, j, k] != obj_id:
                        continue
                    size += 1
                    proj_view[j, k] = 1
        return proj, size

    if axis == 1:
        proj = np.zeros((z, x), dtype=np.uint8)
        proj_view = proj
        for i in range(z):
            for j in range(y):
                for k in range(x):
                    if cutout[i, j, k] != obj_id:
                        continue
                    size += 1
                    proj_view[i, k] = 1
        return proj, size

    proj = np.zeros((z, y), dtype=np.uint8)
    proj_view = proj
    for i in range(z):
        for j in range(y):
            for k in range(x):
                if cutout[i, j, k] != obj_id:
                    continue
                size += 1
                proj_view[i, j] = 1

    return proj, size

def calc_IoA_matrix_2D(
        np.uint32_t[:, :] lab,
        np.uint32_t[:, :] prev_lab,
        np.uint32_t[:] curr_IDs,
        np.uint32_t[:] prev_IDs,
        np.uint32_t[:] prev_areas,
        np.uint32_t[:] curr_areas,
        bint use_union,
):
    """Single-pass IoA matrix between two 2-D label images.

    Parameters
    ----------
    lab, prev_lab : (Y, X) uint32 label images for current and previous frame.
    curr_IDs      : 1-D array of current object labels  (row order of output).
    prev_IDs      : 1-D array of previous object labels (col order of output).
    prev_areas    : pixel area of each entry in prev_IDs.
    curr_areas    : pixel area of each entry in curr_IDs (only used when use_union=True).
    use_union     : if False, denominator is area_prev; if True, denominator is union.

    Returns
    -------
    IoA_matrix : (n_curr, n_prev) float64 array.
    """
    cdef Py_ssize_t n_rows = lab.shape[0]
    cdef Py_ssize_t n_cols = lab.shape[1]
    cdef Py_ssize_t n_curr = curr_IDs.shape[0]
    cdef Py_ssize_t n_prev = prev_IDs.shape[0]
    cdef Py_ssize_t i, j, ci, pi
    cdef unsigned int c, p, max_curr_label = 0, max_prev_label = 0
    cdef int ci_val, pi_val

    for i in range(n_curr):
        if curr_IDs[i] > max_curr_label:
            max_curr_label = curr_IDs[i]
    for i in range(n_prev):
        if prev_IDs[i] > max_prev_label:
            max_prev_label = prev_IDs[i]

    # label -> matrix-index lookup; -1 means "not in the tracked set"
    cdef np.ndarray[np.int32_t, ndim=1] _curr_idx = np.full(max_curr_label + 1, -1, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] _prev_idx = np.full(max_prev_label + 1, -1, dtype=np.int32)
    cdef int[:] curr_idx = _curr_idx
    cdef int[:] prev_idx = _prev_idx

    for i in range(n_curr):
        curr_idx[curr_IDs[i]] = <int>i
    for i in range(n_prev):
        prev_idx[prev_IDs[i]] = <int>i

    cdef np.ndarray[np.uint32_t, ndim=2] _intersections = np.zeros((n_curr, n_prev), dtype=np.uint32)
    cdef unsigned int[:, :] intersections = _intersections

    # Single pass: count overlapping pixels between every (curr, prev) pair
    for i in range(n_rows):
        for j in range(n_cols):
            c = lab[i, j]
            p = prev_lab[i, j]
            if c == 0 or p == 0:
                continue
            if c > max_curr_label or p > max_prev_label:
                continue
            ci_val = curr_idx[c]
            pi_val = prev_idx[p]
            if ci_val < 0 or pi_val < 0:
                continue
            intersections[ci_val, pi_val] += 1

    cdef np.ndarray[np.float64_t, ndim=2] IoA_matrix = np.zeros((n_curr, n_prev), dtype=np.float64)
    cdef double denom_val, I_val

    for ci in range(n_curr):
        for pi in range(n_prev):
            I_val = <double>intersections[ci, pi]
            if I_val == 0.0:
                continue
            if use_union:
                denom_val = <double>(curr_areas[ci] + prev_areas[pi]) - I_val
            else:
                denom_val = <double>prev_areas[pi]
            if denom_val == 0.0:
                continue
            IoA_matrix[ci, pi] = I_val / denom_val

    return IoA_matrix

def calc_IoA_matrix_3D(
        np.uint32_t[:, :, :] lab,
        np.uint32_t[:, :, :] prev_lab,
        np.uint32_t[:] curr_IDs,
        np.uint32_t[:] prev_IDs,
        np.uint32_t[:] prev_areas,
        np.uint32_t[:] curr_areas,
        bint use_union,
):
    """Single-pass IoA matrix between two 3-D label images. See calc_IoA_matrix_2D."""
    cdef Py_ssize_t n_z    = lab.shape[0]
    cdef Py_ssize_t n_rows = lab.shape[1]
    cdef Py_ssize_t n_cols = lab.shape[2]
    cdef Py_ssize_t n_curr = curr_IDs.shape[0]
    cdef Py_ssize_t n_prev = prev_IDs.shape[0]
    cdef Py_ssize_t i, j, k, ci, pi
    cdef unsigned int c, p, max_curr_label = 0, max_prev_label = 0
    cdef int ci_val, pi_val

    for i in range(n_curr):
        if curr_IDs[i] > max_curr_label:
            max_curr_label = curr_IDs[i]
    for i in range(n_prev):
        if prev_IDs[i] > max_prev_label:
            max_prev_label = prev_IDs[i]

    cdef np.ndarray[np.int32_t, ndim=1] _curr_idx = np.full(max_curr_label + 1, -1, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] _prev_idx = np.full(max_prev_label + 1, -1, dtype=np.int32)
    cdef int[:] curr_idx = _curr_idx
    cdef int[:] prev_idx = _prev_idx

    for i in range(n_curr):
        curr_idx[curr_IDs[i]] = <int>i
    for i in range(n_prev):
        prev_idx[prev_IDs[i]] = <int>i

    cdef np.ndarray[np.uint32_t, ndim=2] _intersections = np.zeros((n_curr, n_prev), dtype=np.uint32)
    cdef unsigned int[:, :] intersections = _intersections

    for i in range(n_z):
        for j in range(n_rows):
            for k in range(n_cols):
                c = lab[i, j, k]
                p = prev_lab[i, j, k]
                if c == 0 or p == 0:
                    continue
                if c > max_curr_label or p > max_prev_label:
                    continue
                ci_val = curr_idx[c]
                pi_val = prev_idx[p]
                if ci_val < 0 or pi_val < 0:
                    continue
                intersections[ci_val, pi_val] += 1

    cdef np.ndarray[np.float64_t, ndim=2] IoA_matrix = np.zeros((n_curr, n_prev), dtype=np.float64)
    cdef double denom_val, I_val

    for ci in range(n_curr):
        for pi in range(n_prev):
            I_val = <double>intersections[ci, pi]
            if I_val == 0.0:
                continue
            if use_union:
                denom_val = <double>(curr_areas[ci] + prev_areas[pi]) - I_val
            else:
                denom_val = <double>prev_areas[pi]
            if denom_val == 0.0:
                continue
            IoA_matrix[ci, pi] = I_val / denom_val

    return IoA_matrix


def calc_centroids_2D(
        np.uint32_t[:, :] label_img,
        np.uint32_t[:] labels,
        np.uint32_t[:, :] bboxes,
):
    """Bulk centroid computation restricted to each object's bbox.

    Parameters
    ----------
    label_img : (Y, X) uint32 label image.
    labels    : (n,) uint32 object labels.
    bboxes    : (n, 4) uint32 -> (row_start, row_stop, col_start, col_stop),
                same layout as returned by find_all_objects_2D.

    Returns
    -------
    centroids : (n, 2) float64 -> (mean_row, mean_col), in `label_img` coords.
    """
    cdef Py_ssize_t n_labels = labels.shape[0]
    cdef Py_ssize_t n, i, j
    cdef Py_ssize_t r0, r1, c0, c1
    cdef unsigned int label
    cdef double sum_i, sum_j
    cdef unsigned long long count

    cdef np.ndarray[np.float64_t, ndim=2] centroids = np.empty((n_labels, 2), dtype=np.float64)

    for n in range(n_labels):
        label = labels[n]
        r0 = <Py_ssize_t>bboxes[n, 0]
        r1 = <Py_ssize_t>bboxes[n, 1]
        c0 = <Py_ssize_t>bboxes[n, 2]
        c1 = <Py_ssize_t>bboxes[n, 3]

        sum_i = 0.0
        sum_j = 0.0
        count = 0

        for i in range(r0, r1):
            for j in range(c0, c1):
                if label_img[i, j] == label:
                    sum_i += <double>i
                    sum_j += <double>j
                    count += 1

        centroids[n, 0] = sum_i / count
        centroids[n, 1] = sum_j / count

    return centroids


def calc_centroids_3D(
        np.uint32_t[:, :, :] label_img,
        np.uint32_t[:] labels,
        np.uint32_t[:, :] bboxes,
):
    """Bulk centroid computation restricted to each object's bbox (3D).

    bboxes : (n, 6) uint32 -> (z0, z1, r0, r1, c0, c1), same layout as
             returned by find_all_objects_3D.
    """
    cdef Py_ssize_t n_labels = labels.shape[0]
    cdef Py_ssize_t n, i, j, k
    cdef Py_ssize_t z0, z1, r0, r1, c0, c1
    cdef unsigned int label
    cdef double sum_i, sum_j, sum_k
    cdef unsigned long long count

    cdef np.ndarray[np.float64_t, ndim=2] centroids = np.empty((n_labels, 3), dtype=np.float64)

    for n in range(n_labels):
        label = labels[n]
        z0 = <Py_ssize_t>bboxes[n, 0]
        z1 = <Py_ssize_t>bboxes[n, 1]
        r0 = <Py_ssize_t>bboxes[n, 2]
        r1 = <Py_ssize_t>bboxes[n, 3]
        c0 = <Py_ssize_t>bboxes[n, 4]
        c1 = <Py_ssize_t>bboxes[n, 5]

        sum_i = 0.0
        sum_j = 0.0
        sum_k = 0.0
        count = 0

        for i in range(z0, z1):
            for j in range(r0, r1):
                for k in range(c0, c1):
                    if label_img[i, j, k] == label:
                        sum_i += <double>i
                        sum_j += <double>j
                        sum_k += <double>k
                        count += 1

        centroids[n, 0] = sum_i / count
        centroids[n, 1] = sum_j / count
        centroids[n, 2] = sum_k / count

    return centroids


def color_norm_hsv_style(rgb, alpha_scale=1.0, calc_alpha=True):
    """Normalize already bounded RGB values while preserving hue."""
    cdef np.ndarray[np.float32_t, ndim=3] arr = np.asarray(rgb, dtype=np.float32)
    cdef Py_ssize_t h = arr.shape[0]
    cdef Py_ssize_t w = arr.shape[1]
    cdef Py_ssize_t i, j
    cdef float r, g, b, max_val, alpha_val

    cdef np.ndarray[np.float32_t, ndim=3] norm_rgb = np.empty((h, w, 3), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=3] alpha = np.empty((h, w, 1), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            r = arr[i, j, 0]
            g = arr[i, j, 1]
            b = arr[i, j, 2]
            max_val = r
            if g > max_val:
                max_val = g
            if b > max_val:
                max_val = b

            if max_val < 1e-6:
                max_val = <float>1e-6

            norm_rgb[i, j, 0] = r / max_val
            norm_rgb[i, j, 1] = g / max_val
            norm_rgb[i, j, 2] = b / max_val

            if calc_alpha:
                alpha_val = max_val * <float>alpha_scale
                alpha[i, j, 0] = alpha_val

    if calc_alpha:
        return norm_rgb, alpha
    return norm_rgb, None


def combine_grayscale_images_with_alpha_cy(
        base_img,
        images,
        alphas,
        luts=None,
        base_lut=None,
):
    """Composite already-normalized images using float16 RGB values."""
    cdef np.ndarray[np.float32_t, ndim=2] base_arr = np.asarray(base_img, dtype=np.float32)
    cdef Py_ssize_t h = base_arr.shape[0]
    cdef Py_ssize_t w = base_arr.shape[1]
    cdef Py_ssize_t n_imgs = len(images)
    cdef Py_ssize_t i, j, k, c
    cdef Py_ssize_t n_lut
    cdef float alpha_scale
    cdef float img_val, r, g, b, max_val, alpha_val, mix_val
    cdef int lut_idx

    if n_imgs == 0:
        return np.repeat(base_arr[:, :, None], 3, axis=2).astype(np.float32)

    cdef np.ndarray[np.float32_t, ndim=3] accumulated = np.zeros((h, w, 3), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] total_alpha = np.zeros((h, w), dtype=np.float32)

    cdef np.ndarray[np.float32_t, ndim=2] img_norm
    cdef np.ndarray[np.float32_t, ndim=3] rgb_src
    cdef np.ndarray[np.float32_t, ndim=2] lut_arr

    for i in range(n_imgs):
        img_norm = np.asarray(images[i], dtype=np.float32)
        rgb_src = np.empty((h, w, 3), dtype=np.float32)

        if luts is not None:
            lut_arr = np.asarray(luts[i], dtype=np.float32)
            n_lut = <Py_ssize_t>lut_arr.shape[0]
            for j in range(h):
                for k in range(w):
                    img_val = img_norm[j, k]
                    lut_idx = <int>(img_val * (n_lut - 1))
                    rgb_src[j, k, 0] = lut_arr[lut_idx, 0]
                    rgb_src[j, k, 1] = lut_arr[lut_idx, 1]
                    rgb_src[j, k, 2] = lut_arr[lut_idx, 2]
        else:
            for j in range(h):
                for k in range(w):
                    img_val = img_norm[j, k]
                    rgb_src[j, k, 0] = img_val
                    rgb_src[j, k, 1] = img_val
                    rgb_src[j, k, 2] = img_val

        alpha_scale = <float>alphas[i]

        for j in range(h):
            for k in range(w):
                r = <float>rgb_src[j, k, 0]
                g = <float>rgb_src[j, k, 1]
                b = <float>rgb_src[j, k, 2]
                max_val = r
                if g > max_val:
                    max_val = g
                if b > max_val:
                    max_val = b

                if max_val < 1e-6:
                    max_val = <float>1e-6

                alpha_val = max_val * alpha_scale

                accumulated[j, k, 0] += alpha_val * (r / max_val)
                accumulated[j, k, 1] += alpha_val * (g / max_val)
                accumulated[j, k, 2] += alpha_val * (b / max_val)
                total_alpha[j, k] += alpha_val
                if total_alpha[j, k] > 1.0:
                    total_alpha[j, k] = 1.0

    cdef np.ndarray[np.float32_t, ndim=3] base_rgb
    if base_lut is not None:
        base_rgb = np.empty((h, w, 3), dtype=np.float32)
        lut_arr = np.asarray(base_lut, dtype=np.float32)
        n_lut = <Py_ssize_t>lut_arr.shape[0]
        for j in range(h):
            for k in range(w):
                img_val = base_arr[j, k]
                lut_idx = <int>(img_val * (n_lut - 1))
                base_rgb[j, k, 0] = lut_arr[lut_idx, 0]
                base_rgb[j, k, 1] = lut_arr[lut_idx, 1]
                base_rgb[j, k, 2] = lut_arr[lut_idx, 2]
    else:
        base_rgb = np.repeat(base_arr[:, :, None], 3, axis=2).astype(np.float32)

    cdef np.ndarray[np.float32_t, ndim=3] accumulated_norm
    accumulated_norm, _ = color_norm_hsv_style(accumulated, alpha_scale=1.0, calc_alpha=False)

    cdef np.ndarray[np.float32_t, ndim=3] result = np.empty((h, w, 3), dtype=np.float32)
    for j in range(h):
        for k in range(w):
            alpha_val = total_alpha[j, k]
            for c in range(3):
                mix_val = <float>((<float>base_rgb[j, k, c]) * (1.0 - alpha_val) + accumulated_norm[j, k, c] * alpha_val)
                result[j, k, c] = mix_val

    return result