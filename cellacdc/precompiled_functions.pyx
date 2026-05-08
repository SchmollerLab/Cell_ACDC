# precompiled_functions.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True
# rand change to trigger gh actions: 2
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
        return [], []

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
        return [], []

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