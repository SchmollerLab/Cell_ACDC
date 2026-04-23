# regionprops_helper.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.limits cimport UINT_MAX

def find_all_objects_2D(np.uint32_t[:, :] label_img):
    cdef Py_ssize_t n_rows = label_img.shape[0]
    cdef Py_ssize_t n_cols = label_img.shape[1]
    cdef Py_ssize_t i, j
    cdef unsigned int label, max_label = 0

    # First pass: find max label to allocate C arrays
    for i in range(n_rows):
        for j in range(n_cols):
            label = label_img[i, j]
            if label > max_label:
                max_label = label

    if max_label == 0:
        return []

    cdef np.ndarray[np.uint32_t, ndim=1] _rs = np.full(max_label + 1, UINT_MAX, dtype=np.uint32)
    cdef np.ndarray[np.uint32_t, ndim=1] _re = np.zeros(max_label + 1,          dtype=np.uint32)
    cdef np.ndarray[np.uint32_t, ndim=1] _cs = np.full(max_label + 1, UINT_MAX, dtype=np.uint32)
    cdef np.ndarray[np.uint32_t, ndim=1] _ce = np.zeros(max_label + 1,          dtype=np.uint32)

    cdef unsigned int[:] rs = _rs, re = _re, cs = _cs, ce = _ce

    # Second pass: compute bounding boxes without Python objects in the hot loop
    for i in range(n_rows):
        for j in range(n_cols):
            label = label_img[i, j]
            if label > 0:
                if i     < rs[label]: rs[label] = <unsigned int>i
                if i + 1 > re[label]: re[label] = <unsigned int>(i + 1)
                if j     < cs[label]: cs[label] = <unsigned int>j
                if j + 1 > ce[label]: ce[label] = <unsigned int>(j + 1)

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

    # First pass: find max label
    for i in range(n_z):
        for j in range(n_rows):
            for k in range(n_cols):
                label = label_img[i, j, k]
                if label > max_label:
                    max_label = label

    if max_label == 0:
        return []

    cdef np.ndarray[np.uint32_t, ndim=1] _zs = np.full(max_label + 1, UINT_MAX, dtype=np.uint32)
    cdef np.ndarray[np.uint32_t, ndim=1] _ze = np.zeros(max_label + 1,          dtype=np.uint32)
    cdef np.ndarray[np.uint32_t, ndim=1] _rs = np.full(max_label + 1, UINT_MAX, dtype=np.uint32)
    cdef np.ndarray[np.uint32_t, ndim=1] _re = np.zeros(max_label + 1,          dtype=np.uint32)
    cdef np.ndarray[np.uint32_t, ndim=1] _cs = np.full(max_label + 1, UINT_MAX, dtype=np.uint32)
    cdef np.ndarray[np.uint32_t, ndim=1] _ce = np.zeros(max_label + 1,          dtype=np.uint32)

    cdef unsigned int[:] zs = _zs, ze = _ze, rs = _rs, re = _re, cs = _cs, ce = _ce

    # Second pass: compute bounding boxes
    for i in range(n_z):
        for j in range(n_rows):
            for k in range(n_cols):
                label = label_img[i, j, k]
                if label > 0:
                    if i     < zs[label]: zs[label] = <unsigned int>i
                    if i + 1 > ze[label]: ze[label] = <unsigned int>(i + 1)
                    if j     < rs[label]: rs[label] = <unsigned int>j
                    if j + 1 > re[label]: re[label] = <unsigned int>(j + 1)
                    if k     < cs[label]: cs[label] = <unsigned int>k
                    if k + 1 > ce[label]: ce[label] = <unsigned int>(k + 1)

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