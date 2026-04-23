# regionprops_helper.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.limits cimport INT_MAX

def find_all_objects_2D(np.int32_t[:, :] label_img):
    cdef int n_rows = label_img.shape[0]
    cdef int n_cols = label_img.shape[1]
    cdef int i, j, label, max_label = 0

    # First pass: find max label to allocate C arrays
    for i in range(n_rows):
        for j in range(n_cols):
            label = label_img[i, j]
            if label > max_label:
                max_label = label

    if max_label == 0:
        return []

    cdef np.ndarray[np.int32_t, ndim=1] _rs = np.full(max_label + 1, INT_MAX, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] _re = np.full(max_label + 1, -1,      dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] _cs = np.full(max_label + 1, INT_MAX, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] _ce = np.full(max_label + 1, -1,      dtype=np.int32)

    cdef int[:] rs = _rs, re = _re, cs = _cs, ce = _ce

    # Second pass: compute bounding boxes without Python objects in the hot loop
    for i in range(n_rows):
        for j in range(n_cols):
            label = label_img[i, j]
            if label > 0:
                if i     < rs[label]: rs[label] = i
                if i + 1 > re[label]: re[label] = i + 1
                if j     < cs[label]: cs[label] = j
                if j + 1 > ce[label]: ce[label] = j + 1

    return [
        (lbl, (rs[lbl], re[lbl], cs[lbl], ce[lbl]))
        for lbl in range(1, max_label + 1)
        if re[lbl] != -1
    ]

def find_all_objects_3D(np.int32_t[:, :, :] label_img):
    cdef int n_z = label_img.shape[0]
    cdef int n_rows = label_img.shape[1]
    cdef int n_cols = label_img.shape[2]
    cdef int i, j, k, label, max_label = 0

    # First pass: find max label
    for i in range(n_z):
        for j in range(n_rows):
            for k in range(n_cols):
                label = label_img[i, j, k]
                if label > max_label:
                    max_label = label

    if max_label == 0:
        return []

    cdef np.ndarray[np.int32_t, ndim=1] _zs = np.full(max_label + 1, INT_MAX, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] _ze = np.full(max_label + 1, -1,      dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] _rs = np.full(max_label + 1, INT_MAX, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] _re = np.full(max_label + 1, -1,      dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] _cs = np.full(max_label + 1, INT_MAX, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] _ce = np.full(max_label + 1, -1,      dtype=np.int32)

    cdef int[:] zs = _zs, ze = _ze, rs = _rs, re = _re, cs = _cs, ce = _ce

    # Second pass: compute bounding boxes
    for i in range(n_z):
        for j in range(n_rows):
            for k in range(n_cols):
                label = label_img[i, j, k]
                if label > 0:
                    if i     < zs[label]: zs[label] = i
                    if i + 1 > ze[label]: ze[label] = i + 1
                    if j     < rs[label]: rs[label] = j
                    if j + 1 > re[label]: re[label] = j + 1
                    if k     < cs[label]: cs[label] = k
                    if k + 1 > ce[label]: ce[label] = k + 1

    return [
        (lbl, (zs[lbl], ze[lbl], rs[lbl], re[lbl], cs[lbl], ce[lbl]))
        for lbl in range(1, max_label + 1)
        if ze[lbl] != -1
    ]