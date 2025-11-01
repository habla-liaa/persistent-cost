# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

"""Cython accelerated routines for sparse pivot reduction."""

import numpy as np
cimport numpy as np

ctypedef np.int64_t INT64_t
ctypedef np.uint8_t UINT8_t


cdef inline INT64_t column_pivot(np.ndarray[INT64_t, ndim=1] col):
    cdef Py_ssize_t size = col.shape[0]
    if size == 0:
        return -1
    return col[size - 1]


cdef inline Py_ssize_t find_first_with_pivot(np.ndarray[INT64_t, ndim=1] cpivs, INT64_t pivot):
    cdef Py_ssize_t n = cpivs.shape[0]
    cdef Py_ssize_t idx
    for idx in range(n):
        if cpivs[idx] == pivot:
            return idx
    return -1


cdef np.ndarray[INT64_t, ndim=1] symmetric_difference(
    np.ndarray[INT64_t, ndim=1] a,
    np.ndarray[INT64_t, ndim=1] b,
):
    cdef Py_ssize_t n_a = a.shape[0]
    cdef Py_ssize_t n_b = b.shape[0]
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t j = 0
    cdef Py_ssize_t k = 0

    cdef np.ndarray[INT64_t, ndim=1] out = np.empty(n_a + n_b, dtype=np.int64)

    while i < n_a and j < n_b:
        if a[i] == b[j]:
            i += 1
            j += 1
        elif a[i] < b[j]:
            out[k] = a[i]
            i += 1
            k += 1
        else:
            out[k] = b[j]
            j += 1
            k += 1

    while i < n_a:
        out[k] = a[i]
        i += 1
        k += 1

    while j < n_b:
        out[k] = b[j]
        j += 1
        k += 1

    if k == out.shape[0]:
        return out
    return out[:k].copy()


def do_pivot_sparse(list R_columns, list V_columns, Py_ssize_t n_rows):
    """Perform the sparse pivot reduction working on column index arrays."""

    cdef Py_ssize_t n = len(R_columns)
    cdef np.ndarray[INT64_t, ndim=1] cpivs = np.empty(n, dtype=np.int64)
    cdef np.ndarray[UINT8_t, ndim=1] used = np.zeros(n_rows, dtype=np.uint8)
    cdef np.ndarray[INT64_t, ndim=1] col
    cdef INT64_t pivot
    cdef Py_ssize_t k
    cdef Py_ssize_t j

    for k in range(n):
        col = R_columns[k]
        cpivs[k] = column_pivot(col)

    for k in range(n):
        pivot = cpivs[k]
        while pivot != -1 and pivot < n_rows and used[pivot] == 1:
            j = find_first_with_pivot(cpivs, pivot)
            if j < 0:
                break
            R_columns[k] = symmetric_difference(R_columns[k], R_columns[j])
            V_columns[k] = symmetric_difference(V_columns[k], V_columns[j])
            col = R_columns[k]
            cpivs[k] = column_pivot(col)
            pivot = cpivs[k]

        if pivot != -1 and pivot < n_rows and used[pivot] == 0:
            used[pivot] = 1

    return R_columns, V_columns
