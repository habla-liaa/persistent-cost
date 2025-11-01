# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
"""Cython accelerated routines for the persistent cost pivot algorithm."""

import numpy as np
cimport numpy as np
ctypedef np.uint8_t UINT8_t
ctypedef np.int64_t INT64_t


cdef inline INT64_t column_pivot_k(UINT8_t[:, :] R, Py_ssize_t m, Py_ssize_t k) nogil:
	cdef Py_ssize_t row
	for row in range(m - 1, -1, -1):
		if R[row, k] & 1:
			return row
	return -1

cdef inline Py_ssize_t find_first_with_pivot(INT64_t[:] cpivs, Py_ssize_t n, INT64_t pivot) nogil:
	cdef Py_ssize_t idx
	for idx in range(n):
		if cpivs[idx] == pivot:
			return idx
	return -1
    
cdef inline void add_column_xor(UINT8_t[:, :] M, Py_ssize_t rows, Py_ssize_t j, Py_ssize_t k) nogil:
	cdef Py_ssize_t row
	for row in range(rows):
		M[row, k] ^= M[row, j]


def do_pivot_binary(np.ndarray[np.uint8_t, ndim=2] input_matrix):
	"""Perform the pivot reduction over Z/2Z using a dense binary matrix."""

	cdef np.ndarray[np.uint8_t, ndim=2] R_np = np.array(
		input_matrix, dtype=np.uint8, copy=True, order="C"
	)
	cdef UINT8_t[:, :] R = R_np

	cdef Py_ssize_t m = R_np.shape[0]
	cdef Py_ssize_t n = R_np.shape[1]

	cdef np.ndarray[np.uint8_t, ndim=2] V_np = np.zeros((n, n), dtype=np.uint8)
	cdef UINT8_t[:, :] V = V_np

	cdef np.ndarray[np.int64_t, ndim=1] cpivs_np = np.empty(n, dtype=np.int64)
	cdef INT64_t[:] cpivs = cpivs_np

	cdef np.ndarray[np.uint8_t, ndim=1] used_np = np.zeros(m, dtype=np.uint8)
	cdef UINT8_t[:] used = used_np

	cdef Py_ssize_t idx
	for idx in range(n):
		V[idx, idx] = 1

	cdef Py_ssize_t k, j
	cdef INT64_t pivot

	for k in range(n):
		cpivs[k] = column_pivot_k(R, m, k)

	for k in range(n):
		pivot = cpivs[k]
		while pivot != -1 and pivot < m and used[pivot] == 1:
			j = find_first_with_pivot(cpivs, n, pivot)
			if j < 0:
				break
			add_column_xor(R, m, j, k)
			add_column_xor(V, n, j, k)
			cpivs[k] = column_pivot_k(R, m, k)
			pivot = cpivs[k]

		if pivot != -1 and pivot < m and used[pivot] == 0:
			used[pivot] = 1

	return R_np, V_np
