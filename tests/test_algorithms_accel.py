"""Tests for accelerated pivot implementations."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from persistent_cost.algorithms import available_backends
from persistent_cost.algorithms.dense import (
	do_pivot as do_pivot_python,
)
from persistent_cost.algorithms.dense_fast import (
	do_pivot_cython as do_pivot_dense_cython,
	do_pivot_numba as do_pivot_dense_numba,
)
from persistent_cost.algorithms.sparse import 	do_pivot as do_pivot_python


from persistent_cost.algorithms.sparse_fast import (
	do_pivot_numba as do_pivot_sparse_numba,
	do_pivot_cython as do_pivot_sparse_cython,
)


def _random_binary_matrix(seed: int, shape: tuple[int, int]) -> csc_matrix:
	rng = np.random.default_rng(seed)
	dense = rng.integers(0, 2, size=shape, dtype=np.uint8)
	return sparse.csc_matrix(dense)


def _as_dense(matrix: sparse.spmatrix) -> np.ndarray:
	return np.asarray(matrix.toarray(), dtype=np.uint8) & 1


@pytest.mark.parametrize("shape", [(16, 16), (32, 24)])
def test_numba_backend_matches_python(shape: tuple[int, int]) -> None:
	matrix = _random_binary_matrix(seed=123, shape=shape)
	ref_R, ref_V = do_pivot_python(matrix)
	numba_R, numba_V = do_pivot_sparse_numba(matrix)

	np.testing.assert_array_equal(_as_dense(ref_R), _as_dense(numba_R))
	np.testing.assert_array_equal(_as_dense(ref_V), _as_dense(numba_V))


@pytest.mark.parametrize("shape", [(18, 18), (20, 12)])
def test_cython_backend_matches_python(shape: tuple[int, int]) -> None:
	matrix = _random_binary_matrix(seed=456, shape=shape)
	ref_R, ref_V = do_pivot_python(matrix)
	cython_R, cython_V = do_pivot_sparse_cython(matrix)

	np.testing.assert_array_equal(_as_dense(ref_R), _as_dense(cython_R))
	np.testing.assert_array_equal(_as_dense(ref_V), _as_dense(cython_V))


@pytest.mark.parametrize("shape", [(20, 20), (30, 18)])
def test_sparse_numba_backend_matches_python(shape: tuple[int, int]) -> None:
	matrix = _random_binary_matrix(seed=987, shape=shape)
	ref_R, ref_V = do_pivot_python(matrix)
	numba_R, numba_V = do_pivot_sparse_numba(matrix)

	assert sparse.isspmatrix_csc(numba_R)
	assert sparse.isspmatrix_csc(numba_V)
	np.testing.assert_array_equal(_as_dense(ref_R), _as_dense(numba_R))
	np.testing.assert_array_equal(_as_dense(ref_V), _as_dense(numba_V))


@pytest.mark.parametrize("shape", [(18, 18), (22, 15)])
def test_sparse_cython_backend_matches_python(shape: tuple[int, int]) -> None:
	
	matrix = _random_binary_matrix(seed=654, shape=shape)
	ref_R, ref_V = do_pivot_python(matrix)
	cython_R, cython_V = do_pivot_sparse_cython(matrix)

	assert sparse.isspmatrix_csc(cython_R)
	assert sparse.isspmatrix_csc(cython_V)
	np.testing.assert_array_equal(_as_dense(ref_R), _as_dense(cython_R))
	np.testing.assert_array_equal(_as_dense(ref_V), _as_dense(cython_V))

