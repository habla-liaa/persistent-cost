"""Sparse accelerated pivot variants that avoid densifying matrices."""

from typing import Iterable, Literal, Sequence, List, Tuple, Optional

import numpy as np
from scipy import sparse

SparseBackendLiteral = Literal["numba", "cython"]

try:  # Optional Numba acceleration.
    from numba import njit
    from numba.typed import List as NumbaList
except ImportError:  # pragma: no cover - executed when numba missing.
    njit = None
    NumbaList = None
    NUMBA_AVAILABLE = False
else:
    NUMBA_AVAILABLE = True

try:  # Optional sparse Cython extension.
    from . import _pivot_sparse_cython as _pivot_sparse_cython_module
except ImportError:  # pragma: no cover - executed when extension missing.
    _pivot_sparse_cython_module = None
    CYTHON_AVAILABLE = False
else:
    CYTHON_AVAILABLE = True




def _ensure_binary_csc(matrix: sparse.spmatrix) -> sparse.csc_matrix:
    if not sparse.isspmatrix(matrix):
        raise TypeError("Expected a scipy.sparse matrix as input")
    csc = matrix.tocsc().astype(np.int8).copy()
    if csc.nnz:
        csc.data &= 1
        csc.eliminate_zeros()
    return csc


def _csc_to_column_arrays(matrix: sparse.csc_matrix) -> List[np.ndarray]:
    indices = matrix.indices
    indptr = matrix.indptr
    n_cols = matrix.shape[1]
    return [np.array(indices[indptr[k] : indptr[k + 1]], dtype=np.int64) for k in range(n_cols)]


def _column_arrays_to_csc(columns: Sequence[np.ndarray], shape: Tuple[int, int]) -> sparse.csc_matrix:
    n_cols = len(columns)
    indptr = np.zeros(n_cols + 1, dtype=np.int64)
    nnz = 0
    for idx, column in enumerate(columns, start=1):
        nnz += column.size
        indptr[idx] = nnz
    if nnz == 0:
        return sparse.csc_matrix(shape, dtype=np.int8)
    indices = np.empty(nnz, dtype=np.int64)
    data = np.ones(nnz, dtype=np.int8)
    offset = 0
    for column in columns:
        length = column.size
        if length:
            indices[offset : offset + length] = column
        offset += length
    return sparse.csc_matrix((data, indices, indptr), shape=shape)


def _identity_column_arrays(n_cols: int) -> List[np.ndarray]:
    return [np.array([idx], dtype=np.int64) for idx in range(n_cols)]


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _symdiff_numba(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        n_a = a.size
        n_b = b.size
        out = np.empty(n_a + n_b, dtype=np.int64)
        i = j = k = 0
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
        if k == out.size:
            return out
        return out[:k]

    @njit(cache=True)
    def _find_first_with_pivot(cpivs: np.ndarray, pivot: int) -> int:
        n_cols = cpivs.size
        for idx in range(n_cols):
            if cpivs[idx] == pivot:
                return idx
        return -1

    @njit(cache=True)
    def _column_pivot(column: np.ndarray) -> int:
        if column.size == 0:
            return -1
        return int(column[column.size - 1])

    @njit(cache=True)
    def _do_pivot_sparse_numba_core(columns, v_columns, cpivs: np.ndarray, used: np.ndarray, n_rows: int) -> None:
        n_cols = len(columns)
        for idx in range(n_cols):
            cpivs[idx] = _column_pivot(columns[idx])

        for idx in range(n_cols):
            pivot = cpivs[idx]
            while pivot != -1 and pivot < n_rows and used[pivot] == 1:
                other = _find_first_with_pivot(cpivs, pivot)
                if other < 0:
                    break
                columns[idx] = _symdiff_numba(columns[idx], columns[other])
                v_columns[idx] = _symdiff_numba(v_columns[idx], v_columns[other])
                cpivs[idx] = _column_pivot(columns[idx])
                pivot = cpivs[idx]

            if pivot != -1 and pivot < n_rows and used[pivot] == 0:
                used[pivot] = 1


def _to_numba_list(arrays: Iterable[np.ndarray]):
    if NumbaList is None:
        raise RuntimeError("Numba is not available")
    typed_list = NumbaList()
    for array in arrays:
        typed_list.append(np.array(array, dtype=np.int64))
    return typed_list


def _from_numba_list(typed_list) -> List[np.ndarray]:
    return [np.array(column, dtype=np.int64) for column in typed_list]


def do_pivot_numba(matrix: sparse.spmatrix) -> Tuple[sparse.csc_matrix, sparse.csc_matrix]:
    if not NUMBA_AVAILABLE:  # pragma: no cover - simple import guard.
        raise RuntimeError(
            "Numba is not installed. Install the optional 'accel' extra or add numba manually."
        )

    csc = _ensure_binary_csc(matrix)
    columns = _to_numba_list(_csc_to_column_arrays(csc))
    v_columns = _to_numba_list(_identity_column_arrays(csc.shape[1]))
    cpivs = np.empty(csc.shape[1], dtype=np.int64)
    used = np.zeros(csc.shape[0], dtype=np.uint8)
    _do_pivot_sparse_numba_core(columns, v_columns, cpivs, used, csc.shape[0])
    r_columns = _from_numba_list(columns)
    v_cols = _from_numba_list(v_columns)
    reduced = _column_arrays_to_csc(r_columns, csc.shape)
    transform = _column_arrays_to_csc(v_cols, (csc.shape[1], csc.shape[1]))
    # assert reduced == matrix @ transform

    # np.testing.assert_array_equal(reduced.toarray(), matrix @ transform.toarray())

    return reduced, transform


def do_pivot_cython(matrix: sparse.spmatrix) -> Tuple[sparse.csc_matrix, sparse.csc_matrix]:
    if not CYTHON_AVAILABLE:  # pragma: no cover - simple import guard.
        raise RuntimeError(
            "Cython sparse extension '_pivot_sparse_cython' is not built. Run 'pip install -e .[accel]' or "
            "'python setup.py build_ext --inplace' to compile it."
        )

    csc = _ensure_binary_csc(matrix)
    columns = _csc_to_column_arrays(csc)
    v_columns = _identity_column_arrays(csc.shape[1])
    r_cols, v_cols = _pivot_sparse_cython_module.do_pivot_sparse(columns, v_columns, csc.shape[0])
    reduced = _column_arrays_to_csc(r_cols, csc.shape)
    transform = _column_arrays_to_csc(v_cols, (csc.shape[1], csc.shape[1]))
    return reduced, transform


def do_pivot_fast(
    matrix: sparse.spmatrix, backend: SparseBackendLiteral = "numba"
) -> Tuple[sparse.csc_matrix, sparse.csc_matrix]:
    backend_lc = backend.lower()
    if backend_lc == "numba":
        return do_pivot_numba(matrix)
    if backend_lc == "cython":
        return do_pivot_cython(matrix)
    raise ValueError(f"Unknown backend '{backend}'. Valid choices: numba, cython")


def warm_up_sparse(backends: Optional[Iterable[str]] = None, size: Tuple[int, int] = (32, 32)) -> None:
    """Trigger JIT compilation for the selected sparse backends using a synthetic matrix."""
    rng = np.random.default_rng(821)
    dense = rng.integers(0, 2, size=size, dtype=np.uint8)
    matrix = sparse.csc_matrix(dense, dtype=np.int8)

    try:
        do_pivot_sparse_numba(matrix)
    except RuntimeError:
        pass
