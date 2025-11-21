"""Optimised dense pivot variants using optional accelerators."""

import numpy as np
from scipy import sparse
from typing import Literal, Iterable, Tuple, Optional

BackendLiteral = Literal["numba", "cython"]

try:  # Optional acceleration dependency.
    from numba import njit
except ImportError:  # pragma: no cover - executed when numba missing.
    njit = None
    NUMBA_AVAILABLE = False
else:
    NUMBA_AVAILABLE = True

try:  # Optional Cython extension.
    from . import _pivot_dense_cython as _pivot_cython_module
except ImportError:  # pragma: no cover - executed when extension missing.
    _pivot_cython_module = None  # type: ignore[assignment]
    CYTHON_AVAILABLE = False
else:
    CYTHON_AVAILABLE = True


def _ensure_binary_dense(matrix: sparse.spmatrix) -> np.ndarray:
    """Return a C-contiguous uint8 copy of ``matrix`` reduced modulo two."""

    if not sparse.isspmatrix(matrix):
        raise TypeError("Expected a scipy.sparse matrix as input")
    dense = np.asarray(matrix.toarray(order="C"), dtype=np.uint8)
    dense &= 1
    return dense


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _column_pivot_k_dense_numba(reduced: np.ndarray, column: int) -> int:
        pivot = -1
        rows = reduced.shape[0]
        for row in range(rows - 1, -1, -1):
            if reduced[row, column] & 1:
                pivot = row
                break
        return pivot

    @njit(cache=True)
    def _find_first_with_pivot(cpivs: np.ndarray, pivot: int) -> int:
        n = cpivs.shape[0]
        for idx in range(n):
            if cpivs[idx] == pivot:
                return idx
        return -1

    @njit(cache=True)
    def _add_column_mod2(matrix: np.ndarray, source: int, target: int) -> None:
        rows = matrix.shape[0]
        for row in range(rows):
            matrix[row, target] ^= matrix[row, source]

    @njit(cache=True)
    def _do_pivot_numba_core(reduced: np.ndarray, transform: np.ndarray) -> None:
        n_rows, n_cols = reduced.shape
        cpivs = np.empty(n_cols, dtype=np.int64)
        used = np.zeros(n_rows, dtype=np.uint8)

        for column in range(n_cols):
            cpivs[column] = _column_pivot_k_dense_numba(reduced, column)

        for column in range(n_cols):
            pivot = cpivs[column]
            while pivot != -1 and pivot < n_rows and used[pivot] == 1:
                other = _find_first_with_pivot(cpivs, pivot)
                if other < 0:
                    break
                _add_column_mod2(reduced, other, column)
                _add_column_mod2(transform, other, column)
                cpivs[column] = _column_pivot_k_dense_numba(reduced, column)
                pivot = cpivs[column]

            if pivot != -1 and pivot < n_rows and used[pivot] == 0:
                used[pivot] = 1


def do_pivot_numba(matrix: sparse.spmatrix) -> Tuple[sparse.csc_matrix, sparse.csc_matrix]:
    """Perform the pivot reduction using the Numba backend."""

    if not NUMBA_AVAILABLE:  # pragma: no cover - simple import guard.
        raise RuntimeError(
            "Numba is not installed. Install the optional 'accel' extra or add numba manually."
        )

    dense = _ensure_binary_dense(matrix)
    reduced = np.array(dense, copy=True, order="C", dtype=np.uint8)
    n_cols = reduced.shape[1]
    transform = np.zeros((n_cols, n_cols), dtype=np.uint8)
    np.fill_diagonal(transform, 1)
    _do_pivot_numba_core(reduced, transform)
    return sparse.csc_matrix(reduced, dtype=np.int8), sparse.csc_matrix(transform, dtype=np.int8)


def do_pivot_cython(matrix: sparse.spmatrix) -> Tuple[sparse.csc_matrix, sparse.csc_matrix]:
    """Perform the pivot reduction using the Cython backend."""

    if not CYTHON_AVAILABLE:  # pragma: no cover - simple import guard.
        raise RuntimeError(
            "Cython extension '_pivot_dense_cython' is not built. Run 'pip install -e .[accel]' or "
            "'python setup.py build_ext --inplace' to compile it."
        )

    dense = _ensure_binary_dense(matrix)
    reduced_dense, transform_dense = _pivot_cython_module.do_pivot_binary(dense)
    return (
        sparse.csc_matrix(reduced_dense, dtype=np.int8),
        sparse.csc_matrix(transform_dense, dtype=np.int8),
    )


def do_pivot_fast(
    matrix: sparse.spmatrix, backend: BackendLiteral = "numba"
) -> Tuple[sparse.csc_matrix, sparse.csc_matrix]:
    """Dispatch to the chosen accelerated backend."""

    backend_lc = backend.lower()
    if backend_lc == "numba":
        return do_pivot_numba(matrix)
    if backend_lc == "cython":
        return do_pivot_cython(matrix)
    raise ValueError(f"Unknown backend '{backend}'. Valid choices: numba, cython")


def warm_up(backends: Optional[Iterable[str]] = None, size: Tuple[int, int] = (32, 32)) -> None:
    """Trigger JIT compilation for the selected backends using a synthetic matrix."""

    rng = np.random.default_rng(1234)
    matrix = sparse.csc_matrix(rng.integers(0, 2, size=size, dtype=np.uint8))
    try:
        do_pivot_numba(matrix)
    except RuntimeError:
        pass
