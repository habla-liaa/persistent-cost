"""Baseline sparse pivot implementation and helpers."""

import numpy as np
from scipy import sparse


def births_and_deaths(
    reduced: sparse.spmatrix,
    matrix: sparse.spmatrix,
    epsilons: np.ndarray,
    maxdim: int = 1,
):
    """Compute birth and death times from a reduced boundary matrix."""

    n_cols = reduced.shape[1]
    cpivs = [column_pivot_k(reduced, column) for column in range(n_cols)]
    births: list[float] = []
    deaths: list[float] = []
    dims: list[int] = []

    for column in range(n_cols):
        column_slice = matrix[: column + 1, column].tocsc()
        dim = column_slice.count_nonzero() - 1

        if dim <= maxdim:
            pivot = cpivs[column]
            if pivot == -1:
                if column in cpivs:
                    pivot_col = cpivs.index(column)
                    if epsilons[column] < epsilons[pivot_col]:
                        births.append(epsilons[column])
                        deaths.append(epsilons[pivot_col])
                        dims.append(dim if dim > 0 else 0)
                elif dim < maxdim:
                    births.append(epsilons[column])
                    deaths.append(np.inf)
                    dims.append(dim if dim > 0 else 0)

    return births, deaths, dims


def column_pivot_k(matrix: sparse.spmatrix, column: int) -> int:
    """Return the maximum row index of ``column`` or ``-1`` if it is zero."""

    csc = matrix.tocsc()
    start = csc.indptr[column]
    end = csc.indptr[column + 1]
    rows = csc.indices[start:end]
    if rows.size > 0:
        return int(rows.max())
    return -1


def column_pivots(matrix: sparse.spmatrix) -> list[int]:
    """Return all column pivots for ``matrix``."""

    pivots: list[int] = []
    csc = matrix.tocsc()
    for start, end in zip(csc.indptr[:-1], csc.indptr[1:]):
        if start == end:
            pivots.append(-1)
        else:
            pivots.append(int(csc.indices[end - 1]))
    return pivots


def do_pivot(matrix: sparse.spmatrix) -> tuple[sparse.csc_matrix, sparse.csc_matrix]:
    """Reduce ``matrix`` modulo two using sparse column operations."""

    reduced = matrix.copy().tolil()
    n_cols = reduced.shape[1]
    cpivs = [column_pivot_k(reduced.tocsc(), column) for column in range(n_cols)]
    used_pivots: list[int] = []
    operations: list[tuple[int, int]] = []

    for column in range(n_cols):
        pivot = cpivs[column]
        while pivot in used_pivots and pivot != -1:
            pivot_col = cpivs.index(pivot)
            col_k = reduced[:, column].toarray().ravel()
            col_j = reduced[:, pivot_col].toarray().ravel()
            col_new = (col_k + col_j) % 2
            reduced[:, column] = sparse.csc_matrix(col_new).T
            cpivs[column] = column_pivot_k(reduced, column)
            pivot = cpivs[column]
            operations.append((pivot_col, column))

        if pivot != -1 and pivot not in used_pivots:
            used_pivots.append(pivot)

    transform = sparse.identity(n_cols, format="lil", dtype=int)
    for source, target in operations:
        col_target = transform[:, target].toarray().ravel()
        col_source = transform[:, source].toarray().ravel()
        transform[:, target] = sparse.csc_matrix((col_target + col_source) % 2).T

    return reduced.tocsc(), transform.tocsc()
