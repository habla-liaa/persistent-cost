"""Baseline dense pivot implementation."""

import numpy as np
from scipy import sparse


def column_pivot_k(matrix: np.ndarray, column: int) -> int:
    """Return the pivot index for ``column`` or ``-1`` if the column is zero."""

    rows = matrix.shape[0]
    pivot = -1
    for row in range(rows):
        if matrix[row, column] != 0:
            pivot = row
    return pivot


def do_pivot(matrix: sparse.spmatrix) -> tuple[sparse.csc_matrix, sparse.csc_matrix]:
    """Compute the boundary matrix reduction using dense arithmetic."""

    dense_matrix = np.asarray(matrix.todense())
    reduced = dense_matrix.copy()
    n_cols = reduced.shape[1]
    pivots = [column_pivot_k(dense_matrix, column) for column in range(n_cols)]
    used_pivots: list[int] = []
    operations: list[tuple[int, int]] = []

    for column in range(n_cols):
        pivot = pivots[column]
        while pivot in used_pivots and pivot != -1:
            other = pivots.index(pivot)
            reduced[:, column] = (reduced[:, column] + reduced[:, other]) % 2
            pivots[column] = column_pivot_k(reduced, column)
            pivot = pivots[column]
            operations.append((other, column))

        if pivot != -1 and pivot not in used_pivots:
            used_pivots.append(pivot)

    transform = np.eye(n_cols, dtype=int)
    for source, target in operations:
        transform[:, target] = (transform[:, target] + transform[:, source]) % 2

    return sparse.csc_matrix(reduced), sparse.csc_matrix(transform)
