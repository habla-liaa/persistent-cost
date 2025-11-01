"""Algorithms for persistent_cost: dense, sparse, and accelerated variants."""

from __future__ import annotations

from typing import Sequence

from .dense import do_pivot as dense_do_pivot
from .dense_fast import (
    CYTHON_AVAILABLE as DENSE_CYTHON_AVAILABLE,
    NUMBA_AVAILABLE as DENSE_NUMBA_AVAILABLE,
    do_pivot_cython,
    do_pivot_fast,
    do_pivot_numba,
    warm_up as warm_up_dense,
)
from .sparse import do_pivot as sparse_do_pivot
from .sparse_fast import (
    CYTHON_AVAILABLE as SPARSE_CYTHON_AVAILABLE,
    NUMBA_AVAILABLE as SPARSE_NUMBA_AVAILABLE,
    do_pivot_cython,
    do_pivot_fast,
    do_pivot_numba,
    warm_up_sparse,
)


def available_backends() -> Sequence[str]:
    """Return the list of acceleration backends that are ready to use."""

    backends: list[str] = []
    if DENSE_NUMBA_AVAILABLE or SPARSE_NUMBA_AVAILABLE:
        backends.append("numba")

    if DENSE_CYTHON_AVAILABLE and SPARSE_CYTHON_AVAILABLE:
        backends.append("cython")

    return tuple(backends)


__all__ = [
    "do_pivot_cython",
    "do_pivot_fast",
    "do_pivot_numba",
    "available_backends",
    "warm_up_dense",
    "warm_up_sparse",
]
