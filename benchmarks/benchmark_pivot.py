"""Benchmark helpers for the pivot implementations."""

from __future__ import annotations

import statistics
import sys
import time
from typing import Any, Iterable, Sequence, Union

import numpy as np
from scipy import sparse

import fire
from persistent_cost.algorithms import available_backends

from persistent_cost.algorithms.dense_fast import do_pivot_numba as do_pivot_dense_numba
from persistent_cost.algorithms.dense_fast import do_pivot_cython as do_pivot_dense_cython
from persistent_cost.algorithms.dense import do_pivot as do_pivot_numpy
from persistent_cost.algorithms.sparse_fast import do_pivot_numba as do_pivot_sparse_numba
from persistent_cost.algorithms.sparse_fast import do_pivot_cython as do_pivot_sparse_cython
from persistent_cost.algorithms.sparse import do_pivot as do_pivot_sparse_numpy

from persistent_cost.utils.utils import build_ordered_boundary_matrix

try:  # optional progress bar dependency
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None

try:  # optional persistent homology baseline
    from ripser import ripser
except ImportError:  # pragma: no cover - ripser is optional
    ripser = None


def _normalise_points(points: Union[Sequence[int], str, int]) -> tuple[int, ...]:
    if isinstance(points, str):
        tokens = [token for token in points.replace(",", " ").split() if token]
        if not tokens:
            raise ValueError("No point counts provided.")
        values = [int(token) for token in tokens]
    elif isinstance(points, int):
        values = [points]
    else:
        try:
            values = [int(p) for p in points]  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError(
                "points must be an int, an iterable of ints, or a comma-separated string"
            ) from exc
        if not values:
            raise ValueError("No point counts provided.")

    return tuple(sorted(set(values)))


def _build_boundary_matrix(
    n_points: int,
    ambient_dim: int,
    threshold: float,
    maxdim: int,
    seed: int,
) -> tuple[sparse.csc_matrix, np.ndarray]:
    rng = np.random.default_rng(seed)
    points = rng.random((n_points, ambient_dim))
    matrix, _ = build_ordered_boundary_matrix(
        points=points,
        threshold=threshold,
        maxdim=maxdim,
        field=np.int8,
    )
    return matrix.tocsc(), points


def _progress_iterator(repeats: int, description: str, progress: bool):
    if progress and tqdm is not None:
        return tqdm(range(repeats), desc=description, leave=False)
    if progress:
        print(f"{description}: starting {repeats} repetitions")
    return range(repeats)


def _time_algorithm_with_progress(
    func,
    matrix: sparse.csc_matrix,
    repeats: int,
    description: str,
    progress: bool,
    timeout: float = 10,
) -> tuple[float, tuple[sparse.csc_matrix, sparse.csc_matrix]]:
    iterator = _progress_iterator(repeats, description, progress)
    durations: list[float] = []
    result: tuple[sparse.csc_matrix, sparse.csc_matrix] | None = None
    for _ in iterator:
        start = time.perf_counter()
        result = func(matrix)
        end = time.perf_counter()
        durations.append(end - start)
        if durations[-1] > timeout:
            break
    assert result is not None

    if progress and tqdm is None:
        print(f"{description}: completed")

    return statistics.median(durations), result


def _time_ripser_with_progress(
    points: np.ndarray,
    repeats: int,
    description: str,
    progress: bool,
    timeout: float = 10,
) -> tuple[float, dict[str, Any]]:
    if ripser is None:
        raise RuntimeError("ripser is not available")
    iterator = _progress_iterator(repeats, description, progress)
    durations: list[float] = []
    result: dict[str, Any] | None = None
    for _ in iterator:
        start = time.perf_counter()
        result = ripser(points)
        end = time.perf_counter()
        durations.append(end - start)
        if durations[-1] > timeout:
            break
    assert result is not None
    if progress and tqdm is None:
        print(f"{description}: completed")
    return statistics.median(durations), result


def _assert_factorisation(
    matrix: sparse.csc_matrix,
    reduced: sparse.csc_matrix,
    transform: sparse.csc_matrix,
) -> None:
    product = (matrix @ transform).toarray() % 2
    reduced_dense = reduced.toarray() % 2
    if not np.array_equal(product, reduced_dense):
        raise AssertionError("Reduction check failed: M @ V != R (mod 2)")


def run_benchmark(
    point_counts: Sequence[int],
    repeats: int,
    ambient_dim: int,
    thresholds: Sequence[float],
    maxdim: int,
    seed: int,
    progress: bool,
) -> None:
    backends = tuple(available_backends())

    print("Benchmark parameters:")
    print(f"  points: {list(point_counts)}")
    print(f"  repeats: {repeats}")
    print(f"  ambient_dim: {ambient_dim}")
    print(f"  thresholds: {list(thresholds)}")
    print(f"  maxdim: {maxdim}")
    print(f"  seed: {seed}")
    print(f"  progress: {progress}")
    print(f"  backends: {backends or 'none'}")
    print(
        f"  ripser baseline: {'available' if ripser is not None else 'missing'}")

    results: list[dict[str, float | int | str]] = []

    # Test ripser, dense numpy, sparse numpy, dense numba, sparse numba, dense cython, sparse cython,
    algorithms: list[tuple[str, Any]] = []
    if ripser is not None:
        algorithms.append(("ripser", "sparse", 'ripser'))
    algorithms.append(("numpy", "dense", do_pivot_numpy))
    algorithms.append(("numpy", "sparse", do_pivot_sparse_numpy))
    if "cython" in backends:
        algorithms.append(("cython", "dense", do_pivot_dense_cython))
        algorithms.append(("cython", "sparse", do_pivot_sparse_cython))
    if "numba" in backends:
        algorithms.append(("numba", "dense", do_pivot_dense_numba))
        algorithms.append(("numba", "sparse", do_pivot_sparse_numba))

    from itertools import product
    for idx, (n_points, threshold_value) in enumerate(product(point_counts, thresholds)):
        print(
            f"\n=== Point cloud with {n_points} points and threshold {threshold_value} ===")

        matrix, points = _build_boundary_matrix(
            n_points=n_points,
            ambient_dim=ambient_dim,
            threshold=threshold_value,
            maxdim=maxdim,
            seed=seed + idx,
        )
        matrix = matrix.astype(np.int8)

        rows, cols = matrix.shape
        nnz = matrix.nnz        

        for library, variant, func in algorithms:

            print(f"    Variant: {variant}")
            print(f"    Points: {n_points}")
            print(f"    Threshold: {threshold_value}")
            print(f"    Boundary matrix size: {rows} x {cols}")
            print(f"    Non-zero entries: {nnz}")
            print(f"    Repetitions per algorithm: {repeats}")

            # warm up JIT backends
            if library == "numba":
                if variant == "dense":
                    do_pivot_dense_numba(matrix)
                else:
                    do_pivot_sparse_numba(matrix)

            baseline_time = None

            description = f"n={n_points} | {variant} | {library}"
            if nnz > 200 and library == "python":
                continue
            if nnz > 10000 and library in ("numba", "cython"):
                continue

            print(f"    Running algorithm: {library} ({variant})")

            if func != 'ripser':
                median_time, (reduced, transform) = _time_algorithm_with_progress(
                    func,
                    matrix,
                    repeats,
                    description,
                    progress,
                )
                _assert_factorisation(matrix, reduced, transform)
            else:
                median_time, ripser_result = _time_ripser_with_progress(
                    points,
                    repeats,
                    description,
                    progress,
                )

            if library == "numpy":
                baseline_time = median_time

            speedup = 1.0 if baseline_time is None else baseline_time / median_time

            results.append(
                {
                    "points": n_points,
                    "simplices": rows,
                    "nnz": nnz,
                    "threshold": float(threshold_value),
                    "library": library,
                    "variant": variant,
                    "median": median_time,
                    "speedup": speedup,
                }
            )

    _print_results_table(results)


def _print_results_table(rows: Iterable[dict[str, float | int | str]]) -> None:
    header = (
        f"{'points':>8}  {'simplices':>9}  {'nnz':>10}  {'variant':>8}  "
        f"{'threshold':>10}  {'library':>10}  {'median (s)':>12}  {'speedup':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        points = row["points"]
        simplices = row["simplices"]
        nnz = row["nnz"]
        variant = row["variant"]
        threshold = row["threshold"]
        library = row["library"]
        median = row["median"]
        speedup = row["speedup"]
        print(
            f"{points:8d}  {simplices:9d}  {nnz:10d}  {variant:>8}  {threshold:10.3f}  "
            f"{library:>10}  {median:12.6f}  {speedup:8.2f}"
        )


def benchmark(
    points: Union[Sequence[int], str, int] = (10, 30, 50),
    repeats: int = 5,
    ambient_dim: int = 3,
    thresholds: float = (0.5, 0.6),
    maxdim: int = 1,
    seed: int = 1337,
    progress: bool = True,
) -> None:
    """Benchmark pivot implementations across different point-cloud sizes."""

    point_counts = _normalise_points(points)
    run_benchmark(
        point_counts=point_counts,
        repeats=repeats,
        ambient_dim=ambient_dim,
        thresholds=thresholds,
        maxdim=maxdim,
        seed=seed,
        progress=progress,
    )


if __name__ == "__main__":  # pragma: no cover - manual benchmarking utility.
    fire.Fire({"benchmark": benchmark})
