"""Benchmark helpers for persistence diagram computation."""

from __future__ import annotations

import statistics
import time
from typing import Any, Sequence, Union

import numpy as np
import fire

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from ripser import ripser
except ImportError:
    ripser = None

try:
    import gudhi as gd
except ImportError:
    gd = None

try:
    from persistent_cost.utils.utils import htr
except ImportError:
    htr = None


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
            values = [int(p) for p in points]
        except TypeError as exc:
            raise TypeError(
                "points must be an int, an iterable of ints, or a comma-separated string"
            ) from exc
        if not values:
            raise ValueError("No point counts provided.")

    return tuple(sorted(set(values)))


def _generate_point_cloud(
    n_points: int,
    ambient_dim: int,
    seed: int,
) -> np.ndarray:
    """Generate random point cloud."""
    rng = np.random.default_rng(seed)
    points = rng.random((n_points, ambient_dim))
    return points


def _progress_iterator(repeats: int, description: str, progress: bool):
    if progress and tqdm is not None:
        return tqdm(range(repeats), desc=description, leave=False)
    if progress:
        print(f"{description}: starting {repeats} repetitions")
    return range(repeats)


def _compute_ripser(points: np.ndarray, maxdim: int) -> dict[str, Any]:
    """Compute persistence using ripser."""
    if ripser is None:
        raise RuntimeError("ripser is not available")
    result = ripser(points, maxdim=maxdim)
    return result


def _compute_gudhi(points: np.ndarray, maxdim: int, threshold: float) -> list[tuple[int, tuple[float, float]]]:
    """Compute persistence using gudhi."""
    if gd is None:
        raise RuntimeError("gudhi is not available")
    
    rips_complex = gd.RipsComplex(points=points, max_edge_length=threshold)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=maxdim + 1)
    persistence = simplex_tree.persistence()
    
    return persistence


def _compute_htr(points: np.ndarray, maxdim: int, threshold: float) -> tuple[list, list, list]:
    """Compute persistence using htr."""
    if htr is None:
        raise RuntimeError("htr is not available")
    
    # HTR is a function that returns (births, deaths, dims)
    births, deaths, dims = htr(points=points, threshold=threshold, maxdim=maxdim)
    # Convert to lists with births_deaths_to_dgm
    from persistent_cost.utils.utils import births_deaths_to_dgm
    diagram = births_deaths_to_dgm(births, deaths, dims, maxdim)
    return diagram


def _time_algorithm(
    func,
    *args,
    repeats: int,
    description: str,
    progress: bool,
    timeout: float = 30.0,
) -> tuple[float, Any]:
    """Time an algorithm with multiple repetitions."""
    iterator = _progress_iterator(repeats, description, progress)
    durations: list[float] = []
    result: Any = None
    
    for _ in iterator:
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        durations.append(end - start)
        
        if durations[-1] > timeout:
            break
    
    if progress and tqdm is None:
        print(f"{description}: completed")
    
    return statistics.median(durations), result


def _normalize_diagram(diagram: Any, library: str, maxdim: int) -> dict[int, list[tuple[float, float]]]:
    """Normalize persistence diagrams to common format: {dim: [(birth, death), ...]}"""
    result: dict[int, list[tuple[float, float]]] = {i: [] for i in range(maxdim + 1)}
    
    if library == "ripser":
        for dim in range(maxdim + 1):
            if dim < len(diagram['dgms']):
                for birth, death in diagram['dgms'][dim]:
                    result[dim].append((float(birth), float(death)))
    
    elif library == "gudhi":
        for dim, (birth, death) in diagram:
            if dim <= maxdim:
                result[dim].append((float(birth), float(death)))
    
    elif library == "htr":
        # HTR returns a list of arrays (similar to ripser's dgms)
        for dim in range(maxdim + 1):
            if dim < len(diagram):
                for birth, death in diagram[dim]:
                    result[dim].append((float(birth), float(death)))
    
    # Sort each dimension's bars by birth time
    for dim in result:
        result[dim].sort()
    
    return result


def _compare_diagrams(
    diagram1: dict[int, list[tuple[float, float]]],
    diagram2: dict[int, list[tuple[float, float]]],
    tolerance: float = 1e-6,
) -> tuple[bool, str]:
    """Compare two persistence diagrams."""
    if set(diagram1.keys()) != set(diagram2.keys()):
        return False, f"Different dimensions: {diagram1.keys()} vs {diagram2.keys()}"
    
    for dim in diagram1:
        bars1 = diagram1[dim]
        bars2 = diagram2[dim]
        
        if len(bars1) != len(bars2):
            return False, f"Dimension {dim}: different number of bars ({len(bars1)} vs {len(bars2)})"
        
        for i, ((b1, d1), (b2, d2)) in enumerate(zip(bars1, bars2)):
            if abs(b1 - b2) > tolerance or abs(d1 - d2) > tolerance:
                return False, f"Dimension {dim}, bar {i}: ({b1:.6f}, {d1:.6f}) vs ({b2:.6f}, {d2:.6f})"
    
    return True, "Diagrams match"


def run_benchmark(
    point_counts: Sequence[int],
    repeats: int,
    ambient_dim: int,
    threshold: float,
    maxdim: int,
    seed: int,
    progress: bool,
) -> None:
    """Run benchmark comparing ripser, gudhi, and htr."""
    
    # Check available libraries
    available_libs = []
    if ripser is not None:
        available_libs.append("ripser")
    if gd is not None:
        available_libs.append("gudhi")
    if htr is not None:
        available_libs.append("htr")
    
    if not available_libs:
        print("ERROR: No persistence libraries available!")
        return
    
    print("Benchmark parameters:")
    print(f"  points: {list(point_counts)}")
    print(f"  repeats: {repeats}")
    print(f"  ambient_dim: {ambient_dim}")
    print(f"  threshold: {threshold}")
    print(f"  maxdim: {maxdim}")
    print(f"  seed: {seed}")
    print(f"  progress: {progress}")
    print(f"  available libraries: {available_libs}")
    
    results: list[dict[str, float | int | str]] = []
    
    for idx, n_points in enumerate(point_counts):
        print(f"\n=== Point cloud with {n_points} points ===")
        
        points = _generate_point_cloud(
            n_points=n_points,
            ambient_dim=ambient_dim,
            seed=seed + idx,
        )
        
        diagrams: dict[str, dict[int, list[tuple[float, float]]]] = {}
        
        # Ripser
        if "ripser" in available_libs:
            print(f"  Running ripser...")
            median_time, ripser_result = _time_algorithm(
                _compute_ripser,
                points,
                maxdim,
                repeats=repeats,
                description=f"n={n_points} | ripser",
                progress=progress,
            )
            diagrams["ripser"] = _normalize_diagram(ripser_result, "ripser", maxdim)
            
            results.append({
                "points": n_points,
                "library": "ripser",
                "median": median_time,
                "speedup": 1.0,
            })
            print(f"    Time: {median_time:.6f} s")
        
        # Gudhi
        if "gudhi" in available_libs:
            print(f"  Running gudhi...")
            median_time, gudhi_result = _time_algorithm(
                _compute_gudhi,
                points,
                maxdim,
                threshold,
                repeats=repeats,
                description=f"n={n_points} | gudhi",
                progress=progress,
            )
            diagrams["gudhi"] = _normalize_diagram(gudhi_result, "gudhi", maxdim)
            
            baseline = results[0]["median"] if results else median_time
            speedup = baseline / median_time
            
            results.append({
                "points": n_points,
                "library": "gudhi",
                "median": median_time,
                "speedup": speedup,
            })
            print(f"    Time: {median_time:.6f} s")
        
        # HTR
        if "htr" in available_libs:
            print(f"  Running htr...")
            median_time, htr_result = _time_algorithm(
                _compute_htr,
                points,
                maxdim,
                threshold,
                repeats=repeats,
                description=f"n={n_points} | htr",
                progress=progress,
            )
            diagrams["htr"] = _normalize_diagram(htr_result, "htr", maxdim)
            
            baseline = results[0]["median"] if results else median_time
            speedup = baseline / median_time
            
            results.append({
                "points": n_points,
                "library": "htr",
                "median": median_time,
                "speedup": speedup,
            })
            print(f"    Time: {median_time:.6f} s")
        
        # Compare results
        if len(diagrams) > 1:
            print(f"\n  Comparing results for {n_points} points:")
            libs = list(diagrams.keys())
            for i in range(len(libs)):
                for j in range(i + 1, len(libs)):
                    lib1, lib2 = libs[i], libs[j]
                    match, msg = _compare_diagrams(diagrams[lib1], diagrams[lib2])
                    status = "✓ MATCH" if match else "✗ DIFFER"
                    print(f"    {lib1} vs {lib2}: {status}")
                    if not match:
                        print(f"      {msg}")
                    else:
                        # Print summary of matched features
                        total_bars = sum(len(bars) for bars in diagrams[lib1].values())
                        print(f"      {total_bars} features matched across all dimensions")
    
    _print_results_table(results)


def _print_results_table(rows: list[dict[str, float | int | str]]) -> None:
    """Print results table."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    header = f"{'points':>8}  {'library':>10}  {'median (s)':>12}  {'speedup':>8}"
    print(header)
    print("-" * len(header))
    
    for row in rows:
        points = row["points"]
        library = row["library"]
        median = row["median"]
        speedup = row["speedup"]
        print(f"{points:8d}  {library:>10}  {median:12.6f}  {speedup:8.2f}")


def benchmark(
    points: Union[Sequence[int], str, int] = (10, 20, 50),
    repeats: int = 5,
    ambient_dim: int = 3,
    threshold: float = 2.0,
    maxdim: int = 1,
    seed: int = 1337,
    progress: bool = True,
) -> None:
    """Benchmark persistence computation across different libraries.
    
    Args:
        points: Point cloud sizes to test (default: 10, 20, 50)
        repeats: Number of repetitions per test (default: 5)
        ambient_dim: Ambient dimension of point clouds (default: 3)
        threshold: Maximum edge length for Rips complex (default: 2.0)
        maxdim: Maximum homology dimension (default: 1)
        seed: Random seed for reproducibility (default: 1337)
        progress: Show progress bars (default: True)
    """
    point_counts = _normalise_points(points)
    run_benchmark(
        point_counts=point_counts,
        repeats=repeats,
        ambient_dim=ambient_dim,
        threshold=threshold,
        maxdim=maxdim,
        seed=seed,
        progress=progress,
    )


if __name__ == "__main__":
    fire.Fire({"benchmark": benchmark})
