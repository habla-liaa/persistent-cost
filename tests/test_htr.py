"""
Tests for the htr function comparing results with Ripser.

The htr function computes persistent homology using custom pivot algorithm,
and results should approximately match Ripser's output.
"""

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform
from ripser import ripser

from persistent_cost.utils.utils import htr


def persistence_diagrams_to_arrays(births, deaths, dims, maxdim):
    """
    Convert births, deaths, dims lists to persistence diagrams format like Ripser.
    
    Returns a list of arrays, one per dimension (0 to maxdim).
    Each array has shape (n_features, 2) with columns [birth, death].
    """
    dgms = []
    for d in range(maxdim + 1):
        # Filter features of dimension d
        mask = np.array(dims) == d
        b = np.array(births)[mask]
        dt = np.array(deaths)[mask]
        
        # Stack into (n, 2) array
        if len(b) > 0:
            dgm = np.column_stack([b, dt])
        else:
            dgm = np.empty((0, 2))
        
        dgms.append(dgm)
    
    return dgms


def compare_persistence_diagrams(dgms1, dgms2, maxdim, tolerance=1e-6):
    """
    Compare two sets of persistence diagrams.
    
    Args:
        dgms1: list of arrays (one per dimension)
        dgms2: list of arrays (one per dimension)
        maxdim: maximum dimension to compare
        tolerance: numerical tolerance for comparison
    
    Returns:
        bool: True if diagrams match within tolerance
        str: description of differences if any
    """
    for d in range(maxdim + 1):
        dgm1 = dgms1[d]
        dgm2 = dgms2[d]
        
        # Sort by birth time, then death time
        dgm1_sorted = dgm1[np.lexsort((dgm1[:, 1], dgm1[:, 0]))]
        dgm2_sorted = dgm2[np.lexsort((dgm2[:, 1], dgm2[:, 0]))]
        
        # Check if shapes match
        if dgm1_sorted.shape != dgm2_sorted.shape:
            return False, f"Dimension {d}: shape mismatch {dgm1_sorted.shape} vs {dgm2_sorted.shape}"
        
        # Check if values match within tolerance
        if not np.allclose(dgm1_sorted, dgm2_sorted, rtol=tolerance, atol=tolerance):
            max_diff = np.max(np.abs(dgm1_sorted - dgm2_sorted))
            return False, f"Dimension {d}: values differ by up to {max_diff}"
    
    return True, "All diagrams match"


def test_htr_vs_ripser_triangle():
    """
    Test htr vs Ripser on a simple triangle.
    """
    print("\n" + "=" * 80)
    print("TEST: htr vs Ripser - Triangle (3 points)")
    print("=" * 80)
    
    # Create a triangle
    X = np.array([[0, 0], [1, 0], [0, 1]])
    threshold = 2.0
    maxdim = 1
    
    print(f"\nPoints:\n{X}")
    print(f"Threshold: {threshold}, Max dimension: {maxdim}")
    
    # Compute with htr
    births_htr, deaths_htr, dims_htr = htr(points=X, threshold=threshold, maxdim=maxdim)
    dgms_htr = persistence_diagrams_to_arrays(births_htr, deaths_htr, dims_htr, maxdim)
    
    print(f"\nHTR results:")
    for d in range(maxdim + 1):
        print(f"  H_{d}: {len(dgms_htr[d])} features")
        if len(dgms_htr[d]) > 0:
            print(f"    {dgms_htr[d]}")
    
    # Compute with Ripser
    result_ripser = ripser(X, maxdim=maxdim, thresh=threshold)
    dgms_ripser = result_ripser['dgms']
    
    print(f"\nRipser results:")
    for d in range(maxdim + 1):
        print(f"  H_{d}: {len(dgms_ripser[d])} features")
        if len(dgms_ripser[d]) > 0:
            print(f"    {dgms_ripser[d]}")
    
    # Compare
    match, msg = compare_persistence_diagrams(dgms_htr, dgms_ripser, maxdim, tolerance=1e-5)
    
    print(f"\nComparison: {msg}")
    print("=" * 80)
    
    assert match, f"Persistence diagrams don't match: {msg}"


def test_htr_vs_ripser_square():
    """
    Test htr vs Ripser on a square.
    """
    print("\n" + "=" * 80)
    print("TEST: htr vs Ripser - Square (4 points)")
    print("=" * 80)
    
    # Create a square
    X = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    threshold = 2.0
    maxdim = 1
    
    print(f"\nPoints:\n{X}")
    print(f"Threshold: {threshold}, Max dimension: {maxdim}")
    
    # Compute with htr
    births_htr, deaths_htr, dims_htr = htr(points=X, threshold=threshold, maxdim=maxdim)
    dgms_htr = persistence_diagrams_to_arrays(births_htr, deaths_htr, dims_htr, maxdim)
    
    print(f"\nHTR results:")
    for d in range(maxdim + 1):
        print(f"  H_{d}: {len(dgms_htr[d])} features")
        if len(dgms_htr[d]) > 0:
            print(f"    {dgms_htr[d]}")
    
    # Compute with Ripser
    result_ripser = ripser(X, maxdim=maxdim, thresh=threshold)
    dgms_ripser = result_ripser['dgms']
    
    print(f"\nRipser results:")
    for d in range(maxdim + 1):
        print(f"  H_{d}: {len(dgms_ripser[d])} features")
        if len(dgms_ripser[d]) > 0:
            print(f"    {dgms_ripser[d]}")
    
    # Compare
    match, msg = compare_persistence_diagrams(dgms_htr, dgms_ripser, maxdim, tolerance=1e-5)
    
    print(f"\nComparison: {msg}")
    print("=" * 80)
    
    assert match, f"Persistence diagrams don't match: {msg}"


def test_htr_vs_ripser_circle_points():
    """
    Test htr vs Ripser on points sampled from a circle.
    """
    print("\n" + "=" * 80)
    print("TEST: htr vs Ripser - Circle (10 points)")
    print("=" * 80)
    
    # Create points on a circle
    n = 10
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X = np.column_stack([np.cos(theta), np.sin(theta)])
    
    threshold = 2.0
    maxdim = 1
    
    print(f"\nPoints (first 5):\n{X[:5]}")
    print(f"... ({n} points total)")
    print(f"Threshold: {threshold}, Max dimension: {maxdim}")
    
    # Compute with htr
    births_htr, deaths_htr, dims_htr = htr(points=X, threshold=threshold, maxdim=maxdim)
    dgms_htr = persistence_diagrams_to_arrays(births_htr, deaths_htr, dims_htr, maxdim)
    
    print(f"\nHTR results:")
    for d in range(maxdim + 1):
        print(f"  H_{d}: {len(dgms_htr[d])} features")
        if len(dgms_htr[d]) > 0 and len(dgms_htr[d]) <= 5:
            print(f"    {dgms_htr[d]}")
        elif len(dgms_htr[d]) > 5:
            print(f"    (showing first 5)")
            print(f"    {dgms_htr[d][:5]}")
    
    # Compute with Ripser
    result_ripser = ripser(X, maxdim=maxdim, thresh=threshold)
    dgms_ripser = result_ripser['dgms']
    
    print(f"\nRipser results:")
    for d in range(maxdim + 1):
        print(f"  H_{d}: {len(dgms_ripser[d])} features")
        if len(dgms_ripser[d]) > 0 and len(dgms_ripser[d]) <= 5:
            print(f"    {dgms_ripser[d]}")
        elif len(dgms_ripser[d]) > 5:
            print(f"    (showing first 5)")
            print(f"    {dgms_ripser[d][:5]}")
    
    # Compare
    match, msg = compare_persistence_diagrams(dgms_htr, dgms_ripser, maxdim, tolerance=1e-5)
    
    print(f"\nComparison: {msg}")
    print("=" * 80)
    
    assert match, f"Persistence diagrams don't match: {msg}"


def test_htr_vs_ripser_random_2d():
    """
    Test htr vs Ripser on random 2D points.
    """
    print("\n" + "=" * 80)
    print("TEST: htr vs Ripser - Random 2D points (15 points)")
    print("=" * 80)
    
    # Create random points
    np.random.seed(42)
    n = 15
    X = np.random.rand(n, 2)
    
    threshold = 1.5
    maxdim = 1
    
    print(f"\nPoints (first 5):\n{X[:5]}")
    print(f"... ({n} points total)")
    print(f"Threshold: {threshold}, Max dimension: {maxdim}")
    
    # Compute with htr
    births_htr, deaths_htr, dims_htr = htr(points=X, threshold=threshold, maxdim=maxdim)
    dgms_htr = persistence_diagrams_to_arrays(births_htr, deaths_htr, dims_htr, maxdim)
    
    print(f"\nHTR results:")
    for d in range(maxdim + 1):
        print(f"  H_{d}: {len(dgms_htr[d])} features")
        if len(dgms_htr[d]) > 0 and len(dgms_htr[d]) <= 5:
            print(f"    {dgms_htr[d]}")
        elif len(dgms_htr[d]) > 5:
            print(f"    (showing first 5)")
            print(f"    {dgms_htr[d][:5]}")
    
    # Compute with Ripser
    result_ripser = ripser(X, maxdim=maxdim, thresh=threshold)
    dgms_ripser = result_ripser['dgms']
    
    print(f"\nRipser results:")
    for d in range(maxdim + 1):
        print(f"  H_{d}: {len(dgms_ripser[d])} features")
        if len(dgms_ripser[d]) > 0 and len(dgms_ripser[d]) <= 5:
            print(f"    {dgms_ripser[d]}")
        elif len(dgms_ripser[d]) > 5:
            print(f"    (showing first 5)")
            print(f"    {dgms_ripser[d][:5]}")
    
    # Compare
    match, msg = compare_persistence_diagrams(dgms_htr, dgms_ripser, maxdim, tolerance=1e-5)
    
    print(f"\nComparison: {msg}")
    print("=" * 80)
    
    assert match, f"Persistence diagrams don't match: {msg}"


def test_htr_vs_ripser_3d_points():
    """
    Test htr vs Ripser on random 3D points.
    """
    print("\n" + "=" * 80)
    print("TEST: htr vs Ripser - Random 3D points (12 points)")
    print("=" * 80)
    
    # Create random 3D points
    np.random.seed(123)
    n = 12
    X = np.random.rand(n, 3)
    
    threshold = 1.2
    maxdim = 2
    
    print(f"\nPoints (first 5):\n{X[:5]}")
    print(f"... ({n} points total)")
    print(f"Threshold: {threshold}, Max dimension: {maxdim}")
    
    # Compute with htr
    births_htr, deaths_htr, dims_htr = htr(points=X, threshold=threshold, maxdim=maxdim)
    dgms_htr = persistence_diagrams_to_arrays(births_htr, deaths_htr, dims_htr, maxdim)
    
    print(f"\nHTR results:")
    for d in range(maxdim + 1):
        print(f"  H_{d}: {len(dgms_htr[d])} features")
        if len(dgms_htr[d]) > 0 and len(dgms_htr[d]) <= 5:
            print(f"    {dgms_htr[d]}")
        elif len(dgms_htr[d]) > 5:
            print(f"    (showing first 5)")
            print(f"    {dgms_htr[d][:5]}")
    
    # Compute with Ripser
    result_ripser = ripser(X, maxdim=maxdim, thresh=threshold)
    dgms_ripser = result_ripser['dgms']
    
    print(f"\nRipser results:")
    for d in range(maxdim + 1):
        print(f"  H_{d}: {len(dgms_ripser[d])} features")
        if len(dgms_ripser[d]) > 0 and len(dgms_ripser[d]) <= 5:
            print(f"    {dgms_ripser[d]}")
        elif len(dgms_ripser[d]) > 5:
            print(f"    (showing first 5)")
            print(f"    {dgms_ripser[d][:5]}")
    
    # Compare
    match, msg = compare_persistence_diagrams(dgms_htr, dgms_ripser, maxdim, tolerance=1e-5)
    
    print(f"\nComparison: {msg}")
    print("=" * 80)
    
    assert match, f"Persistence diagrams don't match: {msg}"


def test_htr_vs_ripser_distance_matrix():
    """
    Test that both htr and Ripser produce the same results when using the same point cloud.
    This ensures both are consuming the same distance matrix.
    Also tests that htr can accept a distance_matrix parameter directly.
    """
    print("\n" + "=" * 80)
    print("TEST: htr vs Ripser - Verify same input (distance matrix)")
    print("=" * 80)
    
    # Create test points
    np.random.seed(789)
    n = 8
    X = np.random.rand(n, 2)
    
    threshold = 1.0
    maxdim = 1
    
    print(f"\nPoints:\n{X}")
    print(f"Threshold: {threshold}, Max dimension: {maxdim}")
    
    # Compute distance matrix
    dX = squareform(pdist(X))
    
    print(f"\nDistance matrix (upper triangle):")
    for i in range(min(5, n)):
        print(f"  {dX[i, i:min(i+5, n)]}")
    
    # Compute with htr (uses points)
    births_htr, deaths_htr, dims_htr = htr(points=X, threshold=threshold, maxdim=maxdim)
    dgms_htr = persistence_diagrams_to_arrays(births_htr, deaths_htr, dims_htr, maxdim)
    
    # Compute with htr (uses distance_matrix)
    births_htr_dm, deaths_htr_dm, dims_htr_dm = htr(distance_matrix=dX, threshold=threshold, maxdim=maxdim)
    dgms_htr_dm = persistence_diagrams_to_arrays(births_htr_dm, deaths_htr_dm, dims_htr_dm, maxdim)
    
    # Compute with Ripser (uses points)
    result_ripser = ripser(X, maxdim=maxdim, thresh=threshold)
    dgms_ripser = result_ripser['dgms']
    
    print(f"\nHTR results (from points):")
    for d in range(maxdim + 1):
        print(f"  H_{d}: {len(dgms_htr[d])} features")
        if len(dgms_htr[d]) > 0:
            print(f"    {dgms_htr[d]}")
    
    print(f"\nHTR results (from distance_matrix):")
    for d in range(maxdim + 1):
        print(f"  H_{d}: {len(dgms_htr_dm[d])} features")
        if len(dgms_htr_dm[d]) > 0:
            print(f"    {dgms_htr_dm[d]}")
    
    print(f"\nRipser results:")
    for d in range(maxdim + 1):
        print(f"  H_{d}: {len(dgms_ripser[d])} features")
        if len(dgms_ripser[d]) > 0:
            print(f"    {dgms_ripser[d]}")
    
    # Compare htr(points) with Ripser
    match, msg = compare_persistence_diagrams(dgms_htr, dgms_ripser, maxdim, tolerance=1e-5)
    print(f"\nComparison htr(points) vs Ripser: {msg}")
    assert match, f"Persistence diagrams don't match: {msg}"
    
    # Compare htr(distance_matrix) with htr(points)
    match_dm, msg_dm = compare_persistence_diagrams(dgms_htr_dm, dgms_htr, maxdim, tolerance=1e-5)
    print(f"Comparison htr(distance_matrix) vs htr(points): {msg_dm}")
    assert match_dm, f"Distance matrix results don't match: {msg_dm}"
    
    print("=" * 80)


@pytest.mark.parametrize("n_points,maxdim", [
    (5, 0),
    (5, 1),
    (10, 1),
    (15, 1),
    (8, 2),
])
def test_htr_vs_ripser_parametric(n_points, maxdim):
    """
    Parametric test comparing htr with Ripser for different configurations.
    """
    print(f"\n{'='*80}")
    print(f"TEST: htr vs Ripser - {n_points} points, maxdim={maxdim}")
    print(f"{'='*80}")
    
    # Create random points
    np.random.seed(42 + n_points + maxdim)
    X = np.random.rand(n_points, 2)
    
    threshold = 1.5
    
    print(f"\nConfiguration: {n_points} points, threshold={threshold}, maxdim={maxdim}")
    
    # Compute with htr
    births_htr, deaths_htr, dims_htr = htr(points=X, threshold=threshold, maxdim=maxdim)
    dgms_htr = persistence_diagrams_to_arrays(births_htr, deaths_htr, dims_htr, maxdim)
    
    # Compute with Ripser
    result_ripser = ripser(X, maxdim=maxdim, thresh=threshold)
    dgms_ripser = result_ripser['dgms']
    
    print(f"\nHTR: ", end="")
    for d in range(maxdim + 1):
        print(f"H_{d}={len(dgms_htr[d])} ", end="")
    
    print(f"\nRipser: ", end="")
    for d in range(maxdim + 1):
        print(f"H_{d}={len(dgms_ripser[d])} ", end="")
    print()
    
    # Compare
    match, msg = compare_persistence_diagrams(dgms_htr, dgms_ripser, maxdim, tolerance=1e-5)
    
    print(f"\nComparison: {msg}")
    print(f"{'='*80}")
    
    assert match, f"Persistence diagrams don't match: {msg}"


if __name__ == "__main__":
    # Run tests individually for detailed output
    test_htr_vs_ripser_triangle()
    test_htr_vs_ripser_square()
    test_htr_vs_ripser_circle_points()
    test_htr_vs_ripser_random_2d()
    test_htr_vs_ripser_3d_points()
    test_htr_vs_ripser_distance_matrix()
    
    # Run parametric tests
    for n, dim in [(5, 0), (5, 1), (10, 1), (15, 1), (8, 2)]:
        test_htr_vs_ripser_parametric(n, dim)
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
