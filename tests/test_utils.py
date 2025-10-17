import numpy as np
from scipy.spatial.distance import pdist

from persistent_cost.utils.utils import build_ordered_boundary_matrix, lipschitz_constant

def test_build_ordered_boundary_matrix_2dsquare():
    """
    Test build_ordered_boundary_matrix with a 2D unit square.
    Points: (0,0), (1,0), (1,1), (0,1)
    """
    # Create the 4 corners of a unit square
    X = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ])
    
    # Set threshold large enough to capture all edges and triangles
    # Diagonal of unit square is sqrt(2) ≈ 1.414
    threshold = 2.0
    maxdim = 2
    
    # Build the boundary matrix
    D, simplices_info = build_ordered_boundary_matrix(points=X, threshold=threshold, maxdim=maxdim)
    
    print("\n" + "="*80)
    print("TEST: build_ordered_boundary_matrix with 2D unit square")
    print("="*80)
    
    print("\nInput points (unit square corners):")
    print(X)
    
    print(f"\nThreshold: {threshold}")
    print(f"Max dimension: {maxdim}")
    
    print(f"\nNumber of simplices: {len(simplices_info)}")
    
    print("\nSimplices info:")
    print("-" * 80)
    for i, s in enumerate(simplices_info):
        print(f"Index {i:2d}: dim={s['dim']}, vertices={s['vertices']}, eps={s['eps']:.4f}")
    
    print("\nBoundary matrix D (dense form):")
    print("-" * 80)
    D_dense = D.todense()
    print(D_dense)
    
    print("\nBoundary matrix shape:", D_dense.shape)
    print("Number of non-zero entries:", D.nnz)
    
    print("\n" + "="*80)
    
    # Assertions based on the expected structure
    
    # 1. Total number of simplices
    # 4 vertices + 6 edges + 4 triangles + 1 tetrahedron = 15
    assert len(simplices_info) == 15, f"Expected 15 simplices, got {len(simplices_info)}"
    
    # 2. Matrix dimensions
    assert D.shape == (15, 15), f"Expected shape (15, 15), got {D.shape}"
    assert D.shape[0] == D.shape[1], "Boundary matrix should be square"
    assert D.shape[0] == len(simplices_info), "Matrix dimension should match number of simplices"
    
    # 3. Number of non-zero entries in boundary matrix
    assert D.nnz == 28, f"Expected 28 non-zero entries, got {D.nnz}"
    
    # 4. Check vertices (dimension 0)
    vertices = [s for s in simplices_info if s['dim'] == 0]
    assert len(vertices) == 4, f"Expected 4 vertices, got {len(vertices)}"
    assert all(s['eps'] == 0.0 for s in vertices), "All vertices should have eps=0.0"
    assert vertices[0]['vertices'] == [0]
    assert vertices[1]['vertices'] == [1]
    assert vertices[2]['vertices'] == [2]
    assert vertices[3]['vertices'] == [3]
    
    # 5. Check edges (dimension 1)
    edges = [s for s in simplices_info if s['dim'] == 1]
    assert len(edges) == 6, f"Expected 6 edges, got {len(edges)}"
    
    # Check the 4 square edges (length 1.0)
    square_edges = [e for e in edges if abs(e['eps'] - 1.0) < 1e-6]
    assert len(square_edges) == 4, f"Expected 4 edges of length 1.0, got {len(square_edges)}"
    assert square_edges[0]['vertices'] == [0, 1]  # bottom edge
    assert square_edges[1]['vertices'] == [0, 3]  # left edge
    assert square_edges[2]['vertices'] == [1, 2]  # right edge
    assert square_edges[3]['vertices'] == [2, 3]  # top edge
    
    # Check the 2 diagonal edges (length sqrt(2))
    diagonal_edges = [e for e in edges if abs(e['eps'] - np.sqrt(2)) < 1e-4]
    assert len(diagonal_edges) == 2, f"Expected 2 diagonal edges, got {len(diagonal_edges)}"
    assert diagonal_edges[0]['vertices'] == [0, 2]
    assert diagonal_edges[1]['vertices'] == [1, 3]
    
    # 6. Check triangles (dimension 2)
    triangles = [s for s in simplices_info if s['dim'] == 2]
    assert len(triangles) == 4, f"Expected 4 triangles, got {len(triangles)}"
    assert all(abs(t['eps'] - np.sqrt(2)) < 1e-4 for t in triangles), "All triangles should appear at eps=sqrt(2)"
    assert triangles[0]['vertices'] == [0, 1, 2]
    assert triangles[1]['vertices'] == [0, 1, 3]
    assert triangles[2]['vertices'] == [0, 2, 3]
    assert triangles[3]['vertices'] == [1, 2, 3]
    
    # 7. Check tetrahedron (dimension 3)
    tetrahedra = [s for s in simplices_info if s['dim'] == 3]
    assert len(tetrahedra) == 1, f"Expected 1 tetrahedron, got {len(tetrahedra)}"
    assert tetrahedra[0]['vertices'] == [0, 1, 2, 3]
    assert abs(tetrahedra[0]['eps'] - np.sqrt(2)) < 1e-4
    
    # 8. Check boundary matrix structure
    # The last column (tetrahedron) should have 4 faces (the 4 triangles)
    D_dense = D.todense()
    last_col = D_dense[:, 14]
    assert np.sum(last_col) == 4, "Tetrahedron should have 4 triangular faces"
    
    # Each triangle (columns 10-13) should have 3 edges as boundaries
    for col_idx in range(10, 14):
        col = D_dense[:, col_idx]
        assert np.sum(col) == 3, f"Triangle at index {col_idx} should have 3 edges"
    
    # Each edge (columns 4-9) should have 2 vertices as boundaries
    for col_idx in range(4, 10):
        col = D_dense[:, col_idx]
        assert np.sum(col) == 2, f"Edge at index {col_idx} should have 2 vertices"
    
    # Vertices (columns 0-3) should have no boundary
    for col_idx in range(0, 4):
        col = D_dense[:, col_idx]
        assert np.sum(col) == 0, f"Vertex at index {col_idx} should have no boundary"
    
    # 9. Check that simplices are ordered correctly (eps, dim, vertices)
    for i in range(len(simplices_info) - 1):
        s1, s2 = simplices_info[i], simplices_info[i + 1]
        # Check ordering: (eps, dim, vertices)
        if s1['eps'] < s2['eps']:
            continue  # correct order by eps
        elif s1['eps'] == s2['eps']:
            if s1['dim'] < s2['dim']:
                continue  # correct order by dim
            elif s1['dim'] == s2['dim']:
                assert s1['vertices'] < s2['vertices'], f"Vertices not in order at indices {i}, {i+1}"
        else:
            assert False, f"Simplices not ordered correctly at indices {i}, {i+1}"
    
    print("✓ All assertions passed!")
    print("="*80 + "\n")


def test_lipschitz_constant_scaled_square():
    """
    Test lipschitz_constant with two squares of different sizes.
    X: unit square with side length 1
    Y: scaled square with side length 2
    f: identity mapping [0, 1, 2, 3]
    
    The Lipschitz constant should be 2, as distances in Y are exactly 2x distances in X.
    """
    # X: Unit square (side length 1)
    X = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ])
    
    # Y: Scaled square (side length 2)
    Y = np.array([
        [0, 0],
        [2, 0],
        [2, 2],
        [0, 2]
    ])
    
    # f: identity mapping
    f = np.arange(4)
    
    # Compute condensed distance matrices
    dX = pdist(X)
    dY = pdist(Y)
    
    print("\n" + "="*80)
    print("TEST: lipschitz_constant with scaled squares")
    print("="*80)
    
    print("\nX (unit square):")
    print(X)
    
    print("\nY (scaled square, side length 2):")
    print(Y)
    
    print("\nf (identity mapping):")
    print(f)
    
    print("\nCondensed distance matrix dX:")
    print(dX)
    
    print("\nCondensed distance matrix dY:")
    print(dY)
    
    # Compute Lipschitz constant
    L = lipschitz_constant(dX, dY, f)
    
    print(f"\nComputed Lipschitz constant: {L}")
    
    print("\n" + "="*80)
    
    # Assertions
    # The Lipschitz constant should be exactly 2.0
    # because every distance in Y is exactly 2x the corresponding distance in X
    assert np.isclose(L, 2.0, rtol=1e-10), f"Expected Lipschitz constant ≈ 2.0, got {L}"
    
    # Additional checks
    assert L > 0, "Lipschitz constant should be positive"
    assert np.isfinite(L), "Lipschitz constant should be finite"
    
    # Verify the relationship: dY[i,j] = 2 * dX[i,j] for all i,j
    ratio = dY / dX
    print(f"\nRatio dY/dX for all pairs: {ratio}")
    assert np.allclose(ratio, 2.0), "All distance ratios should be 2.0"
    
    print("\n✓ All assertions passed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_build_ordered_boundary_matrix_2dsquare()
    test_lipschitz_constant_scaled_square()
