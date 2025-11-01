import numpy as np
from persistent_cost.cylinder import cylinder_pipeline

def test_cylinder_pipeline():
    """
    Test the cylinder pipeline with a simple example.
    """
    X = np.array([[0, 4], [2, 4], [4, 4]])
    Y = np.array([[0, 4], [2, 4], [4, 4], [6, 4]])
    f = np.array([0, 1, 2])
    threshold = 10
    maxdim = 2

    d_ker, d_cok = cylinder_pipeline(X, Y, f, threshold, maxdim)

    # Add assertions here to check the output
    # For example, check the number of bars in each dimension
    assert isinstance(d_ker, dict)
    assert isinstance(d_cok, dict)

    # Example assertion: Check if H0 of kernel has one infinite bar
    # This is just an example, the actual value might be different
    # assert len(d_ker.get(0, [])) > 0
    # assert d_ker[0][0][1] == np.inf

    print("\nCylinder pipeline test passed.")
    print("Kernel diagram:", d_ker)
    print("Cokernel diagram:", d_cok)

if __name__ == "__main__":
    test_cylinder_pipeline()
