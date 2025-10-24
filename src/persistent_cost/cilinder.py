from scipy import sparse
from scipy.spatial.distance import pdist, squareform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gudhi as gd
from .utils import matrix_size_from_condensed


def cylindermatrix(dX, dY, f):
    n = matrix_size_from_condensed(dX)
    m = matrix_size_from_condensed(dY)

    f = np.array(f)

    i, j = np.triu_indices(n, k=1)
    f_i, f_j = f[i], f[j]
    
    # dY_fy = d(f(x_i),y_j) para todo i,j
    indices = np.indices((n, m))
    i = indices[0].flatten()
    j = indices[1].flatten()
    f_i = f[i]

    DY_fy = np.ones((n, m), dtype=float) * np.inf

    ijs = [(ii, jj) for ii, jj in zip(i, j) if jj in f_i]
    i, j = zip(*ijs)
    i = np.array(i, dtype=int)
    j = np.array(j, dtype=int)

    f_i = f[i]

    DY_fy[i, j] = squareform(dY)[f_i, j]
    
    D = np.zeros((n + m , n + m ))

    D[0:n, 0:n] = squareform(dX)
    D[n : n + m, n : n + m] = squareform(dY)

    D[0:n, n : n + m] = DY_fy
    D[n : n + m, 0:n] = DY_fy.T

    return D