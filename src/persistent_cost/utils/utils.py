from itertools import combinations
from IPython import embed
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gudhi as gd
# from paper2008_utils import do_pivot


def build_ordered_boundary_matrix(
    points=None, distance_matrix=None, threshold=None, maxdim=None, field=np.int8
):
    """
    Para computar H hasta maxdim, construimos la matriz hasta maxdim+1
    """
    # points or distance_matrix must be provided
    if points is None and distance_matrix is None:
        raise ValueError("Either points or distance_matrix must be provided.")
    # threshold and maxdim must be provided
    if threshold is None or maxdim is None:
        raise ValueError("Both threshold and maxdim must be provided.")

    # Paso 1: complejo de Rips
    rips_complex = gd.RipsComplex(
        points=points, distance_matrix=distance_matrix, max_edge_length=threshold
    )
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=maxdim + 1)
    all_simplices = list(simplex_tree.get_skeleton(maxdim + 1))

    # Paso 2: separar vértices y el resto
    zeroth = [s for s in all_simplices if len(s[0]) == 1]
    higher = [s for s in all_simplices if len(s[0]) >= 2]

    # Paso 3: construir lista de todos los símplices como diccionarios
    simplices_info = []

    for simp, filt in zeroth:
        simplices_info.append({"vertices": list(simp), "dim": 0, "eps": filt})

    for simp, filt in higher:
        simplices_info.append(
            {"vertices": sorted(list(simp)), "dim": len(simp) - 1, "eps": filt}
        )

    # Paso 4: ordenar por (eps, dim, vertices)
    simplices_info.sort(key=lambda s: (s["eps"], s["dim"], s["vertices"]))

    # Paso 5: asignar índices de columna/fila
    for i, s in enumerate(simplices_info):
        s["idx"] = i

    # Paso 6: construir diccionario auxiliar para encontrar índices por conjunto de vértices
    vertices_to_index = {tuple(s["vertices"]): s["idx"]
                         for s in simplices_info}

    # Paso 7: construir matriz de borde dispersa
    m = len(simplices_info)
    D = sparse.lil_matrix((m, m), dtype=field)

    for s in simplices_info:
        if s["dim"] == 0:
            continue  # no tiene borde
        verts = s["vertices"]
        for k in range(len(verts)):
            # quitar vértice k para formar cara
            face = verts[:k] + verts[k + 1:]
            j = vertices_to_index.get(tuple(face))
            if j is not None:
                D[j, s["idx"]] = 1

    return D.tocsr(), simplices_info


def matrix_size_from_condensed(dX):
    n = len(dX)
    return int(0.5 * (np.sqrt(8 * n + 1) - 1) + 1)


def compute_lipschitz_constant(X, Y, f):
    """
    Calcula la constante de Lipschitz antes de la normalización.

    L = max(d_Y(f(x_i), f(x_j)) / d_X(x_i, x_j))

    Args:
        X: puntos del espacio X o dX: matriz de distancias de X condensada
        Y: puntos del espacio Y o dY: matriz de distancias de Y condensada
        f: función de X a Y (índices)
    Returns:
        L: constante de Lipschitz
    """
    from scipy.spatial.distance import squareform

    # If condensed distance matrices are provided
    if not (isinstance(X, np.ndarray) and X.ndim == 2):
        dX = X
        n = matrix_size_from_condensed(dX)
    else:
        dX = pdist(X)
        n = len(X)

    if not (isinstance(Y, np.ndarray) and Y.ndim == 2):
        dY = Y
    else:
        dY = pdist(Y)

    f = np.array(f)
    if n > 1:
        i, j = np.triu_indices(n, k=1)
        f_i, f_j = f[i], f[j]
        dY_ff = squareform(dY)[f_i, f_j]
        # Evitar división por cero
        mask = dX > 0
        L = np.max(dY_ff[mask] / dX[mask]) if np.any(mask) else 0.0
    else:
        L = 1.0
    return L


# def lipschitz_constant(dX, dY, f):

#     f = np.array(f)
#     n = len(f)

#     # dY_ff = d(f(x_i),f(x_j)) para todo i,j
#     i, j = np.triu_indices(n, k=1)
#     f_i, f_j = f[i], f[j]

#     dY_ff = squareform(dY)[f_i, f_j]

#     return float(np.max(dY_ff / dX))

def matrix_size_from_condensed(dX):
    n = len(dX)
    return int(0.5 * (np.sqrt(8 * n + 1) - 1) + 1)


def match_simplices(f, simplices_info_X, simplices_info_Y):
    """
    Devuelve una lista 'correspondencia' tal que
    correspondencia[i] = j si el símplex i de X va al símplex j de Y vía f
    """
    # Crear un diccionario para lookup en Y
    lookup_Y = {tuple(s["vertices"]): s["idx"] for s in simplices_info_Y}

    correspondencia = []
    for s in simplices_info_X:
        vtx_f = tuple(set(f[v] for v in s["vertices"]))
        idx_y = lookup_Y.get(vtx_f, -1)  # -1 si no existe
        correspondencia.append(idx_y)

    return np.array(correspondencia)


def general_position_distance_matrix(X, perturb=1e-7):
    n = len(X)
    Xperturbation = perturb * np.random.rand((n * (n - 1) // 2))
    dX = pdist(X) + Xperturbation
    return dX


def births_and_deaths(R, M, eps, maxdim=1):
    from persistent_cost.algorithms.sparse import column_pivots

    births = []
    deaths = []
    dims = []
    cpivs = column_pivots(R)
    for k in range(R.shape[1]):
        dim = np.sum(M[:k, k] != 0)-1
        if dim <= maxdim:
            if cpivs[k] == -1:
                if k in cpivs:
                    j = cpivs.index(k)
                    if eps[k] < eps[j]:
                        births.append(eps[k])
                        deaths.append(eps[j])
                        dims.append(dim if dim > 0 else 0)
                else:
                    births.append(eps[k])
                    deaths.append(np.inf)
                    dims.append(dim if dim > 0 else 0)

    return births, deaths, dims


# def simplices2df(simplices):
#     df = pd.DataFrame(simplices, columns=['simplex', 'eps'])
#     df['simplex'] = df['simplex'].map(tuple)
#     df['dim'] = df['simplex'].map(len)-1
#     df = df.sort_values(by=['eps', 'dim', 'simplex']).reset_index(drop=True)
#     df['faces'] = df['simplex'].map(lambda x: list(
#         combinations(tuple(sorted(tuple(x))), len(x)-1)))
#     df['step'] = np.cumsum(df['eps'].diff() != 0)
#     return df

# Convert to persistence diagram format (list of arrays per dimension)
def births_deaths_to_dgm(births, deaths, dims, maxdim):
    dgm = []
    for d in range(maxdim + 1):
        mask = np.array(dims) == d
        b = np.array(births)[mask]
        dt = np.array(deaths)[mask]
        if len(b) > 0:
            dgm.append(np.column_stack([b, dt]))
        else:
            dgm.append(np.empty((0, 2)))
    return dgm

from persistent_cost.algorithms.sparse_fast import do_pivot_cython
def htr(points = None, distance_matrix = None, threshold=None, maxdim=None):
    """
    Compute the persistent homology using the HTR algorithm.
    Args:
        points: point cloud data (optional if distance_matrix is provided)
        distance_matrix: square distance matrix (optional if points is provided)
        threshold: maximum distance for Rips complex
        maxdim: maximum homology dimension to compute
    Returns:
        births: list of birth times
        deaths: list of death times
        dims: list of dimensions corresponding to each birth-death pair
    """
    
    if distance_matrix is None and points is not None:
        DX = squareform(pdist(points))
    elif points is None and distance_matrix is not None:
        # distance_matrix should already be a square matrix
        DX = distance_matrix
    else:
        raise ValueError("Either points or distance_matrix must be provided.")
    
    # Ensure threshold and maxdim are provided
    if threshold is None or maxdim is None:
        raise ValueError("Both threshold and maxdim must be provided.")

    M, simplices_X = build_ordered_boundary_matrix(
        distance_matrix=DX, threshold=threshold, maxdim=maxdim)

    R, V = do_pivot_cython(M)
    eps = [s['eps'] for s in simplices_X]
    births, deaths, dims = births_and_deaths(R, M, eps, maxdim)

    return births, deaths, dims



def conematrix_blocks(DX, DY, DY_fy, eps):
    n = len(DX)
    m = len(DY)

    D = np.zeros((n + m + 1, n + m + 1))
    D[0:n, 0:n] = DX
    D[n: n + m, n: n + m] = DY

    D[0:n, n: n + m] = DY_fy
    D[n: n + m, 0:n] = DY_fy.T

    R = np.inf
    # R = max(DX.max(), DY_fy[~np.isinf(DY_fy)].max()) + 1 #instead of np.inf

    D[n + m, n: n + m] = R
    D[n: n + m, n + m] = R

    D[n + m, :n] = eps
    D[:n, n + m] = eps

    return D


def conematrix(dX, dY, f, cone_eps=0.0, max_value=np.inf):

    n = matrix_size_from_condensed(dX)
    m = matrix_size_from_condensed(dY)
    f = np.array(f)

    DY_fy = np.ones((n, m), dtype=float) * max_value

    # dY_fy = d(f(x_i),y_j) para todo i,j
    indices = np.indices((n, m))
    i = indices[0].flatten()
    j = indices[1].flatten()
    f_i = f[i]

    ijs = [(ii, jj) for ii, jj in zip(i, j) if jj in f_i]
    i, j = zip(*ijs)
    i = np.array(i, dtype=int)
    j = np.array(j, dtype=int)

    f_i = f[i]

    DY_fy[i, j] = squareform(dY)[f_i, j]

    # dX     DY_fy
    # DY_fy  dY
    D = conematrix_blocks(squareform(dX), squareform(dY), DY_fy, cone_eps)
    return D

