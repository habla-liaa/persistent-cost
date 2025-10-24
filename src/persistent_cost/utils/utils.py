from scipy import sparse
from scipy.spatial.distance import pdist, squareform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gudhi as gd


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
    vertices_to_index = {tuple(s["vertices"]): s["idx"] for s in simplices_info}

    # Paso 7: construir matriz de borde dispersa
    m = len(simplices_info)
    D = sparse.lil_matrix((m, m), dtype=field)

    for s in simplices_info:
        if s["dim"] == 0:
            continue  # no tiene borde
        verts = s["vertices"]
        for k in range(len(verts)):
            face = verts[:k] + verts[k + 1 :]  # quitar vértice k para formar cara
            j = vertices_to_index.get(tuple(face))
            if j is not None:
                D[j, s["idx"]] = 1

    return D.tocsr(), simplices_info


def matrix_size_from_condensed(dX):
    n = len(dX)
    return int(0.5 * (np.sqrt(8 * n + 1) - 1) + 1)


def lipschitz_constant(dX, dY, f):

    f = np.array(f)
    n = len(f)

    # dY_ff = d(f(x_i),f(x_j)) para todo i,j
    i, j = np.triu_indices(n, k=1)
    f_i, f_j = f[i], f[j]

    dY_ff = squareform(dY)[f_i, f_j]

    return float(np.max(dY_ff / dX))

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