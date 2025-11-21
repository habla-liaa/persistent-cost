import gudhi as gd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from persistent_cost.cone import matrix_size_from_condensed, conematrix

def remove_empty_dims(pairs):
    """Remove empty dimensions from a persistence diagram."""
    return [dim for dim in pairs if len(dim) > 0]

def gudhi_rutine(distance_matrix, maxdim, max_edge_length):
    import gudhi as gd

    rc = gd.RipsComplex(distance_matrix=distance_matrix,
                        max_edge_length=float(max_edge_length))
    st = rc.create_simplex_tree(max_dimension=max(2, maxdim+1))
    st.compute_persistence()
    dgm = [st.persistence_intervals_in_dimension(
        dim) for dim in range(maxdim+1)]
    pairs = st.persistence_pairs()
    simpl2dist = {tuple(a[0]): a[1]
                  for a in st.get_simplices() if len(a[0]) < maxdim+3}
    return dgm, pairs, simpl2dist


def pairs_sort(pairs, maxdim, offset=0):
    pairs_sort = [[] for _ in range(maxdim+1)]
    for p in pairs:
        dim = len(p[0])
        p = (list(map(int, np.sort(p[0])+offset)),
             list(map(int, np.sort(p[1])+offset)))
        pairs_sort[dim-1].append(p)
    return pairs_sort


def pairs2dist(pairs, simpl2dist):
    maxdim = len(pairs)
    dists = [[] for _ in range(maxdim)]
    for dim in range(maxdim):
        for pair in pairs[dim]:
            b = simpl2dist[tuple(pair[0])]
            d = simpl2dist[tuple(pair[1])]
            dists[dim].append((b, d))
        dists[dim] = np.array(dists[dim])
    return dists


def kercoker_bars(pairs_cone, pairs_X, pairs_Y, cone_index):
    """
    Find cokernel and kernel bars in the persistence diagram.
    TODO: optimize
    """

    maxdim = len(pairs_cone) - 1

    pairs_coker = [[] for _ in range(maxdim + 1)]
    pairs_ker = [[] for _ in range(maxdim + 1)]
    missing = [[] for _ in range(maxdim + 1)]

    for dim in range(maxdim + 1):
        for i, pair in enumerate(pairs_cone[dim]):
            newbar = False
            b, d = pair
            if len(d) == 0:
                continue
            # remove cone vertex index with pop
            b = [v for v in b if v != cone_index]
            d = [v for v in d if v != cone_index]

            # coker
            # b_c = b_y_i
            # d_c = d_y_i
            for p in pairs_Y[dim]:
                if len(p[0]) != len(b) or len(p[1]) != len(d):
                    continue
                if p[0] == b and p[1] == d:
                    pairs_coker[dim].append((b, d))
                    newbar = True

            # b_c = b_y_i
            # d_c = b_x_j
            for j in range(len(pairs_X[dim])):
                for i in range(len(pairs_Y[dim])):
                    if len(pairs_Y[dim][i][0]) != len(b) or len(pairs_X[dim][j][0]) != len(d):
                        continue
                    if pairs_Y[dim][i][0] == b and pairs_X[dim][j][0] == d:
                        pairs_coker[dim].append((b, d))
                        newbar = True

            # b_c = b_x_i
            # d_c = d_y_i
            for i in range(len(pairs_X[dim])):
                for j in range(len(pairs_Y[dim])):
                    if len(pairs_X[dim][i][0]) != len(b) or len(pairs_Y[dim][j][1]) != len(d):
                        continue
                    if pairs_X[dim][i][0] == b and pairs_Y[dim][j][1] == d:
                        missing[dim].append(('bx,dy', (b, d)))
                        newbar = True

            # ker
            if dim > 0:
                # b_c = b_x_i (dim-1)
                # d_c = d_x_i (dim-1)
                for p in pairs_X[dim - 1]:
                    if len(p[0]) != len(b) or len(p[1]) != len(d):
                        continue
                    if p[0] == b and p[1] == d:
                        pairs_ker[dim - 1].append((b, d))
                        newbar = True

                # b_c = d_y_i (dim-1)
                # d_c = d_x_j (dim-1)
                for j in range(len(pairs_X[dim - 1])):
                    for i in range(len(pairs_Y[dim - 1])):
                        if len(pairs_Y[dim - 1][i][1]) != len(b) or len(pairs_X[dim - 1][j][1]) != len(d):
                            continue
                        if pairs_Y[dim - 1][i][1] == b and pairs_X[dim - 1][j][1] == d:
                            pairs_ker[dim - 1].append((b, d))
                            newbar = True

                # b_c = b_y_i
                # d_c = d_x_j (dim-1)
                for i in range(len(pairs_Y[dim])):
                    for j in range(len(pairs_X[dim - 1])):
                        if len(pairs_Y[dim][i][0]) != len(b) or len(pairs_X[dim - 1][j][1]) != len(d):
                            continue
                        if pairs_Y[dim][i][0] == b and pairs_X[dim - 1][j][1] == d:
                            missing[dim - 1].append(('by,dx(-1)', (b, d)))
                            newbar = True

            if not newbar:
                missing[dim].append(('null', (b, d)))

    pairs_coker = remove_empty_dims(pairs_coker)
    pairs_ker = remove_empty_dims(pairs_ker)
    missing = remove_empty_dims(missing)
    return pairs_coker, pairs_ker, missing


def cone_pipeline(dX, dY, f, maxdim=1, cone_eps=0.0, return_extra=False):
    """
    TODO: Compute the total persistence diagram using the cone algorithm.

    Parameters
    ----------
    dX : np.array
        Distance matrix of the source space in condensed form.
    dY : np.array
        Distance matrix of the target space in condensed form.
    f : np.array
        Function values.
    cone_eps : float, optional
        Epsilon value for the cone construction, by default 0.0
    tol : float, optional
        Tolerance for numerical comparisons, by default 1e-11
    Returns
    """

    n = matrix_size_from_condensed(dX)
    m = matrix_size_from_condensed(dY)

    f = np.array(f)

    # # dY_ff = d(f(x_i),f(x_j)) para todo i,j
    # i, j = np.triu_indices(n, k=1)
    # f_i, f_j = f[i], f[j]
    # dY_ff = squareform(dY)[f_i, f_j]

    # L = lipschitz(dX, dY_ff)

    # dY = dY / L

    D = conematrix(dX, dY, f, cone_eps, 9999)

    max_edge_length = 3 * max(np.max(squareform(dX)), np.max(squareform(dY)))

    # Guidhi for X
    dgm_X, pairs_X, simpl2dist_X = gudhi_rutine(squareform(dX), maxdim, max_edge_length)

    # Guidhi for Y
    dgm_Y, pairs_Y, simpl2dist_Y = gudhi_rutine(squareform(dY), maxdim, max_edge_length)

    # GUDHI for Cone
    dgm_cone, pairs_cone, simpl2dist_cone = gudhi_rutine(D, maxdim, max_edge_length)

    pairs_cone = pairs_sort(pairs_cone, maxdim)
    pairs_Y = pairs_sort(pairs_Y, maxdim, offset=n)
    pairs_X = pairs_sort(pairs_X, maxdim)

    pairs_coker, pairs_ker, missing = kercoker_bars(
        pairs_cone, pairs_X, pairs_Y, n+m)

    dgm_coker = pairs2dist(pairs_coker, simpl2dist_cone)
    dgm_ker = pairs2dist(pairs_ker, simpl2dist_cone)

    if return_extra:
        return (
            dgm_coker, dgm_ker,  dgm_cone, dgm_X, dgm_Y,
            pairs_coker, pairs_ker, pairs_cone,  pairs_X, pairs_Y,
            missing,
            simpl2dist_cone, simpl2dist_X, simpl2dist_Y, D
        )
    return dgm_coker, dgm_ker, dgm_cone, dgm_X, dgm_Y
