import numpy as np
from scipy.spatial.distance import pdist, squareform
from persistent_cost.utils.utils import conematrix, sort_diagram
from persistent_cost.cone import kercoker_bars


# def kercoker_bars(dgm, dgmX, dgmY, tol=1e-11):
#     """
#     Find cokernel and kernel bars in the persistence diagram.
#     TODO: optimize
#     """
#     maxdim = len(dgm) - 1

#     coker_dgm = [[] for _ in range(maxdim + 1)]
#     ker_dgm = [[] for _ in range(maxdim + 1)]
#     missing = [[] for _ in range(maxdim + 1)]

#     # assert dgm[0][-1][1] == np.inf, "Expected infinite bar in dimension 0 of cone diagram."
#     # dgm[0] = dgm[0][:-1]  # remove the infinite bar in dim 0

#     for dim in range(maxdim + 1):  # dimension cone diagram
#         for i, r in enumerate(dgm[dim]):
#             newbar = False
#             b, d = r
#             # coker
#             # b_c = b_y_i
#             # d_c = d_y_i
#             m = findclose(b, dgmY[dim][:, 0], tol) & findclose(
#                 d, dgmY[dim][:, 1], tol)
#             if sum(m):
#                 # if sum(m) > 1:
#                     # print("Encontre:", sum(m))
#                 coker_dgm[dim].append((b, d))
#                 newbar = True

#             # b_c = b_y_i
#             # d_c = b_x_j
#             if any(findclose(b, dgmY[dim][:, 0], tol)) and any(findclose(d, dgmX[dim][:, 0], tol)):
#                 coker_dgm[dim].append((b, d))
#                 newbar = True

#             # ker
#             if dim > 0:
#                 # b_c = b_x_i (dim-1)
#                 # d_c = d_x_i (dim-1)
#                 m = findclose(
#                     b, dgmX[dim - 1][:, 0], tol) & findclose(d, dgmX[dim - 1][:, 1], tol)
#                 if sum(m):
#                     ker_dgm[dim - 1].append((b, d))
#                     newbar = True

#                 # b_c = d_y_i (dim-1)
#                 # d_c = d_x_j (dim-1)
#                 if any(findclose(b, dgmY[dim - 1][:, 1], tol)) and any(findclose(d, dgmX[dim - 1][:, 1], tol)):
#                     ker_dgm[dim - 1].append((b, d))
#                     newbar = True

#             if not newbar:
#                 # cone_bars.append((k, i, 'sobra', k,  'null', (b, d)))
#                 missing[dim].append(('null', (b, d)))

#     coker_dgm = remove_empty_dims(coker_dgm)
#     ker_dgm = remove_empty_dims(ker_dgm)
#     return coker_dgm, ker_dgm, missing


# def kercoker_bars_(dgm, dgmX, dgmY, cone_eps, tol=1e-11):
#     """
#     Find cokernel and kernel bars in the persistence diagram.
#     TODO: optimize
#     """
#     coker_dgm = [[] for _ in range(len(dgm))]
#     ker_dgm = [[] for _ in range(len(dgm))]

#     assert dgm[0][-1][1] == np.inf, "Expected infinite bar in dimension 0 of cone diagram."
#     dgm[0] = dgm[0][:-1]  # remove the infinite bar in dim 0

#     for k in range(len(dgm)):  # dimension cone diagram
#         for r in dgm[k]:
#             b, d = r
#             if d > cone_eps + tol:
#                 coker_dgm[k].append((b, d))
#                 # ker
#                 if k > 0:
#                     # b_c = b_x_i (dim-1)
#                     # d_c = d_x_i (dim-1)
#                     m = findclose(
#                         b, dgmX[k - 1][:, 0], tol) & findclose(d, dgmX[k - 1][:, 1], tol)
#                     if sum(m):
#                         ker_dgm[k - 1].append((b, d))
#                         coker_dgm[k].remove((b, d))

#                     # b_c = d_y_i (dim-1)
#                     # d_c = d_x_j (dim-1)
#                     if any(findclose(b, dgmY[k - 1][:, 1], tol)) and any(findclose(d, dgmX[k - 1][:, 1], tol)):
#                         ker_dgm[k - 1].append((b, d))
#                         coker_dgm[k].remove((b, d))

#     coker_dgm = remove_empty_dims(coker_dgm)
#     ker_dgm = remove_empty_dims(ker_dgm)
#     return coker_dgm, ker_dgm


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

def cone_pipeline(dX, dY, f, maxdim=1, cone_eps=0.0, tol=1e-11, return_extra=False):
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
    
    f = np.array(f)

    D = conematrix(dX, dY, f, cone_eps, 9999)

    max_edge_length = 3 * max(np.max(squareform(dX)), np.max(squareform(dY)))

    dgm_X, pairs_X, simpl2dist_X = gudhi_rutine(
        squareform(dX), maxdim, max_edge_length)
    
    dgm_Y, pairs_Y, simpl2dist_Y = gudhi_rutine(
        squareform(dY), maxdim, max_edge_length)
    
    dgm_cone, pairs_cone, simpl2dist_cone = gudhi_rutine(
        D, maxdim, max_edge_length)

    dgm_X = sort_diagram(dgm_X)
    dgm_Y = sort_diagram(dgm_Y)
    dgm_cone = sort_diagram(dgm_cone)

    if isinstance(tol, (list, tuple, np.ndarray)): # if is iterable, ndarray or list
        tol_data = {}
        for tol_i in tol:
            tol_data[tol_i] = kercoker_bars(dgm_cone, dgm_X, dgm_Y, tol_i)
        dgm_coker = {tol_i: tol_data[tol_i][0] for tol_i in tol}
        dgm_ker = {tol_i: tol_data[tol_i][1] for tol_i in tol}
        missing = {tol_i: tol_data[tol_i][2] for tol_i in tol}
    else:
        dgm_coker, dgm_ker, missing = kercoker_bars(dgm_cone, dgm_X, dgm_Y, tol)

    if return_extra:
        return dgm_coker, dgm_ker, dgm_cone, dgm_X, dgm_Y, D, missing
    return dgm_coker, dgm_ker, dgm_cone, dgm_X, dgm_Y

