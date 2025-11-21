import numpy as np
from scipy.spatial.distance import pdist, squareform
import torch
from ripser import ripser


def findclose(x, A, tol=1e-5):
    # return ((x + tol) >= A) & ((x - tol) <= A)
    return np.abs(x-A) < tol


def lipschitz(dX, dY):
    return np.max(dY / dX)


def matrix_size_from_condensed(dX):
    n = len(dX)
    return int(0.5 * (np.sqrt(8 * n + 1) - 1) + 1)


def to_condensed_form(i, j, m):
    return m * i + j - ((i + 2) * (i + 1)) // 2.0


def general_position_distance_matrix_(X, perturbation=1e-7):
    n = len(X)
    Xperturbation = perturbation * np.random.rand((n * (n - 1) // 2))
    dX = pdist(X) + Xperturbation
    return dX


def general_position_distance_matrix(X, perturbation=1e-7):
    n = len(X)
    Xperturbation = perturbation * np.random.rand((n * (n - 1) // 2))
    # m = min(np.diff(np.sort(pdist(X))))
    p = pdist(X)
    # if m>1:
    #     p = np.round(p)
    dX = p + Xperturbation
    return dX


def remove_empty_dims(bars):
    bars = [np.array(b) for b in bars]
    lens = list(map(len, bars))  # lengths of each dimension
    for i in range(len(bars)):  # all dims
        if all(l == 0 for l in lens[i:]):
            bars = bars[:i]
            break
    return bars


def kercoker_bars(dgm, dgmX, dgmY, cone_eps, tol=1e-11):
    """
    Find cokernel and kernel bars in the persistence diagram.
    TODO: optimize
    """
    maxdim = len(dgm) - 1

    coker_dgm = [[] for _ in range(maxdim + 1)]
    ker_dgm = [[] for _ in range(maxdim + 1)]
    missing = [[] for _ in range(maxdim + 1)]

    # assert dgm[0][-1][1] == np.inf, "Expected infinite bar in dimension 0 of cone diagram."
    # dgm[0] = dgm[0][:-1]  # remove the infinite bar in dim 0

    for dim in range(maxdim + 1):  # dimension cone diagram
        for i, r in enumerate(dgm[dim]):
            newbar = False
            b, d = r
            if d > cone_eps + tol:
                # coker
                # b_c = b_y_i
                # d_c = d_y_i
                m = findclose(b, dgmY[dim][:, 0], tol) & findclose(
                    d, dgmY[dim][:, 1], tol)
                if sum(m):
                    # if sum(m) > 1:
                        # print("Encontre:", sum(m))
                    coker_dgm[dim].append((b, d))
                    newbar = True

                # b_c = b_y_i
                # d_c = b_x_j
                if any(findclose(b, dgmY[dim][:, 0], tol)) and any(findclose(d, dgmX[dim][:, 0], tol)):
                    coker_dgm[dim].append((b, d))
                    newbar = True

                # ker
                if dim > 0:
                    # b_c = b_x_i (dim-1)
                    # d_c = d_x_i (dim-1)
                    m = findclose(
                        b, dgmX[dim - 1][:, 0], tol) & findclose(d, dgmX[dim - 1][:, 1], tol)
                    if sum(m):
                        ker_dgm[dim - 1].append((b, d))
                        newbar = True

                    # b_c = d_y_i (dim-1)
                    # d_c = d_x_j (dim-1)
                    if any(findclose(b, dgmY[dim - 1][:, 1], tol)) and any(findclose(d, dgmX[dim - 1][:, 1], tol)):
                        ker_dgm[dim - 1].append((b, d))
                        newbar = True

                if not newbar:
                    # cone_bars.append((k, i, 'sobra', k,  'null', (b, d)))
                    missing[dim].append(('null', (b, d)))

    coker_dgm = remove_empty_dims(coker_dgm)
    ker_dgm = remove_empty_dims(ker_dgm)
    return coker_dgm, ker_dgm, missing


def kercoker_bars_(dgm, dgmX, dgmY, cone_eps, tol=1e-11):
    """
    Find cokernel and kernel bars in the persistence diagram.
    TODO: optimize
    """
    coker_dgm = [[] for _ in range(len(dgm))]
    ker_dgm = [[] for _ in range(len(dgm))]

    assert dgm[0][-1][1] == np.inf, "Expected infinite bar in dimension 0 of cone diagram."
    dgm[0] = dgm[0][:-1]  # remove the infinite bar in dim 0

    for k in range(len(dgm)):  # dimension cone diagram
        for r in dgm[k]:
            b, d = r
            if d > cone_eps + tol:
                coker_dgm[k].append((b, d))
                # ker
                if k > 0:
                    # b_c = b_x_i (dim-1)
                    # d_c = d_x_i (dim-1)
                    m = findclose(
                        b, dgmX[k - 1][:, 0], tol) & findclose(d, dgmX[k - 1][:, 1], tol)
                    if sum(m):
                        ker_dgm[k - 1].append((b, d))
                        coker_dgm[k].remove((b, d))

                    # b_c = d_y_i (dim-1)
                    # d_c = d_x_j (dim-1)
                    if any(findclose(b, dgmY[k - 1][:, 1], tol)) and any(findclose(d, dgmX[k - 1][:, 1], tol)):
                        ker_dgm[k - 1].append((b, d))
                        coker_dgm[k].remove((b, d))

    coker_dgm = remove_empty_dims(coker_dgm)
    ker_dgm = remove_empty_dims(ker_dgm)
    return coker_dgm, ker_dgm


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

    thresh = 3 * max(np.max(squareform(dX)), np.max(squareform(dY)))

    dgm_X = ripser(squareform(dX), distance_matrix=True, maxdim=maxdim, thresh=thresh)["dgms"]
    dgm_Y = ripser(squareform(dY), distance_matrix=True, maxdim=maxdim, thresh=thresh)["dgms"]
    dgm_cone = ripser(D, maxdim=maxdim, distance_matrix=True, thresh=thresh)["dgms"]

    dgm_coker, dgm_ker, missing = kercoker_bars(
        dgm_cone, dgm_X, dgm_Y, cone_eps, tol)
    if return_extra:
        return dgm_coker, dgm_ker, dgm_cone, dgm_X, dgm_Y, D, missing
    return dgm_coker, dgm_ker, dgm_cone, dgm_X, dgm_Y

from persistent_cost.utils.utils import htr, births_deaths_to_dgm


def cone_pipeline_htr(dX, dY, f, maxdim=1, cone_eps=0.0, tol=1e-11, threshold=3, return_extra=False):
    """
    Compute the total persistence diagram using the cone algorithm with HTR.

    Parameters
    ----------
    dX : np.array
        Distance matrix of the source space in condensed form.
    dY : np.array
        Distance matrix of the target space in condensed form.
    f : np.array
        Function values.
    maxdim : int, optional
        Maximum homology dimension to compute, by default 1
    cone_eps : float, optional
        Epsilon value for the cone construction, by default 0.0
    tol : float, optional
        Tolerance for numerical comparisons, by default 1e-11
    threshold : float, optional
        Maximum distance for Rips complex, by default 3
    return_extra : bool, optional
        Whether to return extra information, by default False
    
    Returns
    -------
    tuple
        (dgm_coker, dgm_ker, dgm_cone, dgm_X, dgm_Y) or with D and missing if return_extra=True
    """
    
    D = conematrix(dX, dY, f, cone_eps)
    
    # Compute persistence using htr with the new signature
    # htr returns (births, deaths, dims)
    births_X, deaths_X, dims_X = htr(distance_matrix=squareform(dX), threshold=threshold, maxdim=maxdim)
    births_Y, deaths_Y, dims_Y = htr(distance_matrix=squareform(dY), threshold=threshold, maxdim=maxdim)
    births_cone, deaths_cone, dims_cone = htr(distance_matrix=D, threshold=threshold, maxdim=maxdim)
    
    dgm_X = births_deaths_to_dgm(births_X, deaths_X, dims_X, maxdim)
    dgm_Y = births_deaths_to_dgm(births_Y, deaths_Y, dims_Y, maxdim)
    dgm_cone = births_deaths_to_dgm(births_cone, deaths_cone, dims_cone, maxdim)

    dgm_coker, dgm_ker, missing = kercoker_bars(
        dgm_cone, dgm_X, dgm_Y, cone_eps, tol)
    
    if return_extra:
        return dgm_coker, dgm_ker, dgm_cone, dgm_X, dgm_Y, D, missing
    return dgm_coker, dgm_ker, dgm_cone, dgm_X, dgm_Y
