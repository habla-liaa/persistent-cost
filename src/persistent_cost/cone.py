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


def general_position_distance_matrix(X, perturbation=1e-7):
    n = len(X)
    Xperturbation = perturbation * np.random.rand((n * (n - 1) // 2))
    dX = pdist(X) + Xperturbation
    return dX

def remove_empty_dims(bars):
    bars = [np.array(b) for b in bars]
    lens = list(map(len, bars)) # lengths of each dimension
    for i in range(len(bars)): # all dims
        if all(l == 0 for l in lens[i:]): 
            bars = bars[:i]
            break
    return bars


def kercoker_bars_(dgm, dgmX, dgmY, cone_eps, tol=1e-11):
    """
    Find cokernel and kernel bars in the persistence diagram.
    TODO: optimize
    """
    coker_dgm = [[] for _ in range(len(dgm))]
    ker_dgm = [[] for _ in range(len(dgm))]
    
    assert dgm[0][-1][1] == np.inf, "Expected infinite bar in dimension 0 of cone diagram."
    dgm[0] = dgm[0][:-1] # remove the infinite bar in dim 0

    for k in range(len(dgm)): # dimension cone diagram
        count = 0
        for r in dgm[k]:
            b, d = r
            if d > cone_eps + tol:
                # coker
                # b_c = b_y_i
                # d_c = d_y_i
                m = findclose(b, dgmY[k][:, 0], tol) & findclose(d, dgmY[k][:, 1], tol)
                if sum(m):
                    if sum(m)>1:
                        print("Encontre:",sum(m))
                    coker_dgm[k].append((b, d))
                    count +=1

                # b_c = b_y_i
                # d_c = b_x_j
                if any(findclose(b, dgmY[k][:, 0], tol)) and any(findclose(d, dgmX[k][:, 0], tol)):
                    coker_dgm[k].append((b, d))
                    count +=1

                # ker
                if k > 0:
                    # b_c = b_x_i (dim-1)
                    # d_c = d_x_i (dim-1)
                    m = findclose(b, dgmX[k - 1][:, 0], tol) & findclose(d, dgmX[k - 1][:, 1], tol)
                    if sum(m):
                        ker_dgm[k - 1].append((b, d))
                        count +=1

                    # b_c = d_y_i (dim-1)
                    # d_c = d_x_j (dim-1)
                    if any(findclose(b, dgmY[k - 1][:, 1], tol)) and any(findclose(d, dgmX[k - 1][:, 1], tol)):
                        ker_dgm[k - 1].append((b, d))
                        count +=1

        if count != len(dgm[k]):
            print(f"Warning: dimension {k}, found {count} bars, expected {len(dgm[k])} bars.")

    coker_dgm = remove_empty_dims(coker_dgm)
    ker_dgm = remove_empty_dims(ker_dgm)
    return coker_dgm, ker_dgm


def kercoker_bars(dgm, dgmX, dgmY, cone_eps, tol=1e-11):
    """
    Find cokernel and kernel bars in the persistence diagram.
    TODO: optimize
    """
    coker_dgm = [[] for _ in range(len(dgm))]
    ker_dgm = [[] for _ in range(len(dgm))]
    
    assert dgm[0][-1][1] == np.inf, "Expected infinite bar in dimension 0 of cone diagram."
    dgm[0] = dgm[0][:-1] # remove the infinite bar in dim 0
    
    for k in range(len(dgm)): # dimension cone diagram
        for r in dgm[k]:
            b, d = r
            if d > cone_eps + tol:
                coker_dgm[k].append((b, d))                
                # ker
                if k > 0:
                    # b_c = b_x_i (dim-1)
                    # d_c = d_x_i (dim-1)
                    m = findclose(b, dgmX[k - 1][:, 0], tol) & findclose(d, dgmX[k - 1][:, 1], tol)
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
    D[n : n + m, n : n + m] = DY

    D[0:n, n : n + m] = DY_fy
    D[n : n + m, 0:n] = DY_fy.T

    R = np.inf
    # R = max(DX.max(), DY_fy.max()) + 1 instead of np.inf

    D[n + m, n : n + m] = R
    D[n : n + m, n + m] = R

    D[n + m, :n] = eps
    D[:n, n + m] = eps

    return D

def conematrix(dX, dY, f, cone_eps=0.0):

    n = matrix_size_from_condensed(dX)
    m = matrix_size_from_condensed(dY)
    f = np.array(f)

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

    # dX     DY_fy
    # DY_fy  dY
    D = conematrix_blocks(squareform(dX), squareform(dY), DY_fy, cone_eps)
    return D


def cone_pipeline(X, Y, f, maxdim=1, perturbation=1e-7, cone_eps=0.0, tol=1e-11):
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

    dX = general_position_distance_matrix(X, perturbation=perturbation)
    dY = general_position_distance_matrix(Y, perturbation=perturbation)

    n = matrix_size_from_condensed(dX)
    m = matrix_size_from_condensed(dY)

    f = np.array(f)
    
    # dY_ff = d(f(x_i),f(x_j)) para todo i,j
    i, j = np.triu_indices(n, k=1)
    f_i, f_j = f[i], f[j]
    dY_ff = squareform(dY)[f_i, f_j]

    L = lipschitz(dX, dY_ff)

    dY = dY / L

    D = conematrix(dX, dY, f, cone_eps)

    print("Computing persistence diagrams...")
    dgm_X = ripser(squareform(dX), distance_matrix=True, maxdim=maxdim)["dgms"]
    dgm_Y = ripser(squareform(dY), distance_matrix=True, maxdim=maxdim)["dgms"]
    dgm_cone = ripser(D, maxdim=maxdim, distance_matrix=True)["dgms"]
    print("Done.")

    dgm_coker, dgm_ker = kercoker_bars(dgm_cone, dgm_X, dgm_Y, cone_eps, tol)
    return dgm_coker, dgm_ker, dgm_cone, dgm_X, dgm_Y