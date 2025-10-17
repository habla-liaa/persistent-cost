import numpy as np
from scipy.spatial.distance import pdist, squareform
import torch

def findclose(x, A, tol=1e-5):
    return ((x + tol) >= A) & ((x - tol) <= A)


def lipschitz(dX, dY):
    return np.max(dY / dX)


def matrix_size_from_condensed(dX):
    n = len(dX)
    return int(0.5 * (np.sqrt(8 * n + 1) - 1) + 1)


def to_condensed_form(i, j, m):
    return m * i + j - ((i + 2) * (i + 1)) // 2.0


def general_position_distance_matrix(X, perturb=1e-7):
    n = len(X)
    Xperturbation = perturb * np.random.rand((n * (n - 1) // 2))
    dX = pdist(X) + Xperturbation
    return dX


def conematrix(DX, DY, DY_fy, eps):
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


def format_bars(bars):
    bars = [np.array(b) for b in bars]
    lens = list(map(len, bars))
    for i in range(len(bars)):
        if all(l == 0 for l in lens[i:]):
            bars = bars[:i]
            break
    return bars


def kercoker_bars(dgm, dgmX, dgmY, cone_eps, tol=1e-11):
    """
    Find cokernel and kernel bars in the persistence diagram.
    TODO: optimize
    """
    coker_dgm = [[] for _ in range(len(dgm))]
    ker_dgm = [[] for _ in range(len(dgm))]
    for k in range(len(dgm)): # dimension cone diagram
        for r in dgm[k]:
            b, d = r
            if d > cone_eps + tol:
                # coker
                # b_c = b_y_i
                # d_c = d_y_i
                m = findclose(b, dgmY[k][:, 0], tol) & findclose(d, dgmY[k][:, 1], tol)
                if sum(m):
                    coker_dgm[k].append((b, d))

                # b_c = b_y_i
                # d_c = b_x_j
                if any(findclose(b, dgmY[k][:, 0], tol)) and any(findclose(d, dgmX[k][:, 0], tol)):
                    coker_dgm[k].append((b, d))

                # ker
                if k > 0:
                    # b_c = b_x_i (dim-1)
                    # d_c = d_x_i (dim-1)
                    m = findclose(b, dgmX[k - 1][:, 0], tol) & findclose(d, dgmX[k - 1][:, 1], tol)
                    if sum(m):
                        ker_dgm[k - 1].append((b, d))

                    # b_c = d_y_i (dim-1)
                    # d_c = d_x_j (dim-1)
                    if any(findclose(b, dgmY[k - 1][:, 1], tol)) and any(findclose(d, dgmX[k - 1][:, 1], tol)):
                        ker_dgm[k - 1].append((b, d))

    coker_dgm = format_bars(coker_dgm)
    ker_dgm = format_bars(ker_dgm)
    return coker_dgm, ker_dgm


DEBUG = True
def log(*args, **kwargs):
    """
    Log messages if DEBUG is True.
    """
    if DEBUG:
        print(*args, **kwargs)


#####################################################################################
def conematrix_numpy(dX:torch.Tensor, dY:torch.Tensor, f:torch.Tensor, maxdim=1, cone_eps=0.0, tol=1e-11):
    n = matrix_size_from_condensed(dX)
    m = matrix_size_from_condensed(dY)

    f = np.array(f)

    i, j = np.triu_indices(n, k=1)
    f_i, f_j = f[i], f[j]
    
    dY_ff = squareform(dY)[f_i, f_j]
    
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

    L = lipschitz(dX, dY_ff)
    log(f"lipschitz constant: {L:.2f}")

    dY = dY / L

    DY_fy[i, j] = squareform(dY)[f_i, j]

    D = conematrix(squareform(dX), squareform(dY), DY_fy, cone_eps)
    return D
