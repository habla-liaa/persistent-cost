from scipy import sparse
import numpy as np

def births_and_deaths(R, M, eps, maxdim=1):
    n = R.shape[1]
    cpivs = [column_pivot_k(R, k) for k in range(n)]
    births = []
    deaths = []
    dims = []

    for k in range(n):
        # número de entradas no nulas en filas 0..k-1 (slice hasta k)
        col = M[:k+1, k].tocsc()
        # contando los no ceros en esa parte de la columna
        dim = col.count_nonzero() - 1

        if dim <= maxdim:
            if cpivs[k] == -1:
                if k in cpivs:
                    j = cpivs.index(k)
                    if eps[k] < eps[j]:
                        births.append(eps[k])
                        deaths.append(eps[j])
                        dims.append(dim if dim > 0 else 0)
                else:
                    if dim < maxdim:
                        births.append(eps[k])
                        deaths.append(np.inf)
                        dims.append(dim if dim > 0 else 0)

    return births, deaths, dims

def column_pivots(R):
    """Returns the column pivots of a matrix R"""
    rs = []
    # R to csc matrix
    if not R.getformat() == 'csc':
        R = R.tocsc()
    for i, c1 in enumerate(R.indptr[:-1]):
        c2 = R.indptr[i+1]
        if c1 == c2:
            rs.append(-1)
        else:
            rs.append(R.indices[c2-1])


def column_pivot_k_dense(M,k):
    n = M.shape[1]
    piv = -1
    i = 0
    while i < n:
        if M[i,k]!=0:
            piv = i

        i+=1
    return piv

def do_pivot_dense(M):
    M = M.todense()
    R = M.copy()
    n = R.shape[1]
    cpivs = [column_pivot_k_dense(M, k) for k in range(n)]
    P = []
    vs = []

    for k in range(n):
        p = cpivs[k]
        while p in P and p != -1:
            j = cpivs.index(p)  # primera columna con ese pivote p
            
            col_new = (R[:, k] + R[:, j]) % 2

            R[:, k] = col_new

            cpivs[k] = column_pivot_k_dense(R, k)
            p = cpivs[k]

            vs.append((j, k))  # alpha=1 siempre en Z/2Z

        if p != -1 and p not in P:
            P.append(p)

    # Construimos la matriz V de transformaciones elementales
    V = np.eye(n, dtype=int)
    for j, k in vs:
        V[:, k] = (V[:, k] + V[:, j]) % 2

    return sparse.csc_matrix(R), sparse.csc_matrix(V)


def do_pivot(M):
    # M es scipy.sparse.csc_matrix
    R = M.copy().tolil()
    n = R.shape[1]

    cpivs = [column_pivot_k(R.tocsc(), k) for k in range(n)]  # lista pivote columna
    
    P = []   # pivotes ya usados
    vs = []  # lista de operaciones (alpha, j, k)

    for k in range(n):
        p = cpivs[k]
        while p in P and p != -1:
            j = cpivs.index(p)  # primera columna con ese pivote p

            # Operar columnas: R[:, k] = R[:, k] - R[:, j] mod 2 (alpha = 1)
            # En Z/2Z, resta = suma = XOR de vectores columna
            
            col_k = R[:, k].toarray().flatten()
            col_j = R[:, j].toarray().flatten()
            col_new = (col_k + col_j) % 2

            # Actualizar columna k en R
            R[:, k] = sparse.csc_matrix(col_new).T

            # Actualizamos pivote
            R_csc = R.tocsc()
            cpivs[k] = column_pivot_k(R_csc, k)
            p = cpivs[k]

            vs.append((j, k))  # alpha=1 siempre en Z/2Z

        if p != -1 and p not in P:
            P.append(p)

    # Construimos la matriz V de transformaciones elementales
    V = sparse.identity(n, format='lil', dtype=int)
    for j, k in vs:
        col_k = V[:, k].toarray().flatten()
        col_j = V[:, j].toarray().flatten()
        V[:, k] = sparse.csc_matrix((col_k + col_j) % 2).T

    return R.tocsc(), V.tocsc()

def column_pivot_k(M, k):
    """
    Devuelve el máximo índice en la columna k de la matriz dispersa M.
    """
    # M: scipy.sparse in CSC or CSR, formato con indices y indptr
    # k: columna (0-based)
    M_csc = M.tocsc() 
    col_start = M_csc.indptr[k]
    col_end = M_csc.indptr[k+1]
    row_indices = M_csc.indices[col_start:col_end]
    if len(row_indices) > 0:
        return int(max(row_indices))
    else:
        return -1
