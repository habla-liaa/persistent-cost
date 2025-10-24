from scipy import sparse

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

            vs.append((1, j, k))  # alpha=1 siempre en Z/2Z

        if p != -1 and p not in P:
            P.append(p)

    # Construimos la matriz V de transformaciones elementales
    V = sparse.identity(n, format='lil', dtype=int)
    for alpha, j, k in vs:
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