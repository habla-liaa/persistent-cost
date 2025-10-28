from scipy import sparse
from scipy.spatial.distance import pdist, squareform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gudhi as gd
from .utils.utils import matrix_size_from_condensed, match_simplices, build_ordered_boundary_matrix
from .utils.algorithms import column_pivot_k, do_pivot


def cilinder_pipeline(X, Y, f, threshold, maxdim):
    dX = pdist(X)
    dY = pdist(Y)

    PX, simplices_X  = build_ordered_boundary_matrix(distance_matrix=squareform(dX), threshold = threshold, maxdim = 2)
    PY, simplices_Y  = build_ordered_boundary_matrix(distance_matrix=squareform(dY), threshold = threshold, maxdim = 2)

    cilindro = cylindermatrix(dX,dY,f)
    f = np.arange(len(X)) # por cómo armamos la matriz del cilindro,
                        # X corresponde a los primeros elementos 

    PX, simplices_X  = build_ordered_boundary_matrix(distance_matrix=squareform(dX), threshold = threshold, maxdim = 2)
    Pcyl, simplices_cyl  = build_ordered_boundary_matrix(distance_matrix=cilindro, threshold = threshold, maxdim = 2)

    # Armado de las matrices para pasos algebraicos
    D_f = Pcyl
    D_g = PX
    f_simpl = match_simplices(f, simplices_X, simplices_cyl)
    mapping_L = [int(x) for x in f_simpl]

    simplices_Y_dim = {s['idx']:s['dim'] for s in simplices_Y}
    simplices_X_dim = {s['idx']:s['dim'] for s in simplices_X}
    simplices_cyl_dim = {s['idx']:s['dim'] for s in simplices_cyl}

    # Pasos algebraicos para núcleos, imágenes y conúcleos
    R_g, V_g, R_f, V_f = step1(D_g, D_f)
    R_im, V_im = step2(D_f, D_g, mapping_L)
    R_ker, V_ker, cycle_columns_Vim = step3(R_im, V_im, mapping_L)
    R_cok, V_cok = step4(D_f, V_g, R_g, mapping_L)

        # Detección de nacimientos y muertes
    births_ker, bars_ker = detect_births_deaths_kernels(
            R_f, R_g, R_im, R_ker,
            cycle_columns_Vim, simplices_cyl_dim, mapping_L, maxdim=maxdim
        )

    births_cok, bars_cok = detect_births_deaths_cokernels(
            R_f, R_g, R_im, R_cok, simplices_cyl_dim, mapping_L, maxdim=maxdim
        )

    # Diagramas de barras por componente
    d_ker = barcode(births_ker, bars_ker, simplices_X, maxdim=maxdim)[0]
    d_cok = barcode(births_cok, bars_cok, simplices_cyl, maxdim=maxdim)[0]

    return d_ker, d_cok

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

def step1(D_g, D_f):
    """
    Realiza el Step 1 del algoritmo con matrices dispersas.
    Reduce las matrices de borde D_g (para L) y D_f (para K).

    Args:
        D_g: Matriz de borde dispersa del subcomplejo L.
        D_f: Matriz de borde dispersa del complejo K.

    Returns:
        R_g, V_g: Matriz reducida y transformadora para L.
        R_f, V_f: Matriz reducida y transformadora para K.
    """
    R_g, V_g = do_pivot(D_g)
    R_f, V_f = do_pivot(D_f)
    return R_g, V_g, R_f, V_f


def step2(D_f, D_g, mapping_L):
    """
    Construye y reduce D_im reorganizando las filas de D_f para que coincidan
    con el orden de L seguido por K - L.

    Args:
        D_f: Matriz de borde del complejo K (scipy.sparse).
        D_g: Matriz de borde del subcomplejo L (no usado aquí).
        mapping_L: lista o array con los índices de las filas correspondientes a L (0-based).

    Returns:
        R_im, V_im: Matriz reducida y transformadora para D_im.
    """
    total_rows = D_f.shape[0]
    rows_in_L =  [i for i in range(total_rows) if i in mapping_L]
    rows_not_in_L = [i for i in range(total_rows) if i not in rows_in_L]

    # filas reordenadas: primero las de L, luego las que no están en L
    reordered_rows = rows_in_L + rows_not_in_L

    # Indexación por filas en scipy.sparse (convertir a lista)
    D_im = D_f[reordered_rows, :]
    R_im, V_im = do_pivot(D_im)

    return R_im, V_im


def step3(R_im, V_im, mapping_L):
    """
    Construye y reduce D_ker basado en las columnas de R_im que representan ciclos.

    Args:
        R_im: Matriz reducida obtenida de D_im 
        V_im: Matriz de transformación obtenida de la reducción de D_im (scipy.sparse o densa).
        mapping_L: Lista o array con los índices de las filas correspondientes a L (0-based).

    Returns:
        R_ker, V_ker: Matriz reducida y transformadora para D_ker.
        cycle_columns_Vim: Lista de índices de columnas que representan ciclos.
    """
    n_cols = R_im.shape[1]
    # Columnas con todo cero en R_im (columnas que representan ciclos)
    cycle_columns_Vim = [c for c in range(n_cols) if R_im[:, c].getnnz() == 0]

    # Extraer las columnas correspondientes de V_im
    D_ker = V_im[:, cycle_columns_Vim]

    total_rows = R_im.shape[0]
    rows_in_L = [i for i in range(total_rows) if i in mapping_L]
    rows_not_in_L = [i for i in range(total_rows) if i not in rows_in_L]
    reordered_rows = rows_in_L + rows_not_in_L

    # Reordenar filas de D_ker
    D_ker = D_ker[reordered_rows, :]

    # Reducir D_ker
    R_ker, V_ker = do_pivot(D_ker)

    return R_ker, V_ker, cycle_columns_Vim

def step4(D_f, V_g, R_g, mapping_L):
    """
    Construye y reduce D_cok reemplazando columnas de D_f con las correspondientes de V_g.

    Args:
        D_f: Matriz dispersa del complejo K.
        V_g: Matriz de transformación obtenida al reducir D_g.
        R_g: Matriz reducida de D_g (para detectar ciclos).
        mapping_L: Lista o array con índices de las columnas de D_f correspondientes a L (0-based).

    Returns:
        R_cok, V_cok: Matriz reducida y transformadora para D_cok.
    """
    
    # Copia en formato lil para facilitar asignaciones
    D_cok = D_f.tolil()

    # Columnas de R_g que son ciclos (columnas con todo cero)
    cycle_columns = [c for c in range(R_g.shape[1]) if R_g[:, c].getnnz() == 0]

    # Índices en D_f correspondientes a esas columnas
    index_cycle_columns_f = [mapping_L[c] for c in cycle_columns]

    total_rows = D_f.shape[0]
    rows_in_L = [i for i in range(total_rows) if i in mapping_L]
    rows_not_in_L = [i for i in range(total_rows) if i not in rows_in_L]

    for j_idx, idx_f in enumerate(index_cycle_columns_f):
        idx_g = cycle_columns[j_idx]  # columna de V_g con el ciclo
        
        # Poner en filas de L la columna correspondiente de V_g
        # Aseguramos que V_g esté en formato compatible (dense o sparse)
        if hasattr(V_g, "tocsc"):
            col_data = V_g[:, idx_g].toarray().ravel()
        else:
            col_data = V_g[:, idx_g]
        
        for row_idx, val in zip(rows_in_L, col_data):
            D_cok[row_idx, idx_f] = val
        
        # En filas no L poner cero
        for row_idx in rows_not_in_L:
            D_cok[row_idx, idx_f] = 0

    # Convertir a formato CSR para operaciones eficientes
    D_cok = D_cok.tocsr()

    # Reducir D_cok
    R_cok, V_cok = do_pivot(D_cok)

    return R_cok, V_cok

def detect_births_deaths_kernels(R_f, R_g, R_im, R_ker, cycle_columns_Vim, f_dimensions, mapping_L, maxdim):
    """
    Detecta barras del núcleo.

    Args:
        R_f, R_g, R_im, R_ker: matrices reducidas (scipy.sparse).
        cycle_columns_Vim: lista con índices de columnas ciclo en V_im (0-based).
        mapping_L: lista con índices de filas/columnas correspondientes a L (0-based).
        maxdim : hasta qué dimensión de H calcular
        f_dimensions : diccionario con la dimension de cada simplex (key: numero de columna, val: dim)
        g_dimensions : similar con las col de Rg
    Returns:
        births: lista con índices de nacimientos (simplices).
        bars: lista de pares (nacimiento, muerte).
    """
    births = []
    bars = []

    n_f = R_f.shape[1]
    n_g = R_g.shape[1]

    set_mapping_L = set(mapping_L)

    # Nacimientos: simplices en K-L (indices no en mapping_L), columna no nula en R_f,
    # y el pivot en R_im para esa columna es <= cantidad de elementos en L (o sea corresponde a L)
    # dim <= maxdim

    for i in range(n_f):
        if (f_dimensions[i] <= maxdim+1) and (i not in set_mapping_L) and (R_f[:, i].getnnz() > 0):
            pivot = column_pivot_k(R_im, i)
            if 0 <= pivot < len(mapping_L):
                births.append(i) #### la dimension es 1 menos!! la escala es la de sigma, la dim es -1

    # Muertes: simplices en L (índices en mapping_L),
    # negativos en R_g (columna no nula), positivos en R_f (columna nula),
    # y el pivot en R_ker para la columna correspondiente
    for j in range(n_g):
        ### TODO corregir, entender cuál es la
        ### columna de Rker que corresponde al j
        ### para buscar ahi el pivot.
        ## en vim borre algunas, me quede con las que eran ciclos
        ## cycle_columns_Vim
        ## 
        if R_g[:, j].getnnz() > 0 and R_f[:, mapping_L[j]].getnnz() == 0:
            print(j)
            low = column_pivot_k(R_ker, cycle_columns_Vim[mapping_L[j]])
            bars.append((low, j))

    return births, bars

def detect_births_deaths_images(R_g, R_f, R_im, g_dimensions, mapping_L, maxdim):
    """
    Detecta barras de la imagen Im(g→f).

    Args:
        R_g, R_f, R_im: matrices reducidas (scipy.sparse).
        mapping_L: lista con índices de L (0-based).

    Returns:
        births: lista con índices de nacimientos (simplices en K).
        bars: lista de pares (nacimiento, muerte).
    """
    births = []
    bars = []

    # Nacimientos: columnas positivas en R_g, devolver índice según mapping_L
    # dim <= maxdim
    for i in range(R_g.shape[1]):
        if (g_dimensions[i] <= maxdim) and R_g[:, i].getnnz() == 0:  # columna toda cero = positiva
            births.append(mapping_L[i])

    # Muertes: columnas negativas en R_f (no toda cero)
    # y pivot en R_im corresponde a índice de L
    for j in range(R_f.shape[1]):
        if R_f[:, j].getnnz() > 0:
            low = int(column_pivot_k(R_im, j))
            if 0 <= low < len(mapping_L):
                bars.append((mapping_L[low], j))

    return births, bars

def detect_births_deaths_cokernels(R_f, R_g, R_im, R_cok, f_dimensions, mapping_L, maxdim):
    """
    Detecta barras del cociente cokernel.

    Args:
        R_f, R_g, R_im, R_cok: matrices reducidas (scipy.sparse).
        mapping_L: lista con índices de L (0-based).

    Returns:
        births: lista con índices de nacimientos (simplices en K).
        bars: lista de pares (nacimiento, muerte).
    """
    births = []
    bars = []

    # Nacimientos en Cokernel
    # dim <= maxdim

    for i in range(R_f.shape[1]):
        if (f_dimensions[i] <= maxdim) and R_f[:, i].getnnz() == 0:  # columna toda cero = positiva
            if i not in mapping_L:
                births.append(i)
            else:
                # índice de i dentro de mapping_L (0-based)
                idx = mapping_L.index(i)
                if R_g[:, idx].getnnz() > 0:  # negativa en R_g
                    births.append(i)

    # Muertes en Cokernel
    for j in range(R_f.shape[1]):
        if R_f[:, j].getnnz() > 0:
            low_im = column_pivot_k(R_im, j)
            if low_im >= len(mapping_L):  # fila corresponde a K-L (reordenadas)
                low_cok = column_pivot_k(R_cok, j)
                bars.append((low_cok, j))

    return births, bars

def barcode(births, bars, simplices, maxdim):
    """
    Crea una lista de barras a partir de nacimientos y muertes.
    

    Args:
        births: lista de índices de nacimientos (0-based).
        bars: lista de pares (nacimiento, muerte).
        simplices: información de los símplices (lista de diccionarios) 
        en los que se expresan los births y deaths,
        cada uno con claves 'idx', 'vertices', 
        'dim' y 'eps' para indicar la dimensión y escala.

    Returns:
        barcode: diccionario con listas de barras por dimensión,
        cada una con eps de nacimiento y muerte.
        barcode_idx: diccionario _con índices_ de barras por dimensión.
    """
    df_simplices = pd.DataFrame(simplices)
    barcode = {}
    barcode_idx = {}
    births_finite_idx = []

    for d in range(df_simplices['dim'].max()+1):
        if d <= maxdim:
            barcode[d] = []
            barcode_idx[d] = []

    for n,m in bars:
        dim_bar = int(df_simplices.loc[df_simplices['idx'] == n, 'dim'].values[0])
        eps_n = float(df_simplices.loc[df_simplices['idx'] == n, 'eps'].values[0])
        eps_m = float(df_simplices.loc[df_simplices['idx'] == m, 'eps'].values[0])

        barcode[dim_bar].append((eps_n, eps_m))
        barcode_idx[dim_bar].append((n, m))

        births_finite_idx.append(n)

    for n in births:
        # if there is no bar in bars starting at n, create an infinite bar in barcode
        if n not in births_finite_idx:
            dim_bar = int(df_simplices.loc[df_simplices['idx'] == n, 'dim'].values[0])
            eps_n = float(df_simplices.loc[df_simplices['idx'] == n, 'eps'].values[0])
            if dim_bar <= maxdim:
                barcode[dim_bar].append((eps_n, float('inf')))
                barcode_idx[dim_bar].append((n, None))

    return barcode, barcode_idx