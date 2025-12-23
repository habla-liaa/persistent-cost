"""
Benchmark de origen de births/deaths en la matriz CONE.

Este benchmark verifica de dónde provienen los valores de birth y death
en la persistencia del CONE: si vienen de distancias en X, en Y, o no
tienen correspondencia con ninguna.

Setup:
- X: nube de puntos generada
- Y: copia de X
- f: identidad (mapping i -> i)
- CONE: matriz de distancias del cone construida con X, Y, f

Comparamos los births/deaths del diagrama de CONE contra:
- Distancias de X
- Distancias de Y
- Diagrama de persistencia de X
- Diagrama de persistencia de Y
"""

from __future__ import annotations
from pathlib import Path
from typing import Callable
import pandas as pd

import numpy as np
from scipy.spatial.distance import pdist, squareform
import fire
import gudhi as gd
from tqdm import tqdm

from persistent_cost.utils.utils import compute_lipschitz_constant, conematrix

MAX_VALUE = 9999.0

# =============================================================================
# Generadores de nubes de puntos (copiados de benchmark_numerical_precision.py)
# =============================================================================

def sample_sphere(n: int, dim: int = 3, radius: float = 1.0, seed: int = 42) -> np.ndarray:
    """Muestreo uniforme de una esfera de dimensión dim-1 embebida en R^dim."""
    rng = np.random.default_rng(seed)
    points = rng.standard_normal((n, dim))
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = radius * points / norms
    return points


def sample_torus(n: int, R: float = 2.0, r: float = 1.0, seed: int = 42) -> np.ndarray:
    """Muestreo uniforme de un toro en R^3."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    return np.column_stack([x, y, z])


def sample_circle(n: int, radius: float = 1.0, seed: int = 42) -> np.ndarray:
    """Muestreo uniforme de un círculo en R^2."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack([x, y])


def sample_uniform_cube(n: int, dim: int = 3, seed: int = 42) -> np.ndarray:
    """Muestreo uniforme del cubo [0, 1]^dim."""
    rng = np.random.default_rng(seed)
    return rng.random((n, dim))


def sample_gaussian(n: int, dim: int = 3, seed: int = 42) -> np.ndarray:
    """Muestreo de una distribución gaussiana estándar."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim))


def sample_two_circles(n: int, seed: int = 42) -> np.ndarray:
    """Dos círculos separados."""
    rng = np.random.default_rng(seed)
    n1 = n // 2
    n2 = n - n1
    theta1 = rng.uniform(0, 2 * np.pi, n1)
    circle1 = np.column_stack([np.cos(theta1), np.sin(theta1)])
    theta2 = rng.uniform(0, 2 * np.pi, n2)
    circle2 = np.column_stack([np.cos(theta2) + 3, np.sin(theta2)])
    return np.vstack([circle1, circle2])


def sample_eight_figure(n: int, seed: int = 42) -> np.ndarray:
    """Figura de ocho."""
    rng = np.random.default_rng(seed)
    n1 = n // 2
    n2 = n - n1
    theta1 = rng.uniform(0, 2 * np.pi, n1)
    circle1 = np.column_stack([np.cos(theta1) - 1, np.sin(theta1)])
    theta2 = rng.uniform(0, 2 * np.pi, n2)
    circle2 = np.column_stack([np.cos(theta2) + 1, np.sin(theta2)])
    return np.vstack([circle1, circle2])


def sample_clusters(n: int, n_clusters: int = 3, dim: int = 2, seed: int = 42) -> np.ndarray:
    """Clusters gaussianos separados."""
    rng = np.random.default_rng(seed)
    points_per_cluster = n // n_clusters
    remaining = n - points_per_cluster * n_clusters
    clusters = []
    for i in range(n_clusters):
        center = rng.uniform(-5, 5, dim)
        n_pts = points_per_cluster + (1 if i < remaining else 0)
        cluster = center + 0.3 * rng.standard_normal((n_pts, dim))
        clusters.append(cluster)
    return np.vstack(clusters)


def sample_annulus(n: int, r_inner: float = 0.5, r_outer: float = 1.0, seed: int = 42) -> np.ndarray:
    """Anillo en R^2."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    r = np.sqrt(rng.uniform(r_inner**2, r_outer**2, n))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])


def sample_linked_circles(n: int, seed: int = 42) -> np.ndarray:
    """Dos círculos enlazados en R^3."""
    rng = np.random.default_rng(seed)
    n1 = n // 2
    n2 = n - n1
    theta1 = rng.uniform(0, 2 * np.pi, n1)
    circle1 = np.column_stack([np.cos(theta1), np.sin(theta1), np.zeros(n1)])
    theta2 = rng.uniform(0, 2 * np.pi, n2)
    circle2 = np.column_stack(
        [np.cos(theta2) + 0.5, np.zeros(n2), np.sin(theta2)])
    return np.vstack([circle1, circle2])


POINT_CLOUD_GENERATORS: dict[str, Callable] = {
    'sphere_2d': lambda n, seed: sample_sphere(n, dim=2, seed=seed),
    'sphere_3d': lambda n, seed: sample_sphere(n, dim=3, seed=seed),
    'sphere_4d': lambda n, seed: sample_sphere(n, dim=4, seed=seed),
    'torus': lambda n, seed: sample_torus(n, seed=seed),
    'circle': lambda n, seed: sample_circle(n, seed=seed),
    'uniform_2d': lambda n, seed: sample_uniform_cube(n, dim=2, seed=seed),
    'uniform_3d': lambda n, seed: sample_uniform_cube(n, dim=3, seed=seed),
    'gaussian_2d': lambda n, seed: sample_gaussian(n, dim=2, seed=seed),
    'gaussian_3d': lambda n, seed: sample_gaussian(n, dim=3, seed=seed),
    'two_circles': lambda n, seed: sample_two_circles(n, seed=seed),
    'eight_figure': lambda n, seed: sample_eight_figure(n, seed=seed),
    'clusters_2d': lambda n, seed: sample_clusters(n, dim=2, seed=seed),
    'clusters_3d': lambda n, seed: sample_clusters(n, dim=3, seed=seed),
    'annulus': lambda n, seed: sample_annulus(n, seed=seed),
    'linked_circles': lambda n, seed: sample_linked_circles(n, seed=seed),
}


def compute_persistence_gudhi(distance_matrix: np.ndarray, maxdim: int = 1) -> list:
    """
    Computa persistencia usando Gudhi desde una matriz de distancias.

    Returns:
        Lista de tuplas (dim, (birth, death))
    """
    if gd is None:
        raise RuntimeError("gudhi no está disponible")

    # Aceptar matrices cuadradas o vectores condensados
    if distance_matrix.ndim == 1:
        distance_matrix = squareform(distance_matrix)

    finite_mask = np.isfinite(distance_matrix)
    max_edge = float(np.max(distance_matrix[finite_mask])) * 1.1

    rips = gd.RipsComplex(
        distance_matrix=distance_matrix,
        max_edge_length=max_edge,
    )
    st = rips.create_simplex_tree(max_dimension=maxdim + 1)
    persistence = st.persistence()

    return [(dim, (birth, death)) for dim, (birth, death) in persistence if dim <= maxdim+1]


def compute_barcodes(distance_matrix: np.ndarray, maxdim: int = 1) -> list:
    """Devuelve una lista de barras como diccionarios (dim, birth, death)."""
    bars = []
    for dim, (birth, death) in compute_persistence_gudhi(distance_matrix, maxdim):
        bars.append({
            'dim': int(dim),
            'birth': float(birth),
            'death': float(death),
        })
    return bars


def match_value_to_bars(value: float, bars: list, tol: float) -> list:
    """Encuentra coincidencias (birth/death) dentro de la tolerancia en una lista de barras."""
    if not np.isfinite(value):
        return []

    matches = []
    for bar in bars:
        if np.isfinite(bar['birth']) and abs(bar['birth'] - value) <= tol:
            matches.append({
                'bar_dim': bar['dim'],
                'match_kind': 'birth',
                'bar_birth': bar['birth'],
                'bar_death': bar['death'],
                'delta': float(bar['birth'] - value),
            })
        if np.isfinite(bar['death']) and abs(bar['death'] - value) <= tol:
            matches.append({
                'bar_dim': bar['dim'],
                'match_kind': 'death',
                'bar_birth': bar['birth'],
                'bar_death': bar['death'],
                'delta': float(bar['death'] - value),
            })
    return matches

def analyze_cone(
    dX: np.ndarray,
    dY: np.ndarray,
    f: np.ndarray,
    cone_eps: float = 0.001,
    max_value: float = 9999.0,
    maxdim: int = 1,
    tol: float = 1e-10,
    match_tol: float = 1e-7,
) -> list:
    """
    Analiza de dónde provienen los births/deaths del CONE.

    Args:
        X: Nube de puntos X
        Y: Nube de puntos Y
        f: Mapping de X a Y
        cone_eps: Epsilon del cone
        max_value: Valor máximo para la matriz
        maxdim: Dimensión máxima de homología
        tol: Tolerancia para comparación de valores (distancias)
        match_tol: Tolerancia para hacer match con births/deaths de X e Y

    Returns:
        Diccionario con estadísticas de origen
    """

    # Distancias únicas (incluyendo 0)
    distances_X = np.sort(np.unique(dX))
    distances_Y = np.sort(np.unique(dY))

    # Construir matriz CONE usando la función del paquete
    D_cone = conematrix(dX, dY, f, cone_eps, max_value)

    # Calcular persistencia de X, Y y del CONE
    bars_X = compute_barcodes(squareform(dX), maxdim)
    bars_Y = compute_barcodes(squareform(dY), maxdim)
    persistence_cone = compute_barcodes(D_cone, maxdim)
    # Analizar cada birth y death
    data = []

    for bar_id, bar in enumerate(persistence_cone):
        for value, bod in ((bar['birth'], 'birth'), (bar['death'], 'death')):
            matches_x = [{**m, 'diagram': 'X'} for m in match_value_to_bars(value, bars_X, match_tol)]
            matches_y = [{**m, 'diagram': 'Y'} for m in match_value_to_bars(value, bars_Y, match_tol)]
            matches = matches_x + matches_y

            data.append({
                'bar_id': bar_id,
                'dim': bar['dim'],
                'isinf': bool(np.isinf(value)),
                'bod': bod,
                'value': float(value),
                'cone_eps': bool(np.any(np.abs(value - cone_eps) < tol)),
                'zero': bool(np.abs(value) < tol),
                'X_distance': bool(np.any(np.abs(distances_X - value) < tol)),
                'Y_distance': bool(np.any(np.abs(distances_Y - value) < tol)),
                'max_value': bool(np.any(np.abs(value - max_value) < tol)),
                'matches': matches,
                'n_matches': len(matches),
                'match_in_X': any(m['diagram'] == 'X' for m in matches),
                'match_in_Y': any(m['diagram'] == 'Y' for m in matches),
            })

    return data


def run_cone_bars_benchmark(
    generators: list[str] | None = None,
    n_points_list: list[int] | None = None,
    n_repeats: int = 3,
    maxdim: int = 1,
    cone_eps: float = 0.001,
    seed: int = 42,
    verbose: bool = True,
    progress: bool = True,
    mapping_types: list[str] | None = None,
    match_tol: float = 1e-7,
) -> list:
    """
    Ejecuta el benchmark de origen de valores CONE.

    Args:
        generators: Lista de generadores a usar (None = todos)
        n_points_list: Lista de tamaños de nube (None = [20, 50, 100])
        n_repeats: Número de repeticiones por configuración
        maxdim: Dimensión máxima de homología
        cone_eps: Epsilon del cone
        seed: Semilla base para reproducibilidad
        verbose: Mostrar resultados detallados
        progress: Mostrar barra de progreso
        mapping_types: ['inclusion', 'projection']
        match_tol: tolerancia para matching con barras de X/Y

    Returns:
        Lista de barsResult
    """
    if gd is None:
        print("ERROR: gudhi no está disponible")
        return []

    if generators is None:
        generators = list(POINT_CLOUD_GENERATORS.keys())

    if mapping_types is None:
        mapping_types = ['inclusion', 'projection']

    mapping_types = [m.strip().lower() for m in mapping_types]
    mapping_types = [m for m in mapping_types if m in {'inclusion', 'projection'}]
    if not mapping_types:
        print("ERROR: No hay tipos de mapeo válidos (use 'inclusion' o 'projection')")
        return []

    if n_points_list is None:
        n_points_list = [20, 50]

    # Validar generadores
    invalid = [g for g in generators if g not in POINT_CLOUD_GENERATORS]
    if invalid:
        print(f"WARNING: Generadores no reconocidos: {invalid}")
        generators = [g for g in generators if g in POINT_CLOUD_GENERATORS]

    if not generators:
        print("ERROR: No hay generadores válidos")
        return []

    # Valores de m a explorar (5 puntos en el rango) solo para inclusión
    n_m_values = 3

    def m_values_count(mapping: str) -> int:
        return n_m_values if mapping == 'inclusion' else 1
    
    L_scale_values = [0.1, 0.5, 1.0]

    total_iters = 0
    for mapping in mapping_types:
        total_iters += len(generators) * n_repeats * m_values_count(mapping) * len(L_scale_values) * len(n_points_list)

    if verbose:
        print("=" * 70)
        print("BENCHMARK DE ORIGEN DE VALORES CONE")
        print("=" * 70)
        print(f"Generadores: {generators}")
        print(f"Tamaños de nube: {n_points_list}")
        print(f"Valores de m (solo inclusión): 5 puntos en [1, n_points]")
        print(f"Repeticiones: {n_repeats}")
        print(f"Dimensión máxima: H{maxdim}")
        print(f"Cone epsilon: {cone_eps}")
        print(f"Mappings: {mapping_types} (inclusión: X⊂Y, projection: Y=proj(X))")
        print(f"Semilla base: {seed}")
        print(f"Total de pruebas: {total_iters}")
        print("=" * 70)


    if progress and tqdm is not None:
        pbar = tqdm(total=total_iters, desc="Analizando orígenes")
    else:
        pbar = None

    current_seed = seed
    results = []
    for mapping in mapping_types:
        for gen_name in generators:
            generator = POINT_CLOUD_GENERATORS[gen_name]

            for n_points in n_points_list:
                if mapping == 'inclusion':
                    m_values = np.linspace(1, n_points-1, n_m_values, dtype=int)
                    m_values = np.unique(m_values)
                else:  # projection: no filtramos elementos, usamos todos
                    m_values = np.array([n_points], dtype=int)
                
                for L_scale in L_scale_values:
                    for m in m_values:
                        for rep in range(n_repeats):
                            current_seed += 1

                            try:
                                base_points = generator(n_points, current_seed)

                                if mapping == 'inclusion':
                                    Y = base_points
                                    X = base_points[:m].copy()
                                elif mapping == 'projection':
                                    if base_points.shape[1] < 2:
                                        raise ValueError("Proyección requiere dimensión >=2 para eliminar una")
                                    Y = base_points[:, :-1]
                                    X = base_points  # no filtramos
                                    m = int(X.shape[0])
                                else:
                                    raise ValueError(f"Tipo de mapping no soportado: {mapping}")

                                f = np.arange(X.shape[0])

                                # Calcular distancias
                                dX = pdist(X)
                                dY = pdist(Y)

                                # Lipschitz normalizado
                                L = compute_lipschitz_constant(dX, dY, f)    
                                dY = dY / L * L_scale

                                # Analizar orígenes
                                data = analyze_cone(
                                    dX, dY, f,
                                    cone_eps=cone_eps,
                                    max_value=MAX_VALUE,
                                    maxdim=maxdim,
                                    match_tol=match_tol,
                                )

                                # add field to each dict in data
                                for d in data:
                                    d['generator'] = gen_name
                                    d['n_bars'] = len(data) // 2  # cada barra tiene birth y death
                                    d['n_points'] = n_points
                                    d['m'] = int(m)
                                    d['L_scale'] = L_scale
                                    d['L_lipschitz'] = L
                                    d['seed'] = current_seed
                                    d['mapping'] = mapping

                                results.extend(data)

                            except Exception as e:
                                if verbose:
                                    print(
                                        f"ERROR en {gen_name}, mapping={mapping}, n={n_points}, m={m}, seed={current_seed}: {e}")

                            if pbar is not None:
                                pbar.update(1)

    if pbar is not None:
        pbar.close()

    return results


def main(
    generators: str | None = None,
    n_points: str = "20,50",
    n_repeats: int = 3,
    maxdim: int = 2,
    cone_eps: float = 0.001,
    seed: int = 42,
    progress: bool = True,
    list_generators: bool = False,
    mapping_types: str | None = None,
    match_tol: float = 1e-11,
    output_path: str = "cone_bars_benchmark.json",
) -> None:
    """
    Benchmark de origen de valores CONE.

    Analiza de dónde provienen los births/deaths en la persistencia del CONE
    y genera un dataframe donde cada fila es un birth/death de una barra del
    CONE. Incluye coincidencias con barras de persistencia de X e Y dentro
    de una tolerancia y el tipo de mapping empleado.

    Args:
        generators: Generadores a usar, separados por coma (None = todos)
        n_points: Tamaños de nube, separados por coma
        n_repeats: Número de repeticiones por configuración
        maxdim: Dimensión máxima de homología
        cone_eps: Epsilon del cone
        seed: Semilla base para reproducibilidad
        progress: Mostrar barra de progreso
        list_generators: Solo listar generadores disponibles
        mapping_types: 'inclusion', 'projection' o lista separada por coma
        match_tol: tolerancia para matching con barras de X/Y
        output_path: Archivo de salida para el plot (None = mostrar)
    Ejemplo:
        python benchmark_cone_bars.py --n_points=20,50,100 --n_repeats=50
        python benchmark_cone_bars.py --generators=sphere_3d,torus --output=plot.png
    """
    if list_generators:
        print("Generadores disponibles:")
        for name in sorted(POINT_CLOUD_GENERATORS.keys()):
            print(f"  - {name}")
        return

    # Parsear argumentos
    gen_list = None
    if generators:
        if isinstance(generators, str):
            gen_list = [g.strip() for g in generators.split(",")]
        else:
            gen_list = list(generators)

    if isinstance(n_points, str):
        n_points_list = [int(x.strip()) for x in n_points.split(",")]
    elif isinstance(n_points, (list, tuple)):
        n_points_list = [int(x) for x in n_points]
    else:
        n_points_list = [int(n_points)]

    if mapping_types is None:
        mapping_list = None
    elif isinstance(mapping_types, str):
        mapping_list = [m.strip() for m in mapping_types.split(',')]
    else:
        mapping_list = list(mapping_types)

    # Ejecutar benchmark
    results = run_cone_bars_benchmark(
        generators=gen_list,
        n_points_list=n_points_list,
        n_repeats=n_repeats,
        maxdim=maxdim,
        cone_eps=cone_eps,
        seed=seed,
        verbose=True,
        progress=progress,
        mapping_types=mapping_list,
        match_tol=match_tol,
    )

    df = pd.DataFrame(results)

    df.to_json(output_path, orient="records", indent=2)
    print(f"Resultados guardados en {output_path}")

if __name__ == "__main__":
    fire.Fire(main)
