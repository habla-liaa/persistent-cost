"""
Benchmark de precisión numérica para Ripser.

Este benchmark verifica que los valores de birth y death reportados por Ripser
correspondan exactamente a distancias reales en la matriz de distancias de la
nube de puntos original.

La idea es que en un complejo de Rips filtrado:
- Cada birth corresponde al momento en que un simplejo aparece
- Para H0: births son 0, deaths son distancias entre puntos
- Para H1: births y deaths son distancias entre pares de puntos
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
import fire
from dataclasses import dataclass
from typing import Callable
import warnings

try:
    from ripser import ripser
except ImportError:
    ripser = None

try:
    import gudhi as gd
except ImportError:
    gd = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


# =============================================================================
# Generadores de nubes de puntos
# =============================================================================

def sample_sphere(n: int, dim: int = 3, radius: float = 1.0, seed: int = 42) -> np.ndarray:
    """Muestreo uniforme de una esfera de dimensión dim-1 embebida en R^dim."""
    rng = np.random.default_rng(seed)
    # Generar puntos gaussianos y normalizar
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
    """Dos círculos separados (para generar H0 con 2 componentes)."""
    rng = np.random.default_rng(seed)
    n1 = n // 2
    n2 = n - n1
    
    theta1 = rng.uniform(0, 2 * np.pi, n1)
    circle1 = np.column_stack([np.cos(theta1), np.sin(theta1)])
    
    theta2 = rng.uniform(0, 2 * np.pi, n2)
    circle2 = np.column_stack([np.cos(theta2) + 3, np.sin(theta2)])
    
    return np.vstack([circle1, circle2])


def sample_eight_figure(n: int, seed: int = 42) -> np.ndarray:
    """Figura de ocho (dos círculos tangentes, un H1 feature)."""
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
    """Anillo (disco con hueco) en R^2."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    # Muestreo uniforme en el área del anillo
    r = np.sqrt(rng.uniform(r_inner**2, r_outer**2, n))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])


def sample_linked_circles(n: int, seed: int = 42) -> np.ndarray:
    """Dos círculos enlazados en R^3 (Hopf link)."""
    rng = np.random.default_rng(seed)
    n1 = n // 2
    n2 = n - n1
    
    # Primer círculo en el plano xy
    theta1 = rng.uniform(0, 2 * np.pi, n1)
    circle1 = np.column_stack([np.cos(theta1), np.sin(theta1), np.zeros(n1)])
    
    # Segundo círculo en el plano xz, desplazado
    theta2 = rng.uniform(0, 2 * np.pi, n2)
    circle2 = np.column_stack([np.cos(theta2) + 0.5, np.zeros(n2), np.sin(theta2)])
    
    return np.vstack([circle1, circle2])


# Registro de generadores
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


# =============================================================================
# Verificación de precisión numérica
# =============================================================================

@dataclass
class PrecisionResult:
    """Resultado de verificación de precisión para una nube de puntos."""
    library: str  # 'ripser' o 'gudhi'
    generator: str
    n_points: int
    seed: int
    n_bars_total: int
    n_bars_checked: int
    max_error: float
    mean_error: float
    all_errors: list[float]  # todos los errores individuales


def find_closest_distance(value: float, distances: np.ndarray) -> tuple[float, float]:
    """
    Encuentra la distancia más cercana a un valor dado.
    Retorna (distancia_más_cercana, error).
    """
    if len(distances) == 0:
        return np.inf, np.inf
    
    errors = np.abs(distances - value)
    idx = np.argmin(errors)
    return distances[idx], errors[idx]


def check_ripser_precision(
    points: np.ndarray,
    maxdim: int = 1,
) -> tuple[dict, np.ndarray]:
    """
    Verifica que los births y deaths de Ripser correspondan a distancias reales.
    
    Args:
        points: Nube de puntos (n x d)
        maxdim: Dimensión máxima de homología
    
    Returns:
        dict con resultados de verificación, matriz de distancias usada
    """
    if ripser is None:
        raise RuntimeError("ripser no está disponible")
    
    # Calcular matriz de distancias
    dist_condensed = pdist(points)
    all_distances = np.sort(np.unique(dist_condensed))
    
    # Agregar 0 a las distancias (para births en H0)
    all_distances = np.concatenate([[0.0], all_distances])
    
    # Calcular persistencia con Ripser
    result = ripser(points, maxdim=maxdim)
    diagrams = result['dgms']
    
    # Verificar cada birth y death
    n_bars_total = 0
    n_bars_checked = 0
    all_errors = []
    
    for dim, dgm in enumerate(diagrams):
        for birth, death in dgm:
            n_bars_total += 1
            
            # Verificar birth
            if not np.isinf(birth):
                n_bars_checked += 1
                _, error = find_closest_distance(birth, all_distances)
                all_errors.append(error)
            
            # Verificar death (si no es infinito)
            if not np.isinf(death):
                n_bars_checked += 1
                _, error = find_closest_distance(death, all_distances)
                all_errors.append(error)
    
    max_error = max(all_errors) if all_errors else 0.0
    mean_error = np.mean(all_errors) if all_errors else 0.0
    
    return {
        'n_bars_total': n_bars_total,
        'n_bars_checked': n_bars_checked,
        'max_error': max_error,
        'mean_error': mean_error,
        'all_errors': all_errors,
    }, all_distances


def check_gudhi_precision(
    points: np.ndarray,
    maxdim: int = 1,
) -> tuple[dict, np.ndarray]:
    """
    Verifica que los births y deaths de Gudhi correspondan a distancias reales.
    
    Args:
        points: Nube de puntos (n x d)
        maxdim: Dimensión máxima de homología
    
    Returns:
        dict con resultados de verificación, matriz de distancias usada
    """
    if gd is None:
        raise RuntimeError("gudhi no está disponible")
    
    # Calcular matriz de distancias
    dist_condensed = pdist(points)
    all_distances = np.sort(np.unique(dist_condensed))
    
    # Agregar 0 a las distancias (para births en H0)
    all_distances = np.concatenate([[0.0], all_distances])
    
    # Calcular persistencia con Gudhi
    # Usar un threshold grande para capturar toda la persistencia
    max_edge = float(np.max(dist_condensed)) * 1.1
    rips_complex = gd.RipsComplex(points=points, max_edge_length=max_edge)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=maxdim + 1)
    persistence = simplex_tree.persistence()
    
    # Verificar cada birth y death
    n_bars_total = 0
    n_bars_checked = 0
    all_errors = []
    
    for dim, (birth, death) in persistence:
        if dim > maxdim:
            continue
        n_bars_total += 1
        
        # Verificar birth
        if not np.isinf(birth):
            n_bars_checked += 1
            _, error = find_closest_distance(birth, all_distances)
            all_errors.append(error)
        
        # Verificar death (si no es infinito)
        if not np.isinf(death):
            n_bars_checked += 1
            _, error = find_closest_distance(death, all_distances)
            all_errors.append(error)
    
    max_error = max(all_errors) if all_errors else 0.0
    mean_error = np.mean(all_errors) if all_errors else 0.0
    
    return {
        'n_bars_total': n_bars_total,
        'n_bars_checked': n_bars_checked,
        'max_error': max_error,
        'mean_error': mean_error,
        'all_errors': all_errors,
    }, all_distances


def run_precision_benchmark(
    generators: list[str] | None = None,
    n_points_list: list[int] | None = None,
    n_repeats: int = 10,
    maxdim: int = 1,
    seed: int = 42,
    verbose: bool = True,
    progress: bool = True,
    libraries: list[str] | None = None,
) -> list[PrecisionResult]:
    """
    Ejecuta el benchmark de precisión numérica.
    
    Args:
        generators: Lista de generadores a usar (None = todos)
        n_points_list: Lista de tamaños de nube (None = [20, 50, 100])
        n_repeats: Número de repeticiones por configuración
        maxdim: Dimensión máxima de homología
        seed: Semilla base para reproducibilidad
        verbose: Mostrar resultados detallados
        progress: Mostrar barra de progreso
        libraries: Lista de librerías a usar (None = todas disponibles)
    
    Returns:
        Lista de PrecisionResult
    """
    # Determinar librerías disponibles
    available_libs = []
    if ripser is not None:
        available_libs.append('ripser')
    if gd is not None:
        available_libs.append('gudhi')
    
    if not available_libs:
        print("ERROR: Ni ripser ni gudhi están disponibles")
        return []
    
    # Filtrar librerías solicitadas
    if libraries is None:
        libraries = available_libs
    else:
        libraries = [lib for lib in libraries if lib in available_libs]
        if not libraries:
            print(f"ERROR: Librerías solicitadas no disponibles. Disponibles: {available_libs}")
            return []
    
    if generators is None:
        generators = list(POINT_CLOUD_GENERATORS.keys())
    
    if n_points_list is None:
        n_points_list = [20, 50, 100]
    
    # Validar generadores
    invalid = [g for g in generators if g not in POINT_CLOUD_GENERATORS]
    if invalid:
        print(f"WARNING: Generadores no reconocidos: {invalid}")
        generators = [g for g in generators if g in POINT_CLOUD_GENERATORS]
    
    if not generators:
        print("ERROR: No hay generadores válidos")
        return []
    
    # Calcular total de iteraciones
    total_iters = len(libraries) * len(generators) * len(n_points_list) * n_repeats
    
    if verbose:
        print("=" * 70)
        print("BENCHMARK DE PRECISIÓN NUMÉRICA")
        print("=" * 70)
        print(f"Librerías: {libraries}")
        print(f"Generadores: {generators}")
        print(f"Tamaños de nube: {n_points_list}")
        print(f"Repeticiones: {n_repeats}")
        print(f"Dimensión máxima: H{maxdim}")
        print(f"Semilla base: {seed}")
        print(f"Total de pruebas: {total_iters}")
        print("=" * 70)
    
    results: list[PrecisionResult] = []
    
    # Iterador con progreso opcional
    if progress and tqdm is not None:
        pbar = tqdm(total=total_iters, desc="Verificando precisión")
    else:
        pbar = None
    
    current_seed = seed
    
    for lib_name in libraries:
        if lib_name == 'ripser':
            check_func = check_ripser_precision
        else:
            check_func = check_gudhi_precision
        
        for gen_name in generators:
            generator = POINT_CLOUD_GENERATORS[gen_name]
            
            for n_points in n_points_list:
                for rep in range(n_repeats):
                    current_seed += 1
                    
                    try:
                        # Generar nube de puntos
                        points = generator(n_points, current_seed)
                        
                        # Verificar precisión
                        check_result, _ = check_func(
                            points,
                            maxdim=maxdim,
                        )
                        
                        result = PrecisionResult(
                            library=lib_name,
                            generator=gen_name,
                            n_points=n_points,
                            seed=current_seed,
                            **check_result
                        )
                        results.append(result)
                        
                    except Exception as e:
                        if verbose:
                            print(f"ERROR en {lib_name}/{gen_name}, n={n_points}, seed={current_seed}: {e}")
                    
                    if pbar is not None:
                        pbar.update(1)
    
    if pbar is not None:
        pbar.close()
    
    return results


def print_results_summary(results: list[PrecisionResult]) -> None:
    """Imprime un resumen de los resultados del benchmark."""
    if not results:
        print("No hay resultados para mostrar")
        return
    
    print("\n" + "=" * 70)
    print("RESUMEN DE RESULTADOS")
    print("=" * 70)
    
    # Estadísticas generales
    total_bars = sum(r.n_bars_checked for r in results)
    all_errors = [e for r in results for e in r.all_errors]
    
    print(f"\nTotal de valores verificados: {total_bars}")
    
    # Por librería
    libraries = sorted(set(r.library for r in results))
    print("\n" + "-" * 70)
    print("RESULTADOS POR LIBRERÍA")
    print("-" * 70)
    
    for lib in libraries:
        lib_results = [r for r in results if r.library == lib]
        lib_errors = [e for r in lib_results for e in r.all_errors]
        if lib_errors:
            print(f"\n{lib.upper()}:")
            print(f"  Error máximo:  {max(lib_errors):.2e}")
            print(f"  Error medio:   {np.mean(lib_errors):.2e}")
            print(f"  Error mediana: {np.median(lib_errors):.2e}")
            print(f"  Error mínimo:  {min(lib_errors):.2e}")
    
    # Por generador
    print("\n" + "-" * 70)
    print("RESULTADOS POR GENERADOR")
    print("-" * 70)
    
    generators = sorted(set(r.generator for r in results))
    
    header = f"{'Generador':<18} {'Tests':>6} {'Valores':>8} {'Max Err':>12} {'Mean Err':>12}"
    print(header)
    print("-" * len(header))
    
    for gen in generators:
        gen_results = [r for r in results if r.generator == gen]
        n_tests = len(gen_results)
        n_values = sum(r.n_bars_checked for r in gen_results)
        max_err = max(r.max_error for r in gen_results)
        mean_err = np.mean([r.mean_error for r in gen_results])
        
        print(f"{gen:<18} {n_tests:>6} {n_values:>8} {max_err:>12.2e} {mean_err:>12.2e}")


def plot_results(results: list[PrecisionResult], output: str | None = None) -> None:
    """Genera los plots de los resultados del benchmark."""
    if not HAS_PLOTTING:
        print("WARNING: matplotlib y seaborn no disponibles, no se generará plot")
        print("Instalar con: pip install matplotlib seaborn")
        return
    
    import pandas as pd
    
    # Preparar datos para el plot
    data = []
    for r in results:
        # Agregar pequeño epsilon para evitar log(0)
        max_err = r.max_error if r.max_error > 0 else 1e-16
        data.append({
            'n': r.n_points,
            'library': r.library,
            'generator': r.generator,
            'max_error': max_err,
            'max_error_raw': r.max_error,
            'mean_error': r.mean_error,
        })
    
    df = pd.DataFrame(data)
    
    # Obtener librerías disponibles
    libraries = sorted(df['library'].unique())
    n_libs = len(libraries)
    
    # Crear figura con layout: filas por librería, 2 columnas, leyenda abajo
    fig = plt.figure(figsize=(16, 5 * n_libs + 2))
    gs = fig.add_gridspec(n_libs + 1, 2, height_ratios=[5] * n_libs + [1], hspace=0.35, wspace=0.15)
    
    # Obtener paleta de colores para generadores
    generators_sorted = sorted(df['generator'].unique())
    n_generators = len(generators_sorted)
    palette = sns.color_palette("husl", n_generators)
    palette_dict = {gen: palette[i] for i, gen in enumerate(generators_sorted)}
    
    n_repeats = len(df) // (n_libs * n_generators * len(df['n'].unique()))
    
    # Crear plots para cada librería
    for lib_idx, lib_name in enumerate(libraries):
        df_lib = df[df['library'] == lib_name]
        
        # Stripplot (columna izquierda)
        ax1 = fig.add_subplot(gs[lib_idx, 0])
        sns.stripplot(
            data=df_lib, 
            x='n', 
            y='max_error', 
            hue='generator',
            hue_order=generators_sorted,
            palette=palette_dict,
            dodge=True,
            alpha=0.7,
            size=5,
            ax=ax1,
            legend=False,
        )
        ax1.set_yscale('log')
        ax1.set_xlabel('Número de puntos (n)', fontsize=11)
        ax1.set_ylabel('Error máximo', fontsize=11)
        ax1.set_title(f'{lib_name.upper()} - Error por configuración', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Boxplot (columna derecha)
        ax2 = fig.add_subplot(gs[lib_idx, 1])
        sns.boxplot(
            data=df_lib,
            x='n',
            y='max_error',
            ax=ax2,
            color='lightblue' if lib_name == 'ripser' else 'lightgreen',
        )
      
        ax2.set_yscale('log')
        ax2.set_xlabel('Número de puntos (n)', fontsize=11)
        ax2.set_ylabel('Error máximo', fontsize=11)
        ax2.set_title(f'{lib_name.upper()} - Distribución por n', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
    # Leyenda en la fila inferior
    ax_legend = fig.add_subplot(gs[n_libs, :])
    ax_legend.axis('off')
    
    handles = [plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=palette_dict[gen], markersize=8, label=gen)
               for gen in generators_sorted]
    
    ax_legend.legend(handles=handles, loc='center', ncol=5, fontsize=9,
                     title='Tipo de nube de puntos', title_fontsize=10,
                     frameon=True, fancybox=True)
    
    plt.suptitle(
        f'Precisión numérica: Ripser vs Gudhi\n({n_repeats} repeticiones × {n_generators} generadores)',
        fontsize=14,
        y=0.99,
    )
    
    plt.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.08)
    
    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print(f"Plot guardado en: {output}")
    else:
        plt.show()


def main(
    generators: str | None = None,
    n_points: str = "20,50,100",
    n_repeats: int = 50,
    maxdim: int = 1,
    seed: int = 42,
    output: str | None = None,
    progress: bool = True,
    list_generators: bool = False,
) -> None:
    """
    Benchmark de precisión numérica para Ripser y Gudhi.
    
    Verifica que los valores de birth y death reportados por las librerías
    correspondan a distancias reales en la matriz de distancias.
    Ejecuta el benchmark completo y genera un plot comparativo.
    
    Args:
        generators: Generadores a usar, separados por coma (None = todos)
        n_points: Tamaños de nube, separados por coma
        n_repeats: Número de repeticiones por configuración
        maxdim: Dimensión máxima de homología
        seed: Semilla base para reproducibilidad
        output: Archivo de salida para guardar el plot (None = mostrar)
        progress: Mostrar barra de progreso
        list_generators: Solo listar generadores disponibles
    
    Ejemplo:
        python benchmark_numerical_precision.py --n_points=20,50,100 --n_repeats=50
        python benchmark_numerical_precision.py --generators=sphere_3d,torus --output=plot.png
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
    
    # Parsear n_points (puede ser string, tuple, o lista)
    if isinstance(n_points, str):
        n_points_list = [int(x.strip()) for x in n_points.split(",")]
    elif isinstance(n_points, (list, tuple)):
        n_points_list = [int(x) for x in n_points]
    else:
        n_points_list = [int(n_points)]
    
    # Ejecutar benchmark
    results = run_precision_benchmark(
        generators=gen_list,
        n_points_list=n_points_list,
        n_repeats=n_repeats,
        maxdim=maxdim,
        seed=seed,
        verbose=True,
        progress=progress,
    )
    
    if not results:
        print("No hay resultados")
        return
    
    # Mostrar resumen
    print_results_summary(results)
    
    # Generar plot
    plot_results(results, output=output)


if __name__ == "__main__":
    fire.Fire(main)
