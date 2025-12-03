"""
Script principal para ejecutar todos los experimentos y guardar resultados.
"""

from typing import Union
import json
import pickle
from pathlib import Path
from IPython import embed
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import math
import fire

from generate_spaces import EXPERIMENTS

# Importar los pipelines desde persistent_cost
from persistent_cost.cone import cone_pipeline
from persistent_cost.cone_pairs import cone_pipeline as cone_pairs_pipeline
from persistent_cost.cone_htr import cone_pipeline_htr
from persistent_cost.cone_gd import cone_pipeline as cone_gd_pipeline
from persistent_cost.cylinder import cylinder_dgm, cylinder_pipeline
from persistent_cost.utils.utils import compute_lipschitz_constant, sort_diagram


class SafeJSONEncoder(json.JSONEncoder):
    """
    Encoder personalizado que convierte valores infinitos/NaN a null.
    """

    def default(self, obj):
        if isinstance(obj, (float, np.floating)):
            if math.isinf(obj) or math.isnan(obj):
                return None
            return float(obj)
        elif isinstance(obj, (int, np.integer)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def convert_infinity_to_null(obj):
    """
    Convierte recursivamente valores Infinity a null para serialización JSON válida.
    """
    if isinstance(obj, dict):
        return {k: convert_infinity_to_null(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # Manejar tanto listas como tuplas
        return [convert_infinity_to_null(item) for item in obj]
    elif isinstance(obj, (float, np.floating)):
        # Manejar tanto float de Python como numpy float
        if math.isinf(obj):
            return None
        elif math.isnan(obj):
            return None
        return float(obj)
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        # Convertir numpy array a lista y procesar
        return convert_infinity_to_null(obj.tolist())
    return obj


def execute_cone_algorithm(algorithm_name, pipeline_func, dX, dY, f, X, Y, maxdim, cone_eps, tol, verbose):
    """
    Ejecuta un algoritmo cone genérico y formatea los resultados.

    Args:
        algorithm_name: Nombre del algoritmo para logging ('cone', 'cone_pairs', 'cone_htr', 'cone_gd')
        pipeline_func: Función pipeline a ejecutar
        dX, dY, f: Datos de entrada
        X, Y: Espacios originales (para guardar en resultados)
        maxdim: Dimensión homológica máxima
        cone_eps: Epsilon para construcción del cono
        verbose: Mostrar información detallada

    Returns:
        dict con resultados formateados
    """
    if verbose:
        print(f"\nEjecutando {algorithm_name}...")

    results_tuple = pipeline_func(
        dX, dY, f,
        maxdim=maxdim,
        cone_eps=cone_eps,
        tol=tol,
        return_extra=True
    )

    # Todos los algoritmos retornan al menos: (dgm_coker, dgm_ker, dgm_cone, dgm_X, dgm_Y, ...)
    # cone y cone_gd retornan 7 elementos: dgm_coker, dgm_ker, dgm_cone, dgm_X, dgm_Y, D, missing
    # cone_htr retorna 7 elementos: dgm_coker, dgm_ker, dgm_cone, dgm_X, dgm_Y, D, missing
    # cone_pairs retorna 15 elementos: incluye pairs y simpl2dist

    if len(results_tuple) >= 7:
        dgm_coker, dgm_ker, dgm_cone, dgm_X, dgm_Y = results_tuple[:5]
        missing = results_tuple[6] if len(results_tuple) > 6 else []
    else:
        # Fallback por si acaso
        dgm_coker, dgm_ker, dgm_cone, dgm_X, dgm_Y = results_tuple[:5]
        missing = []

    # Calcular cylinder_dgm para comparación
    dgm_cylinder = cylinder_dgm(dX, dY, f, maxdim)

    # Sort diagrams lexicográficamente
    dgm_cylinder = sort_diagram(dgm_cylinder)

    # Formatear resultados
    result_dict = {
        'X': X.tolist(),
        'Y': Y.tolist(),
        'f': f if isinstance(f, list) else f.tolist(),
        'dgm_coker': dgm_coker,
        'dgm_ker': dgm_ker,
        'dgm_cone': dgm_cone,
        'dgm_cylinder': dgm_cylinder,
        'dgm_X': dgm_X,
        'dgm_Y': dgm_Y,
        'missing': missing,
    }

    if verbose:
        print(
            f"  {algorithm_name} completado. Ker bars: {len(dgm_ker)}, Coker bars: {len(dgm_coker)}")

    return result_dict


def run_single_experiment(experiment_name, n, dim=2, threshold=3.0, maxdim=2, cone_eps=0.0, seed=42, tol=1e-11, verbose=True,
                          run_cone=True, run_cone_pairs=True, run_cone_htr=True, run_cone_gd=True, run_cylinder=True):
    """
    Ejecuta un experimento individual con los métodos disponibles.

    Args:
        experiment_name: nombre del experimento
        n: número de puntos
        dim: dimensión del espacio
        threshold: threshold para construcción de complejos
        maxdim: dimensión homológica máxima
        seed: semilla para reproducibilidad
        verbose: mostrar información detallada
        run_cone: ejecutar método cone (default: True)
        run_cone_pairs: ejecutar método cone_pairs (default: True)
        run_htr: ejecutar método cone_htr (default: True)
        run_cone_gd: ejecutar método cone_gd (default: True)
        run_cylinder: ejecutar método cylinder (default: True)

    Returns:
        dict con todos los resultados del experimento
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Experimento: {experiment_name}, n={n}, dim={dim}")
        print(f"{'='*60}")

    # Generar espacios
    experiment_func = EXPERIMENTS[experiment_name]

    # Manejar casos especiales
    if experiment_name == 'toro_proyecta':
        X, Y, f = experiment_func(n=n, seed=seed)
    elif experiment_name == 'circulo_en_toro':
        X, Y, f = experiment_func(n=n, seed=seed)
    else:
        X, Y, f = experiment_func(n=n, dim=dim, seed=seed)

    if verbose:
        print(f"Tamaños: X={X.shape}, Y={Y.shape}, f={len(f)}")

    # Calcular constante de Lipschitz
    L = compute_lipschitz_constant(X, Y, f)
    if verbose:
        print(f"Constante de Lipschitz (antes de normalización): {L:.6f}")

    # Calcular matrices de distancia
    dX = pdist(X)
    dY = pdist(Y)

    dY = dY / L

    results = {
        'experiment_name': experiment_name,
        'n': n,
        'dim': dim,
        'lipschitz_constant': float(L),
        'X_shape': X.shape,
        'Y_shape': Y.shape,
        'seed': seed,
        'cone_eps': cone_eps,
    }

    # Ejecutar cone (método 1)
    if run_cone:
        results['cone'] = execute_cone_algorithm(
            'cone', cone_pipeline, dX, dY, f, X, Y, maxdim, cone_eps, tol, verbose
        )

    # Ejecutar cone_pairs (método 2)
    if run_cone_pairs:
        results['cone_pairs'] = execute_cone_algorithm(
            'cone_pairs', cone_pairs_pipeline, dX, dY, f, X, Y, maxdim, cone_eps, tol, verbose
        )

    # Ejecutar cone_htr (método con HTR)
    if run_cone_htr:
        results['cone_htr'] = execute_cone_algorithm(
            'cone_htr', cone_pipeline_htr, dX, dY, f, X, Y, maxdim, cone_eps, tol, verbose
        )

    # Ejecutar cone_gd (método con Gudhi)
    if run_cone_gd:
        results['cone_gd'] = execute_cone_algorithm(
            'cone_gd', cone_gd_pipeline, dX, dY, f, X, Y, maxdim, cone_eps, tol, verbose
        )

    # Ejecutar cylinder
    # if run_cylinder:
    #     if verbose:
    #         print("\nEjecutando cylinder...")
    #     try:
    #         dgm_ker_cyl, dgm_coker_cyl = cylinder_pipeline(
    #             dX, dY, f,
    #             threshold=threshold,
    #             maxdim=maxdim,
    #             verbose=False
    #         )

    #         # También calcular cylinder_dgm para comparación
    #         cylinder_dgm_cyl = cylinder_dgm(dX, dY, f, maxdim)

    #         results['cylinder'] = {
    #             'X': X.tolist(),
    #             'Y': Y.tolist(),
    #             'f': f.tolist(),
    #             'dgm_ker': [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in dgm_ker_cyl],
    #             'dgm_coker': [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in dgm_coker_cyl],
    #             'dgm_cylinder': [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in cylinder_dgm_cyl],
    #             'dgm_X': [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in dgm_X],
    #             'dgm_Y': [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in dgm_Y],
    #             'missing': [],  # Cylinder no calcula missing bars
    #         }
    #         if verbose:
    #             print(f"  Cylinder completado. Ker bars: {len(dgm_ker_cyl)}, Coker bars: {len(dgm_coker_cyl)}")
    #     except Exception as e:
    #         print(f"  Error en cylinder: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         results['cylinder'] = {'error': str(e)}

    return results


def save_results(results, output_dir='results'):
    """
    Guarda los resultados en formato JSON y pickle.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    experiment_name = results['experiment_name']
    n = results['n']

    # Nombre base del archivo
    base_name = f"{experiment_name}_n{n}_seed{results.get('seed', '')}_eps{results.get('cone_eps', 0)}"

    # Guardar JSON (convertir Infinity a null para compatibilidad)
    json_path = output_path / f"{base_name}.json"
    with open(json_path, 'w') as f:
        json_results = convert_infinity_to_null(results)
        json.dump(json_results, f, indent=2, cls=SafeJSONEncoder)

    print(f"\nResultados guardados en:")
    print(f"  - {json_path}")

    return json_path


def main(
    n: tuple = (20, 50),
    dim: int = 2,
    maxdim: int = 2,
    seed: int = 42,
    cone_eps: float = 0.0,
    tol: Union[float, tuple] = (1e-11, 1e-2),  # accepts float or range
    experiments: tuple = None,
    cone: bool = True,
    cone_pairs: bool = True,
    cone_htr: bool = True,
    cone_gd: bool = True,
    cylinder: bool = True,
    output_dir: str = 'results',
):
    """
    Función principal que ejecuta todos los experimentos.

    Args:
        n: Valores de n para los experimentos (default: (20, 50))
        dim: Dimensión del espacio (default: 2)
        maxdim: Dimensión homológica máxima (default: 2)
        seed: Semilla para reproducibilidad (default: 42)
        experiments: Tupla con nombres de experimentos a ejecutar (default: None = todos)
        cone: Ejecutar método cone (default: True)
        cone_pairs: Ejecutar método cone_pairs (default: True)
        htr: Ejecutar método cone_htr (default: True)
        cone_gd: Ejecutar método cone_gd (default: True)
        cylinder: Ejecutar método cylinder (default: True)

    Examples:
        # Ejecutar todos los métodos con valores por defecto
        python run_experiments.py main

        # Ejecutar solo htr
        python run_experiments.py main --cone=False --cone_pairs=False --cone_gd=False --cylinder=False

        # Valores personalizados de n
        python run_experiments.py main --n 10 30 100

        # Experimentos específicos
        python run_experiments.py main --experiments circulo_embebido esfera_proyecta
    """
    # Convertir n a lista si es necesario
    if isinstance(n, int):
        n_values = [n]
    else:
        n_values = list(n)

    if isinstance(tol, (list, tuple)) and len(tol) == 2:
        tol = np.logspace(math.log10(tol[0]), math.log10(tol[1]), num=10)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determinar qué experimentos ejecutar
    if experiments:
        experiments_list = list(experiments) if not isinstance(
            experiments, str) else [experiments]
        experiments_to_run = {k: EXPERIMENTS[k]
                              for k in experiments_list if k in EXPERIMENTS}
        if not experiments_to_run:
            print(f"Error: Ninguno de los experimentos especificados existe.")
            print(f"Disponibles: {list(EXPERIMENTS.keys())}")
            return
    else:
        experiments_to_run = EXPERIMENTS

    print(f"\nMétodos activos:")
    print(f"  - cone: {cone}")
    print(f"  - cone_pairs: {cone_pairs}")
    print(f"  - htr: {cone_htr}")
    print(f"  - cone_gd: {cone_gd}")
    print(f"  - cylinder: {cylinder}")
    print(f"\nExperimentos a ejecutar: {list(experiments_to_run.keys())}")
    print(f"Valores de n: {n_values}\n")
    print(f"Valor de cone_eps: {cone_eps}\n")
    print(f"Semilla: {seed}\n")

    # Ejecutar todos los experimentos
    all_results = []

    for experiment_name in experiments_to_run.keys():
        for n_val in n_values:
            print(f"\n{'#'*60}")
            print(f"# Experimento: {experiment_name}, n={n_val}")
            print(f"{'#'*60}")

            results = run_single_experiment(
                experiment_name=experiment_name,
                n=n_val,
                dim=dim,
                maxdim=maxdim,
                cone_eps=cone_eps,
                tol=tol,
                seed=seed,
                verbose=True,
                run_cone=cone,
                run_cone_pairs=cone_pairs,
                run_cone_htr=cone_htr,
                run_cone_gd=cone_gd,
                run_cylinder=cylinder
            )

            # Guardar resultados
            save_results(results, output_dir)

            # Generar reporte
            # generate_report(results, output_dir)

            all_results.append(results)

    # Guardar resumen de todos los experimentos (convertir Infinity a null)
    summary_path = output_path / 'summary.json'
    with open(summary_path, 'w') as f:
        summary_results = convert_infinity_to_null(all_results)
        json.dump(summary_results, f, indent=2, cls=SafeJSONEncoder)

    print(f"\n{'='*60}")
    print(f"TODOS LOS EXPERIMENTOS COMPLETADOS")
    print(f"Resumen guardado en: {summary_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    fire.Fire()
