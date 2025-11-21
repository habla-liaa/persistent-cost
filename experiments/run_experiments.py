"""
Script principal para ejecutar todos los experimentos y guardar resultados.
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import math
import fire

from generate_spaces import EXPERIMENTS
from visualization import (
    # plot_persistence_diagrams,
    format_bars_for_output,
    classify_cone_bars
)

# Importar los pipelines desde persistent_cost
from persistent_cost.cone import cone_pipeline, cone_pipeline_htr
from persistent_cost.cone2 import cone_pipeline as cone2_pipeline
from persistent_cost.cylinder import cylinder_dgm, cylinder_pipeline
from persistent_cost.utils.utils import compute_lipschitz_constant


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


def run_single_experiment(experiment_name, n, dim=2, threshold=3.0, maxdim=2, seed=42, verbose=True,
                          run_cone=True, run_cone2=True, run_htr=True, run_cylinder=True):
    """
    Ejecuta un experimento individual con los tres métodos (cone, cone2, cylinder).
    
    Args:
        experiment_name: nombre del experimento
        n: número de puntos
        dim: dimensión del espacio
        threshold: threshold para construcción de complejos
        maxdim: dimensión homológica máxima
        seed: semilla para reproducibilidad
        verbose: mostrar información detallada
        run_cone: ejecutar método cone (default: True)
        run_cone2: ejecutar método cone2 (default: True)
        run_htr: ejecutar método cone_htr (default: True)
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
        'timestamp': datetime.now().isoformat(),
    }
    
    # Ejecutar cone (método 1)
    if run_cone:
        if verbose:
            print("\nEjecutando cone (método 1)...")
        
        cone_results = cone_pipeline(
            dX, dY, f, 
            maxdim=maxdim, 
            cone_eps=0.0, 
            return_extra=True
        )
        dgm_coker_cone, dgm_ker_cone, dgm_cone, dgm_X_cone, dgm_Y_cone, D_cone, missing_cone = cone_results            
        
        # También calcular cylinder para comparación
        cylinder_dgm_ = cylinder_dgm(dX, dY, f, maxdim)
        
        results['cone'] = {
            'X': X.tolist(),
            'Y': Y.tolist(),
            'f': f if isinstance(f, list) else f.tolist(),
            'dgm_coker': [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in dgm_coker_cone],
            'dgm_ker': [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in dgm_ker_cone],
            'dgm_cone': [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in dgm_cone],
            'dgm_cylinder': [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in cylinder_dgm_],
            'dgm_X': [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in dgm_X_cone],
            'dgm_Y': [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in dgm_Y_cone],
            'missing': missing_cone,
        }
        if verbose:
            print(f"  Cone completado. Ker bars: {len(dgm_ker_cone)}, Coker bars: {len(dgm_coker_cone)}")
    
    # Ejecutar cone2 (método 2)
    if run_cone2:
        if verbose:
            print("\nEjecutando cone2 (método 2)...")
        
        cone2_results = cone2_pipeline(
            dX, dY, f,
            maxdim=maxdim,
            cone_eps=0.0,
            return_extra=True
        )
        (dgm_coker_cone2, dgm_ker_cone2, dgm_cone2, dgm_X_cone2, dgm_Y_cone2,
         pairs_coker, pairs_ker, pairs_cone, pairs_X, pairs_Y,
         missing_cone2, simpl2dist_cone, simpl2dist_X, simpl2dist_Y, D_cone2) = cone2_results
        
        # También calcular cylinder para comparación
        cylinder_dgm_2 = cylinder_dgm(dX, dY, f, maxdim)
        
        results['cone2'] = {
            'X': X.tolist(),
            'Y': Y.tolist(),
            'f': f,
            'dgm_coker': [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in dgm_coker_cone2],
            'dgm_ker': [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in dgm_ker_cone2],
            'dgm_cone': [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in dgm_cone2],
            'dgm_cylinder': [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in cylinder_dgm_2],
            'dgm_X': [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in dgm_X_cone2],
            'dgm_Y': [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in dgm_Y_cone2],
            'missing': missing_cone2,
        }
        if verbose:
            print(f"  Cone2 completado. Ker bars: {len(dgm_ker_cone2)}, Coker bars: {len(dgm_coker_cone2)}")
    
    # Ejecutar cone_htr (método con HTR)
    if run_htr:
        if verbose:
            print("\nEjecutando cone_htr (método con HTR)...")
        
        cone_htr_results = cone_pipeline_htr(
            dX, dY, f, 
            maxdim=maxdim, 
            cone_eps=0.0,
            threshold=threshold,
            return_extra=True
        )
        dgm_coker_htr, dgm_ker_htr, dgm_cone_htr, dgm_X_htr, dgm_Y_htr, D_htr, missing_htr = cone_htr_results
        
        # También calcular cylinder para comparación
        cylinder_dgm_htr = cylinder_dgm(dX, dY, f, maxdim)
        
        results['cone_htr'] = {
            'X': X.tolist(),
            'Y': Y.tolist(),
            'f': f.tolist(),
            'dgm_coker': [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in dgm_coker_htr],
            'dgm_ker': [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in dgm_ker_htr],
            'dgm_cone': [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in dgm_cone_htr],
            'dgm_cylinder': [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in cylinder_dgm_htr],
            'dgm_X': [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in dgm_X_htr],
            'dgm_Y': [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in dgm_Y_htr],
            'missing': missing_htr,
        }
        if verbose:
            print(f"  Cone_htr completado. Ker bars: {len(dgm_ker_htr)}, Coker bars: {len(dgm_coker_htr)}")
    
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
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Nombre base del archivo
    base_name = f"{experiment_name}_n{n}_{timestamp}"
    
    # Guardar JSON (convertir Infinity a null para compatibilidad)
    json_path = output_path / f"{base_name}.json"
    with open(json_path, 'w') as f:
        json_results = convert_infinity_to_null(results)
        json.dump(json_results, f, indent=2, cls=SafeJSONEncoder)
    
    # Guardar pickle (para objetos numpy complejos si es necesario)
    pickle_path = output_path / f"{base_name}.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResultados guardados en:")
    print(f"  - {json_path}")
    print(f"  - {pickle_path}")
    
    return json_path, pickle_path


def generate_report(results, output_dir='results'):
    """
    Genera un reporte completo del experimento incluyendo gráficos.
    """
    experiment_name = results['experiment_name']
    n = results['n']
    
    # Crear subdirectorio para este experimento
    output_path = Path(output_dir)
    exp_dir = output_path / f"{experiment_name}_n{n}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Generar reporte de texto
    report_path = exp_dir / 'report.txt'
    with open(report_path, 'w') as f:
        f.write(f"REPORTE DE EXPERIMENTO\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Experimento: {experiment_name}\n")
        f.write(f"Tamaño n: {n}\n")
        f.write(f"Dimensión: {results.get('dim', 'N/A')}\n")
        f.write(f"Constante de Lipschitz: {results['lipschitz_constant']:.6f}\n")
        f.write(f"Timestamp: {results['timestamp']}\n\n")
        
        # Reportar resultados de cada método
        for method in ['cone', 'cone2', 'cone_htr', 'cylinder']:
            if method not in results:
                continue
            
            f.write(f"\n{method.upper()}\n")
            f.write(f"{'-'*40}\n")
            
            method_data = results[method]
            if 'error' in method_data:
                f.write(f"Error: {method_data['error']}\n")
                continue
            
            # Barras del kernel
            if 'dgm_ker' in method_data:
                f.write(f"\nBarras del Kernel:\n")
                bars_ker = format_bars_for_output(method_data['dgm_ker'])
                f.write(bars_ker)
            
            # Barras del cokernel
            if 'dgm_coker' in method_data:
                f.write(f"\nBarras del Cokernel:\n")
                bars_coker = format_bars_for_output(method_data['dgm_coker'])
                f.write(bars_coker)
            
            # Para cone, reportar clasificación de barras
            if method == 'cone' and 'dgm_cone' in method_data:
                f.write(f"\nClasificación de barras del cono:\n")
                classification = classify_cone_bars(
                    method_data['dgm_cone'],
                    method_data['dgm_ker'],
                    method_data['dgm_coker']
                )
                f.write(classification)
    
    print(f"Reporte generado en: {report_path}")
    
    # # Generar gráficos
    # try:
    #     # plot_persistence_diagrams(results, exp_dir)
    #     # print(f"Gráficos generados en: {exp_dir}")
    # except Exception as e:
    #     print(f"Error generando gráficos: {e}")
    
    return report_path


def main(
    n: tuple = (20, 50),
    dim: int = 2,
    maxdim: int = 2,
    threshold: float = 3.0,
    seed: int = 42,
    experiments: tuple = None,
    cone: bool = True,
    cone2: bool = True,
    htr: bool = True,
    cylinder: bool = True,
    output_dir: str = 'results',
):
    """
    Función principal que ejecuta todos los experimentos.
    
    Args:
        n: Valores de n para los experimentos (default: (20, 50))
        dim: Dimensión del espacio (default: 2)
        maxdim: Dimensión homológica máxima (default: 2)
        threshold: Threshold para construcción de complejos (default: 3.0)
        seed: Semilla para reproducibilidad (default: 42)
        experiments: Tupla con nombres de experimentos a ejecutar (default: None = todos)
        cone: Ejecutar método cone (default: True)
        cone2: Ejecutar método cone2 (default: True)
        htr: Ejecutar método cone_htr (default: True)
        cylinder: Ejecutar método cylinder (default: True)
    
    Examples:
        # Ejecutar todos los métodos con valores por defecto
        python run_experiments.py main
        
        # Ejecutar solo htr
        python run_experiments.py main --cone=False --cone2=False --cylinder=False
        
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
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determinar qué experimentos ejecutar
    if experiments:
        experiments_list = list(experiments) if not isinstance(experiments, str) else [experiments]
        experiments_to_run = {k: EXPERIMENTS[k] for k in experiments_list if k in EXPERIMENTS}
        if not experiments_to_run:
            print(f"Error: Ninguno de los experimentos especificados existe.")
            print(f"Disponibles: {list(EXPERIMENTS.keys())}")
            return
    else:
        experiments_to_run = EXPERIMENTS
    
    print(f"\nMétodos activos:")
    print(f"  - cone: {cone}")
    print(f"  - cone2: {cone2}")
    print(f"  - htr: {htr}")
    print(f"  - cylinder: {cylinder}")
    print(f"\nExperimentos a ejecutar: {list(experiments_to_run.keys())}")
    print(f"Valores de n: {n_values}\n")
    
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
                threshold=threshold,
                maxdim=maxdim,
                seed=seed,
                verbose=True,
                run_cone=cone,
                run_cone2=cone2,
                run_htr=htr,
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
