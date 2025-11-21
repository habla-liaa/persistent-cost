"""
Script para analizar y comparar resultados de experimentos ya ejecutados.
"""

import os
import json
import glob
import numpy as np
import pandas as pd
from collections import defaultdict


def load_results_from_dir(results_dir='results'):
    """
    Carga todos los resultados JSON del directorio.
    """
    json_files = glob.glob(os.path.join(results_dir, '*.json'))
    json_files = [f for f in json_files if not f.endswith('summary.json')]
    
    results = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            results.append(json.load(f))
    
    return results


def compute_statistics(dgm):
    """
    Calcula estadísticas básicas de un diagrama de persistencia.
    """
    stats = {
        'n_bars': 0,
        'n_finite': 0,
        'n_infinite': 0,
        'total_persistence': 0.0,
        'avg_persistence': 0.0,
        'max_persistence': 0.0,
    }
    
    all_bars = []
    for dim_bars in dgm:
        if isinstance(dim_bars, list):
            dim_bars = np.array(dim_bars) if dim_bars else np.empty((0, 2))
        all_bars.extend(dim_bars)
    
    if len(all_bars) == 0:
        return stats
    
    all_bars = np.array(all_bars)
    stats['n_bars'] = len(all_bars)
    
    # Separar finitas e infinitas
    finite_mask = ~np.isinf(all_bars[:, 1])
    stats['n_finite'] = finite_mask.sum()
    stats['n_infinite'] = (~finite_mask).sum()
    
    if stats['n_finite'] > 0:
        finite_bars = all_bars[finite_mask]
        persistences = finite_bars[:, 1] - finite_bars[:, 0]
        stats['total_persistence'] = persistences.sum()
        stats['avg_persistence'] = persistences.mean()
        stats['max_persistence'] = persistences.max()
    
    return stats


def create_comparison_dataframe(results_list):
    """
    Crea un DataFrame de pandas con comparación de resultados.
    """
    rows = []
    
    for result in results_list:
        exp_name = result['experiment_name']
        n = result['n']
        L = result['lipschitz_constant']
        
        for method in ['cone', 'cone2', 'cylinder']:
            if method not in result or 'error' in result[method]:
                continue
            
            method_data = result[method]
            
            # Estadísticas de kernel
            ker_stats = compute_statistics(method_data.get('dgm_ker', []))
            
            # Estadísticas de cokernel
            coker_stats = compute_statistics(method_data.get('dgm_coker', []))
            
            row = {
                'experimento': exp_name,
                'n': n,
                'metodo': method,
                'lipschitz': L,
                'ker_n_bars': ker_stats['n_bars'],
                'ker_total_pers': ker_stats['total_persistence'],
                'ker_avg_pers': ker_stats['avg_persistence'],
                'ker_max_pers': ker_stats['max_persistence'],
                'coker_n_bars': coker_stats['n_bars'],
                'coker_total_pers': coker_stats['total_persistence'],
                'coker_avg_pers': coker_stats['avg_persistence'],
                'coker_max_pers': coker_stats['max_persistence'],
            }
            
            rows.append(row)
    
    return pd.DataFrame(rows)


def compare_methods(results_list, output_path='results/method_comparison.txt'):
    """
    Compara los resultados de diferentes métodos.
    """
    # Agrupar por experimento y n
    grouped = defaultdict(lambda: defaultdict(dict))
    
    for result in results_list:
        exp_name = result['experiment_name']
        n = result['n']
        key = (exp_name, n)
        
        for method in ['cone', 'cone2', 'cylinder']:
            if method not in result or 'error' in result[method]:
                continue
            
            method_data = result[method]
            grouped[key][method] = method_data
    
    with open(output_path, 'w') as f:
        f.write("COMPARACIÓN DE MÉTODOS\n")
        f.write("="*80 + "\n\n")
        
        for (exp_name, n), methods in sorted(grouped.items()):
            f.write(f"\n{exp_name} (n={n})\n")
            f.write("-"*60 + "\n")
            
            # Comparar número de barras
            f.write(f"{'Método':<15} {'Ker bars':<12} {'Coker bars':<12}\n")
            
            for method_name, method_data in methods.items():
                n_ker = sum(len(b) if isinstance(b, (list, np.ndarray)) else 0 
                           for b in method_data.get('dgm_ker', []))
                n_coker = sum(len(b) if isinstance(b, (list, np.ndarray)) else 0 
                             for b in method_data.get('dgm_coker', []))
                
                f.write(f"{method_name:<15} {n_ker:<12} {n_coker:<12}\n")
            
            f.write("\n")
    
    print(f"Comparación guardada en: {output_path}")


def analyze_experiment_type(results_list, experiment_name, output_path=None):
    """
    Analiza todos los resultados de un tipo de experimento específico.
    """
    filtered = [r for r in results_list if r['experiment_name'] == experiment_name]
    
    if not filtered:
        print(f"No se encontraron resultados para {experiment_name}")
        return
    
    if output_path is None:
        output_path = f'results/analysis_{experiment_name}.txt'
    
    with open(output_path, 'w') as f:
        f.write(f"ANÁLISIS: {experiment_name}\n")
        f.write("="*80 + "\n\n")
        
        for result in sorted(filtered, key=lambda x: x['n']):
            n = result['n']
            L = result['lipschitz_constant']
            
            f.write(f"\nn = {n}, Lipschitz = {L:.6f}\n")
            f.write("-"*60 + "\n")
            
            for method in ['cone', 'cone2', 'cylinder']:
                if method not in result or 'error' in result[method]:
                    continue
                
                f.write(f"\n{method.upper()}:\n")
                
                method_data = result[method]
                ker_stats = compute_statistics(method_data.get('dgm_ker', []))
                coker_stats = compute_statistics(method_data.get('dgm_coker', []))
                
                f.write(f"  Kernel: {ker_stats['n_bars']} barras, ")
                f.write(f"persistencia total = {ker_stats['total_persistence']:.4f}\n")
                
                f.write(f"  Cokernel: {coker_stats['n_bars']} barras, ")
                f.write(f"persistencia total = {coker_stats['total_persistence']:.4f}\n")
    
    print(f"Análisis guardado en: {output_path}")


def main():
    """
    Función principal de análisis.
    """
    print("Cargando resultados...")
    results = load_results_from_dir('results')
    
    if not results:
        print("No se encontraron resultados en el directorio 'results/'")
        return
    
    print(f"Cargados {len(results)} resultados")
    
    # Crear DataFrame
    print("\nCreando DataFrame...")
    df = create_comparison_dataframe(results)
    
    # Guardar CSV
    csv_path = 'results/comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"DataFrame guardado en: {csv_path}")
    
    # Mostrar resumen
    print("\nResumen por experimento:")
    print(df.groupby(['experimento', 'metodo'])['ker_n_bars'].mean())
    
    # Comparar métodos
    print("\nGenerando comparación de métodos...")
    compare_methods(results)
    
    # Analizar cada tipo de experimento
    print("\nGenerando análisis por experimento...")
    experiment_names = df['experimento'].unique()
    for exp_name in experiment_names:
        analyze_experiment_type(results, exp_name)
    
    print("\n" + "="*60)
    print("Análisis completado!")
    print("="*60)


if __name__ == '__main__':
    main()
