"""
Script interactivo para explorar resultados de experimentos.

Uso:
    python explore_results.py
"""

import os
import json
import glob
import numpy as np
from tabulate import tabulate


def list_available_results(results_dir='results'):
    """Lista todos los resultados disponibles."""
    json_files = glob.glob(os.path.join(results_dir, '*_n*.json'))
    json_files = [f for f in json_files if not f.endswith('summary.json')]
    
    if not json_files:
        print("No se encontraron resultados.")
        return []
    
    results = []
    for json_file in sorted(json_files):
        with open(json_file, 'r') as f:
            data = json.load(f)
            results.append({
                'file': os.path.basename(json_file),
                'experiment': data['experiment_name'],
                'n': data['n'],
                'lipschitz': data['lipschitz_constant'],
                'timestamp': data.get('timestamp', 'N/A'),
            })
    
    return results


def display_results_table(results):
    """Muestra una tabla de resultados."""
    if not results:
        print("No hay resultados para mostrar.")
        return
    
    table_data = []
    for i, r in enumerate(results):
        table_data.append([
            i + 1,
            r['experiment'],
            r['n'],
            f"{r['lipschitz']:.4f}",
            r['timestamp'][:19] if len(r['timestamp']) > 19 else r['timestamp'],
        ])
    
    headers = ['#', 'Experimento', 'n', 'Lipschitz', 'Timestamp']
    print("\nResultados disponibles:")
    print(tabulate(table_data, headers=headers, tablefmt='grid'))


def load_and_display_result(filename, results_dir='results'):
    """Carga y muestra un resultado específico."""
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"Experimento: {data['experiment_name']}")
    print(f"n = {data['n']}, dim = {data.get('dim', 'N/A')}")
    print(f"Constante de Lipschitz: {data['lipschitz_constant']:.6f}")
    print(f"{'='*70}\n")
    
    # Mostrar resultados por método
    for method in ['cone', 'cone2', 'cylinder']:
        if method not in data:
            continue
        
        method_data = data[method]
        
        if 'error' in method_data:
            print(f"{method.upper()}: Error - {method_data['error']}")
            continue
        
        print(f"\n{method.upper()}:")
        print("-" * 50)
        
        # Contar barras por dimensión
        if 'dgm_ker' in method_data:
            ker_counts = count_bars_by_dim(method_data['dgm_ker'])
            print(f"Kernel:")
            for dim, count in enumerate(ker_counts):
                if count > 0:
                    print(f"  H{dim}: {count} barras")
        
        if 'dgm_coker' in method_data:
            coker_counts = count_bars_by_dim(method_data['dgm_coker'])
            print(f"Cokernel:")
            for dim, count in enumerate(coker_counts):
                if count > 0:
                    print(f"  H{dim}: {count} barras")
        
        # Mostrar algunas barras de ejemplo
        if 'dgm_ker' in method_data and ker_counts:
            print(f"\nEjemplos de barras del Kernel:")
            show_sample_bars(method_data['dgm_ker'], max_samples=3)
        
        if 'dgm_coker' in method_data and coker_counts:
            print(f"\nEjemplos de barras del Cokernel:")
            show_sample_bars(method_data['dgm_coker'], max_samples=3)


def count_bars_by_dim(dgm):
    """Cuenta barras por dimensión."""
    counts = []
    for dim_bars in dgm:
        if isinstance(dim_bars, list):
            counts.append(len(dim_bars))
        else:
            counts.append(0)
    return counts


def show_sample_bars(dgm, max_samples=3):
    """Muestra algunas barras de ejemplo."""
    for dim, bars in enumerate(dgm):
        if isinstance(bars, list):
            bars = np.array(bars) if bars else np.empty((0, 2))
        
        if len(bars) == 0:
            continue
        
        n_show = min(len(bars), max_samples)
        print(f"  Dimensión {dim}:")
        for i in range(n_show):
            birth, death = bars[i]
            if np.isinf(death):
                print(f"    ({birth:.4f}, ∞)")
            else:
                persistence = death - birth
                print(f"    ({birth:.4f}, {death:.4f}) [pers={persistence:.4f}]")
        
        if len(bars) > max_samples:
            print(f"    ... y {len(bars) - max_samples} más")


def compare_methods_for_experiment(exp_name, n, results_dir='results'):
    """Compara métodos para un experimento específico."""
    pattern = f"{exp_name}_n{n}_*.json"
    files = glob.glob(os.path.join(results_dir, pattern))
    
    if not files:
        print(f"No se encontraron resultados para {exp_name} con n={n}")
        return
    
    # Tomar el más reciente
    files.sort()
    filepath = files[-1]
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"Comparación de métodos: {exp_name} (n={n})")
    print(f"{'='*70}\n")
    
    table_data = []
    for method in ['cone', 'cone2', 'cylinder']:
        if method not in data or 'error' in data[method]:
            continue
        
        method_data = data[method]
        
        ker_counts = count_bars_by_dim(method_data.get('dgm_ker', []))
        coker_counts = count_bars_by_dim(method_data.get('dgm_coker', []))
        
        total_ker = sum(ker_counts)
        total_coker = sum(coker_counts)
        
        table_data.append([
            method,
            total_ker,
            ', '.join(f"{c}" for c in ker_counts if c > 0),
            total_coker,
            ', '.join(f"{c}" for c in coker_counts if c > 0),
        ])
    
    headers = ['Método', 'Total Ker', 'Ker por dim', 'Total Coker', 'Coker por dim']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))


def interactive_menu():
    """Menú interactivo principal."""
    results_dir = 'results'
    
    while True:
        print("\n" + "="*70)
        print("EXPLORADOR DE RESULTADOS")
        print("="*70)
        print("\n1. Listar todos los resultados")
        print("2. Ver resultado específico")
        print("3. Comparar métodos para un experimento")
        print("4. Salir")
        
        choice = input("\nSelecciona una opción: ").strip()
        
        if choice == '1':
            results = list_available_results(results_dir)
            display_results_table(results)
        
        elif choice == '2':
            results = list_available_results(results_dir)
            if results:
                display_results_table(results)
                idx = input("\nIngresa el número del resultado (o 'c' para cancelar): ").strip()
                if idx != 'c':
                    try:
                        idx = int(idx) - 1
                        if 0 <= idx < len(results):
                            load_and_display_result(results[idx]['file'], results_dir)
                        else:
                            print("Índice fuera de rango.")
                    except ValueError:
                        print("Entrada inválida.")
        
        elif choice == '3':
            exp_name = input("Nombre del experimento: ").strip()
            n = input("Valor de n: ").strip()
            try:
                n = int(n)
                compare_methods_for_experiment(exp_name, n, results_dir)
            except ValueError:
                print("n debe ser un número entero.")
        
        elif choice == '4':
            print("\n¡Hasta luego!")
            break
        
        else:
            print("Opción inválida.")


def main():
    """Función principal."""
    import sys
    
    if len(sys.argv) > 1:
        # Modo comando
        if sys.argv[1] == 'list':
            results = list_available_results()
            display_results_table(results)
        elif sys.argv[1] == 'show' and len(sys.argv) > 2:
            load_and_display_result(sys.argv[2])
        elif sys.argv[1] == 'compare' and len(sys.argv) > 3:
            compare_methods_for_experiment(sys.argv[2], int(sys.argv[3]))
        else:
            print("Uso:")
            print("  python explore_results.py              # Modo interactivo")
            print("  python explore_results.py list         # Listar resultados")
            print("  python explore_results.py show <file>  # Mostrar resultado")
            print("  python explore_results.py compare <exp> <n>  # Comparar métodos")
    else:
        # Modo interactivo
        try:
            interactive_menu()
        except KeyboardInterrupt:
            print("\n\n¡Hasta luego!")


if __name__ == '__main__':
    main()
