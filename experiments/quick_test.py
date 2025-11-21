#!/usr/bin/env python3
"""
Script rápido para ejecutar un experimento individual.

Uso:
    python quick_test.py inclusion_punto 50
    python quick_test.py producto 100
"""

import sys
from run_experiments import run_single_experiment, save_results, generate_report


def main():
    if len(sys.argv) < 2:
        print("Uso: python quick_test.py <experimento> [n=50] [dim=2]")
        print("\nExperimentos disponibles:")
        print("  - inclusion_punto")
        print("  - producto")
        print("  - suspension")
        print("  - toro_proyecta")
        print("  - circulo_en_toro")
        print("  - muestreo_random")
        sys.exit(1)
    
    experiment_name = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    dim = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    
    print(f"\nEjecutando experimento: {experiment_name} con n={n}, dim={dim}")
    print("="*60)
    
    results = run_single_experiment(
        experiment_name=experiment_name,
        n=n,
        dim=dim,
        threshold=3.0,
        maxdim=2,
        seed=42,
        verbose=True
    )
    
    # Guardar y generar reporte
    save_results(results, output_dir='results')
    generate_report(results, output_dir='results')
    
    print("\n" + "="*60)
    print("Experimento completado!")
    print("="*60)


if __name__ == '__main__':
    main()
