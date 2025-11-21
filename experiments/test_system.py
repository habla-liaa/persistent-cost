#!/usr/bin/env python3
"""
Test rápido para verificar que todo el sistema funciona.

Este script ejecuta un experimento pequeño para verificar:
1. Que todos los módulos se importan correctamente
2. Que los generadores de espacios funcionan
3. Que los tres métodos (cone, cone2, cylinder) se ejecutan
4. Que la visualización y guardado funcionan
"""

import sys
import os


from persistent_cost.utils.utils import compute_lipschitz_constant

def test_imports():
    """Verifica que todos los imports funcionen."""
    print("Verificando imports...", end=" ")
    try:
        import numpy as np
        import scipy
        import matplotlib
        from scipy.spatial.distance import pdist
        import gudhi
        
        # Imports de persistent_cost
        from persistent_cost.cone import cone_pipeline
        from persistent_cost.cone2 import cone_pipeline as cone2_pipeline
        from persistent_cost.cylinder import cylinder_pipeline
        
        
        # Imports locales
        from generate_spaces import EXPERIMENTS
        from visualization import plot_persistence_diagrams, format_bars_for_output
        from run_experiments import run_single_experiment
        
        print("✓ OK")
        return True
    except ImportError as e:
        print(f"✗ FALLO: {e}")
        return False


def test_space_generation():
    """Verifica que la generación de espacios funcione."""
    print("Verificando generación de espacios...", end=" ")
    try:
        from generate_spaces import producto
        
        X, Y, f = producto(n=10, dim=2, seed=42)
        
        assert X.shape == (10, 2), f"Forma de X incorrecta: {X.shape}"
        assert Y.shape == (10, 2), f"Forma de Y incorrecta: {Y.shape}"
        assert len(f) == 10, f"Longitud de f incorrecta: {len(f)}"
        
        print("✓ OK")
        return True
    except Exception as e:
        print(f"✗ FALLO: {e}")
        return False


def test_lipschitz():
    """Verifica el cálculo de Lipschitz."""
    print("Verificando cálculo de Lipschitz...", end=" ")
    try:
        from generate_spaces import producto
        from persistent_cost.utils.utils import compute_lipschitz_constant 
        
        X, Y, f = producto(n=10, dim=2, seed=42)
        L = compute_lipschitz_constant(X, Y, f)
        
        assert L > 0, f"Lipschitz debe ser positivo: {L}"
        assert not np.isnan(L), "Lipschitz es NaN"
        assert not np.isinf(L), "Lipschitz es infinito"
        
        print(f"✓ OK (L={L:.4f})")
        return True
    except Exception as e:
        print(f"✗ FALLO: {e}")
        return False


def test_cone():
    """Verifica que cone funcione."""
    print("Verificando método cone...", end=" ")
    try:
        from persistent_cost.cone import cone_pipeline
        from generate_spaces import producto
        from scipy.spatial.distance import pdist
        
        X, Y, f = producto(n=10, dim=2, seed=42)
        dX = pdist(X)
        dY = pdist(Y)
        
        L = compute_lipschitz_constant(X, Y, f)
        dY = dY / L
        
        dgm_coker, dgm_ker, dgm_cone, dgm_X, dgm_Y = cone_pipeline(
            dX, dY, f, maxdim=1, cone_eps=0.0
        )
        
        assert len(dgm_X) > 0, "dgm_X vacío"
        assert len(dgm_Y) > 0, "dgm_Y vacío"
        assert len(dgm_cone) > 0, "dgm_cone vacío"
        
        print("✓ OK")
        return True
    except Exception as e:
        print(f"✗ FALLO: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cone2():
    """Verifica que cone2 funcione."""
    print("Verificando método cone2...", end=" ")
    try:
        from persistent_cost.cone2 import cone_pipeline
        from generate_spaces import producto
        from scipy.spatial.distance import pdist
        
        X, Y, f = producto(n=10, dim=2, seed=42)
        dX = pdist(X)
        dY = pdist(Y)

        L = compute_lipschitz_constant(X, Y, f)
        dY = dY / L
        
        dgm_coker, dgm_ker, dgm_cone, dgm_X, dgm_Y = cone_pipeline(
            dX, dY, f, maxdim=1, cone_eps=0.0
        )
        
        assert len(dgm_X) > 0, "dgm_X vacío"
        assert len(dgm_Y) > 0, "dgm_Y vacío"
        assert len(dgm_cone) > 0, "dgm_cone vacío"
        
        print("✓ OK")
        return True
    except Exception as e:
        print(f"✗ FALLO: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cylinder():
    """Verifica que cylinder funcione."""
    print("Verificando método cylinder...", end=" ")
    try:
        from persistent_cost.cylinder import cylinder_pipeline
        from generate_spaces import producto
        from scipy.spatial.distance import pdist
        
        X, Y, f = producto(n=10, dim=2, seed=42)

        dX = pdist(X)
        dY = pdist(Y)
        
        L = compute_lipschitz_constant(dX, dY, f)
        dY = dY / L
        
        dgm_ker, dgm_coker = cylinder_pipeline(
            dX, dY, f, threshold=2.0, maxdim=1, verbose=False
        )
        
        assert len(dgm_ker) >= 0, "dgm_ker inválido"
        assert len(dgm_coker) >= 0, "dgm_coker inválido"
        
        print("✓ OK")
        return True
    except Exception as e:
        print(f"✗ FALLO: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_experiment():
    """Ejecuta un experimento completo pequeño."""
    print("\nEjecutando experimento de prueba completo...")
    try:
        from run_experiments import run_single_experiment
        
        results = run_single_experiment(
            experiment_name='producto',
            n=15,  # Muy pequeño para rapidez
            dim=2,
            threshold=2.0,
            maxdim=1,
            seed=42,
            verbose=False
        )
        
        # Verificar estructura de resultados
        assert 'experiment_name' in results
        assert 'lipschitz_constant' in results
        assert 'cone' in results or 'cone2' in results or 'cylinder' in results
        
        print("✓ Experimento completo ejecutado exitosamente")
        return True
    except Exception as e:
        print(f"✗ FALLO: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Ejecuta todos los tests."""
    print("="*70)
    print("VERIFICACIÓN DEL SISTEMA DE EXPERIMENTOS")
    print("="*70)
    print()
    
    tests = [
        test_imports,
        test_space_generation,
        test_lipschitz,
        test_cone,
        test_cone2,
        test_cylinder,
        test_full_experiment,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"Error crítico en {test.__name__}: {e}")
            results.append(False)
    
    print()
    print("="*70)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✓ TODOS LOS TESTS PASARON ({passed}/{total})")
        print("="*70)
        print("\nEl sistema está listo para usar.")
        print("Puedes ejecutar:")
        print("  ./run_all.sh         # Para ejecutar todos los experimentos")
        print("  ./quick_test.py producto 50  # Para un experimento individual")
        return 0
    else:
        print(f"✗ ALGUNOS TESTS FALLARON ({passed}/{total} pasaron)")
        print("="*70)
        print("\nRevisá los errores arriba y asegurate de que:")
        print("  1. persistent_cost está instalado: pip install -e .")
        print("  2. Todas las dependencias están instaladas")
        print("  3. Los módulos cone, cone2 y cylinder funcionan")
        return 1


if __name__ == '__main__':
    import numpy as np
    sys.exit(main())
