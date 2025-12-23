"""
Paquete de experimentos para persistent_cost.

Este paquete contiene herramientas para ejecutar experimentos comparativos
de los métodos cone, cone2 y cylinder.
"""

from ..src.persistent_cost.utils.generate_spaces import (
    inclusion_punto,
    producto,
    suspension,
    toro_proyecta,
    circulo_en_toro,
    muestreo_random,
)

from persistent_cost.utils.utils import compute_lipschitz_constant

from .run_experiments import (
    run_single_experiment,
    save_results,
    generate_report,
)

from .visualization import (
    plot_persistence_diagrams,
    format_bars_for_output,
    classify_cone_bars,
    plot_barcode,
)

__all__ = [
    # Generación de espacios
    'EXPERIMENTS',
    'inclusion_punto',
    'producto',
    'suspension',
    'toro_proyecta',
    'circulo_en_toro',
    'muestreo_random',
    'compute_lipschitz_constant',
    
    # Ejecución de experimentos
    'run_single_experiment',
    'save_results',
    'generate_report',
    
    # Visualización
    'plot_persistence_diagrams',
    'format_bars_for_output',
    'classify_cone_bars',
    'plot_barcode',
]

__version__ = '0.1.0'
