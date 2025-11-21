"""
Configuración para los experimentos.

Ajusta estos valores para cambiar el comportamiento de los experimentos.
"""

# Configuración de tamaños
DEFAULT_N_VALUES = [50, 100]
DEFAULT_DIM = 2
DEFAULT_MAXDIM = 2

# Configuración de threshold para cylinder
DEFAULT_THRESHOLD = 3.0

# Semilla para reproducibilidad
DEFAULT_SEED = 42

# Configuración de cone
DEFAULT_CONE_EPS = 0.0
DEFAULT_TOL = 1e-11

# Directorios
OUTPUT_DIR = 'results'

# Configuración de visualización
PLOT_DPI = 150
PLOT_FIGSIZE = (15, 10)
MAX_BARS_BARCODE = 50

# Colores para dimensiones en plots
DIMENSION_COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']

# Configuración de análisis
COMPARISON_METRICS = [
    'n_bars',
    'total_persistence',
    'avg_persistence',
    'max_persistence',
]

# Verbosidad por defecto
VERBOSE = True

# Configuración de guardado
SAVE_JSON = True
SAVE_PICKLE = True
SAVE_PLOTS = True

# Formatos de archivo
TIMESTAMP_FORMAT = '%Y%m%d_%H%M%S'
