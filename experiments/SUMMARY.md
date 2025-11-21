# Resumen de Archivos Creados - Experimentos

## Estructura completa

```
experiments/
├── EXPERIMENTOS.md           # Especificación original (ya existía)
├── README.md                 # Documentación completa
├── QUICKSTART.md            # Guía de inicio rápido
│
├── __init__.py              # Paquete Python
├── config.py                # Configuración centralizada
│
├── generate_spaces.py       # Generación de espacios X, Y, f
├── run_experiments.py       # Script principal de ejecución
├── visualization.py         # Funciones de visualización
├── analyze_results.py       # Análisis post-ejecución
├── explore_results.py       # Explorador interactivo
│
├── quick_test.py           # Script para pruebas rápidas (ejecutable)
├── run_all.sh              # Script bash para ejecutar todo (ejecutable)
│
└── results/                # Directorio de resultados (se crea al ejecutar)
    ├── <experimento>_n<n>_<timestamp>.json
    ├── <experimento>_n<n>_<timestamp>.pkl
    ├── <experimento>_n<n>/
    │   ├── report.txt
    │   ├── cone_diagrams.png
    │   ├── cone2_diagrams.png
    │   └── cylinder_diagrams.png
    ├── comparison.csv
    ├── method_comparison.txt
    └── summary.json
```

## Archivos principales

### 1. `generate_spaces.py`
Genera los 6 experimentos especificados:
- `inclusion_punto`: Inclusión de punto en nube
- `producto`: Identidad
- `suspension`: Suspensión
- `toro_proyecta`: Proyección de toro
- `circulo_en_toro`: Círculo en toro
- `muestreo_random`: Submuestreo aleatorio

Cada función retorna `(X, Y, f)` y calcula la constante de Lipschitz.

### 2. `run_experiments.py`
Script principal que:
- Ejecuta los 3 métodos (cone, cone2, cylinder) para cada experimento
- Calcula diagramas de persistencia para X, Y, Cone, Kernel, Cokernel
- Guarda resultados en JSON y pickle
- Genera reportes y gráficos

### 3. `visualization.py`
Funciones de visualización:
- `plot_persistence_diagrams()`: 6 diagramas por método
- `plot_barcode()`: Código de barras
- `format_bars_for_output()`: Formato texto
- `classify_cone_bars()`: Clasificación kernel/cokernel/desapareadas

### 4. `analyze_results.py`
Análisis post-ejecución:
- Crea DataFrame de pandas con todos los resultados
- Genera CSV comparativo
- Calcula estadísticas (persistencias, número de barras)
- Compara métodos

### 5. `explore_results.py`
Explorador interactivo:
- Menú interactivo para navegar resultados
- Comparación de métodos
- Visualización de barras individuales
- Modo comando y modo interactivo

## Scripts auxiliares

### `quick_test.py` (ejecutable)
Para pruebas rápidas de un solo experimento:
```bash
./quick_test.py producto 50
```

### `run_all.sh` (ejecutable)
Ejecuta todo el pipeline:
```bash
./run_all.sh
```
1. Ejecuta `run_experiments.py`
2. Ejecuta `analyze_results.py`

## Archivos de configuración

### `config.py`
Centraliza todos los parámetros:
- Tamaños de nubes: `DEFAULT_N_VALUES = [50, 100]`
- Dimensiones: `DEFAULT_DIM = 2`, `DEFAULT_MAXDIM = 2`
- Threshold: `DEFAULT_THRESHOLD = 3.0`
- Semilla: `DEFAULT_SEED = 42`
- Parámetros de visualización
- Directorios

### `__init__.py`
Exporta todas las funciones públicas para usar como paquete:
```python
from experiments import run_single_experiment, EXPERIMENTS
```

## Documentación

### `README.md`
Documentación completa:
- Descripción de experimentos
- Instrucciones de uso
- Formato de resultados
- Interpretación de diagramas
- Dependencias

### `QUICKSTART.md`
Guía de inicio rápido:
- Instalación
- Ejemplos básicos
- Troubleshooting
- Casos de uso comunes

## Flujo de trabajo típico

### 1. Ejecutar todos los experimentos
```bash
cd experiments
./run_all.sh
```

### 2. Explorar resultados
```bash
python explore_results.py
```

### 3. Analizar específicamente
```python
from experiments import run_single_experiment

results = run_single_experiment('producto', n=50)
```

## Formato de salida

### JSON (estructurado)
```json
{
  "experiment_name": "producto",
  "n": 50,
  "lipschitz_constant": 1.234,
  "cone": {
    "dgm_ker": [[...], ...],
    "dgm_coker": [[...], ...],
    "dgm_cone": [[...], ...],
    "dgm_X": [[...], ...],
    "dgm_Y": [[...], ...]
  },
  "cone2": {...},
  "cylinder": {...}
}
```

### Reportes de texto
```
REPORTE DE EXPERIMENTO
=============================================================

Experimento: producto
Tamaño n: 50
Constante de Lipschitz: 1.234567

CONE
----------------------------------------
Barras del Kernel:
  Dimensión 0: 5 barras
    [0] (0.1234, 0.5678) - persistencia: 0.4444
  ...
```

### Gráficos PNG
6 subplots por método:
1. Espacio X
2. Espacio Y  
3. Cono
4. Kernel
5. Cokernel
6. Info

## Extensibilidad

### Añadir un nuevo experimento

En `generate_spaces.py`:
```python
def mi_experimento(n=50, dim=2, seed=42):
    X = ...  # Generar X
    Y = ...  # Generar Y
    f = ...  # Definir función
    return X, Y, f

EXPERIMENTS['mi_experimento'] = mi_experimento
```

### Cambiar parámetros globales

Editar `config.py`:
```python
DEFAULT_N_VALUES = [30, 60, 100]
DEFAULT_MAXDIM = 3
```

### Añadir nueva métrica

En `analyze_results.py`, función `compute_statistics()`:
```python
stats['mi_metrica'] = calcular_mi_metrica(dgm)
```

## Características destacadas

✓ **Reproducibilidad**: Semillas fijas para resultados consistentes
✓ **Modularidad**: Funciones independientes y reutilizables
✓ **Flexibilidad**: Configuración centralizada y extensible
✓ **Documentación**: 3 niveles (README, QUICKSTART, docstrings)
✓ **Múltiples formatos**: JSON, pickle, CSV, TXT, PNG
✓ **Análisis completo**: Estadísticas, comparaciones, visualizaciones
✓ **Uso interactivo**: Scripts de línea de comando y Python
✓ **Manejo de errores**: Try-catch en ejecución de métodos

## Dependencias

Core:
- numpy
- scipy  
- matplotlib
- gudhi
- persistent_cost (cone, cone2, cylinder)

Análisis:
- pandas
- tabulate (para explore_results.py)

Todas están en requirements.txt o pyproject.toml del proyecto principal.

## Notas de implementación

1. **Normalización de Lipschitz**: Los métodos cone/cone2 normalizan dY por L
2. **Manejo de infinitos**: Barras infinitas se marcan con ∞ y se plotean como △
3. **Condensed form**: Las matrices de distancia se pasan en forma condensada a cone/cone2
4. **Sparse matrices**: cylinder usa matrices dispersas internamente
5. **Threshold**: Solo cylinder usa threshold; cone/cone2 filtran completamente
6. **Missing bars**: cone reporta barras no clasificadas como ker/coker

## Próximos pasos sugeridos

1. Ejecutar `./run_all.sh` para generar resultados base
2. Usar `explore_results.py` para navegar interactivamente
3. Comparar resultados entre métodos
4. Ajustar parámetros en `config.py` según necesidad
5. Extender con nuevos experimentos si se requiere
