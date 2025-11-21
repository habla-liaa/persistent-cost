# Experimentos de Persistent Cost

Este directorio contiene scripts para ejecutar experimentos comparativos de los métodos `cone`, `cone2` y `cylinder` del paquete `persistent_cost`.

## Estructura

- `generate_spaces.py`: Funciones para generar los espacios X, Y y la función f para cada experimento
- `run_experiments.py`: Script principal que ejecuta todos los experimentos
- `visualization.py`: Funciones para visualización y generación de reportes
- `EXPERIMENTOS.md`: Especificación detallada de los experimentos

## Experimentos Implementados

1. **inclusion_punto**: Inclusión de un punto en una nube aleatoria
2. **producto**: Producto (identidad de una nube en sí misma)
3. **suspension**: Suspensión (nube mapeada a un punto)
4. **toro_proyecta**: Proyección de un toro a sus primeras dos coordenadas
5. **circulo_en_toro**: Círculo incluido en un muestreo del toro
6. **muestreo_random**: Inclusión de un submuestreo en una nube aleatoria

## Uso

### Ejecutar todos los experimentos

```bash
python run_experiments.py
```

Esto ejecutará todos los experimentos con `n=50` y `n=100`, guardando los resultados en el directorio `results/`.

### Ejecutar un experimento individual

```python
from run_experiments import run_single_experiment

results = run_single_experiment(
    experiment_name='producto',
    n=50,
    dim=2,
    threshold=3.0,
    maxdim=2,
    seed=42
)
```

### Personalizar parámetros

Edita las variables en la función `main()` de `run_experiments.py`:

```python
n_values = [50, 100]  # Tamaños de nubes
dim = 2               # Dimensión de las nubes
maxdim = 2            # Dimensión homológica máxima
threshold = 3.0       # Threshold para cylinder
seed = 42             # Semilla para reproducibilidad
```

## Resultados

Los resultados se guardan en el directorio `results/` con la siguiente estructura:

```
results/
├── <experimento>_n<n>_<timestamp>.json    # Resultados en JSON
├── <experimento>_n<n>_<timestamp>.pkl     # Resultados en pickle (backup)
├── <experimento>_n<n>/
│   ├── report.txt                         # Reporte detallado
│   ├── cone_diagrams.png                  # Diagramas para método cone
│   ├── cone2_diagrams.png                 # Diagramas para método cone2
│   └── cylinder_diagrams.png              # Diagramas para método cylinder
└── summary.json                           # Resumen de todos los experimentos
```

**Nota**: Para visualizar y analizar los resultados, usa el visualizador web `results_viewer.html` en lugar de los scripts de análisis Python.

### Formato de salida

Cada resultado JSON incluye:

- **Constante de Lipschitz** antes de la normalización
- **Tamaños** de los espacios X e Y
- **Diagramas de persistencia** para cada método (cone, cone2, cylinder):
  - Espacio X (`dgm_X`)
  - Espacio Y (`dgm_Y`)
  - Cono (`dgm_cone`)
  - Kernel (`dgm_ker`)
  - Cokernel (`dgm_coker`)
  - Missing (`missing`) - barras no clasificadas (solo cone/cone2)
- **Lista de barras** por dimensión con nacimiento y muerte

## Visualización

Usa el visualizador web `results_viewer.html` para explorar los resultados de forma interactiva.

### Características del visualizador:
- **Carga múltiple**: Carga varios archivos JSON simultáneamente
- **Navegación**: Filtra por experimento y tamaño de muestra
- **Tabs por método**: Cambia entre cone, cone2 y cylinder
- **Diagramas interactivos**: 6 diagramas por método:
  1. Espacio X
  2. Espacio Y
  3. Cono
  4. Kernel
  5. Cokernel
  6. Missing (si aplica)
- **Estadísticas**: Para cada diagrama:
  - Total de barras (finitas e infinitas)
  - Persistencia total, promedio y máxima
  - Conteo por dimensión homológica
- **Lista de barras**: Detalle de cada barra con (nacimiento, muerte, persistencia)

### Colores por dimensión:
- 🔴 Rojo: H₀ (componentes conexas)
- 🔵 Azul: H₁ (ciclos/loops)
- 🟢 Verde: H₂ (cavidades)
- 🟠 Naranja: H₃
- 🟣 Violeta: H₄

Las barras infinitas se muestran como triángulos (△) en el borde superior del diagrama.

## Dependencias

### Para ejecutar experimentos:
- numpy
- scipy
- matplotlib
- gudhi
- persistent_cost (con módulos cone, cone2, cylinder)

### Para visualizar resultados:
- Navegador web moderno (Chrome, Firefox, Safari, Edge)
- No requiere dependencias adicionales (HTML+JavaScript puro)

## Notas

- Los experimentos con toro (`toro_proyecta`, `circulo_en_toro`) están en dimensión 3 por definición
- El parámetro `threshold` solo se usa para el método `cylinder`
- Los métodos `cone` y `cone2` usan Ripser y GUDHI respectivamente
- Todos los experimentos usan semilla fija para reproducibilidad
