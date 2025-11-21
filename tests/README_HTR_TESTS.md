# Tests para la función HTR

## Resumen

Se han creado tests comprehensivos para la función `htr` (Homología de Tipo Rips) que comparan sus resultados con los de Ripser, la implementación estándar de referencia para computar homología persistente.

## Archivos creados/modificados

### Nuevo archivo de tests: `tests/test_htr.py`

Este archivo contiene 11 tests que verifican que `htr` produce los mismos resultados que Ripser:

1. **test_htr_vs_ripser_triangle**: Triángulo simple (3 puntos)
2. **test_htr_vs_ripser_square**: Cuadrado (4 puntos)
3. **test_htr_vs_ripser_circle_points**: Puntos en un círculo (10 puntos)
4. **test_htr_vs_ripser_random_2d**: Puntos aleatorios en 2D (15 puntos)
5. **test_htr_vs_ripser_3d_points**: Puntos aleatorios en 3D (12 puntos)
6. **test_htr_vs_ripser_distance_matrix**: Verifica que ambos consumen la misma matriz de distancias y que `htr` puede aceptar tanto `points` como `distance_matrix`
7-11. **test_htr_vs_ripser_parametric**: Tests parametrizados con diferentes configuraciones

### Signatura actualizada de `htr`

La función `htr` ahora acepta los siguientes argumentos:

```python
def htr(points=None, distance_matrix=None, threshold=None, maxdim=None):
    """
    Compute the persistent homology using the HTR algorithm.
    Args:
        points: point cloud data (optional if distance_matrix is provided)
        distance_matrix: square distance matrix (optional if points is provided)
        threshold: maximum distance for Rips complex
        maxdim: maximum homology dimension to compute
    Returns:
        births: list of birth times
        deaths: list of death times
        dims: list of dimensions corresponding to each birth-death pair
    """
```

**Cambios principales:**
- Primer argumento ahora es `points=None` (antes era `X`)
- Nuevo argumento opcional `distance_matrix=None` para pasar directamente una matriz de distancias cuadrada
- Se debe proporcionar exactamente uno de `points` o `distance_matrix`

### Correcciones realizadas

1. **Corrección de la función `htr` en `utils.py`**:
   - Actualizado para manejar correctamente el parámetro `distance_matrix`
   - Cuando se proporciona `distance_matrix`, se usa directamente (debe ser una matriz cuadrada)
   - Cuando se proporciona `points`, se calcula la matriz de distancias internamente

2. **Corrección de type hints para Python 3.8**:
   - Reemplazados `tuple[...]` → `Tuple[...]`
   - Reemplazados `list[...]` → `List[...]`
   - Reemplazados `... | None` → `Optional[...]`
   - Archivos corregidos:
     - `src/persistent_cost/algorithms/dense.py`
     - `src/persistent_cost/algorithms/dense_fast.py`
     - `src/persistent_cost/algorithms/sparse.py`
     - `src/persistent_cost/algorithms/sparse_fast.py`

## Funcionalidad de los tests

### Función auxiliar: `persistence_diagrams_to_arrays`
Convierte las listas de births, deaths, dims en el formato de diagramas de persistencia compatible con Ripser (lista de arrays, uno por dimensión).

### Función auxiliar: `compare_persistence_diagrams`
Compara dos conjuntos de diagramas de persistencia:
- Verifica que el número de features coincida por dimensión
- Ordena los features por tiempo de nacimiento y muerte
- Compara valores dentro de una tolerancia numérica (1e-5 por defecto)
- Retorna un booleano y un mensaje descriptivo

## Casos de prueba

Los tests cubren:
- **Geometrías simples**: triángulos, cuadrados
- **Estructuras topológicas**: círculos (con H₁ no trivial)
- **Puntos aleatorios**: 2D y 3D
- **Diferentes dimensiones homológicas**: H₀, H₁, H₂
- **Diferentes tamaños**: de 3 a 15 puntos
- **Verificación de consistencia**: 
  - Ambos algoritmos (htr y Ripser) consumen los mismos datos de entrada
  - `htr` produce los mismos resultados cuando recibe `points` o `distance_matrix`

## Ejemplos de uso

```python
import numpy as np
from persistent_cost.utils.utils import htr

# Opción 1: Pasar puntos directamente
X = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
births, deaths, dims = htr(points=X, threshold=2.0, maxdim=1)

# Opción 2: Pasar matriz de distancias
from scipy.spatial.distance import pdist, squareform
dX = squareform(pdist(X))
births, deaths, dims = htr(distance_matrix=dX, threshold=2.0, maxdim=1)
```

## Cómo ejecutar los tests

```bash
# Ejecutar todos los tests de htr
pytest tests/test_htr.py -v

# Ejecutar un test específico
pytest tests/test_htr.py::test_htr_vs_ripser_triangle -v

# Ejecutar con output detallado
pytest tests/test_htr.py -v -s

# Ejecutar directamente el script (muestra output formateado)
python tests/test_htr.py
```

## Resultados

✅ **Todos los tests pasan exitosamente**

Los 11 tests confirman que `htr` produce los mismos diagramas de persistencia que Ripser dentro de la tolerancia numérica especificada (1e-5).

## Ejemplo de output

```
HTR results:
  H_0: 4 features
    [[ 0. inf]
     [ 0.  1.]
     [ 0.  1.]
     [ 0.  1.]]
  H_1: 1 features
    [[1.         1.41421356]]

Ripser results:
  H_0: 4 features
    [[ 0.  1.]
     [ 0.  1.]
     [ 0.  1.]
     [ 0. inf]]
  H_1: 1 features
    [[1.         1.41421354]]

Comparison: All diagrams match
```

## Nota sobre el orden

Los features dentro de cada dimensión pueden aparecer en diferente orden entre HTR y Ripser (por ejemplo, el punto infinito puede estar al principio o al final), pero los tests ordenan ambos conjuntos antes de compararlos, asegurando que la comparación sea robusta.
