# Guía de Inicio Rápido - Experimentos

## Instalación de dependencias

Asegúrate de tener instalado el paquete persistent_cost:

```bash
cd /home/miles/Dropbox/PABLo/proyectos/tda/persistent-cost
pip install -e '.[accel]'
```

## Ejecución rápida

### Opción 1: Ejecutar todos los experimentos

```bash
cd experiments
./run_all.sh
```

o

```bash
cd experiments
python run_experiments.py
```

Esto ejecutará los 6 experimentos con n=50 y n=100 (12 ejecuciones totales).

### Opción 2: Ejecutar un experimento específico

```bash
cd experiments
python quick_test.py producto 50
```

Experimentos disponibles:
- `inclusion_punto`
- `producto`
- `suspension`
- `toro_proyecta`
- `circulo_en_toro`
- `muestreo_random`

### Opción 3: Visualizar resultados

Abre el visualizador web en tu navegador:

```bash
cd experiments
open results_viewer.html  # macOS
xdg-open results_viewer.html  # Linux
start results_viewer.html  # Windows
```

Luego carga los archivos JSON desde el botón "Cargar Resultados JSON".

### Opción 4: Desde Python

```python
import sys
sys.path.append('/home/miles/Dropbox/PABLo/proyectos/tda/persistent-cost')

from experiments import run_single_experiment, generate_report

results = run_single_experiment(
    experiment_name='producto',
    n=50,
    dim=2,
    maxdim=2,
    seed=42
)

generate_report(results, output_dir='results')
```

## Visualización de resultados

### Visualizador Web (Recomendado)

Abre `results_viewer.html` en tu navegador y carga los archivos JSON:

1. Abre el archivo HTML en cualquier navegador moderno
2. Haz clic en "Cargar Resultados JSON"
3. Selecciona uno o varios archivos JSON del directorio `results/`
4. Navega entre experimentos usando los selectores
5. Cambia entre métodos (cone, cone2, cylinder) con las pestañas
6. Explora los diagramas y estadísticas de cada componente

### Archivos generados automáticamente

Los scripts de ejecución también generan:

## Estructura de resultados

```
results/
├── <experimento>_n<n>_<timestamp>.json    # Datos en JSON (úsalos en el visualizador)
├── <experimento>_n<n>_<timestamp>.pkl     # Datos en pickle (backup)
├── <experimento>_n<n>/
│   ├── report.txt                         # Reporte textual
│   ├── cone_diagrams.png                  # Visualización cone
│   ├── cone2_diagrams.png                 # Visualización cone2
│   └── cylinder_diagrams.png              # Visualización cylinder
└── summary.json                           # Resumen general
```

## Interpretación de resultados

### Visualizador Web

Cada gráfico muestra 6 subplots:
1. **X**: Homología del espacio fuente
2. **Y**: Homología del espacio destino
3. **Cono/Cilindro**: Homología del espacio de mapeo
4. **Kernel**: Clases que mueren al aplicar f
5. **Cokernel**: Clases que nacen al aplicar f
6. **Info**: Información del método usado

### Colores por dimensión

- 🔴 Rojo: H₀ (componentes conexas)
- 🔵 Azul: H₁ (ciclos/loops)
- 🟢 Verde: H₂ (cavidades)

### Interpretación de barras

Una barra (b, d) significa:
- **b** (birth): La clase homológica nace en filtración b
- **d** (death): La clase homológica muere en filtración d
- **d - b**: Persistencia (estabilidad de la característica)

Barras con triángulos (△) tienen muerte infinita.

## Personalización

Edita `config.py` para cambiar:
- Tamaños de nubes (`DEFAULT_N_VALUES`)
- Dimensiones (`DEFAULT_DIM`, `DEFAULT_MAXDIM`)
- Threshold para cylinder (`DEFAULT_THRESHOLD`)
- Parámetros de visualización
- Directorio de salida

## Troubleshooting

### Error: "No module named 'persistent_cost'"

Instala el paquete:
```bash
cd /home/miles/Dropbox/PABLo/proyectos/tda/persistent-cost
pip install -e .
```

### Error: "Cython extension not built"

Instala con aceleradores:
```bash
pip install -e '.[accel]'
```

### Experimentos muy lentos

Para pruebas rápidas, usa n pequeño:
```bash
python quick_test.py producto 20
```

### Memoria insuficiente

Reduce `maxdim` o `n` en `run_experiments.py`:
```python
maxdim = 1  # Solo H₀ y H₁
n_values = [30, 50]  # Nubes más pequeñas
```

## Ejemplos de uso avanzado

### Comparar solo dos métodos

Edita `run_experiments.py` y comenta el método no deseado en `run_single_experiment()`.

### Cambiar semilla para diferentes muestras

```python
results = run_single_experiment(
    experiment_name='producto',
    n=50,
    seed=123  # Semilla diferente
)
```

### Generar solo gráficos sin reejecutar

```python
import json
from visualization import plot_persistence_diagrams

with open('results/producto_n50_<timestamp>.json') as f:
    results = json.load(f)

plot_persistence_diagrams(results, 'results/producto_n50')
```

## Notas importantes

1. **Reproducibilidad**: Todos los experimentos usan semillas fijas (seed=42 por defecto)
2. **Normalización**: Los métodos cone/cone2 normalizan dY por la constante de Lipschitz
3. **Threshold**: Solo cylinder usa threshold; cone/cone2 computan toda la filtración
4. **Tiempo**: Experimentos con n=100 pueden tomar varios minutos cada uno
5. **Espacio**: Los resultados pueden ocupar varios MB por experimento

## Contacto y soporte

Para preguntas sobre los experimentos o resultados, consulta:
- `EXPERIMENTOS.md` - Especificación detallada
- `README.md` - Documentación completa
- Código fuente en `generate_spaces.py`, `run_experiments.py`, `visualization.py`
