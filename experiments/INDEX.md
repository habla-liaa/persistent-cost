# Índice de Documentación - Experimentos

## 📚 Guías de Usuario

| Documento | Propósito | Nivel |
|-----------|-----------|-------|
| **QUICKSTART.md** | Guía de inicio rápido, primeros pasos | 🟢 Principiante |
| **README.md** | Documentación completa, referencia | 🟡 Intermedio |
| **SUMMARY.md** | Resumen técnico del sistema | 🔴 Avanzado |
| **EXPERIMENTOS.md** | Especificación original de experimentos | 📋 Referencia |

## 🔧 Archivos de Código

### Scripts Ejecutables
- `test_system.py` - Verificación de instalación
- `quick_test.py` - Ejecutar un experimento individual
- `run_all.sh` - Ejecutar todos los experimentos
- `run_experiments.py` - Script principal de ejecución

### Módulos de Python
- `generate_spaces.py` - Generación de espacios experimentales
- `visualization.py` - Funciones de visualización (PNG)
- `config.py` - Configuración centralizada
- `__init__.py` - Exportaciones del paquete

### Visualizador Web
- `results_viewer.html` - Visualizador interactivo HTML+JavaScript
- `viewer.js` - Lógica del visualizador

## 🚀 Inicio Rápido

### 1. Verificar instalación
```bash
./test_system.py
```

### 3. Ejecutar un experimento de prueba
```bash
./quick_test.py producto 20
```

### 4. Visualizar resultados
Abre `results_viewer.html` en tu navegador y carga los archivos JSON generados.

## 📖 Rutas de Aprendizaje

### Para Principiantes
1. Lee **QUICKSTART.md**
2. Ejecuta `./test_system.py`
3. Prueba `./quick_test.py producto 20`
4. Explora resultados con `explore_results.py`

### Para Usuarios Intermedios
1. Lee **README.md**
2. Revisa **EXPERIMENTOS.md** para entender los experimentos
3. Ejecuta `./run_all.sh`
4. Abre `results_viewer.html` para visualizar resultados
5. Modifica `config.py` según necesidades

### Para Desarrolladores
1. Lee **SUMMARY.md**
2. Examina el código fuente
3. Extiende `generate_spaces.py` con nuevos experimentos
4. Personaliza `visualization.py` para nuevos gráficos
5. Contribuye mejoras

## 🎯 Casos de Uso Comunes

### Ejecutar un experimento específico
```bash
./quick_test.py <experimento> <n>
```
Ver: QUICKSTART.md, sección "Ejecución rápida"

### Visualizar y explorar resultados
Abre `results_viewer.html` en tu navegador y carga los JSON
Ver: README.md, sección "Visualización"

### Generar gráficos PNG
Los gráficos se generan automáticamente al ejecutar experimentos
Ver: visualization.py

### Añadir un nuevo experimento
Editar `generate_spaces.py`
Ver: SUMMARY.md, sección "Añadir un nuevo experimento"

### Cambiar parámetros globalmente
Editar `config.py`
Ver: README.md, sección "Personalización"

## 🔍 Referencia Rápida

### Experimentos Disponibles
1. `inclusion_punto` - Inclusión de punto en nube
2. `producto` - Identidad de nube en sí misma
3. `suspension` - Suspensión (nube → punto)
4. `toro_proyecta` - Proyección de toro 3D → 2D
5. `circulo_en_toro` - Círculo incluido en toro
6. `muestreo_random` - Submuestreo aleatorio

### Métodos Implementados
- **cone** - Usa Ripser, más rápido
- **cone2** - Usa GUDHI, más preciso
- **cylinder** - Método algebraico

### Formatos de Salida
- JSON - Datos estructurados
- Pickle - Objetos Python completos
- CSV - Tablas comparativas
- TXT - Reportes legibles
- PNG - Visualizaciones

### Estructura de Resultados
```
results/
├── <experimento>_n<n>_<timestamp>.{json,pkl}
├── <experimento>_n<n>/
│   ├── report.txt
│   └── {cone,cone2,cylinder}_diagrams.png
└── summary.json
```

## 🔍 Visualización Interactiva

### Visualizador Web (`results_viewer.html`)

**Características:**
- Carga múltiple de archivos JSON
- Navegación por experimento y tamaño
- Tabs para cada método (cone, cone2, cylinder)
- 6 diagramas por método: X, Y, Cono, Ker, Coker, Missing
- Estadísticas en tiempo real
- Lista completa de barras por dimensión

**Uso:**
1. Abre `results_viewer.html` en cualquier navegador moderno
2. Carga archivos JSON desde el botón de carga
3. Selecciona experimento y n en los selectores
4. Explora diagramas y estadísticas

**No requiere servidor web** - funciona directamente desde el sistema de archivos.

## 🆘 Ayuda y Soporte

### Problemas Comunes

**Error: "No module named 'persistent_cost'"**
```bash
cd .. && pip install -e .
```

**Error: "Cython extension not built"**
```bash
cd .. && pip install -e '.[accel]'
```

**Tests fallan**
```bash
./test_system.py  # Para diagnóstico
```

**Resultados inesperados**
- Verifica semilla: `seed=42` por defecto
- Revisa parámetros en `config.py`
- Consulta logs en reportes de texto

### Recursos Adicionales

**Dentro del proyecto:**
- Docstrings en cada función
- Comentarios en código
- Tests unitarios en `../tests/`

**Documentación del paquete:**
- `../README.md` - README principal
- `../src/persistent_cost/` - Código fuente

## 📊 Interpretación de Resultados

### Diagramas de Persistencia
- **Eje X**: Birth (nacimiento)
- **Eje Y**: Death (muerte)
- **Diagonal**: Línea de referencia
- **Triángulos (△)**: Barras infinitas
- **Colores en visualizador web**:
  - 🔴 Rojo: H₀ (componentes conexas)
  - 🔵 Azul: H₁ (ciclos)
  - 🟢 Verde: H₂ (cavidades)
  - 🟠 Naranja: H₃
  - 🟣 Violeta: H₄

### Interpretación de Barras
- **(b, d)**: Nace en b, muere en d
- **d - b**: Persistencia (importancia)
- **d = ∞**: Característica persiste indefinidamente

### Kernel vs Cokernel
- **Kernel**: Clases que mueren al aplicar f
- **Cokernel**: Clases que nacen al aplicar f
- **Desapareadas**: Barras no clasificadas

## 🔄 Flujo de Trabajo Completo

```
1. Preparación
   ├─ Leer QUICKSTART.md
   ├─ Ejecutar test_system.py
   └─ Configurar config.py

2. Ejecución
   ├─ Prueba: quick_test.py
   ├─ Completo: run_all.sh
   └─ Personalizado: run_experiments.py

3. Visualización
   ├─ Web: results_viewer.html (recomendado)
   ├─ PNG: *_diagrams.png
   └─ Texto: report.txt

4. Análisis
   ├─ Explorar diagramas en visualizador
   ├─ Comparar métodos por tabs
   └─ Revisar estadísticas por componente
```

## 📝 Notas Importantes

- ⚠️ Experimentos con n=100 pueden tomar varios minutos
- 💾 Los resultados pueden ocupar varios MB
- 🎲 Usa semillas consistentes para reproducibilidad
- 🔧 Ajusta parámetros en `config.py`, no en código
- 📊 Revisa logs en `report.txt` para diagnósticos

## 📅 Mantenimiento

### Limpieza de Resultados
```bash
rm -rf results/*
```

### Actualización del Sistema
```bash
cd .. && git pull && pip install -e '.[accel]'
```

### Verificación Post-Actualización
```bash
./test_system.py
```

---

**Última actualización**: Noviembre 2025
**Versión**: 0.1.0
**Mantenedor**: PABLo Team
