#!/bin/bash
# Script para ejecutar todos los experimentos en lote

echo "======================================"
echo "Ejecutando todos los experimentos"
echo "======================================"

cd "$(dirname "$0")"

# Crear directorio de resultados si no existe
mkdir -p results

echo ""
echo "Iniciando ejecución..."
echo ""

python3 run_experiments.py

echo ""
echo "======================================"
echo "Ejecución completada"
echo "======================================"
echo ""
echo "Todos los resultados están en el directorio 'results/'"
echo "Abre 'results_viewer.html' en tu navegador para visualizarlos"
echo ""
