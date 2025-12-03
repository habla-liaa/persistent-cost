#!/bin/bash
# Script para ejecutar todos los experimentos en lote

python experiments/run_experiments.py main --cylinder=False --cone_pairs=False --cone_htr=False --output_dir="docs/results/"
python experiments/run_experiments.py main --cylinder=False --cone_pairs=False --cone_htr=False --output_dir="docs/results/" --cone_eps=0.01
python experiments/run_experiments.py main --cylinder=False --cone_pairs=False --cone_htr=False --output_dir="docs/results/" --cone_eps=0.001