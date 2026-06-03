#!/bin/bash
set -e

echo "====================================================="
echo "  Producción Científica: Búsqueda del Crossover      "
echo "  Arquitectura: Lattice Boltzmann en CUDA (GPU)      "
echo "====================================================="

# Limpiar resultados anteriores para evitar contaminación de datos
rm -f datos-cuda_T*.csv

echo "Compilando cuda_lbm.cu..."
nvcc -O3 cuda_lbm.cu -o mpemba_cuda
echo "Compilación exitosa."
echo "-----------------------------------------------------"

# Temperaturas iniciales clave para revelar la dinámica de la memoria
TEMPERATURES=(30 95)

for T in "${TEMPERATURES[@]}"; do
    ./mpemba_cuda $T
done

echo "-----------------------------------------------------"
echo "Simulaciones finalizadas. Generando gráfica..."
python3 plot_sweep.py

echo "Proceso finalizado."
