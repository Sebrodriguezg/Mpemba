#!/bin/bash

# Salir inmediatamente si algún comando crítico falla
set -e

echo "================================================="
echo " Iniciando Benchmarking HPC - Efecto Mpemba LBM  "
echo "================================================="

# Archivo donde se guardarán los tiempos
OUT_FILE="benchmark_results.csv"

# Crear el encabezado del archivo CSV
echo "Method,Threads,Time" > $OUT_FILE

# 1. Ejecutar Serial (Baseline)
echo "[1/4] Ejecutando código Serial baseline..."
./mpemba_serial >> $OUT_FILE

# Definir la cantidad de hilos/procesos a evaluar
# Ajusta estos números si tu computadora tiene más núcleos (ej: 1 2 4 8 16)
RECURSOS=(2 4 8 16 32 64 128)

# 2. Ejecutar OpenMP
echo "[2/4] Ejecutando rutinas de OpenMP..."
for p in "${RECURSOS[@]}"; do
    echo "  -> Evaluando con $p hilos..."
    # Pasamos el número de hilos mediante la variable de entorno estándar
    OMP_NUM_THREADS=$p ./mpemba_omp >> $OUT_FILE
done

# 3. Ejecutar MPI
echo "[3/4] Ejecutando rutinas de MPI..."
for p in "${RECURSOS[@]}"; do
    echo "  -> Evaluando con $p procesos distribuidos..."
    # Usamos mpirun (o mpiexec) con el flag -np (number of processes)
    # El flag --oversubscribe es útil si pruebas en tu laptop con más procesos que núcleos físicos
    mpirun --allow-run-as-root -np $p --oversubscribe ./mpemba_mpi >> $OUT_FILE
done

# 4. Ejecutar CUDA
echo "[4/4] Ejecutando kernel de CUDA (GPU)..."
./mpemba_cuda >> $OUT_FILE

echo "================================================="
echo " Benchmarking completado. Resultados en $OUT_FILE"
echo "================================================="

# 5. Llamar a Python para graficar
echo "Generando gráficas de rendimiento..."
python3 plot_performance.py

echo "Proceso finalizado exitosamente."
