#!/usr/bin/env bash
# run_bench.sh -- benchmark de escalado de T1 (Arnoldi-Lindblad MPI+OpenMP).
#
# Mide el tiempo de pared del calculo de modos lentos variando:
#   (A) hilos OpenMP (1 rango)            -> escalado de memoria compartida
#   (B) rangos MPI   (1 hilo por rango)   -> escalado de datos distribuidos
#   (C) hibrido  (rangos x hilos)         -> uso combinado
# Escribe results/t1_bench.csv con columnas: modo,N,d,ranks,threads,spmv,wall
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$HERE/build/arnoldi_lindblad_mpi"
CFG="$HERE/../configs/ising_bench.ini"
OUT="$HERE/../results/t1_bench.csv"
MPIRUN="${MPIRUN:-mpirun}"

echo "modo,N,d,ranks,threads,spmv,wall" > "$OUT"

run() {  # $1=modo $2=ranks $3=threads
    local modo=$1 ranks=$2 threads=$3
    local line
    line=$(OMP_NUM_THREADS=$threads "$MPIRUN" --oversubscribe -np "$ranks" "$BIN" "$CFG" 2>/dev/null | grep '^BENCH,')
    # BENCH,N,d,ranks,threads,spmv,wall
    echo "$modo,${line#BENCH,}" | tee -a "$OUT"
}

echo "=== (A) OpenMP: 1 rango, hilos 1/2/4/8 ==="
for t in 1 2 4 8; do run openmp 1 "$t"; done

echo "=== (B) MPI: 1 hilo, rangos 1/2/4 ==="
for r in 1 2 4; do run mpi "$r" 1; done

echo "=== (C) Hibrido: rangos x hilos ==="
run hibrido 2 2
run hibrido 2 4
run hibrido 4 2

echo "Resultados -> $OUT"
