#!/usr/bin/env bash
# run_bench.sh -- escalado de T3a (trayectorias cuanticas MPI+OpenMP).
# Caso embarrassingly parallel: se espera escalado casi ideal.
# Escribe results/t3a_bench.csv: modo,N,d,ranks,threads,M,wall
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$HERE/build/qtraj_mpi"
CFG="$HERE/../configs/ising_bench.ini"
OUT="$HERE/../results/t3a_bench.csv"
MPIRUN="${MPIRUN:-mpirun}"

echo "modo,N,d,ranks,threads,M,wall" > "$OUT"
run() {  # modo ranks threads
    local line
    line=$(OMP_NUM_THREADS=$3 "$MPIRUN" --oversubscribe -np "$2" "$BIN" "$CFG" 2>/dev/null | grep '^BENCH,')
    echo "$1,${line#BENCH,}" | tee -a "$OUT"
}
echo "=== (A) OpenMP: 1 rango, hilos 1/2/4/8 ==="
for t in 1 2 4 8; do run openmp 1 "$t"; done
echo "=== (B) MPI: 1 hilo, rangos 1/2/4/8 ==="
for r in 1 2 4 8; do run mpi "$r" 1; done
echo "Resultados -> $OUT"
