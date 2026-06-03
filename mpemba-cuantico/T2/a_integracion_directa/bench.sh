#!/bin/bash
# bench.sh -- benchmark del entregable T2(a): RK4 directo, serial vs OpenMP.
# Mide escalado fuerte (vs hilos) y escalado con el tamano del sistema (vs N).
set -e
cd "$(dirname "$0")"
BIN=./rk4_evolution
mkdir -p results
CFG=/tmp/bench_t2a.ini

mkcfg() { printf "[model]\nN=%s\nh=0.5\nJ=1.0\ngamma=0.4\nT=0.8\n[run]\nT0=5.0\nt_max=%s\ndt=0.01\nlog_every=%s\n" "$1" "$2" "$3" > "$CFG"; }

echo "=== escalado fuerte: N=7, vs nro de hilos ==="
mkcfg 7 2.0 10
echo "N,threads,steps,wall_s,dhs_final" > results/scaling_threads.csv
for p in 1 2 4 6 8 12 16; do
  echo "  hilos=$p ..."
  OMP_NUM_THREADS=$p $BIN "$CFG" >> results/scaling_threads.csv
done

echo "=== escalado con el tamano: 8 hilos, vs N ==="
echo "N,threads,steps,wall_s,dhs_final" > results/scaling_size.csv
for N in 4 5 6 7; do
  echo "  N=$N ..."
  mkcfg "$N" 2.0 10
  OMP_NUM_THREADS=8 $BIN "$CFG" >> results/scaling_size.csv
done

echo "=== serie vs paralelo: tiempo de computo vs tamano del sistema (N), 1 hilo vs 8 hilos ==="
# misma malla d=2^N y mismo nro de pasos; lo unico que cambia es OMP_NUM_THREADS.
# produce dos curvas T_serie(N) y T_paralelo(N) -> evidencia la mejora al crecer la malla.
echo "N,threads,steps,wall_s,dhs_final" > results/serial_vs_parallel.csv
for N in 4 5 6 7; do
  mkcfg "$N" 1.0 1000000          # t_max=1.0, dt=0.01 -> 100 pasos (sin volcado de curva)
  for p in 1 8; do
    echo "  N=$N hilos=$p ..."
    OMP_NUM_THREADS=$p $BIN "$CFG" >> results/serial_vs_parallel.csv
  done
done

echo "=== curva fisica de relajacion: N=7, 8 hilos ==="
mkcfg 7 6.0 5
OMP_NUM_THREADS=8 $BIN "$CFG" results/curve.csv > /dev/null

echo "DONE"
