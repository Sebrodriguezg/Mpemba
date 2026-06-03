#!/bin/bash
# bench.sh -- benchmark T3(a): trayectorias cuanticas con MPI.
# Escalado fuerte sobre rangos MPI (debe ser casi ideal) + error estadistico vs M.
set -e
cd "$(dirname "$0")"
BIN=./trajectories_mpi
mkdir -p results
C=/tmp/bench_t3.ini
MPIRUN="mpirun.openmpi --oversubscribe"

echo "=== escalado fuerte: N=7, M=2000, vs nro de rangos MPI ==="
printf "[model]\nN=7\nh=0.5\nJ=1.0\ngamma=0.4\nT=0.8\n[run]\nt_max=3.0\ndt=0.02\nM=2000\nlog_every=20\n" > $C
echo "N,ranks,M,wall_s,dhs" > results/scaling_ranks.csv
for p in 1 2 4 8 16; do
  echo "  rangos=$p ..."
  $MPIRUN -np $p $BIN $C >> results/scaling_ranks.csv
done

echo "=== error estadistico vs M: N=6, 8 rangos (ref exacta = dhs_exact.txt) ==="
echo "N,ranks,M,wall_s,dhs" > results/error_vs_M.csv
for M in 250 500 1000 2000 4000 8000 16000; do
  echo "  M=$M ..."
  printf "[model]\nN=6\nh=0.5\nJ=1.0\ngamma=0.4\nT=0.8\n[run]\nt_max=4.0\ndt=0.02\nM=%s\nlog_every=20\n" "$M" > $C
  $MPIRUN -np 8 $BIN $C >> results/error_vs_M.csv
done

echo "=== serie vs paralelo: tiempo de computo vs nro de trayectorias M, 1 rango vs 8 rangos ==="
# el trabajo MC crece con M; serie (1 rango) vs paralelo (8 rangos) -> dos rectas
# que divergen, evidenciando la mejora (escalado casi ideal, sin comunicacion).
echo "N,ranks,M,wall_s,dhs" > results/serial_vs_parallel.csv
for M in 500 1000 2000 4000; do
  printf "[model]\nN=6\nh=0.5\nJ=1.0\ngamma=0.4\nT=0.8\n[run]\nt_max=4.0\ndt=0.02\nM=%s\nlog_every=20\n" "$M" > $C
  for p in 1 8; do
    echo "  M=$M rangos=$p ..."
    $MPIRUN -np $p $BIN $C >> results/serial_vs_parallel.csv
  done
done

echo "=== curva fisica: N=8 (d=256), 8 rangos, M=4000 ==="
printf "[model]\nN=8\nh=0.5\nJ=1.0\ngamma=0.4\nT=0.8\n[run]\nt_max=6.0\ndt=0.02\nM=4000\nlog_every=10\n" > $C
$MPIRUN -np 8 $BIN $C results/curve.csv > /dev/null

echo "DONE"
