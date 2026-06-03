#!/bin/bash
# bench.sh -- benchmark T2(b): Krylov. Escalado fuerte + comparacion de metodo (a vs b).
set -e
cd "$(dirname "$0")"
KRY=./krylov_evolution
RK4=../a_integracion_directa/rk4_evolution
mkdir -p results
C=/tmp/bench_t2b.ini

echo "=== escalado fuerte: Krylov N=7, vs hilos ==="
printf "[model]\nN=7\nh=0.5\nJ=1.0\ngamma=0.4\nT=0.8\n[run]\nT0=5.0\nt_max=4.0\ntau=0.5\nm=30\n" > $C
echo "N,threads,m,tau,nchunks,Lapplies,wall_s,dhs" > results/scaling_threads.csv
for p in 1 2 4 6 8 12 16; do
  echo "  hilos=$p ..."
  OMP_NUM_THREADS=$p $KRY $C >> results/scaling_threads.csv
done

echo "=== comparacion de metodo a vs b: precision vs trabajo (N=6, t_max=3, 8 hilos) ==="
export OMP_NUM_THREADS=8
# RK4 a varios dt
echo "method,param,Lapplies,wall_s,dhs" > results/method_compare.csv
for dt in 0.05 0.02 0.01 0.005 0.002 0.001; do
  printf "[model]\nN=6\nh=0.5\nJ=1.0\ngamma=0.4\nT=0.8\n[run]\nT0=5.0\nt_max=3.0\ndt=%s\nlog_every=100000\n" "$dt" > $C
  line=$($RK4 $C)                       # N,threads,steps,wall,dhs
  steps=$(echo $line | cut -d, -f3); wall=$(echo $line | cut -d, -f4); dhs=$(echo $line | cut -d, -f5)
  echo "RK4,dt=$dt,$((steps*4)),$wall,$dhs" >> results/method_compare.csv
done
# Krylov a varios (tau,m)
for cfg in "1.0 10" "1.0 20" "0.5 20" "0.5 30" "0.25 30" "0.25 40"; do
  tau=$(echo $cfg | cut -d' ' -f1); m=$(echo $cfg | cut -d' ' -f2)
  printf "[model]\nN=6\nh=0.5\nJ=1.0\ngamma=0.4\nT=0.8\n[run]\nT0=5.0\nt_max=3.0\ntau=%s\nm=%s\n" "$tau" "$m" > $C
  line=$($KRY $C)                       # N,threads,m,tau,nchunks,Lapplies,wall,dhs
  La=$(echo $line | cut -d, -f6); wall=$(echo $line | cut -d, -f7); dhs=$(echo $line | cut -d, -f8)
  echo "Krylov,tau=$tau/m=$m,$La,$wall,$dhs" >> results/method_compare.csv
done

echo "=== serie vs paralelo: tiempo de computo vs tamano del sistema (N), 1 hilo vs 8 hilos ==="
# mismo (tau,m) y mismo tramo temporal; solo cambia OMP_NUM_THREADS -> dos curvas.
echo "N,threads,m,tau,nchunks,Lapplies,wall_s,dhs" > results/serial_vs_parallel.csv
for N in 4 5 6 7; do
  printf "[model]\nN=%s\nh=0.5\nJ=1.0\ngamma=0.4\nT=0.8\n[run]\nT0=5.0\nt_max=3.0\ntau=0.5\nm=20\n" "$N" > $C
  for p in 1 8; do
    echo "  N=$N hilos=$p ..."
    OMP_NUM_THREADS=$p $KRY $C >> results/serial_vs_parallel.csv
  done
done

echo "=== curva fisica: Krylov N=7 ==="
printf "[model]\nN=7\nh=0.5\nJ=1.0\ngamma=0.4\nT=0.8\n[run]\nT0=5.0\nt_max=6.0\ntau=0.25\nm=30\n" > $C
$KRY $C results/curve.csv > /dev/null

echo "DONE"
