# T1 — Modos lentos del Liouvilliano (Arnoldi-Lindblad)

Extrae los autovalores del Liouvilliano $\mathcal{L}$ más cercanos a 0 (estado
estacionario $\lambda_1=0$ y modos lentos $\lambda_2,\lambda_3,\dots$) y sus
autooperadores derecho/izquierdo, para diagnosticar el efecto Mpemba y calcular
los solapamientos $a_k=\mathrm{Tr}(l_k^\dagger\rho_0)$.

**Método** (Minganti & Huybrechts, *Quantum* **6**, 649, 2022): Arnoldi sobre el
propagador $P(\tau)=e^{\tau\mathcal{L}}$ —cuya acción se calcula *matrix-free*
integrando $\dot V=\mathcal{L}[V]$ con RK4— en vez de diagonalizar o invertir
$\mathcal{L}$. Como $\mu_k=e^{\tau\lambda_k}$ y $\operatorname{Re}\lambda_k\le0$,
los modos lentos de $\mathcal{L}$ son los autovalores *dominantes en módulo* de
$P$, que Arnoldi extrae primero ("faster than the clock").

## Serial (Python)
```bash
cd python
python3 validate_t1.py     # valida vs diagonalizacion densa (oraculo)
```
Genera `results/t1_validation.csv` y `results/t1_eigs.png`.
**Resultado:** error en autovalores lentos $10^{-13}$–$2.6\times10^{-5}$ vs el
oráculo denso; *faster-than-the-clock* de hasta **8.8×** en SpMV (Λ-3 niveles).

## Paralelo (C++ MPI + OpenMP)
```bash
cd cpp && mkdir -p build && cd build
cmake .. -DCMAKE_CXX_COMPILER=g++-15 && make
OMP_NUM_THREADS=4 mpirun --oversubscribe -np 2 ./arnoldi_lindblad_mpi ../../configs/ising_n4.ini
```
Valida contra el oráculo (N=4: $\lambda_2=-0.69558$, coincide a 6 cifras).

**Paralelización:** Arnoldi es secuencialmente acoplado → el paralelismo vive en
el **SpMV** (hotspot). OpenMP paraleliza el `matmul`; MPI distribuye las filas
del `matmul` y reconstruye con `Allgatherv` (datos distribuidos, §8.1 del informe).

**Benchmark** (`./run_bench.sh` → `plot_bench.py`, Ising N=7, d=128):

| workers | OpenMP speedup | MPI speedup |
|---|---|---|
| 1 | 1.0× | 1.0× |
| 2 | 1.79× | 1.81× |
| 4 | 2.89× | 2.71× |
| 8 | 2.95× | — |

Plateau a 8 workers por ancho de banda de memoria (matmul memory-bound). MPI
escala como OpenMP pese a la comunicación `Allgatherv`. Figura: `results/t1_scaling.png`.
