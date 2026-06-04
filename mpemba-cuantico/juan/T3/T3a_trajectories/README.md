# T3a — Trayectorias cuánticas (Monte Carlo wavefunction)

Régimen de muchos cuerpos por **saltos cuánticos**: se propagan $M$ vectores de
estado $|\psi\rangle\in\mathbb{C}^d$ ($d=2^N$) y se recupera
$\rho=\mathbb{E}[|\psi\rangle\langle\psi|]$. La memoria pasa de $4^N$ a $2^N$,
alcanzando $N$ inviables para la ecuación maestra. Error estadístico $\sim M^{-1/2}$.
Ref.: Daley, *Adv. Phys.* **63**, 77 (2014), arXiv:1404.5028.

## Serial + paralelo (Python)
```bash
cd python
python3 validate_t3a.py     # convergencia M^-1/2 vs ecuacion maestra + speedup
```
**Resultado:** pendiente log-log del error $= -0.488$ (teórico $-0.5$);
multiprocessing **4.6×** vs serial (TLS, M=4000). Figura `results/t3a_convergence.png`.

## Paralelo (C++ MPI + OpenMP, matrix-free many-body)
`qtraj_mpi.cpp` aplica $H_{\text{eff}}$ y los saltos *matrix-free* sobre el vector
de estado (coste $O(N\,d)$, sin almacenar matrices), llegando a N grande.
```bash
cd cpp && mkdir -p build && cd build && cmake .. -DCMAKE_CXX_COMPILER=g++-15 && make
cd ..
OMP_NUM_THREADS=4 mpirun --oversubscribe -np 2 ./build/qtraj_mpi configs/ising_traj.ini
```

**Validación** (N=3, vs ecuación maestra densa, `python/cross_validate_t3a.py`):
error máx $8\times10^{-3}$, dentro del error estadístico $M^{-1/2}$ (M=20000).

**Paralelización:** cada trayectoria es independiente → *embarrassingly parallel*.
MPI reparte las $M$ trayectorias entre rangos (`MPI_Reduce` final), OpenMP las
reparte entre hilos dentro del rango. **Escalado casi ideal** (`run_bench.sh`, N=10, d=1024):

| workers | OpenMP | MPI |
|---|---|---|
| 2 | 1.95× | 1.96× |
| 4 | 3.72× | 3.73× |
| 8 | 4.77× | 4.64× |

Contraste con T1: aquí no hay acoplamiento secuencial → escala mucho mejor que el
SpMV distribuido de Arnoldi. Figuras `results/t3a_scaling.png`, `results/t3a_curves.png`.
