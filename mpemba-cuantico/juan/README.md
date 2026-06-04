# Efecto Mpemba cuántico en HPC — Tareas T1 y T3

Implementación **serial y paralela** (MPI + OpenMP / hilos) de las tareas
computacionales del estudio numérico del efecto Mpemba cuántico (informe base
`mpemba_cuantico.pdf`, §7.3), con validación contra soluciones exactas y
benchmarks de escalado. Trabajo para **Sistemas Distribuidos 2026**.

## Estructura

```
mpemba-cuantico-tareas/
├── doc/                    Documento LaTeX (5 capitulos) -> doc/main.pdf
│   ├── main.tex
│   ├── chapters/           01 intro · 02 T1 · 03 T3a · 04 T3b · 05 conclusiones
│   └── figures/            figuras generadas por el codigo
├── common/                 ORACULO validado (qmpe.py, models.py) reutilizado
├── T1/                     Modos lentos del Liouvilliano (Arnoldi-Lindblad)
│   ├── python/   serial (numpy) + validacion vs denso
│   ├── cpp/      paralelo MPI+OpenMP (SpMV distribuido) + benchmark
│   ├── configs/  · results/
└── T3/
    ├── T3a_trajectories/   Trayectorias cuanticas (saltos), memoria 2^N
    │   ├── python/ serial + multiprocessing + validacion
    │   ├── cpp/    paralelo MPI+OpenMP (matrix-free many-body)
    │   ├── configs/ · results/
    └── T3b_tensor_networks/  TEBD disipativo (superket MPS), coste poly(chi)
        ├── python/ serial + paralelo por hilos (bonds disjuntos)
        ├── configs/ · results/
```

## Resumen de resultados (todos de ejecuciones reales)

| Tarea | Validación vs exacto | Escalado (4 workers) | Alcance |
|---|---|---|---|
| **T1** Arnoldi-Lindblad | error $\lambda$ $10^{-13}$–$10^{-5}$; *faster-than-clock* 8.8× | OpenMP 2.9× / MPI 2.7× | $N\le7$ (denso) |
| **T3a** trayectorias | convergencia $M^{-1/2}$ (pend. −0.488) | OpenMP 3.7× / MPI 3.7× (≈ideal) | $N=10$, memoria $2^N$ |
| **T3b** redes tensoriales | error → $6\times10^{-7}$ con $\chi$ | hilos 2.7× | $N=32$ ($4^{32}\approx10^{19}$) |

**Lección central:** la estructura del acoplamiento del algoritmo determina la
escalabilidad — T1 (SpMV acoplado) < T3b (estructurado) < T3a (embarrassingly parallel).

## Requisitos

- Python 3.10+ con `numpy`, `matplotlib`
- C++17, OpenMPI, OpenMP (g++-15), CMake 3.16+; LAPACK (Accelerate en macOS) para T1
- LaTeX (pdflatex/latexmk) para el documento

## Reproducir todo

```bash
# T1
(cd T1/python && python3 validate_t1.py)
(cd T1/cpp && mkdir -p build && cd build && cmake .. -DCMAKE_CXX_COMPILER=g++-15 && make && cd .. && ./run_bench.sh && python3 plot_bench.py)
# T3a
(cd T3/T3a_trajectories/python && python3 validate_t3a.py)
(cd T3/T3a_trajectories/cpp && mkdir -p build && cd build && cmake .. -DCMAKE_CXX_COMPILER=g++-15 && make && cd .. && ./run_bench.sh && python3 plot_t3a.py)
# T3b
(cd T3/T3b_tensor_networks/python && python3 validate_t3b.py && VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 bench_t3b.py)
# Documento
(cd doc && latexmk -pdf main.tex)
```

Cada subcarpeta tiene su propio `README.md` con detalles del método y los resultados.
