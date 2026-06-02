# Marco cuántico del efecto Mpemba (QMpE)

Marco completo del **efecto Mpemba cuántico**: informe teórico-numérico a nivel de
artículo más su capa computacional ejecutable y verificada, con las
**costuras de paralelización** preparadas para escalar a HPC.

> Objetivo: poder **estudiar** el fenómeno en pocos cuerpos (Python) y tener un
> **núcleo HPC** (C++ MPI+OpenMP) listo para extender a muchos cuerpos.

---

## Estructura

```
mpemba-cuantico/
├── doc/                    # informe (nivel artículo) + bibliografía
│   ├── mpemba_cuantico.tex # teoría, espectral, numérico HPC + implementación de referencia
│   ├── referencias.bib     # 48 referencias
│   └── mpemba_cuantico.pdf
├── python/                 # implementación de referencia (estudio + validación)
│   ├── qmpe.py             # NÚCLEO: vectorización, espectro, evolución, distancias, QFI
│   ├── models.py           # modelos resolubles: TLS, Λ-3 niveles, Ising disipativo
│   ├── run_demo.py         # validación + diagnóstico + efecto Mpemba fuerte (figuras/CSV)
│   ├── trajectories.py     # trayectorias cuánticas (T3a) en paralelo (multiprocessing)
│   ├── plot_hpc.py         # analiza la salida del núcleo C++ y detecta cruces
│   └── requirements.txt
├── cpp/                    # núcleo HPC (parallelization-ready)
│   ├── qmpe_hpc.cpp        # SpMV matrix-free + RK4 + barrido MPI + OpenMP
│   ├── CMakeLists.txt
│   └── config.ini
├── figures/                # figuras de validación (usadas por el informe)
└── README.md
```

Para compilar el informe: `cd doc && pdflatex mpemba_cuantico && bibtex mpemba_cuantico && pdflatex mpemba_cuantico && pdflatex mpemba_cuantico`.

---

## Mapa código ↔ informe

| Sección del `.tex` | Concepto | Implementación |
|---|---|---|
| §2 GKSL | $\mathcal L[\rho]=-i[H,\rho]+\sum_\mu\mathcal D[L_\mu]\rho$ | `qmpe.apply_liouvillian`, `cpp:apply_lindblad` |
| §3 Espectro | $\lambda_k$, $\hat r_k$, $\hat l_k$, biortonormalidad | `qmpe.Spectrum` |
| §3 ec. (spectral) | $\rho_t=\rho_{ss}+\sum_k e^{\lambda_k t}a_k\hat r_k$ | `qmpe.evolve_spectral` |
| §4 distancias | $D_{tr}, D_{HS}, D_{KL}, S(\rho\|\rho_{ss})$ | `qmpe.trace_distance`, … ; `cpp:d_hs` |
| §4 criterio fuerte | $a_2=\operatorname{Tr}(\hat l_2^\dagger\rho_0)=0$ | `qmpe.Spectrum.overlaps`, `run_demo.strong_mpemba` |
| §5 QFI | $F_T=\sum_i(\partial_T p_i)^2/p_i$ | `qmpe.qfi_temperature` |
| §6 modelos | TLS, Λ, Ising | `models.py`, `cpp:build_ising` |
| §7.1 vectorización | $\mathbb L=\dots$ (Kronecker) | `qmpe.build_liouvillian` |
| §7.2 matrix-free | SpMV sin materializar $\mathbb L$ | `qmpe.apply_liouvillian`, `cpp:apply_lindblad` |
| §7.3 T2 evolución | RK4 / acción del exponencial | `qmpe.evolve_rk4`, `cpp:rk4_step` |
| §7.3 T3a trayectorias | Monte Carlo wavefunction | `trajectories.py` |
| §8 paralelización | MPI (preparaciones) + OpenMP (SpMV) | `cpp:qmpe_hpc.cpp` |
| §8 Algoritmo 1 | diagnóstico espectral + curvas | `run_demo.py` |

---

## Cómo estudiarlo (Python)

```bash
source ../../.venv_mpemba/bin/activate      # numpy + matplotlib
cd python
python run_demo.py        # (1) valida TLS vs analítico  (err ~1e-15)
                          # (2) diagnóstico espectral (autovalores lentos, a_k)
                          # (3) efecto Mpemba fuerte (cruce de distancias)
                          # (4) escalado d^2 del Liouvilliano
python trajectories.py    # valida trayectorias cuánticas (T3a) vs ec. maestra
```
Salida en `python/results/` (figuras PNG + `mpemba_curves.csv`).

**Resultados de validación esperados:**
- TLS: `max|p_num − p_ana| ≈ 7e-15` (RK4 matrix-free exacto).
- Trayectorias: error estadístico `≈ M^-1/2`.
- Mpemba fuerte: cruce detectado `t* ≈ 0.36` (Λ-3 niveles).

---

## Cómo correr el núcleo HPC (C++)

```bash
cd cpp && mkdir -p build && cd build
cmake .. -DMPI_CXX_COMPILER=/usr/bin/mpicxx.openmpi -DCMAKE_CXX_COMPILER=/usr/bin/mpicxx.openmpi
make
OMP_NUM_THREADS=4 mpirun.openmpi --oversubscribe -np 2 ./qmpe_hpc ../config.ini
python ../../python/plot_hpc.py results     # grafica y detecta cruces
```

Edita `cpp/config.ini` para cambiar `N` (espines), el baño y la lista de
preparaciones `T0_list`. Recuerda: $d=2^N$, $\dim(\mathbb L)=4^N$.

---

## Costuras de paralelización (qué falta para escalar)

El núcleo C++ ya implementa **dos** de los tres niveles del informe (§8.1):

1. **OpenMP intranodo** sobre el SpMV / producto de matrices — `// [PARALELIZAR]` en `matmul`.
2. **MPI internodo** sobre el barrido de preparaciones — reparto de `T0_list` entre rangos.

Marcadas como `seam` para la siguiente fase (ver comentarios en `qmpe_hpc.cpp`):

3. **Matrix-free disperso**: sustituir las matrices densas por almacenamiento
   disperso (CSR) de $H$ y $\{L_\mu\}$ y un SpMV que no materialice $\mathbb L$.
   Habilita $N\gtrsim 8$.
4. **Trayectorias cuánticas (T3a)**: portar `trajectories.py` a C++/MPI repartiendo
   las $M$ trayectorias entre rangos (escala casi perfecta, memoria $d$ en vez de $d^2$).
5. **Modos lentos (T1) con SLEPc/PETSc**: problema de autovalores interior
   (`shift-invert` en $\sigma=0$) para obtener $\lambda_2,\lambda_3,\hat l_2$ sin
   diagonalización densa. Sustituye el espectro denso de `qmpe.Spectrum`.
6. **GPU**: el SpMV y las contracciones tensoriales (cuSPARSE / redes tensoriales)
   para el régimen de muchos cuerpos (T3b).

---

## Notas de validación física

- El núcleo HPC usa $D_{HS}$ (no requiere diagonalizar) como diagnóstico; la
  distancia de traza $D_{tr}$ requiere autovalores (LAPACK) y está disponible en
  la versión Python.
- La interpretación física del cruce depende de comparar $T_0$ con $T_{\text{bath}}$:
  $T_0>T_{\text{bath}}$ ⇒ enfriamiento (Mpemba directo); $T_0<T_{\text{bath}}$ ⇒
  calentamiento (Mpemba inverso, §5.2 del informe).
- Contrastar siempre contra los modelos resolubles antes de escalar.
