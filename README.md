# Mpemba-X v2

**Multi-framework Mpemba-effect simulation suite** — extended with the most
recent literature (through 2026), now driven by experimental configuration
files and capable of visualizing the temperature field of a water mass.

CPU-only parallel implementation (OpenMP + MPI; no CUDA). Built and tested
with OpenMPI 4.1 and GCC 13 on Ubuntu 24.

> **Marco cuántico (QMpE):** el efecto Mpemba cuántico tiene su propio marco
> teórico-numérico, con informe a nivel de artículo y código paralelo, en la
> carpeta [`mpemba-cuantico/`](mpemba-cuantico/) (ver su `README.md`).

---

## What's new in v2 (vs v1)

1. **INI configuration files**: every module is now driven by a config file
   in `configs/`. No more long command-line flags. You set the mass, geometry,
   initial temperatures, bath temperature etc. in plain text and pass the
   file as the only argument.
2. **Water mass and temperature visualization**: the new `mpemba_water_fourier`
   module solves the macroscopic heat equation on a 1D water column, with
   skin supersolidity and hydrogen-bond memory (Zhang et al. 2014), and
   exports the full $T(x, t)$ field as a heatmap-ready CSV. The Python
   visualizer renders a heatmap + cooling-curve comparison + Mpemba crossover
   detection.
3. **Five new physics modules** based on the updated bibliography:
   * `water_fourier` — Zhang et al., *PCCP* **16**, 22995 (2014)
   * `water_md` — Jin & Goddard, *JPCC* **119**, 2622 (2015); Naserifar &
     Goddard, *PNAS* **116**, 1998 (2019)
   * `langevin_inverse` — Kumar, Chétrite & Bechhoefer, *PNAS* **119**,
     e2118484119 (2022) [inverse Mpemba / anomalous heating]
   * `quantum_lindblad` — Carollo, Lasanta & Lesanovsky, *PRL* **127**,
     060401 (2021); Chattopadhyay, Santos & Misra, arXiv 2601.05046 (2026)
     [Davies qubit + QFI thermometric Mpemba]
   * `thermomajorization` — Vu & Hayakawa, *PRL* **134**, 107101 (2025)
     [universal Mpemba diagnostic under ALL monotone metrics simultaneously]

---

## Summary of all 9 frameworks

| # | Module                       | Paper                                            | What it tracks                       | ME observable?       |
|---|------------------------------|--------------------------------------------------|--------------------------------------|----------------------|
| 1 | `markovian`                  | Lu & Raz, PNAS 2017                              | $D_e(t)$, $D_{L_1}$, $D_{KL}$        | ✓ at $t \approx 22.4$ |
| 2 | `klich_raz`                  | Klich, Raz et al. PRX 2019                       | $a_2(T)$, Mpemba index histogram     | ✓ sign change of $a_2$ |
| 3 | `granular_analytic`          | Lasanta et al. PRL 2017                          | $T(t),\ a_2(t)$                      | ✓ at $t^* \approx 0.18$ |
| 4 | `langevin`                   | Kumar & Bechhoefer, Nature 2020                  | $D_{L_1}(t)$ for 3 quenches          | ✓ at $t^* \approx 0.033$ |
| 5 | `langevin_inverse`           | Kumar, Chétrite & Bechhoefer PNAS 2022           | heating curves for 2 cold inits      | conditional; effect is weak |
| 6 | `water_fourier`              | Zhang et al., PCCP 2014                          | $T(x, t)$ field, mass, enthalpy      | depends on $C_{\rm mem}$ |
| 7 | `water_md`                   | Jin & Goddard, JPCC 2015                         | coordination, $T(t)$ of mW particles | structural diagnostic |
| 8 | `quantum_lindblad`           | Carollo PRL 2021; Chattopadhyay arXiv 2026       | $p_1(t)$, QFI, trace distance        | ✓ metrological gain    |
| 9 | `thermomajorization`         | Vu & Hayakawa, PRL 2025                          | Lorenz curves, dominance, crossings  | **universal** ✓        |

---

## Configuration files (`configs/`)

All physical parameters live here. Edit one file, run one binary, get one
result you can compare to your experiment.

```
configs/
├── markov_3state.ini             # Lu-Raz Markovian 3-state
├── markov_ising.ini              # Lu-Raz Ising chain
├── klich_raz_rem.ini             # REM random barriers
├── granular.ini                  # Lasanta moment ODE
├── colloid_kumar.ini             # Forward Mpemba (Nature 2020)
├── colloid_inverse.ini           # Inverse Mpemba (PNAS 2022)
├── water_zhang_default.ini       # Default Zhang Fourier water
├── water_demo.ini                # Demo with visible crossover
├── water_lab_100g.ini            # EDITABLE TEMPLATE for your experiment
├── water_md_jin_goddard.ini      # Coarse-grained MD water
└── quantum_qubit.ini             # Lindblad qubit + QFI
```

### How to edit `water_lab_100g.ini` for YOUR experimental conditions

```ini
[water]
mass_g            = 100.0          ; ← YOUR measured mass in grams
tube_length_m     = 0.08           ; ← effective column height (m)
skin_thickness_m  = 0.003          ; skin layer (~3 mm typical)

[preparation_hot]
T_initial_C = 80.0                 ; ← YOUR hot sample initial T

[preparation_cold]
T_initial_C = 25.0                 ; ← YOUR cold sample initial T

[bath]
T_bath_C       = -18.0             ; ← YOUR freezer temperature
h_drain_W_m2_K = 25.0              ; tunable to match your cooling rate
```

Then run:
```
OMP_NUM_THREADS=4 mpirun --allow-run-as-root --oversubscribe -np 2 \
    ./build/mpemba_water_fourier configs/water_lab_100g.ini
python3 analysis/plot_water_fourier.py results/water_lab_100g
```

---

## How to build

```
mkdir -p build && cd build
cmake .. -DMPI_CXX_COMPILER=/usr/bin/mpicxx.openmpi \
         -DMPI_C_COMPILER=/usr/bin/mpicc.openmpi
make -j4
```

This produces 10 binaries:
- `mpemba_markovian`, `mpemba_klich_raz`, `mpemba_granular_analytic`,
  `mpemba_granular_dsmc`, `mpemba_langevin`, `mpemba_langevin_inverse`,
  `mpemba_water_fourier`, `mpemba_water_md`, `mpemba_quantum_lindblad`,
  `mpemba_thermomajorization`.

Dependencies: GCC 13+ (C++17), OpenMPI 4+, CMake 3.16+, Python 3 with
`numpy` and `matplotlib`.

---

## How to run (after build)

Each binary takes ONE argument: the path to its config file.

```
# Module 1: Markovian
OMP_NUM_THREADS=2 mpirun --allow-run-as-root --oversubscribe -np 4 \
    ./build/mpemba_markovian configs/markov_3state.ini

# Module 2: Klich-Raz
OMP_NUM_THREADS=2 mpirun --allow-run-as-root --oversubscribe -np 4 \
    ./build/mpemba_klich_raz configs/klich_raz_rem.ini

# Module 3: Granular analytic
./build/mpemba_granular_analytic configs/granular.ini

# Module 4: Langevin colloid
OMP_NUM_THREADS=4 mpirun --allow-run-as-root --oversubscribe -np 2 \
    ./build/mpemba_langevin configs/colloid_kumar.ini

# Module 5: Inverse Mpemba (anomalous heating)
OMP_NUM_THREADS=4 mpirun --allow-run-as-root --oversubscribe -np 2 \
    ./build/mpemba_langevin_inverse configs/colloid_inverse.ini

# Module 6: Water Fourier (Zhang) -- WITH HEATMAP VISUALIZATION
OMP_NUM_THREADS=4 mpirun --allow-run-as-root --oversubscribe -np 2 \
    ./build/mpemba_water_fourier configs/water_demo.ini
python3 analysis/plot_water_fourier.py results/water_demo

# Module 7: Water MD (Jin-Goddard mW)
OMP_NUM_THREADS=4 mpirun --allow-run-as-root --oversubscribe -np 2 \
    ./build/mpemba_water_md configs/water_md_jin_goddard.ini

# Module 8: Quantum Lindblad + QFI
OMP_NUM_THREADS=2 mpirun --allow-run-as-root --oversubscribe -np 4 \
    ./build/mpemba_quantum_lindblad configs/quantum_qubit.ini

# Plots
python3 analysis/plot_markovian.py
python3 analysis/plot_klich_raz.py
python3 analysis/plot_granular_analytic.py
python3 analysis/plot_langevin.py
python3 analysis/plot_quantum.py
python3 analysis/plot_inverse.py
python3 analysis/plot_summary.py    # combined 3x3 figure
```

---

## What the water-Fourier module gives you (visualization)

The flagship new feature. Running `mpemba_water_fourier configs/water_demo.ini`
produces:

* `timeseries_hot.csv`, `timeseries_cold.csv`: averaged quantities vs time
  (mean T, skin T, bulk T, enthalpy, mass, RMS distance to bath)
* `field_hot.csv`, `field_cold.csv`: full 2D arrays $T(x, t)$, one row per
  saved time step
* Running `analysis/plot_water_fourier.py results/water_demo` produces a
  9-panel figure with:
  - heatmap of $T(x, t)$ for both samples
  - profile snapshots at selected times
  - mean T vs t, mass conservation, total enthalpy
  - log-scale cooling curve with **automatic Mpemba crossover detection**
  - Burridge-Linden $\Delta E_H / \Delta E_C$ ratio

This is what you can compare directly to your lab thermograms.

---

## Theoretical foundation: thermomajorization (Vu-Hayakawa 2025)

The crown jewel of the v2 update is the universal diagnostic in `thermomajorization`.

> **Theorem (Vu & Hayakawa, PRL 134, 107101 (2025))**
> Let $p_h, p_w$ be two distributions over states with energies $\{E_i\}$,
> and let $\gamma = e^{-\beta E}/Z$ be the bath Gibbs distribution at $T_b$.
> Then $p_h$ "thermomajorizes" $p_w$ (denoted $p_h \succ_{T_b} p_w$) iff
> $D_f(p_h \| \gamma) \ge D_f(p_w \| \gamma)$ for **every** Gibbs-contractive
> $f$-divergence simultaneously (KL, total variation, $\alpha$-Rényi, …).

This is implemented in `core/thermomajorization.hpp` and used both by the
`markovian` module (automatic diagnostic each time step) and as a standalone
post-processor (`mpemba_thermomajorization`).

A change of sign of the min/max gap of the Lorenz curve = a robust certificate
of the Mpemba effect, valid under any distance you might choose.

---

## Performance reference (test machine: 4-core Intel)

| Module               | Config                              | Wall time |
|----------------------|-------------------------------------|-----------|
| markovian            | 3-state, 8 T_inits, 4 ranks         | 8 s       |
| klich_raz            | L=12, 8000 realizations             | 0.2 s     |
| granular_analytic    | serial                              | <0.1 s    |
| langevin             | 200k trajectories, 0.5 s sim time   | 140 s     |
| langevin_inverse     | 200k trajectories, 1.0 s sim time   | 92 s      |
| water_fourier        | 100 g, 40 min, N=60                 | 0.07 s    |
| water_md             | N=128, 2500 steps                   | 1 s       |
| quantum_lindblad     | 5 T_inits, 6000 steps               | 0.004 s   |

---

## Limitations and honest caveats

1. **Water Fourier** (Zhang 2014 model) is a phenomenological model. The
   parameter `C_mem` controlling the H-bond memory must be tuned to match
   real experiments. For physical magnitudes (~ 1e4 J/m³) the memory term is
   small compared to bulk conduction and does not by itself produce a Mpemba
   crossover. Larger values (~ 1e7 J/m³) do produce a crossover but represent
   an effective coupling. The Burridge-Linden 2016 critique applies: do not
   over-interpret crossovers without comparison to experiment.

2. **Water MD** is coarse-grained (Stillinger-Weber two-body only; three-body
   omitted for performance). It captures relaxation of coordination number
   but is not quantitative for ice nucleation kinetics. For production
   simulation use LAMMPS with full mW.

3. **Inverse Mpemba** is generically weaker than the forward effect
   (Kumar-Chetrite 2022 emphasize this in their abstract). The configuration
   we ship may not show a crossover; user should tune the tilt and potential
   parameters.

4. **Spin glass module (Baity-Jesi 2019)** still pending — requires Janus II-
   style multispin coding. The Klich-Raz REM serves as a related proxy.

5. **DSMC granular** (kept from v1) does not show the crossover at the
   parameters used. The analytic moment-ODE version (Lasanta) does and is
   what should be used to demonstrate the effect.

---

## File layout

```
mpemba-x-v2/
├── CMakeLists.txt
├── core/                       # header-only utilities
│   ├── config_parser.hpp        # NEW: INI parser
│   ├── thermomajorization.hpp   # NEW: Vu-Hayakawa universal diagnostic
│   ├── distance_metrics.hpp
│   ├── jacobi_eigen.hpp
│   ├── philox_rng.hpp
│   └── csv_io.hpp
├── configs/                    # all experimental configurations
├── modules/                    # 9 physics modules
│   ├── markovian/
│   ├── klich_raz/
│   ├── granular/               # main_analytic.cpp + main_dsmc.cpp
│   ├── langevin/
│   ├── langevin_inverse/       # NEW
│   ├── water_fourier/          # NEW
│   ├── water_md/               # NEW
│   ├── quantum_lindblad/       # NEW
│   └── thermomajorization/     # NEW
├── analysis/                   # Python plot scripts
├── build/                      # compiled binaries
└── results/                    # generated data and figures
```

---

## References (papers in your project knowledge)

1. Lu & Raz, *PNAS* **114**, 5083 (2017) [`lu2017.pdf`, `luraz2017nonequilibriumthermodynamicsofthemarkovianmpembaeffectanditsinverse.pdf`]
2. Klich, Raz, Hirschberg & Vucelja, *PRX* **9**, 021060 (2019) [`PhysRevX_9_021060.pdf`]
3. Lasanta, Vega Reyes, Prados & Santos, *PRL* **119**, 148001 (2017) [`lasanta2017.pdf`]
4. Biswas, Prasad, Raz & Rajesh, *PRE* **102**, 012906 (2020) [`biswas2020.pdf`]
5. Kumar & Bechhoefer, *Nature* **584**, 64 (2020) [`kumar2020.pdf`]
6. Kumar, Chétrite & Bechhoefer, *PNAS* **119**, e2118484119 (2022) [`kumaretal2022anomalousheatinginacolloidalsystem.pdf`]
7. Baity-Jesi et al., *PNAS* **116**, 15350 (2019) [`baityjesi2019.pdf`, `baityjesietal2019thempembaeffectinspinglassesisapersistentmemoryeffect.pdf`]
8. Carollo, Lasanta & Lesanovsky, *PRL* **127**, 060401 (2021) [`carollo2021.pdf`]
9. Zhang et al., *PCCP* **16**, 22995 (2014) [`c4cp03669g.pdf`]
10. Jin & Goddard, *JPCC* **119**, 2622 (2015) [`mechanismsunderlyingthempembaeffectinwaterfrommoleculardynamicssimulations.pdf`]
11. Naserifar & Goddard, *PNAS* **116**, 1998 (2019) [`naserifargoddard2019liquidwaterisadynamicpolydispersebranchedpolymer.pdf`]
12. Vu & Hayakawa, *PRL* **134**, 107101 (2025) [`2502_00123v3.pdf`]
13. Chattopadhyay, Santos & Misra, arXiv:2601.05046 (2026) [`2601_05046v1.pdf`]
14. Li & Yang, arXiv:2604.14740 (2026) [`2604_14740v2.pdf`]
15. Burridge & Linden, *Sci. Rep.* **6**, 37665 (2016) [`srep37665.pdf`]
16. Hallstadius & Burridge, *Proc. R. Soc. A* **476**, 20190829 (2020) [`rspa_2019_0829.pdf`]
17. (review compiled from these sources) [`Búsqueda_de_Papers_Efecto_Mpemba.pdf`]
