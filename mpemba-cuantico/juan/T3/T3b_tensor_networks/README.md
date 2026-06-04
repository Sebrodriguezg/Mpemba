# T3b — Redes tensoriales (TEBD disipativo)

Régimen de muchos cuerpos con **redes tensoriales**. La matriz densidad se
escribe como un *superket* $|\rho\rangle\rangle$ —un MPS con dimensión física 4
por sitio (índice doblado fila/columna de cada qubit)— y la ecuación maestra
$\partial_t|\rho\rangle\rangle=\mathcal{L}|\rho\rangle\rangle$ se integra con
**TEBD**: se trotteriza $e^{dt\mathcal{L}}$ en compuertas locales de 2 sitios y
se trunca el rango de enlace a $\chi$ por SVD. Coste polinómico en $\chi$.
Refs.: Zwolak & Vidal, *PRL* **93**, 207205 (2004); Weimer *et al.*, *RMP* **93**,
015008 (2021), arXiv:1907.07079.

## Serial (Python)
`mpdo_tebd.py` construye los generadores de bond reutilizando
`common.qmpe.build_liouvillian` (con permutación de índices al orden site-local),
exponencia las compuertas y evoluciona el MPS con Trotter de 2º orden (Strang).

```bash
cd python
python3 validate_t3b.py     # TEBD vs ecuacion maestra exacta + convergencia en chi
```
**Validación** (N=4 vs exacto): el error cae con $\chi$ —
$6\times10^{-2}\ (\chi{=}1)\to3\times10^{-3}\ (\chi{=}4)\to6\times10^{-7}\ (\chi{=}16)$.
Figura `results/t3b_validation.png`.

## Paralelo (Python, threads)
`tebd_parallel.py`: dentro de cada capa de Trotter los bonds **pares** (o
**impares**) son disjuntos → sus compuertas (cada una una SVD $O(\chi^3)$) se
aplican en paralelo con `ThreadPoolExecutor`; numpy libera el GIL en la SVD, así
que escalan sin coste de serialización.

```bash
cd python
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 bench_t3b.py
```
(BLAS a 1 hilo para aislar el paralelismo de compuertas.)

**Escalado** (N=20, χ=48): 1.84× (2 hilos), 2.72× (4), 3.61× (8). Es el patrón de
paralelismo *estructurado* de las redes tensoriales — intermedio entre el SpMV
acoplado de T1 y las trayectorias independientes de T3a.

**Muchos cuerpos** (`bench_t3b.py`): N=32 ($4^{32}\approx1.8\times10^{19}$,
imposible en denso) se evoluciona con χ=24 en ~3.6 s. Figuras
`results/t3b_scaling.png`, `results/t3b_largeN.png`.

**Mapeo a MPI** (no implementado): repartir segmentos contiguos de la cadena
entre rangos con intercambio de los tensores de frontera por capa.
