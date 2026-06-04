#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tebd_parallel.py  --  Tarea T3b PARALELA: TEBD disipativo con paralelismo de
compuertas dentro de cada capa de Trotter.

Idea de paralelizacion: en una capa de Trotter, las compuertas sobre bonds
DISJUNTOS no comparten tensores y pueden aplicarse simultaneamente:
  * capa par   : bonds (0,1), (2,3), (4,5), ...   -> independientes
  * capa impar : bonds (1,2), (3,4), (5,6), ...   -> independientes
Cada compuerta hace una SVD de coste O(chi^3); numpy libera el GIL durante la
SVD/BLAS, de modo que se paralelizan con HILOS (ThreadPoolExecutor) sin coste de
serializacion (a diferencia de multiprocessing). Es el patron de paralelismo
"estructurado" de las redes tensoriales, distinto del SpMV acoplado (T1) y de las
trayectorias independientes (T3a).

Para muchos nodos, este esquema se mapea a MPI repartiendo segmentos contiguos
de la cadena entre rangos, con comunicacion de frontera entre segmentos (no
implementado aqui: requiere intercambio de los tensores de borde por capa).
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mpdo_tebd as tn  # noqa: E402


def _apply_layer(mps, bonds_idx, gates, chi, executor):
    """Aplica en paralelo las compuertas de un conjunto de bonds disjuntos."""
    if executor is None:
        for k in bonds_idx:
            mps.apply_gate(k, gates[k], chi)
    else:
        list(executor.map(lambda k: mps.apply_gate(k, gates[k], chi), bonds_idx))


def evolve_tebd_parallel(N, J, h, gamma, T, p0, t_max, dt, chi,
                         log_every=20, n_threads=1,
                         observable=tn.excitation_density):
    """Igual que mpdo_tebd.evolve_tebd pero con las compuertas de cada capa
    aplicadas en paralelo (n_threads hilos). Devuelve (times, vals, chi_max, wall)."""
    bonds = tn.ising_bond_generators(N, J, h, gamma, T)
    G_half = [tn.expm_gate(Lb, dt / 2) for Lb in bonds]
    G_full = [tn.expm_gate(Lb, dt) for Lb in bonds]

    a, b = np.sqrt(1 - p0), np.sqrt(p0)
    rho1 = np.outer([a, b], np.conj([a, b]))
    mps = tn.SuperketMPS.product([rho1.copy() for _ in range(N)])

    even = list(range(0, N - 1, 2))
    odd = list(range(1, N - 1, 2))

    executor = ThreadPoolExecutor(max_workers=n_threads) if n_threads > 1 else None
    n_steps = int(np.ceil(t_max / dt))
    times, vals = [], []
    chi_max = 1
    t0 = time.time()
    for step in range(n_steps + 1):
        if step % log_every == 0:
            times.append(step * dt)
            vals.append(observable(mps))
        if step == n_steps:
            break
        _apply_layer(mps, even, G_half, chi, executor)
        _apply_layer(mps, odd, G_full, chi, executor)
        _apply_layer(mps, even, G_half, chi, executor)
        chi_max = max(chi_max, mps.max_bond())
    wall = time.time() - t0
    if executor is not None:
        executor.shutdown()
    return np.array(times), np.array(vals), chi_max, wall
