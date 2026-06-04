#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qtraj.py  --  Tarea T3a: trayectorias cuanticas (Monte Carlo wavefunction).

Metodo de saltos cuanticos (Dalibard-Castin-Molmer; ver Daley, Adv. Phys. 63,
77 (2014), arXiv:1404.5028). En vez de propagar el operador densidad rho
(dimension d^2), se propagan M vectores de estado |psi> (dimension d) y se
promedia rho = E[|psi><psi|]. Esto reduce la memoria de d^2 a d y es
TRIVIALMENTE PARALELIZABLE: cada trayectoria es independiente.

Algoritmo (por trayectoria, paso dt):
  H_eff = H - (i/2) sum_mu L_mu^dag L_mu                (no hermitiano)
  |psi'> = |psi> - i dt H_eff |psi>                     (propagacion determinista)
  dp = 1 - <psi'|psi'>                                  (prob. total de salto)
  si r < dp (r ~ U[0,1]):  SALTO
      elegir canal mu con prob ~ <psi|L_mu^dag L_mu|psi>
      |psi> = L_mu|psi> / || L_mu|psi> ||
  si no:
      |psi> = |psi'> / || |psi'> ||
El error estadistico del promedio decae como M^{-1/2}.

Este modulo ofrece DOS implementaciones de paralelizacion:
  * serial          (un proceso)            -> referencia
  * multiprocessing (varios procesos)       -> memoria compartida en un nodo
La version C++ (../cpp) reparte las M trayectorias entre rangos MPI.
"""

from __future__ import annotations
import os
import sys
import numpy as np
from multiprocessing import Pool, cpu_count

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def heff(H, Ls):
    """Hamiltoniano efectivo no-hermitiano H_eff = H - (i/2) sum L^dag L."""
    He = H.astype(complex).copy()
    for Lk in Ls:
        He = He - 0.5j * (Lk.conj().T @ Lk)
    return He


def one_trajectory(H, Ls, psi0, t_max, dt, log_idx, seed, accumulate_rho=True):
    """Una trayectoria estocastica.

    Si accumulate_rho: devuelve rho(t) muestreado = |psi><psi| en los pasos
    log_idx (array (len(log_idx), d, d)). Si no, devuelve solo las poblaciones
    diagonales |psi_i|^2 (array (len(log_idx), d)) -- mucho mas barato en memoria
    y suficiente para diagnosticos diagonales.
    """
    rng = np.random.default_rng(seed)
    He = heff(H, Ls)
    psi = psi0.astype(complex).copy()
    d = psi.shape[0]
    n_steps = int(np.ceil(t_max / dt))
    log_set = {idx: j for j, idx in enumerate(log_idx)}
    if accumulate_rho:
        out = np.zeros((len(log_idx), d, d), dtype=complex)
    else:
        out = np.zeros((len(log_idx), d), dtype=float)

    for step in range(n_steps + 1):
        if step in log_set:
            j = log_set[step]
            if accumulate_rho:
                out[j] = np.outer(psi, psi.conj())
            else:
                out[j] = np.abs(psi) ** 2
        if step == n_steps:
            break
        psi_det = psi - 1j * dt * (He @ psi)
        norm2 = np.real(np.vdot(psi_det, psi_det))
        if rng.random() > norm2:                       # SALTO
            weights = np.array([np.real(np.vdot(psi, Lk.conj().T @ (Lk @ psi)))
                                for Lk in Ls])
            weights = np.clip(weights, 0, None)
            mu = rng.choice(len(Ls), p=weights / weights.sum())
            phi = Ls[mu] @ psi
            psi = phi / np.linalg.norm(phi)
        else:
            psi = psi_det / np.sqrt(norm2)
    return out


# --- worker para multiprocessing (argumentos empacados) ---
def _worker(args):
    H, Ls, psi0, t_max, dt, log_idx, seed, acc = args
    return one_trajectory(H, Ls, psi0, t_max, dt, log_idx, seed, acc)


def evolve_serial(H, Ls, psi0, t_max, dt, M=2000, log_every=20,
                  base_seed=12345, accumulate_rho=True):
    """Promedia M trayectorias en UN solo proceso (referencia serial)."""
    n_steps = int(np.ceil(t_max / dt))
    log_idx = list(range(0, n_steps + 1, log_every))
    times = np.array([i * dt for i in log_idx])
    d = psi0.shape[0]
    shape = (len(log_idx), d, d) if accumulate_rho else (len(log_idx), d)
    acc = np.zeros(shape, dtype=complex if accumulate_rho else float)
    for m in range(M):
        acc += one_trajectory(H, Ls, psi0, t_max, dt, log_idx,
                              base_seed + m, accumulate_rho)
    acc /= M
    return times, acc


def evolve_parallel(H, Ls, psi0, t_max, dt, M=2000, log_every=20,
                    n_workers=None, base_seed=12345, accumulate_rho=True):
    """Promedia M trayectorias con multiprocessing (memoria compartida en un nodo).

    Cada worker simula un subconjunto de trayectorias; el promedio es la
    reduccion final. Equivale al patron MPI de la version C++ pero con procesos
    locales.
    """
    n_steps = int(np.ceil(t_max / dt))
    log_idx = list(range(0, n_steps + 1, log_every))
    times = np.array([i * dt for i in log_idx])
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    d = psi0.shape[0]
    shape = (len(log_idx), d, d) if accumulate_rho else (len(log_idx), d)
    acc = np.zeros(shape, dtype=complex if accumulate_rho else float)
    tasks = [(H, Ls, psi0, t_max, dt, log_idx, base_seed + m, accumulate_rho)
             for m in range(M)]
    with Pool(n_workers) as pool:
        for traj in pool.imap_unordered(_worker, tasks, chunksize=16):
            acc += traj
    acc /= M
    return times, acc
