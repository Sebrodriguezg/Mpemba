#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trajectories.py  --  Trayectorias cuanticas (Monte Carlo wavefunction), tarea T3a.

Implementa el metodo de saltos cuanticos (quantum jumps) del informe (Sec 7.3):
en lugar de propagar el operador densidad rho (dimension d^2), se propagan M
vectores de estado |psi> (dimension d) y se promedia rho = E[|psi><psi|]. Esto
reduce la memoria de d^2 a d y es TRIVIALMENTE PARALELIZABLE: cada trayectoria es
independiente (embarrassingly parallel). Aqui se paraleliza con multiprocessing;
en HPC se reparten las trayectorias entre rangos MPI/nodos (ver ../cpp y el
informe, Seccion 8.1, nivel 2 de paralelismo).

Algoritmo (por trayectoria):
  H_eff = H - (i/2) sum_mu L_mu† L_mu          (no hermitiano)
  paso dt:
    |psi'> = |psi> - i dt H_eff |psi>          (propagacion determinista)
    dp = 1 - <psi'|psi'>                        (prob. total de salto)
    si r < dp (r uniforme):                     SALTO
        elegir canal mu con prob ~ <psi|L_mu†L_mu|psi> dt / dp
        |psi> = L_mu|psi> / ||L_mu|psi>||
    si no:
        |psi> = |psi'> / |||psi'>||

El error estadistico decae como M^{-1/2}.
"""

from __future__ import annotations
import numpy as np
from multiprocessing import Pool, cpu_count


def _heff(H, Ls):
    He = H.astype(complex).copy()
    for Lk in Ls:
        He = He - 0.5j * (Lk.conj().T @ Lk)
    return He


def _one_trajectory(args):
    """Una trayectoria estocastica. Devuelve rho(t) muestreado = |psi><psi|."""
    H, Ls, psi0, t_max, dt, log_idx, seed = args
    rng = np.random.default_rng(seed)
    He = _heff(H, Ls)
    psi = psi0.astype(complex).copy()
    d = psi.shape[0]
    n_steps = int(np.ceil(t_max / dt))
    out = np.zeros((len(log_idx), d, d), dtype=complex)
    log_set = {idx: j for j, idx in enumerate(log_idx)}

    for step in range(n_steps + 1):
        if step in log_set:
            out[log_set[step]] = np.outer(psi, psi.conj())
        if step == n_steps:
            break
        psi_det = psi - 1j * dt * (He @ psi)
        norm2 = np.real(np.vdot(psi_det, psi_det))
        if rng.random() > norm2:
            # SALTO: elegir canal segun pesos <psi|L†L|psi>
            weights = np.array([np.real(np.vdot(psi, Lk.conj().T @ (Lk @ psi)))
                                for Lk in Ls])
            weights = np.clip(weights, 0, None)
            mu = rng.choice(len(Ls), p=weights / weights.sum())
            phi = Ls[mu] @ psi
            psi = phi / np.linalg.norm(phi)
        else:
            psi = psi_det / np.sqrt(norm2)
    return out


def evolve_trajectories(H, Ls, psi0, t_max, dt, M=2000, log_every=20,
                        n_workers=None, base_seed=12345):
    """Promedia M trayectorias en paralelo. Devuelve (times, rhos).

    rhos[n] = (1/M) sum_traj |psi_traj(t_n)><psi_traj(t_n)|  ->  rho(t_n).
    """
    n_steps = int(np.ceil(t_max / dt))
    log_idx = list(range(0, n_steps + 1, log_every))
    times = np.array([i * dt for i in log_idx])
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    tasks = [(H, Ls, psi0, t_max, dt, log_idx, base_seed + m) for m in range(M)]
    d = psi0.shape[0]
    acc = np.zeros((len(log_idx), d, d), dtype=complex)

    with Pool(n_workers) as pool:
        for traj in pool.imap_unordered(_one_trajectory, tasks, chunksize=16):
            acc += traj
    acc /= M
    return times, acc


# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Validacion: trayectorias vs ecuacion maestra (TLS).
    import time as _time
    import models, qmpe

    H, Ls, info = models.tls(omega0=1.0, gamma=1.0, T=0.5)
    psi0 = np.array([0.0, 1.0], dtype=complex)   # parte en |1> (excitado)
    rho0 = np.outer(psi0, psi0.conj())

    t_max, dt, M = 6.0, 2e-3, 4000
    print(f"Trayectorias cuanticas (T3a): M={M}, workers={max(1,cpu_count()-1)}")
    t0 = _time.time()
    times, rhos = evolve_trajectories(H, Ls, psi0, t_max, dt, M=M, log_every=50)
    print(f"  tiempo: {_time.time()-t0:.2f} s")

    p_traj = np.real(rhos[:, 1, 1])
    p_ana = models.tls_population_analytic(1.0, times, info["Gamma"], info["p_eq"])
    err = np.max(np.abs(p_traj - p_ana))
    print(f"  max|p_traj - p_ana| = {err:.3e}  (esperado ~ M^-1/2 = {M**-0.5:.3e})")
    print(f"  -> {'OK' if err < 5 * M**-0.5 else 'REVISAR'}")
