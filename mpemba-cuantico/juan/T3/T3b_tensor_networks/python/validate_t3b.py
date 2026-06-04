#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_t3b.py  --  Validacion de la tarea T3b (TEBD disipativo).

1. Correctitud: n_exc(t) del TEBD coincide con la ecuacion maestra EXACTA
   (oraculo `common.qmpe`) para N pequeno (N=4) al aumentar chi.
2. Convergencia en chi: el error decae al crecer la dimension de enlace.

Genera:  results/t3b_validation.csv,  results/t3b_validation.png
"""
from __future__ import annotations
import os
import sys
import csv
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import qmpe, models  # noqa: E402
import mpdo_tebd as tn  # noqa: E402

RESULTS = os.path.abspath(os.path.join(_HERE, "..", "results"))
os.makedirs(RESULTS, exist_ok=True)


def exact_nexc(N, J, h, gamma, T, p0, times):
    """n_exc(t) exacto via evolucion densa de la ecuacion maestra."""
    H, Ls, info = models.dissipative_ising(N=N, J=J, h=h, gamma=gamma, T=T)
    a, b = np.sqrt(1 - p0), np.sqrt(p0)
    psi1 = np.array([a, b], dtype=complex)
    psi = psi1
    for _ in range(N - 1):
        psi = np.kron(psi, psi1)
    rho0 = np.outer(psi, psi.conj())
    # operador densidad de excitacion
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)
    d = 2 ** N
    Nop = np.zeros((d, d), dtype=complex)
    for i in range(N):
        mats = [np.eye(2, dtype=complex)] * N
        mats[i] = P1
        op = mats[0]
        for k in range(1, N):
            op = np.kron(op, mats[k])
        Nop += op
    Nop /= N
    t_grid, rhos = qmpe.evolve_rk4(H, Ls, rho0, t_max=times[-1] + 1e-9, dt=2e-3, log_every=1)
    n_exact = np.array([np.real(np.trace(r @ Nop)) for r in rhos])
    return np.interp(times, t_grid, n_exact)


def main():
    N, J, h, gamma, T, p0 = 4, 1.0, 0.5, 0.4, 0.8, 0.9
    t_max, dt, log_every = 4.0, 1e-2, 20

    print("=" * 64)
    print(f"  T3b -- TEBD disipativo vs ecuacion maestra exacta (N={N})")
    print("=" * 64)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    rows = []
    times_ref = None
    n_ref = None
    for chi in [1, 2, 4, 8, 16]:
        times, vals, chi_max = tn.evolve_tebd(N, J, h, gamma, T, p0, t_max, dt,
                                              chi=chi, log_every=log_every)
        if n_ref is None:
            times_ref = times
            n_ref = exact_nexc(N, J, h, gamma, T, p0, times)
        err = float(np.max(np.abs(vals - n_ref)))
        rows.append({"chi": chi, "chi_max_alcanzado": chi_max, "max_err": err})
        print(f"  chi={chi:2d} (alcanzado {chi_max:2d})   max|n_TEBD - n_exacto| = {err:.3e}")
        ax1.plot(times, vals, "--", lw=1, label=f"TEBD chi={chi}")

    ax1.plot(times_ref, n_ref, "k-", lw=2, label="exacto (ec. maestra)")
    ax1.set_xlabel("t"); ax1.set_ylabel(r"$n_{exc}(t)$")
    ax1.set_title(f"T3b: TEBD vs exacto (Ising N={N}, $p_0$={p0})")
    ax1.legend(fontsize=8)

    chis = [r["chi"] for r in rows]
    errs = [r["max_err"] for r in rows]
    ax2.semilogy(chis, errs, "o-")
    ax2.set_xlabel(r"$\chi$ (dimension de enlace)")
    ax2.set_ylabel("max error vs exacto")
    ax2.set_title("Convergencia en $\\chi$")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    figpath = os.path.join(RESULTS, "t3b_validation.png")
    fig.savefig(figpath, dpi=130)
    print(f"\nFigura -> {figpath}")

    csvpath = os.path.join(RESULTS, "t3b_validation.csv")
    with open(csvpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["chi", "chi_max_alcanzado", "max_err"])
        w.writeheader(); w.writerows(rows)
    print(f"CSV    -> {csvpath}")
    best = min(r["max_err"] for r in rows)
    print("\n  Veredicto:", "OK" if best < 1e-3 else "REVISAR", f"(mejor error {best:.2e})")


if __name__ == "__main__":
    main()
