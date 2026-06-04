#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_t3a.py  --  Validacion de la tarea T3a (trayectorias cuanticas).

1. Convergencia: el error del promedio de M trayectorias frente a la solucion
   EXACTA de la ecuacion maestra (oraculo `common.qmpe`) decae como M^{-1/2}.
2. Speedup: tiempo serial vs multiprocessing (memoria compartida en un nodo).

Genera:  results/t3a_convergence.csv,  results/t3a_convergence.png
"""

from __future__ import annotations
import os
import sys
import csv
import time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import qmpe, models  # noqa: E402
import qtraj  # noqa: E402

RESULTS = os.path.abspath(os.path.join(_HERE, "..", "results"))
os.makedirs(RESULTS, exist_ok=True)


def main():
    # Modelo TLS (Davies qubit): tiene solucion exacta de la poblacion excitada.
    H, Ls, info = models.tls(omega0=1.0, gamma=1.0, T=0.5)
    psi0 = np.array([0.0, 1.0], dtype=complex)        # parte en |1> (excitado)
    rho0 = np.outer(psi0, psi0.conj())
    t_max, dt = 6.0, 2e-3

    # --- oraculo: ecuacion maestra exacta (suma espectral) ---
    sp = qmpe.Spectrum(H, Ls)
    log_every = 50
    n_steps = int(np.ceil(t_max / dt))
    log_idx = list(range(0, n_steps + 1, log_every))
    times = np.array([i * dt for i in log_idx])
    rhos_exact = qmpe.evolve_spectral(sp, rho0, times)
    p_exact = np.real(rhos_exact[:, 1, 1])            # poblacion excitada exacta

    print("=" * 64)
    print("  T3a -- trayectorias cuanticas: convergencia M^{-1/2}")
    print("=" * 64)

    Ms = [125, 250, 500, 1000, 2000, 4000, 8000]
    rows = []
    for M in Ms:
        t_times, pops = qtraj.evolve_parallel(
            H, Ls, psi0, t_max, dt, M=M, log_every=log_every,
            accumulate_rho=False)              # solo poblaciones (barato)
        p_traj = pops[:, 1]                    # poblacion excitada promedio
        err = float(np.sqrt(np.mean((p_traj - p_exact) ** 2)))   # RMS
        rows.append({"M": M, "rms_err": err, "M_inv_sqrt": M ** -0.5})
        print(f"  M={M:5d}   RMS error = {err:.3e}   (M^-1/2 = {M**-0.5:.3e})")

    # ajuste log-log: pendiente esperada ~ -0.5
    logM = np.log(np.array([r["M"] for r in rows]))
    logE = np.log(np.array([r["rms_err"] for r in rows]))
    slope = np.polyfit(logM, logE, 1)[0]
    print(f"\n  Pendiente log-log del error vs M: {slope:.3f}  (esperado ~ -0.5)")

    # --- speedup serial vs multiprocessing ---
    print("\n  --- speedup serial vs multiprocessing (M=4000) ---")
    Mb = 4000
    t0 = time.time()
    qtraj.evolve_serial(H, Ls, psi0, t_max, dt, M=Mb, log_every=log_every,
                        accumulate_rho=False)
    t_ser = time.time() - t0
    t0 = time.time()
    qtraj.evolve_parallel(H, Ls, psi0, t_max, dt, M=Mb, log_every=log_every,
                          accumulate_rho=False)
    t_par = time.time() - t0
    print(f"  serial          = {t_ser:.2f} s")
    print(f"  multiprocessing = {t_par:.2f} s   speedup = {t_ser/t_par:.2f}x")

    # --- figura ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    ax1.plot(times, p_exact, "k-", lw=2, label="ecuacion maestra (exacta)")
    for M in [125, 1000, 8000]:
        _, pops = qtraj.evolve_parallel(H, Ls, psi0, t_max, dt, M=M,
                                        log_every=log_every, accumulate_rho=False)
        ax1.plot(times, pops[:, 1], "--", lw=1, label=f"M={M}")
    ax1.set_xlabel("t"); ax1.set_ylabel(r"$p_1(t)$ (poblacion excitada)")
    ax1.set_title("T3a: trayectorias vs ecuacion maestra (TLS)")
    ax1.legend(fontsize=8)

    Marr = np.array([r["M"] for r in rows])
    Earr = np.array([r["rms_err"] for r in rows])
    ax2.loglog(Marr, Earr, "o-", label="RMS error")
    ax2.loglog(Marr, Earr[0] * (Marr / Marr[0]) ** -0.5, "k--",
               label=r"$\propto M^{-1/2}$")
    ax2.set_xlabel("M (numero de trayectorias)"); ax2.set_ylabel("RMS error")
    ax2.set_title(f"Convergencia (pendiente {slope:.2f})")
    ax2.legend()
    fig.tight_layout()
    figpath = os.path.join(RESULTS, "t3a_convergence.png")
    fig.savefig(figpath, dpi=130)
    print(f"\nFigura -> {figpath}")

    csvpath = os.path.join(RESULTS, "t3a_convergence.csv")
    with open(csvpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["M", "rms_err", "M_inv_sqrt"])
        w.writeheader(); w.writerows(rows)
    print(f"CSV    -> {csvpath}")
    print("\n  Veredicto:", "OK" if abs(slope + 0.5) < 0.15 else "REVISAR",
          f"(pendiente {slope:.3f})")


if __name__ == "__main__":
    main()
