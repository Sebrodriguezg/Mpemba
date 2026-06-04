#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_t1.py  --  Validacion de la tarea T1 (Arnoldi-Lindblad) contra el
oraculo denso (`common.qmpe.Spectrum`, diagonalizacion completa con numpy).

Verifica, sobre los modelos resolubles del informe (Seccion 6):
  1. Que los autovalores lentos lambda_2, lambda_3 coinciden con la
     diagonalizacion densa a tolerancia controlada.
  2. Que el estado estacionario coincide (distancia HS ~ 0).
  3. Que los solapamientos a_k = Tr(l_k^dag rho0) reproducen los del oraculo.
  4. "Faster than the clock": coste en SpMV de Arnoldi-Lindblad vs evolucion
     directa hasta el estacionario.

Genera:  results/t1_validation.csv  y  results/t1_eigs.png
"""

from __future__ import annotations
import os
import sys
import csv
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import qmpe, models  # noqa: E402
import arnoldi_lindblad as al    # noqa: E402

RESULTS = os.path.abspath(os.path.join(_HERE, "..", "results"))
os.makedirs(RESULTS, exist_ok=True)


def build_models():
    """Devuelve una lista de (nombre, H, Ls, rho0) de los modelos del Seccion 6."""
    out = []

    # TLS (Davies qubit) -- un solo modo, relajacion monotona (sin cruce)
    H, Ls, info = models.tls(omega0=1.0, gamma=1.0, T=0.5)
    rho0 = np.diag([0.1, 0.9]).astype(complex)
    out.append(("TLS", H, Ls, rho0))

    # Lambda de 3 niveles -- dos canales con tasas distintas (admite Mpemba)
    H, Ls, info = models.lambda_three_level(gamma1=1.5, gamma2=0.3)
    rho0 = np.diag([0.7, 0.2, 0.1]).astype(complex)
    out.append(("Lambda-3", H, Ls, rho0))

    # Ising disipativo N=2,3 -- el caso de muchos cuerpos en miniatura
    for N in (2, 3):
        H, Ls, info = models.dissipative_ising(N=N, J=1.0, h=0.5, gamma=0.4, T=0.8)
        d = H.shape[0]
        rho0 = np.eye(d, dtype=complex) / d
        out.append((f"Ising-N{N}", H, Ls, rho0))

    return out


def match_eigs(lam_ref, lam_test):
    """Empareja cada autovalor de test con el de referencia mas cercano y
    devuelve el error maximo de los modos lentos (los k mas cercanos a 0)."""
    errs = []
    for lt in lam_test:
        j = int(np.argmin(np.abs(lam_ref - lt)))
        errs.append(abs(lam_ref[j] - lt))
    return float(np.max(errs)) if errs else np.nan


def main():
    rows = []
    print("=" * 70)
    print("  T1 -- Arnoldi-Lindblad  vs  diagonalizacion densa (oraculo)")
    print("=" * 70)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))

    for ax, (name, H, Ls, rho0) in zip(axes, build_models()):
        d = H.shape[0]
        # --- oraculo denso ---
        sp = qmpe.Spectrum(H, Ls)
        lam_dense = sp.eigvals
        a_dense = sp.overlaps(rho0)

        # --- Arnoldi-Lindblad ---
        kmodes = min(6, d * d)
        res = al.slow_modes(H, Ls, k=kmodes, dt=2e-3, m=min(40, d * d), restarts=4)
        lam_al = res.eigvals
        a_al = res.overlaps(rho0)

        # errores
        err_lam = match_eigs(lam_dense, lam_al)
        err_ss = qmpe.hs_distance(res.rho_ss, sp.rho_ss)
        # error en a_2 (el relevante para Mpemba fuerte)
        err_a2 = abs(abs(a_al[1]) - abs(a_dense[1])) if len(a_al) > 1 else np.nan

        # faster-than-the-clock (solo para sistemas pequenos)
        cost = al.steady_state_cost_comparison(H, Ls, dt=2e-3, tol=1e-7)

        print(f"\n[{name}]  d={d}  dim(L)={d*d}")
        print(f"  lambda_2 denso   = {lam_dense[1]:.6f}")
        print(f"  lambda_2 Arnoldi = {lam_al[1]:.6f}")
        print(f"  max|d-lambda_lentos| = {err_lam:.2e}")
        print(f"  HS(rho_ss_AL, rho_ss_denso) = {err_ss:.2e}")
        print(f"  |a2|: denso={abs(a_dense[1]):.4f}  AL={abs(a_al[1]):.4f}  err={err_a2:.2e}")
        print(f"  faster-than-clock: SpMV evol={cost['n_spmv_evolution']}  "
              f"AL={cost['n_spmv_arnoldi']}  speedup={cost['speedup']:.1f}x")

        rows.append({
            "modelo": name, "d": d, "dimL": d * d,
            "lambda2_re_dense": lam_dense[1].real, "lambda2_im_dense": lam_dense[1].imag,
            "lambda2_re_AL": lam_al[1].real, "lambda2_im_AL": lam_al[1].imag,
            "err_lambda_slow": err_lam, "err_rho_ss_HS": err_ss, "err_a2": err_a2,
            "spmv_evolution": cost["n_spmv_evolution"], "spmv_arnoldi": cost["n_spmv_arnoldi"],
            "speedup_ftc": cost["speedup"],
        })

        # plot: espectro en el plano complejo
        ax.scatter(lam_dense.real, lam_dense.imag, s=80, facecolors="none",
                   edgecolors="C0", label="denso (oraculo)")
        ax.scatter(lam_al.real, lam_al.imag, s=20, c="C3", marker="x",
                   label="Arnoldi-Lindblad")
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_title(f"{name}  (d={d})")
        ax.set_xlabel(r"Re$(\lambda)$")
        ax.set_ylabel(r"Im$(\lambda)$")
        ax.legend(fontsize=8)

    fig.suptitle("T1: espectro de modos lentos -- Arnoldi-Lindblad vs diagonalizacion densa")
    fig.tight_layout()
    figpath = os.path.join(RESULTS, "t1_eigs.png")
    fig.savefig(figpath, dpi=130)
    print(f"\nFigura -> {figpath}")

    csvpath = os.path.join(RESULTS, "t1_validation.csv")
    with open(csvpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"CSV    -> {csvpath}")

    max_err = max(r["err_lambda_slow"] for r in rows)
    print("\n" + "=" * 70)
    print(f"  Veredicto: max error en autovalores lentos = {max_err:.2e}")
    print("  OK" if max_err < 1e-3 else "  REVISAR (error alto)")
    print("=" * 70)


if __name__ == "__main__":
    main()
