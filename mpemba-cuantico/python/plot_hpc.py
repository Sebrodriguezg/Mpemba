#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_hpc.py  --  Analiza la salida del nucleo HPC en C++ (qmpe_hpc).

Lee los CSV results/qmpe_T0_*.csv (curvas D_HS(t) por preparacion), las grafica
en escala log y detecta cruces de Mpemba entre todos los pares.

Uso:  python plot_hpc.py [results_dir]
"""
import os, sys, glob, re, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    with open(path) as f:
        r = csv.reader(f); next(r)
        rows = [(float(a), float(b)) for a, b in r]
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1]


def main(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "qmpe_T0_*.csv")),
                   key=lambda p: float(re.search(r"T0_([\d.]+)\.csv", p).group(1)))
    if not files:
        print(f"no hay CSV en {results_dir}"); return 1
    data = {}
    for f in files:
        T0 = float(re.search(r"T0_([\d.]+)\.csv", f).group(1))
        data[T0] = load(f)

    fig, ax = plt.subplots(figsize=(7, 4.6))
    cmap = plt.get_cmap("plasma")
    T0s = sorted(data)
    for T0 in T0s:
        t, D = data[T0]
        c = cmap((np.log10(T0) - np.log10(T0s[0])) /
                 max(1e-9, np.log10(T0s[-1]) - np.log10(T0s[0])))
        ax.semilogy(t, D, color=c, lw=1.7, label=f"$T_0={T0}$")
    ax.set_xlabel("t"); ax.set_ylabel(r"$D_{HS}(\rho_t \| \rho_{ss})$")
    ax.set_title("Nucleo HPC: relajacion de la cadena de Ising disipativa")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    out = os.path.join(results_dir, "qmpe_hpc_curves.png")
    fig.tight_layout(); fig.savefig(out, dpi=140)
    print(f"-> {out}")

    # deteccion de cruces entre pares (caliente parte mas lejos y adelanta)
    print("Cruces de Mpemba detectados (D(0) mayor -> termina por debajo):")
    found = False
    for i, Ta in enumerate(T0s):
        for Tb in T0s[i + 1:]:
            ta, Da = data[Ta]; tb, Db = data[Tb]
            n = min(len(Da), len(Db))
            diff = Da[:n] - Db[:n]
            if diff[0] > 0 and diff[-1] < 0:   # caliente(Ta) empieza lejos, acaba cerca
                kcross = np.where(diff < 0)[0][0]
                print(f"  T0={Ta} vs T0={Tb}: cruce ~ t={ta[kcross]:.3f}")
                found = True
            elif diff[0] < 0 and diff[-1] > 0:
                kcross = np.where(diff > 0)[0][0]
                print(f"  T0={Tb} vs T0={Ta}: cruce ~ t={tb[kcross]:.3f}")
                found = True
    if not found:
        print("  (ninguno con esta parametrizacion; ajustar h, gamma, T_bath o T0_list)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "results"))
