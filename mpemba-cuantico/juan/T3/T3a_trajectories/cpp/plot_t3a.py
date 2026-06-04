#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_t3a.py -- escalado y curvas de relajacion de T3a (trayectorias)."""
import os
import csv
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))


def plot_scaling():
    path = os.path.join(RES, "t3a_bench.csv")
    if not os.path.exists(path):
        return
    rows = list(csv.DictReader(open(path)))
    omp = sorted((int(r["threads"]), float(r["wall"])) for r in rows if r["modo"] == "openmp")
    mpi = sorted((int(r["ranks"]), float(r["wall"])) for r in rows if r["modo"] == "mpi")
    N, d = rows[0]["N"], rows[0]["d"]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    if omp:
        t1 = omp[0][1]; ax.plot([w for w, _ in omp], [t1 / t for _, t in omp], "o-", label="OpenMP (hilos)")
    if mpi:
        t1 = mpi[0][1]; ax.plot([w for w, _ in mpi], [t1 / t for _, t in mpi], "s-", label="MPI (rangos)")
    wmax = max([w for w, _ in omp] + [w for w, _ in mpi])
    ax.plot(range(1, wmax + 1), range(1, wmax + 1), "k--", lw=0.8, label="ideal")
    ax.set_xlabel("workers"); ax.set_ylabel("speedup $t_1/t_p$")
    ax.set_title(f"T3a trayectorias: escalado (Ising N={N}, d={d})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(RES, "t3a_scaling.png"); fig.savefig(out, dpi=130)
    print("Figura ->", out)


def plot_curves():
    files = sorted(glob.glob(os.path.join(RES, "t3a_p0_*.csv")))
    if not files:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    # estimar el estacionario como el promedio de los valores finales
    finals = []
    series = []
    for f in files:
        data = np.loadtxt(f, delimiter=",", skiprows=1)
        p0 = f.split("t3a_p0_")[1].replace(".csv", "")
        series.append((p0, data[:, 0], data[:, 1]))
        finals.append(data[-1, 1])
    n_ss = float(np.mean(finals))
    for p0, t, n in series:
        ax1.plot(t, n, label=f"$p_0$={p0}")
        ax2.semilogy(t, np.abs(n - n_ss) + 1e-6, label=f"$p_0$={p0}")
    ax1.axhline(n_ss, color="gray", ls=":", lw=1, label="estacionario")
    ax1.set_xlabel("t"); ax1.set_ylabel(r"$n_{exc}(t)$")
    ax1.set_title("Relajacion de distintas preparaciones (Ising N=8)")
    ax1.legend(fontsize=8)
    ax2.set_xlabel("t"); ax2.set_ylabel(r"$|n_{exc}(t)-n_{ss}|$")
    ax2.set_title("Distancia al estacionario (escala log)")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(RES, "t3a_curves.png"); fig.savefig(out, dpi=130)
    print("Figura ->", out)


if __name__ == "__main__":
    plot_scaling()
    plot_curves()
