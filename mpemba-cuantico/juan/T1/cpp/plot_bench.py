#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_bench.py -- grafica el escalado de T1 (Arnoldi-Lindblad MPI+OpenMP).

Lee results/t1_bench.csv y produce results/t1_scaling.png con speedup vs
numero de workers para los modos OpenMP (memoria compartida) y MPI (datos
distribuidos), junto a la linea ideal.
"""
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def main():
    rows = load(os.path.join(RES, "t1_bench.csv"))
    omp = [(int(r["threads"]), float(r["wall"])) for r in rows if r["modo"] == "openmp"]
    mpi = [(int(r["ranks"]), float(r["wall"])) for r in rows if r["modo"] == "mpi"]
    omp.sort(); mpi.sort()

    t1_omp = omp[0][1]
    t1_mpi = mpi[0][1]
    N = rows[0]["N"]; d = rows[0]["d"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # speedup
    wx = [w for w, _ in omp]
    ax1.plot([w for w, _ in omp], [t1_omp / t for _, t in omp], "o-", label="OpenMP (hilos)")
    ax1.plot([w for w, _ in mpi], [t1_mpi / t for _, t in mpi], "s-", label="MPI (rangos)")
    ax1.plot(wx, wx, "k--", lw=0.8, label="ideal")
    ax1.set_xlabel("workers (hilos / rangos)")
    ax1.set_ylabel("speedup  $t_1 / t_p$")
    ax1.set_title(f"T1 Arnoldi-Lindblad: speedup (Ising N={N}, d={d})")
    ax1.legend(); ax1.grid(alpha=0.3)

    # wall time
    ax2.plot([w for w, _ in omp], [t for _, t in omp], "o-", label="OpenMP")
    ax2.plot([w for w, _ in mpi], [t for _, t in mpi], "s-", label="MPI")
    ax2.set_xlabel("workers")
    ax2.set_ylabel("tiempo de pared [s]")
    ax2.set_title("Tiempo absoluto")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(RES, "t1_scaling.png")
    fig.savefig(out, dpi=130)
    print("Figura ->", out)


if __name__ == "__main__":
    main()
