#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_t3b.py  --  escalado de T3b (TEBD disipativo paralelo por compuertas) y
demostracion de muchos cuerpos (N grande, inviable para metodos exactos).

IMPORTANTE: ejecutar con BLAS de 1 hilo para aislar el paralelismo de compuertas:
  VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 bench_t3b.py

Genera: results/t3b_bench.csv, results/t3b_scaling.png, results/t3b_largeN.png
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

import tebd_parallel as tp  # noqa: E402

RESULTS = os.path.abspath(os.path.join(_HERE, "..", "results"))
os.makedirs(RESULTS, exist_ok=True)


def scaling():
    N, chi = 20, 48
    t_max, dt = 2.0, 2e-2
    print("=" * 60)
    print(f"  T3b escalado: TEBD paralelo por compuertas (N={N}, chi={chi})")
    print("=" * 60)
    rows = []
    base = None
    for nt in [1, 2, 4, 8]:
        t, v, cm, wall = tp.evolve_tebd_parallel(
            N, 1.0, 0.5, 0.4, 0.8, 0.9, t_max, dt, chi=chi, log_every=10, n_threads=nt)
        if base is None:
            base = wall
        rows.append({"threads": nt, "wall": wall, "speedup": base / wall})
        print(f"  threads={nt}  wall={wall:.2f}s  speedup={base/wall:.2f}x  (chi_max={cm})")
    with open(os.path.join(RESULTS, "t3b_bench.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["threads", "wall", "speedup"])
        w.writeheader(); w.writerows(rows)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    th = [r["threads"] for r in rows]
    ax.plot(th, [r["speedup"] for r in rows], "o-", label="TEBD (hilos, bonds disjuntos)")
    ax.plot(th, th, "k--", lw=0.8, label="ideal")
    ax.set_xlabel("hilos"); ax.set_ylabel("speedup $t_1/t_p$")
    ax.set_title(f"T3b: escalado por compuertas (N={N}, $\\chi$={chi})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "t3b_scaling.png"), dpi=130)
    print("Figura -> results/t3b_scaling.png")


def large_N():
    """Demostracion: N grande, varias preparaciones. El Liouvilliano denso seria
    4^N (p.ej. N=32 -> 4^32 ~ 1.8e19) -- imposible; TEBD lo maneja con chi acotado."""
    N, chi = 32, 24
    t_max, dt = 4.0, 2e-2
    print("\n" + "=" * 60)
    print(f"  T3b muchos cuerpos: N={N} (4^N = {4**N:.2e}, INVIABLE en denso)")
    print("=" * 60)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    finals = []
    series = []
    for p0 in [0.1, 0.5, 0.9]:
        t, v, cm, wall = tp.evolve_tebd_parallel(
            N, 1.0, 0.5, 0.4, 0.8, p0, t_max, dt, chi=chi, log_every=10, n_threads=4)
        series.append((p0, t, v)); finals.append(v[-1])
        print(f"  p0={p0}: n_exc {v[0]:.3f} -> {v[-1]:.3f}  (chi_max={cm}, wall={wall:.1f}s)")
    n_ss = float(np.mean(finals))
    for p0, t, v in series:
        ax.plot(t, v, label=f"$p_0$={p0}")
    ax.axhline(n_ss, color="gray", ls=":", lw=1, label="estacionario")
    ax.set_xlabel("t"); ax.set_ylabel(r"$n_{exc}(t)$")
    ax.set_title(f"T3b: relajacion de Ising disipativo N={N} ($\\chi$={chi})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "t3b_largeN.png"), dpi=130)
    print("Figura -> results/t3b_largeN.png")


if __name__ == "__main__":
    scaling()
    large_N()
