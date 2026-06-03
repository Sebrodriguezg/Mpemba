#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_t3.py -- figuras del entregable T3(a): trayectorias cuanticas con MPI.

results/:
  fig_t3_speedup.png    escalado fuerte sobre rangos MPI (casi ideal)
  fig_t3_error_M.png    error estadistico vs M (~ M^-1/2)
  fig_t3_relax.png      curva de relajacion (N=8)
  t3_metrics.txt
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def load(p):
    with open(p) as f: return list(csv.DictReader(f))


def fig_speedup(m):
    rows = sorted(load(os.path.join(RES, "scaling_ranks.csv")), key=lambda r: int(r["ranks"]))
    p = np.array([int(r["ranks"]) for r in rows]); T = np.array([float(r["wall_s"]) for r in rows])
    S = T[p == 1][0] / T; E = S / p
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(p, S, "o-", lw=1.9, label="medido (MPI)"); ax[0].plot(p, p, "k--", alpha=0.5, label="ideal $S=p$")
    ax[0].set_xlabel("rangos MPI $p$"); ax[0].set_ylabel("$S(p)=T_1/T_p$")
    ax[0].set_title("(a) Escalado fuerte de trayectorias (N=7, M=2000)"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(p, 100 * E, "s-", color="C2", lw=1.9); ax[1].axhline(100, color="k", ls="--", alpha=0.5)
    ax[1].set_xlabel("rangos MPI $p$"); ax[1].set_ylabel("$E(p)$ [%]"); ax[1].set_ylim(0, 115)
    ax[1].set_title("(b) Eficiencia (casi ideal: embarrassingly parallel)"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_t3_speedup.png"), dpi=140)
    m["S_max"] = float(S.max()); m["p_max"] = int(p.max())
    m["E_at_8"] = float(100 * E[p == 8][0]) if 8 in p else None
    m["E_at_max"] = float(100 * E[-1])


def fig_error(m):
    rows = load(os.path.join(RES, "error_vs_M.csv"))
    exact = float(open(os.path.join(RES, "dhs_exact.txt")).read().strip())
    # agrupar por M y calcular el RMS del error sobre las semillas
    byM = {}
    for r in rows:
        byM.setdefault(int(r["M"]), []).append(abs(float(r["dhs"]) - exact))
    M = np.array(sorted(byM), float)
    err = np.array([np.sqrt(np.mean(np.array(byM[int(x)]) ** 2)) for x in M])
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.loglog(M, err, "o-", color="C2", lw=1.8, label="error RMS (4 semillas)")
    ax.loglog(M, err[0] * (M / M[0]) ** -0.5, "k--", alpha=0.6, label=r"$\propto M^{-1/2}$")
    ax.set_xlabel("nº de trayectorias $M$"); ax.set_ylabel("error en $D_{HS}$ final")
    ax.set_title("Convergencia estadistica de Monte Carlo ($M^{-1/2}$)")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_t3_error_M.png"), dpi=140)
    slope = np.polyfit(np.log(M), np.log(np.clip(err, 1e-9, None)), 1)[0]
    m["mc_slope"] = float(slope); m["exact"] = exact


def fig_serial_parallel(m):
    """Serie (1 rango) vs paralelo (8 rangos): tiempo de computo vs nro de
    trayectorias M. Dos rectas que divergen -> mejora casi constante (~ #cores)."""
    path = os.path.join(RES, "serial_vs_parallel.csv")
    if not os.path.exists(path): return
    rows = load(path)
    Ms = sorted({int(r["M"]) for r in rows})
    def series(p):
        d = {int(r["M"]): float(r["wall_s"]) for r in rows if int(r["ranks"]) == p}
        return np.array([d[x] for x in Ms])
    M = np.array(Ms, float); Ts, Tp = series(1), series(8); S = Ts / Tp
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(M, Ts, "o-", color="C3", lw=1.9, label="serie (1 rango)")
    ax[0].plot(M, Tp, "s-", color="C0", lw=1.9, label="paralelo (8 rangos)")
    ax[0].set_xlabel("nº de trayectorias $M$"); ax[0].set_ylabel("tiempo de computo [s]")
    ax[0].set_title("(a) Serie vs paralelo (trayectorias MPI, N=6)")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(M, S, "^-", color="C2", lw=1.9)
    ax[1].axhline(8, color="k", ls="--", alpha=0.5, label="8 rangos (ideal)")
    ax[1].set_xlabel("nº de trayectorias $M$"); ax[1].set_ylabel(r"speedup $T_{\rm serie}/T_{\rm par}$")
    ax[1].set_title("(b) Mejora ~constante (embarrassingly parallel)")
    ax[1].legend(); ax[1].grid(alpha=0.3); ax[1].set_ylim(0, 9)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_t3_serial_parallel.png"), dpi=140)
    m["speedup_at_Mmax"] = float(S[-1])
    m["t_serie_Mmax"] = float(Ts[-1]); m["t_par_Mmax"] = float(Tp[-1])


def fig_relax():
    p = os.path.join(RES, "curve.csv")
    if not os.path.exists(p): return
    rows = load(p); t = np.array([float(r["t"]) for r in rows]); D = np.array([float(r["D_HS"]) for r in rows])
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.plot(t, D, "-", color="C2", lw=1.6)
    ax.set_xlabel("t"); ax.set_ylabel(r"$D_{HS}(\rho_t\|\rho_{ss})$")
    ax.set_title("Relajacion por trayectorias (N=8, d=256, M=4000)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_t3_relax.png"), dpi=140)


if __name__ == "__main__":
    m = {}
    fig_speedup(m); fig_error(m); fig_serial_parallel(m); fig_relax()
    with open(os.path.join(RES, "t3_metrics.txt"), "w") as f:
        for k, v in m.items(): f.write(f"{k} = {v}\n")
    print("metricas:", m)
