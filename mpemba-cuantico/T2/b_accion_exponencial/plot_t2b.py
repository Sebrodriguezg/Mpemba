#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_t2b.py -- figuras del entregable T2(b): Krylov (accion del exponencial).

Produce en results/:
  fig_t2b_speedup.png        escalado fuerte (serial vs OpenMP)
  fig_t2b_method_compare.png comparacion a vs b: error vs trabajo (#aplicaciones de L)
  fig_t2b_relax.png          curva de relajacion
  t2b_metrics.txt            numeros clave
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fig_speedup(metrics):
    rows = sorted(load(os.path.join(RES, "scaling_threads.csv")), key=lambda r: int(r["threads"]))
    p = np.array([int(r["threads"]) for r in rows])
    T = np.array([float(r["wall_s"]) for r in rows])
    S = T[p == 1][0] / T; E = S / p
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(p, S, "o-", lw=1.8, label="medido"); ax[0].plot(p, p, "k--", alpha=0.5, label="ideal")
    ax[0].set_xlabel("hilos $p$"); ax[0].set_ylabel("$S(p)=T_1/T_p$")
    ax[0].set_title("(a) Escalado fuerte (Krylov, N=7)"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(p, 100 * E, "s-", color="C3", lw=1.8); ax[1].axhline(100, color="k", ls="--", alpha=0.5)
    ax[1].set_xlabel("hilos $p$"); ax[1].set_ylabel("$E(p)$ [%]"); ax[1].set_ylim(0, 110)
    ax[1].set_title("(b) Eficiencia"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_t2b_speedup.png"), dpi=140)
    metrics["S_max"] = float(S.max()); metrics["E_at_8"] = float(100 * E[p == 8][0])


def fig_method(metrics):
    rows = load(os.path.join(RES, "method_compare.csv"))
    rk4 = [r for r in rows if r["method"] == "RK4"]
    kry = [r for r in rows if r["method"] == "Krylov"]
    # referencia EXACTA (evolucion espectral) si esta disponible
    exact_path = os.path.join(RES, "dhs_exact.txt")
    if os.path.exists(exact_path):
        dhs_ref = float(open(exact_path).read().strip())
    else:
        dhs_ref = float(max(rk4, key=lambda r: int(r["Lapplies"]))["dhs"])

    def pts(rs):
        L = np.array([int(r["Lapplies"]) for r in rs], float)
        err = np.array([abs(float(r["dhs"]) - dhs_ref) for r in rs])
        err = np.clip(err, 1e-16, None)
        o = np.argsort(L); return L[o], err[o]

    Lr, Er = pts(rk4)
    Lk, Ek = pts(kry)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.loglog(Lr, Er, "o-", color="C0", lw=1.8, label="(a) RK4 directo")
    ax.loglog(Lk, Ek, "s-", color="C3", lw=1.8, label="(b) Krylov exponencial")
    ax.set_xlabel("trabajo = nº de aplicaciones de $\\mathcal{L}$")
    ax.set_ylabel("error en $D_{HS}$ final (vs ref.)")
    ax.set_title("Comparacion a vs b: precision alcanzada por unidad de trabajo")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_t2b_method_compare.png"), dpi=140)

    # metrica: trabajo para llegar a error < 1e-6 con cada metodo
    def work_for(L, E, tol=1e-6):
        ok = L[E < tol]; return int(ok.min()) if len(ok) else None
    metrics["L_rk4_1e-6"] = work_for(Lr, Er)
    metrics["L_kry_1e-6"] = work_for(Lk, Ek)


def fig_serial_parallel(metrics):
    """Serie (1 hilo) vs paralelo (8 hilos): tiempo de computo vs tamano N (Krylov)."""
    path = os.path.join(RES, "serial_vs_parallel.csv")
    if not os.path.exists(path): return
    rows = load(path)
    Ns = sorted({int(r["N"]) for r in rows})
    def series(p):
        d = {int(r["N"]): float(r["wall_s"]) for r in rows if int(r["threads"]) == p}
        return np.array([d[n] for n in Ns])
    N = np.array(Ns, float); Ts, Tp = series(1), series(8); S = Ts / Tp
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].semilogy(N, Ts, "o-", color="C3", lw=1.9, label="serie (1 hilo)")
    ax[0].semilogy(N, Tp, "s-", color="C0", lw=1.9, label="paralelo (8 hilos)")
    ax[0].set_xlabel("N (espines, malla $d=2^N$)"); ax[0].set_ylabel("tiempo de computo [s]")
    ax[0].set_title(r"(a) Serie vs paralelo (Krylov, $\tau$=0.5, m=20)"); ax[0].set_xticks(N)
    ax[0].legend(); ax[0].grid(alpha=0.3, which="both")
    ax[1].plot(N, S, "^-", color="C2", lw=1.9)
    ax[1].set_xlabel("N (espines)"); ax[1].set_ylabel(r"speedup $T_{\rm serie}/T_{\rm par}$")
    ax[1].set_title("(b) Mejora por paralelizacion"); ax[1].set_xticks(N); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_t2b_serial_parallel.png"), dpi=140)
    metrics["speedup_at_Nmax"] = float(S[-1])
    metrics["t_serie_Nmax"] = float(Ts[-1]); metrics["t_par_Nmax"] = float(Tp[-1])


def fig_relax():
    path = os.path.join(RES, "curve.csv")
    if not os.path.exists(path): return
    rows = load(path)
    t = np.array([float(r["t"]) for r in rows]); D = np.array([float(r["D_HS"]) for r in rows])
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.semilogy(t, D, "s-", ms=3, lw=1.4, color="C3")
    ax.set_xlabel("t"); ax.set_ylabel(r"$D_{HS}(\rho_t\|\rho_{ss})$")
    ax.set_title("Relajacion por Krylov (N=7, $\\tau$=0.25 grande, m=30)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_t2b_relax.png"), dpi=140)


if __name__ == "__main__":
    m = {}
    fig_speedup(m); fig_method(m); fig_serial_parallel(m); fig_relax()
    with open(os.path.join(RES, "t2b_metrics.txt"), "w") as f:
        for k, v in m.items(): f.write(f"{k} = {v}\n")
    print("metricas:", m)
