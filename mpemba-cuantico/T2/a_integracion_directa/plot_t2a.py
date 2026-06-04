#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_t2a.py -- figuras y validacion del entregable T2(a): RK4 directo.

Produce en results/:
  fig_t2a_speedup.png      speedup S(p)=T1/Tp y eficiencia E(p)=S/p (escalado fuerte)
  fig_t2a_size.png         tiempo y tiempo/paso vs tamano del sistema (N)
  fig_t2a_relax.png        curva fisica de relajacion D_HS(t)
  fig_t2a_convergence.png  convergencia del RK4: error vs dt (orden 4)
  t2a_metrics.txt          numeros clave para el informe

Reutiliza el nucleo validado de context/python para el test de convergencia
(verdad de referencia = evolucion espectral exacta).
"""
import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
sys.path.insert(0, os.path.join(HERE, "..", "..", "context", "python"))


def load_csv(path):
    with open(path) as f:
        r = csv.DictReader(f)
        return [{k: float(v) for k, v in row.items()} for row in r]


# ---------------------------------------------------------------------
def fig_speedup(metrics):
    rows = sorted(load_csv(os.path.join(RES, "scaling_threads.csv")), key=lambda r: r["threads"])
    p = np.array([r["threads"] for r in rows])
    T = np.array([r["wall_s"] for r in rows])
    T1 = T[p == 1][0]
    S = T1 / T
    E = S / p

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(p, S, "o-", lw=1.8, label="medido")
    ax[0].plot(p, p, "k--", alpha=0.5, label="ideal $S=p$")
    ax[0].set_xlabel("hilos $p$"); ax[0].set_ylabel("speedup $S(p)=T_1/T_p$")
    ax[0].set_title("(a) Escalado fuerte (RK4, N=7)"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(p, 100 * E, "s-", color="C3", lw=1.8)
    ax[1].axhline(100, color="k", ls="--", alpha=0.5)
    ax[1].set_xlabel("hilos $p$"); ax[1].set_ylabel("eficiencia $E(p)=S/p$ [%]")
    ax[1].set_title("(b) Eficiencia paralela"); ax[1].grid(alpha=0.3); ax[1].set_ylim(0, 110)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_t2a_speedup.png"), dpi=140)

    # Ley de Amdahl: 1/S = f + (1-f)/p. Se ajusta SOLO en el rango de nucleos
    # FISICOS (p<=8); los puntos 12/16 son hyperthreads y degradan (no es Amdahl).
    mask = p <= 8
    x = 1.0 / p[mask]; y = 1.0 / S[mask]
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    f_serial = coef[0]                 # interseccion ~ fraccion serie
    metrics["S_max"] = float(S.max()); metrics["p_at_Smax"] = int(p[np.argmax(S)])
    metrics["E_at_8"] = float(100 * E[p == 8][0]) if 8 in p else float("nan")
    metrics["f_serial_amdahl"] = float(max(0.0, f_serial))
    metrics["T1"] = float(T1)
    return metrics


def fig_size(metrics):
    rows = sorted(load_csv(os.path.join(RES, "scaling_size.csv")), key=lambda r: r["N"])
    N = np.array([r["N"] for r in rows])
    T = np.array([r["wall_s"] for r in rows])
    steps = np.array([r["steps"] for r in rows])
    d = 2 ** N
    per_step = T / steps
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].semilogy(N, T, "o-", lw=1.8)
    ax[0].set_xlabel("N (espines)"); ax[0].set_ylabel("tiempo total [s] (8 hilos)")
    ax[0].set_title("(a) Coste vs tamano del sistema"); ax[0].grid(alpha=0.3, which="both")
    # coste por paso ~ d^3 (matmul denso); referencia
    ax[1].loglog(d, per_step, "o-", lw=1.8, label="medido")
    ref = per_step[-1] * (d / d[-1]) ** 3
    ax[1].loglog(d, ref, "k--", alpha=0.5, label=r"$\propto d^3$")
    ax[1].set_xlabel("$d=2^N$"); ax[1].set_ylabel("tiempo por paso [s]")
    ax[1].set_title("(b) Coste por paso $\\sim d^3$"); ax[1].legend(); ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_t2a_size.png"), dpi=140)


def fig_serial_parallel(metrics):
    """Serie (1 hilo) vs paralelo (8 hilos): tiempo de computo vs tamano N.
    Las dos curvas en una misma grafica evidencian la mejora al crecer la malla."""
    path = os.path.join(RES, "serial_vs_parallel.csv")
    if not os.path.exists(path):
        return
    rows = load_csv(path)
    Ns = sorted({int(r["N"]) for r in rows})
    def series(p):
        d = {int(r["N"]): r["wall_s"] for r in rows if int(r["threads"]) == p}
        return np.array([d[n] for n in Ns])
    N = np.array(Ns, float)
    Ts, Tp = series(1), series(8)
    S = Ts / Tp
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].semilogy(N, Ts, "o-", color="C3", lw=1.9, label="serie (1 hilo)")
    ax[0].semilogy(N, Tp, "s-", color="C0", lw=1.9, label="paralelo (8 hilos)")
    ax[0].set_xlabel("N (espines, malla $d=2^N$)"); ax[0].set_ylabel("tiempo de computo [s]")
    ax[0].set_title("(a) Serie vs paralelo (RK4, 100 pasos)"); ax[0].set_xticks(N)
    ax[0].legend(); ax[0].grid(alpha=0.3, which="both")
    ax[1].plot(N, S, "^-", color="C2", lw=1.9)
    ax[1].set_xlabel("N (espines)"); ax[1].set_ylabel(r"speedup $T_{\rm serie}/T_{\rm par}$")
    ax[1].set_title("(b) Mejora por paralelizacion"); ax[1].set_xticks(N); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_t2a_serial_parallel.png"), dpi=140)
    metrics["speedup_at_Nmax"] = float(S[-1])
    metrics["t_serie_Nmax"] = float(Ts[-1]); metrics["t_par_Nmax"] = float(Tp[-1])


def fig_relax():
    path = os.path.join(RES, "curve.csv")
    if not os.path.exists(path):
        return
    rows = load_csv(path)
    t = np.array([r["t"] for r in rows]); D = np.array([r["D_HS"] for r in rows])
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.semilogy(t, D, lw=1.8)
    ax.set_xlabel("t"); ax.set_ylabel(r"$D_{HS}(\rho_t \| \rho_{ss})$")
    ax.set_title("Relajacion (RK4 directo, N=7, $T_0$=5 al estacionario verdadero)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_t2a_relax.png"), dpi=140)


def fig_convergence(metrics):
    """Convergencia del RK4: error a tiempo fijo vs dt -> pendiente ~4."""
    try:
        import models, qmpe
    except Exception as e:
        print("  (sin context/python, salto convergencia):", e); return
    H, Ls, info = models.dissipative_ising(N=2, J=1.0, h=0.5, gamma=0.4, T=0.8)
    rho0 = models.gibbs_state(H, 5.0)
    sp = qmpe.Spectrum(H, Ls)
    Tend = 2.0
    exact = qmpe.evolve_spectral(sp, rho0, np.array([Tend]))[0]
    dts = np.array([0.2, 0.1, 0.05, 0.025, 0.0125])
    errs = []
    for dt in dts:
        _, rhos = qmpe.evolve_rk4(H, Ls, rho0, Tend, dt, log_every=10 ** 9)
        # evolve_rk4 loguea el paso 0 y nada mas con log_every enorme -> reintegro al final
        # tomamos el ultimo estado integrando manualmente
        rho = rho0.astype(complex).copy()
        nst = int(np.ceil(Tend / dt))
        for _ in range(nst):
            k1 = qmpe.apply_liouvillian(H, Ls, rho)
            k2 = qmpe.apply_liouvillian(H, Ls, rho + 0.5 * dt * k1)
            k3 = qmpe.apply_liouvillian(H, Ls, rho + 0.5 * dt * k2)
            k4 = qmpe.apply_liouvillian(H, Ls, rho + dt * k3)
            rho = rho + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        errs.append(np.linalg.norm(rho - exact))
    errs = np.array(errs)
    slope = np.polyfit(np.log(dts), np.log(errs), 1)[0]
    metrics["rk4_order"] = float(slope)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.loglog(dts, errs, "o-", lw=1.8, label=f"error (pendiente={slope:.2f})")
    ax.loglog(dts, errs[-1] * (dts / dts[-1]) ** 4, "k--", alpha=0.6, label=r"orden 4 ($\propto \Delta t^4$)")
    ax.set_xlabel(r"$\Delta t$"); ax.set_ylabel(r"$\|\rho_{RK4}(T)-\rho_{exact}(T)\|_F$")
    ax.set_title("Convergencia del integrador RK4 (vs evolucion espectral exacta)")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_t2a_convergence.png"), dpi=140)


if __name__ == "__main__":
    metrics = {}
    fig_speedup(metrics)
    fig_size(metrics)
    fig_serial_parallel(metrics)
    fig_relax()
    fig_convergence(metrics)
    with open(os.path.join(RES, "t2a_metrics.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k} = {v}\n")
    print("metricas:", metrics)
    print("figuras en", RES)
