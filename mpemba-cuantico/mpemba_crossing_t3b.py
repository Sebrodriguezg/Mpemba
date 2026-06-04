#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mpemba_crossing_t3b.py -- cruce de Mpemba con REDES TENSORIALES (T3b de Juan).

Reusa el evolucionador TEBD de Juan (juan/.../mpdo_tebd.py) SIN MODIFICARLO: solo
le pasamos un observable a medida que contrae el superket-MPS a rho y devuelve la
distancia de Hilbert-Schmidt al estado estacionario verdadero (mismo rho_ss exacto
que usan T2/T3, via qmpe.Spectrum), para que el t* sea comparable.

Preparaciones (mismas que los otros metodos):
  |+>^N  -> p0 = 0.5  (parte lejos)      |0>^N -> p0 = 0.0  (parte cerca)

Genera juan/T3/T3b_tensor_networks/results/fig_t3b_mpemba.png  (+ CSV).
"""
import os, sys, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "juan"))                                  # 'common'
sys.path.insert(0, os.path.join(HERE, "juan", "T3", "T3b_tensor_networks", "python"))
from common import qmpe, models           # oraculo (identico a context/python)
import mpdo_tebd as tn                     # TEBD de Juan (sin tocar)

N, J, h, gamma, Tbath = 5, 1.0, 0.5, 0.4, 0.8
CHI = 32                                   # exacto para N=5 (cota = 16)
T_MAX, DT, LOG = 6.0, 0.02, 6

# --- estado estacionario verdadero (nucleo de L), exacto ---
H, Ls, info = models.dissipative_ising(N=N, J=J, h=h, gamma=gamma, T=Tbath)
sp = qmpe.Spectrum(H, Ls); rss = sp.rho_ss; d = sp.d

# --- contraccion del superket-MPS de Juan a rho (solo lectura de A[k]) ---
def mps_to_rho(mps):
    v = mps.A[0].reshape(mps.A[0].shape[1], mps.A[0].shape[2])   # (4, D)  (Dl=1)
    for k in range(1, mps.N):
        Ak = mps.A[k]
        v = np.tensordot(v, Ak, axes=(1, 0))                     # (4^k, 4, D')
        v = v.reshape(v.shape[0] * 4, Ak.shape[2])
    vec = v.reshape(-1)                                          # (4^N,)
    t = vec.reshape([2, 2] * N)                                  # ejes [a0,b0,a1,b1,...]
    perm = list(range(0, 2 * N, 2)) + list(range(1, 2 * N, 2))   # [a..., b...]
    rho = np.transpose(t, perm).reshape(d, d)
    return rho / np.trace(rho)

def dhs_obs(mps):
    return qmpe.hs_distance(mps_to_rho(mps), rss)

def first_crossing(ts, Dfar, Dnear):
    diff = np.asarray(Dfar) - np.asarray(Dnear)
    if diff[0] <= 0: return None
    idx = np.where((diff[:-1] > 0) & (diff[1:] <= 0))[0]
    if len(idx) == 0: return None
    k = idx[0]
    return float(ts[k] - diff[k] * (ts[k+1]-ts[k]) / (diff[k+1]-diff[k]))

if __name__ == "__main__":
    print(f"T3b (TEBD de Juan) N={N} chi={CHI}; NESS exacto "
          f"(||L[rss]||={np.linalg.norm(qmpe.apply_liouvillian(H, Ls, rss)):.1e}).")
    ts, Dfar, _  = tn.evolve_tebd(N, J, h, gamma, Tbath, p0=0.5, t_max=T_MAX, dt=DT,
                                  chi=CHI, log_every=LOG, observable=dhs_obs)
    _,  Dnear, _ = tn.evolve_tebd(N, J, h, gamma, Tbath, p0=0.0, t_max=T_MAX, dt=DT,
                                  chi=CHI, log_every=LOG, observable=dhs_obs)
    ts = np.array(ts); Dfar = np.array(Dfar); Dnear = np.array(Dnear)
    tstar = first_crossing(ts, Dfar, Dnear)
    print(f"  D0(+)={Dfar[0]:.3f}  D0(0)={Dnear[0]:.3f}  t*={tstar}")

    outdir = os.path.join(HERE, "juan", "T3", "T3b_tensor_networks", "results")
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.semilogy(ts, Dfar,  "r-", lw=2.0, label=fr"$|+\rangle^{{\otimes N}}$ (parte lejos), $D_0$={Dfar[0]:.2f}")
    ax.semilogy(ts, Dnear, "b-", lw=2.0, label=fr"$|0\rangle^{{\otimes N}}$ (parte cerca), $D_0$={Dnear[0]:.2f}")
    if tstar is not None:
        yc = float(np.interp(tstar, ts, Dfar))
        ax.axvline(tstar, color="green", ls="--", lw=1.8, label=fr"$t^*={tstar:.2f}$")
        ax.plot([tstar], [yc], "go", ms=10, zorder=5)
        ax.annotate(fr"cruce $t^*={tstar:.2f}$", xy=(tstar, yc), xytext=(tstar+0.7, yc*5),
                    fontsize=9, arrowprops=dict(arrowstyle="->", color="green"))
    ax.set_xlabel("t"); ax.set_ylabel(r"$D_{HS}(\rho_t \,\|\, \rho_{ss})$")
    ax.set_title(f"Efecto Mpemba por redes tensoriales / TEBD (N={N}, $\\chi$={CHI})")
    ax.legend(fontsize=8.5, loc="upper right"); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_t3b_mpemba.png"), dpi=140)
    with open(os.path.join(outdir, "mpemba_crossing.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t", "D_far_plus", "D_near_0"])
        for i in range(len(ts)): w.writerow([ts[i], Dfar[i], Dnear[i]])
    print(f"  -> {outdir}/fig_t3b_mpemba.png")
