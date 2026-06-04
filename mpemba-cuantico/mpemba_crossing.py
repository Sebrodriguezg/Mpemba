#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mpemba_crossing.py -- figuras del CRUCE de Mpemba (la curva que parte lejos
adelanta a la que parte cerca) para los tres metodos de Sebastian, resaltando
el tiempo de cruce t*.

Mismo modelo de Ising disipativo y misma pareja de preparaciones en los tres,
para que las figuras sean comparables:
  - |+>^N  : todos los espines en |+>  -> parte MAS LEJOS del estacionario
  - |0>^N  : todos los espines en |0>  -> parte mas cerca
La referencia es el estado estacionario VERDADERO (nucleo de L, no Gibbs).

Genera (N=5, d=32):
  T2/a_integracion_directa/results/fig_t2a_mpemba.png   (metodo RK4)
  T2/b_accion_exponencial/results/fig_t2b_mpemba.png    (metodo Krylov)
  T3/integration/results/fig_t3_mpemba.png              (metodo trayectorias MCWF)
y un CSV de las curvas junto a cada figura.
"""
import os, sys, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def expm(A):
    """Exponencial de matriz pequena (m x m) por eigendescomposicion."""
    w, V = np.linalg.eig(A)
    return (V * np.exp(w)) @ np.linalg.inv(V)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "context", "python"))
import models, qmpe

# ---------------------------------------------------------------- modelo comun
N, J, h, gamma, Tbath = 5, 1.0, 0.5, 0.4, 0.8
H, Ls, info = models.dissipative_ising(N=N, J=J, h=h, gamma=gamma, T=Tbath)
sp = qmpe.Spectrum(H, Ls)
rss = sp.rho_ss                      # estado estacionario VERDADERO (L[rss]=0)
d = sp.d

# preparaciones puras (producto)
ket0 = np.array([1, 0], complex); ketp = np.array([1, 1], complex) / np.sqrt(2)
def prod_state(single):
    psi = np.array([1], complex)
    for _ in range(N): psi = np.kron(psi, single)
    return psi
psi_far  = prod_state(ketp)          # |+>^N : parte LEJOS
psi_near = prod_state(ket0)          # |0>^N : parte cerca
rho_far  = np.outer(psi_far,  psi_far.conj())
rho_near = np.outer(psi_near, psi_near.conj())

T_MAX, DT = 6.0, 0.01
LOG = 12

# ---------------------------------------------------------------- evolventes
def evolve_rk4(rho0):
    ts, rhos = qmpe.evolve_rk4(H, Ls, rho0, T_MAX, DT, log_every=LOG)
    return np.array(ts), [np.array([qmpe.hs_distance(r, rss) for r in rhos])][0]

def _Lact(V, Lds, LdL):
    out = -1j * (H @ V - V @ H)
    for mu in range(len(Ls)):
        out = out + Ls[mu] @ V @ Lds[mu] - 0.5 * (LdL[mu] @ V + V @ LdL[mu])
    return out

def evolve_krylov(rho0, tau=0.25, m=30):
    Lds = [L.conj().T for L in Ls]; LdL = [Lds[i] @ Ls[i] for i in range(len(Ls))]
    rho = rho0.astype(complex).copy()
    ts = [0.0]; Ds = [qmpe.hs_distance(rho, rss)]
    nchunks = int(round(T_MAX / tau))
    for c in range(nchunks):
        beta = np.sqrt(np.vdot(rho, rho).real)
        Q = [rho / beta]; Hess = np.zeros((m + 1, m), complex); mm = m
        for j in range(m):
            w = _Lact(Q[j], Lds, LdL)
            for i in range(j + 1):
                hij = np.vdot(Q[i], w); Hess[i, j] = hij; w = w - hij * Q[i]
            hj = np.sqrt(np.vdot(w, w).real); Hess[j + 1, j] = hj
            if hj < 1e-12: mm = j + 1; break
            Q.append(w / hj)
        E = expm(Hess[:mm, :mm] * tau)
        rho = beta * sum(E[i, 0] * Q[i] for i in range(mm))
        ts.append((c + 1) * tau); Ds.append(qmpe.hs_distance(rho, rss))
    return np.array(ts), np.array(Ds)

def evolve_mcwf(psi0, M=4000, seed=1):
    LdL = [L.conj().T @ L for L in Ls]
    Heff = H - 0.5j * sum(LdL)
    nst = int(round(T_MAX / DT))
    logs = list(range(0, nst + 1, LOG))
    acc = np.zeros((len(logs), d, d), complex)
    rng = np.random.default_rng(seed)
    f = lambda v: -1j * (Heff @ v)
    for _ in range(M):
        psi = psi0.astype(complex).copy(); li = 0
        for s in range(nst + 1):
            if li < len(logs) and s == logs[li]:
                acc[li] += np.outer(psi, psi.conj()); li += 1
            if s == nst: break
            k1 = f(psi); k2 = f(psi + 0.5*DT*k1); k3 = f(psi + 0.5*DT*k2); k4 = f(psi + DT*k3)
            psip = psi + DT * (k1 + 2*k2 + 2*k3 + k4) / 6.0
            n2 = np.vdot(psip, psip).real
            if rng.random() < 1.0 - n2:                       # salto
                ws = [np.vdot(L @ psi, L @ psi).real for L in Ls]; tot = sum(ws)
                r = rng.random() * tot; cc = 0.0; mu = 0
                for mu in range(len(Ls)):
                    cc += ws[mu]
                    if r <= cc: break
                Lp = Ls[mu] @ psi; psi = Lp / np.sqrt(np.vdot(Lp, Lp).real)
            else:
                psi = psip / np.sqrt(n2)
    acc /= M
    ts = np.array([s * DT for s in logs])
    Ds = np.array([qmpe.hs_distance(acc[i], rss) for i in range(len(logs))])
    return ts, Ds

# ---------------------------------------------------------------- figura comun
def first_crossing(ts, Dfar, Dnear):
    """Primer cruce DESCENDENTE: la curva 'far' (arriba al inicio) cae por debajo
    de 'near'. Robusto frente al ruido del suelo de los metodos estocasticos
    (ignora los cruces espurios posteriores cuando ambas estan en el ruido)."""
    diff = np.asarray(Dfar) - np.asarray(Dnear)          # >0 si far esta arriba
    if diff[0] <= 0:
        return None
    idx = np.where((diff[:-1] > 0) & (diff[1:] <= 0))[0]  # transiciones + -> -
    if len(idx) == 0:
        return None
    k = idx[0]
    return float(ts[k] - diff[k] * (ts[k+1] - ts[k]) / (diff[k+1] - diff[k]))

def make_figure(ts, Dfar, Dnear, out_png, out_csv, method_title, extra=None):
    tstar = first_crossing(ts, Dfar, Dnear)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.semilogy(ts, Dfar,  "r-", lw=2.0, label=fr"$|+\rangle^{{\otimes N}}$ (parte lejos), $D_0$={Dfar[0]:.2f}")
    ax.semilogy(ts, Dnear, "b-", lw=2.0, label=fr"$|0\rangle^{{\otimes N}}$ (parte cerca), $D_0$={Dnear[0]:.2f}")
    if tstar is not None:
        yc = float(np.interp(tstar, ts, Dfar))
        ax.axvline(tstar, color="green", ls="--", lw=1.8, label=fr"$t^*={tstar:.2f}$")
        ax.plot([tstar], [yc], "go", ms=10, zorder=5)
        ax.annotate(fr"cruce $t^*={tstar:.2f}$", xy=(tstar, yc),
                    xytext=(tstar + 0.6, yc * 6), fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="green"))
    ax.set_xlabel("t"); ax.set_ylabel(r"$D_{HS}(\rho_t \,\|\, \rho_{ss})$")
    ax.set_title(method_title)
    ax.legend(fontsize=8.5, loc="upper right"); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)
    with open(out_csv, "w", newline="") as fcsv:
        w = csv.writer(fcsv); w.writerow(["t", "D_far_plus", "D_near_0"])
        for i in range(len(ts)): w.writerow([ts[i], Dfar[i], Dnear[i]])
    print(f"  {os.path.basename(out_png)}: t*={tstar:.4f}  D0(+)={Dfar[0]:.3f}  D0(0)={Dnear[0]:.3f}")
    return tstar

if __name__ == "__main__":
    print(f"Modelo Ising disipativo N={N} (d={d}); NESS verdadero (||L[rss]||="
          f"{np.linalg.norm(qmpe.apply_liouvillian(H, Ls, rss)):.1e}).")
    base = HERE

    print("[T2a] RK4 ...")
    ts, Df = evolve_rk4(rho_far); _, Dn = evolve_rk4(rho_near)
    make_figure(ts, Df, Dn,
                f"{base}/T2/a_integracion_directa/results/fig_t2a_mpemba.png",
                f"{base}/T2/a_integracion_directa/results/mpemba_crossing.csv",
                f"Efecto Mpemba por RK4 directo (Ising disipativo, N={N})")

    print("[T2b] Krylov ...")
    ts, Df = evolve_krylov(rho_far); _, Dn = evolve_krylov(rho_near)
    make_figure(ts, Df, Dn,
                f"{base}/T2/b_accion_exponencial/results/fig_t2b_mpemba.png",
                f"{base}/T2/b_accion_exponencial/results/mpemba_crossing.csv",
                f"Efecto Mpemba por Krylov (Ising disipativo, N={N})")

    print("[T3] trayectorias MCWF (M=4000) ...")
    ts, Df = evolve_mcwf(psi_far); _, Dn = evolve_mcwf(psi_near)
    make_figure(ts, Df, Dn,
                f"{base}/T3/integration/results/fig_t3_mpemba.png",
                f"{base}/T3/integration/results/mpemba_crossing.csv",
                f"Efecto Mpemba por trayectorias cuanticas (N={N}, M=4000)")
    print("listo.")
