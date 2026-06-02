#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_demo.py  --  Demostracion y validacion del nucleo QMpE.

Ejecuta el pipeline de diagnostico del informe (Seccion 8, Algoritmo 1) sobre
los modelos resolubles y produce:

  1. VALIDACION: TLS numerico (RK4 matrix-free y espectral) vs solucion analitica.
  2. DIAGNOSTICO ESPECTRAL: autovalores lentos y solapamientos a_k (criterio de
     Mpemba fuerte a_2 = 0).
  3. EFECTO MPEMBA: cruce de curvas de distancia en el sistema Lambda de 3 niveles.
  4. Figuras PNG + CSV en ./results/

Uso:
    python run_demo.py
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import qmpe
import models

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)


# =====================================================================
def validate_tls():
    """(1) El integrador matrix-free reproduce la solucion analitica del TLS."""
    print("\n=== (1) VALIDACION: TLS numerico vs analitico ===")
    H, Ls, info = models.tls(omega0=1.0, gamma=1.0, T=0.5)
    Gamma, p_eq = info["Gamma"], info["p_eq"]
    p0 = 0.9
    rho0 = models.thermal_population_state(2, p0)

    t_max, dt = 6.0, 1e-3
    times, rhos = qmpe.evolve_rk4(H, Ls, rho0, t_max, dt, log_every=20)
    p_num = np.real(rhos[:, 1, 1])
    p_ana = models.tls_population_analytic(p0, times, Gamma, p_eq)
    err = np.max(np.abs(p_num - p_ana))
    print(f"  Gamma = {Gamma:.6f}   p_eq = {p_eq:.6f}")
    print(f"  max|p_num - p_ana| = {err:.3e}   ->  {'OK' if err < 1e-5 else 'FALLA'}")

    # comprobacion fisica: traza y positividad
    traces = np.array([np.real(np.trace(r)) for r in rhos])
    print(f"  max|Tr(rho)-1| = {np.max(np.abs(traces - 1)):.3e}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(times, p_ana, "k-", lw=2, label="analitico")
    ax.plot(times[::3], p_num[::3], "ro", ms=4, label="RK4 matrix-free")
    ax.axhline(p_eq, color="gray", ls="--", label="$p_{eq}$")
    ax.set_xlabel("t"); ax.set_ylabel("$p_1(t)$")
    ax.set_title(f"TLS: validacion del integrador (err={err:.1e})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/01_tls_validation.png", dpi=140)
    print(f"  -> {OUT}/01_tls_validation.png")
    return err < 1e-5


# =====================================================================
def spectral_diagnosis():
    """(2) Tarea T1: autovalores lentos y solapamientos a_k."""
    print("\n=== (2) DIAGNOSTICO ESPECTRAL (T1): Lambda de 3 niveles ===")
    H, Ls, info = models.lambda_three_level(omega=1.0, gamma1=1.5, gamma2=0.3, T=0.6)
    sp = qmpe.Spectrum(H, Ls)
    print(f"  Autovalores lambda_k (ordenados por |Re|):")
    for k in range(min(6, sp.d * sp.d)):
        lam = sp.eigvals[k]
        print(f"    lambda_{k+1} = {lam.real:+.5f} {lam.imag:+.5f}i")
    print(f"  tau = 1/|Re(lambda_2)| = {sp.relaxation_time():.5f}")

    # Solapamientos para varias preparaciones
    print("  Solapamiento a_2 = Tr(l_2† rho0) por preparacion:")
    for p in [0.95, 0.7, 0.5]:
        rho0 = models.thermal_population_state(3, p)
        a = sp.overlaps(rho0)
        print(f"    p_exc={p:.2f}:  a_2 = {a[1]:+.4f}   |a_2| = {abs(a[1]):.4f}")
    return sp


# =====================================================================
def _hermitian_traceless_mode(sp, k):
    """Devuelve el modo derecho r_k hermitizado, sin traza y normalizado en HS."""
    r = sp.r_ops[k]
    r = 0.5 * (r + r.conj().T)               # parte hermitica
    r = r - np.trace(r) / sp.d * np.eye(sp.d)  # quitar traza
    nrm = np.sqrt(np.real(np.trace(r @ r)))
    return r / nrm if nrm > 1e-12 else r


def _max_coeff_positive(rho_ss, mode, safety=0.85):
    """Mayor |c| tal que rho_ss + c*mode siga siendo semidefinida positiva."""
    ev_ss = np.linalg.eigvalsh(rho_ss)
    ev_mode = np.linalg.eigvalsh(mode)
    span = max(np.max(np.abs(ev_mode)), 1e-12)
    return safety * np.min(ev_ss) / span


def strong_mpemba(sp):
    """(3) Efecto Mpemba FUERTE (Carollo): construccion espectral determinista.

    Mecanismo del informe (caja 'Idea fisica central'): se construyen dos estados
    validos que difieren en su solapamiento con los modos de relajacion.
      - rho_caliente: excita un modo RAPIDO con gran amplitud -> parte LEJOS pero
        relaja rapido (poco solapamiento con el modo lento, a_slow ~ 0).
      - rho_templado: excita el modo LENTO -> parte mas cerca pero relaja lento.
    La no-ortogonalidad de los modos garantiza el cruce.
    """
    print("\n=== (3) EFECTO MPEMBA FUERTE (Carollo): geometria espectral ===")
    rho_ss = sp.rho_ss

    # Identificar modos reales (Im~0) ordenados por |Re| creciente
    real_idx = [k for k in range(1, sp.d * sp.d)
                if abs(sp.eigvals[k].imag) < 1e-6]
    k_slow, k_fast = real_idx[0], real_idx[-1]
    lam_slow = sp.eigvals[k_slow].real
    lam_fast = sp.eigvals[k_fast].real
    print(f"  modo lento : lambda_{k_slow+1} = {lam_slow:+.5f}  (tau={1/abs(lam_slow):.2f})")
    print(f"  modo rapido: lambda_{k_fast+1} = {lam_fast:+.5f}  (tau={1/abs(lam_fast):.2f})")

    r_slow = _hermitian_traceless_mode(sp, k_slow)
    r_fast = _hermitian_traceless_mode(sp, k_fast)

    c_slow = _max_coeff_positive(rho_ss, r_slow)
    c_fast = _max_coeff_positive(rho_ss, r_fast)

    # rho_caliente: modo rapido con AMPLITUD PLENA (parte lejos), nada de lento
    rho_hot = rho_ss + c_fast * r_fast
    # rho_templado: modo lento con amplitud MENOR (parte mas cerca), nada de rapido
    rho_warm = rho_ss + 0.5 * c_slow * r_slow

    for nm, rr in [("caliente", rho_hot), ("templado", rho_warm)]:
        ev = np.linalg.eigvalsh(rr)
        assert ev.min() > -1e-9, f"{nm} no es positivo (min ev={ev.min():.2e})"

    d_hot0 = qmpe.trace_distance(rho_hot, rho_ss)
    d_warm0 = qmpe.trace_distance(rho_warm, rho_ss)
    print(f"  D(0): caliente={d_hot0:.4f}  templado={d_warm0:.4f}  "
          f"(caliente parte {'MAS LEJOS' if d_hot0 > d_warm0 else 'mas cerca'})")

    times = np.linspace(0, max(6 / abs(lam_fast), 4 / abs(lam_slow)), 500)
    rhos_hot = qmpe.evolve_spectral(sp, rho_hot, times)
    rhos_warm = qmpe.evolve_spectral(sp, rho_warm, times)
    D_hot = np.array([qmpe.trace_distance(r, rho_ss) for r in rhos_hot])
    D_warm = np.array([qmpe.trace_distance(r, rho_ss) for r in rhos_warm])

    tstar = qmpe.crossing_time(times, D_hot, D_warm)
    if tstar is not None:
        print(f"  -> CRUCE en t* = {tstar:.4f}: la caliente adelanta a la templada (Mpemba)")
    else:
        print("  -> sin cruce detectado")

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.semilogy(times, D_hot, "r-", lw=1.9, label=f"caliente (modo rapido), $D_0$={d_hot0:.2f}")
    ax.semilogy(times, D_warm, "b-", lw=1.9, label=f"templado (modo lento), $D_0$={d_warm0:.2f}")
    ax.semilogy(times, d_hot0 * np.exp(lam_fast * times), "r:", alpha=0.5,
                label=r"$\sim e^{Re(\lambda_{fast})t}$")
    ax.semilogy(times, d_warm0 * np.exp(lam_slow * times), "b:", alpha=0.5,
                label=r"$\sim e^{Re(\lambda_{slow})t}$")
    if tstar is not None:
        ax.axvline(tstar, color="green", ls="--", label=f"$t^*={tstar:.2f}$")
    ax.set_xlabel("t"); ax.set_ylabel(r"$D_{tr}(\rho_t \| \rho_{ss})$")
    ax.set_title("Efecto Mpemba fuerte cuantico (construccion espectral, Carollo 2021)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(f"{OUT}/03_mpemba_strong.png", dpi=140)
    print(f"  -> {OUT}/03_mpemba_strong.png")

    import csv
    with open(f"{OUT}/mpemba_curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "D_caliente", "D_templado"])
        for n, t in enumerate(times):
            w.writerow([t, D_hot[n], D_warm[n]])
    print(f"  -> {OUT}/mpemba_curves.csv")
    return tstar


# =====================================================================
def ising_scaling():
    """(4) Demuestra el escalado d^2: construye el Liouvilliano de Ising."""
    print("\n=== (4) ESCALADO HPC: Liouvilliano de Ising disipativo ===")
    for N in [2, 3, 4]:
        H, Ls, info = models.dissipative_ising(N=N, J=1.0, h=0.5, gamma=0.4, T=0.8)
        d = info["d"]
        print(f"  N={N}:  d=2^N={d},  dim(L)=d^2 x d^2 = {d*d} x {d*d} "
              f"= {d**4} elementos densos")
    print("  -> para N grande, d^2 fuerza matrix-free + trayectorias/SLEPc (ver ../cpp)")


# =====================================================================
if __name__ == "__main__":
    print("#" * 68)
    print("# Demostracion del nucleo QMpE  (acompana a mpemba_cuantico.tex)")
    print("#" * 68)
    ok = validate_tls()
    sp = spectral_diagnosis()
    strong_mpemba(sp)
    ising_scaling()
    print("\nListo. Figuras y CSV en", OUT)
    print("Estado de validacion del integrador:", "OK" if ok else "REVISAR")
