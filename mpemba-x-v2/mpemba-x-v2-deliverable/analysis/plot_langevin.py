#!/usr/bin/env python3
"""Plot Langevin colloid Mpemba effect (Kumar & Bechhoefer 2020).

Reads results/langevin/distances.csv and produces a four-panel figure:
  (a) D_L1(t) for hot/warm/cold initial conditions, log scale
  (b) D_KL(t) for hot/warm/cold initial conditions, log scale
  (c) Asymmetric double-well potential U(x) at start
  (d) Initial Boltzmann distributions at T_h, T_w, T_c

Annotates the Mpemba crossover (D_L1_hot < D_L1_warm at some t > 0).
"""
import os
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_distances(path):
    cols = {"t": [], "L1_h": [], "L1_w": [], "L1_c": [],
            "KL_h": [], "KL_w": [], "KL_c": []}
    with open(path) as f:
        r = csv.reader(f); header = next(r)
        for row in r:
            cols["t"].append(float(row[0]))
            cols["L1_h"].append(float(row[1]))
            cols["L1_w"].append(float(row[2]))
            cols["L1_c"].append(float(row[3]))
            cols["KL_h"].append(float(row[4]))
            cols["KL_w"].append(float(row[5]))
            cols["KL_c"].append(float(row[6]))
    return {k: np.asarray(v) for k, v in cols.items()}

def double_well(x, U0=6.0, h=0.8, L=1.0):
    u = x / L
    return U0 * (u**4 - 2 * u**2) + h * u

def boltzmann_1d(xs, T, U_func):
    U = U_func(xs)
    p = np.exp(-(U - U.min()) / T)
    p /= np.trapezoid(p, xs)
    return p

def find_crossover(t, y_h, y_w):
    """Find first t such that y_h drops below y_w (Mpemba crossover)."""
    for i in range(1, len(t)):
        if y_h[i-1] > y_w[i-1] and y_h[i] <= y_w[i]:
            return t[i], y_h[i], y_w[i]
    return None

def main(csv_path="results/langevin/distances.csv",
         out="results/langevin/plot_langevin.png",
         U0=6.0, h=0.8, Th=1000.0, Tw=12.0, Tc=1.0):
    if not os.path.exists(csv_path):
        print(f"missing {csv_path}", file=sys.stderr); return 1
    d = load_distances(csv_path)

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.27)
    ax_L1 = fig.add_subplot(gs[0, 0])
    ax_KL = fig.add_subplot(gs[0, 1])
    ax_U  = fig.add_subplot(gs[1, 0])
    ax_pi = fig.add_subplot(gs[1, 1])

    # --- D_L1(t) ---
    ax_L1.plot(d["t"], d["L1_h"], '-', color='crimson',  lw=2.0, label=f"$T_h={Th:g}$ (hot)")
    ax_L1.plot(d["t"], d["L1_w"], '-', color='orange',   lw=2.0, label=f"$T_w={Tw:g}$ (warm)")
    ax_L1.plot(d["t"], d["L1_c"], '-', color='steelblue',lw=2.0, label=f"$T_c={Tc:g}$ (cold)")
    cross = find_crossover(d["t"], d["L1_h"], d["L1_w"])
    if cross:
        tc, _, _ = cross
        ax_L1.axvline(tc, color='k', ls='--', alpha=0.5)
        ax_L1.text(tc * 1.05, 0.7 * max(d["L1_h"].max(), d["L1_w"].max()),
                   f"Mpemba\ncrossover\n$t \\approx {tc:.4f}$",
                   fontsize=10)
    ax_L1.set_xlabel("time")
    ax_L1.set_ylabel("$D_{L_1}(t)$  [L1 distance to $\\pi_b$]")
    ax_L1.set_title("(a) $L_1$ distance: hot crosses below warm = ME")
    ax_L1.set_yscale("log")
    ax_L1.legend(fontsize=9, loc="lower left")
    ax_L1.grid(True, alpha=0.3)

    # --- D_KL(t) ---
    ax_KL.plot(d["t"], d["KL_h"], '-', color='crimson',  lw=2.0, label="hot")
    ax_KL.plot(d["t"], d["KL_w"], '-', color='orange',   lw=2.0, label="warm")
    ax_KL.plot(d["t"], d["KL_c"], '-', color='steelblue',lw=2.0, label="cold")
    cross_KL = find_crossover(d["t"], d["KL_h"], d["KL_w"])
    if cross_KL:
        ax_KL.axvline(cross_KL[0], color='k', ls='--', alpha=0.5)
    ax_KL.set_xlabel("time")
    ax_KL.set_ylabel("$D_{KL}(t)$")
    ax_KL.set_title("(b) Kullback-Leibler — same physics, independent metric")
    ax_KL.set_yscale("log")
    ax_KL.legend(fontsize=9)
    ax_KL.grid(True, alpha=0.3)

    # --- Potential U(x) ---
    xs = np.linspace(-3, 3, 400)
    Us = double_well(xs, U0=U0, h=h)
    ax_U.plot(xs, Us, 'k-', lw=2)
    ax_U.set_xlabel("$x$")
    ax_U.set_ylabel("$U(x)$")
    ax_U.set_title(f"(c) Asymmetric double-well  ($U_0={U0}, h={h}$)")
    ax_U.axhline(0, color='gray', lw=0.5)
    # Mark the two wells
    # roots of dU/dx = 4 U0 (u^3 - u) + h = 0 (with L=1)
    # for small h these are near u = -1, +1
    ax_U.annotate("ground", xy=(-1.05, -U0 + h*(-1)), xytext=(-2.6, -2),
                  arrowprops=dict(arrowstyle="->"), fontsize=10)
    ax_U.annotate("metastable", xy=(0.95, -U0 + h*1), xytext=(1.4, 0.5),
                  arrowprops=dict(arrowstyle="->"), fontsize=10)
    ax_U.grid(True, alpha=0.3)

    # --- Initial distributions ---
    xs2 = np.linspace(-3, 3, 400)
    Uf  = lambda x: double_well(x, U0=U0, h=h)
    pih = boltzmann_1d(xs2, Th, Uf)
    piw = boltzmann_1d(xs2, Tw, Uf)
    pic = boltzmann_1d(xs2, Tc, Uf)
    ax_pi.plot(xs2, pih, color='crimson',   lw=2.0, label=f"$\\pi(x; T_h={Th:g})$")
    ax_pi.plot(xs2, piw, color='orange',    lw=2.0, label=f"$\\pi(x; T_w={Tw:g})$")
    ax_pi.plot(xs2, pic, color='steelblue', lw=2.0, label=f"$\\pi(x; T_c={Tc:g})$")
    ax_pi.set_xlabel("$x$")
    ax_pi.set_ylabel("$\\pi(x;T)$")
    ax_pi.set_title("(d) Initial Boltzmann distributions")
    ax_pi.legend(fontsize=9)
    ax_pi.grid(True, alpha=0.3)

    fig.suptitle("Mpemba effect in a colloidal Langevin system\n"
                 "(Kumar & Bechhoefer, Nature 584, 64 (2020))",
                 fontsize=13, fontweight='bold')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f"saved: {out}")
    if cross:
        print(f"D_L1 Mpemba crossover at t={cross[0]:.4g}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
