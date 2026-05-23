#!/usr/bin/env python3
"""Plot Lasanta analytic Mpemba effect (granular fluid, analytic ODE).

Reads results/granular_analytic/T_a2_A.csv and T_a2_B.csv.
Reproduces Fig. 4 of Lasanta et al., PRL 119, 148001 (2017).
"""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(p):
    t, T, a2 = [], [], []
    with open(p) as f:
        r = csv.reader(f); next(r)
        for row in r:
            t.append(float(row[0])); T.append(float(row[1])); a2.append(float(row[2]))
    return np.asarray(t), np.asarray(T), np.asarray(a2)

def main(in_dir="results/granular_analytic",
         out="results/granular_analytic/plot_lasanta_analytic.png"):
    tA, TA, aA = load(os.path.join(in_dir, "T_a2_A.csv"))
    tB, TB, aB = load(os.path.join(in_dir, "T_a2_B.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel (a): T(t)
    axes[0].plot(tA, TA, color='crimson',   lw=2,
                 label='A: $T_0=1.00, a_{2,0}=+0.50$ (hot)')
    axes[0].plot(tB, TB, color='steelblue', lw=2,
                 label='B: $T_0=0.99, a_{2,0}=-0.35$ (cold)')
    cross = None
    for i in range(1, len(tA)):
        if TA[i-1] > TB[i-1] and TA[i] <= TB[i]:
            cross = tA[i]; break
    if cross is not None:
        axes[0].axvline(cross, color='k', ls='--', alpha=0.6)
        axes[0].text(cross * 1.05, 0.7, f"$t^* \\approx {cross:.3f}$",
                     fontsize=11)
        # Inset zoom
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins = inset_axes(axes[0], width="40%", height="40%",
                           loc="upper right")
        mask = (tA >= 0.05) & (tA <= 0.5)
        axins.plot(tA[mask], TA[mask], color='crimson',   lw=1.6)
        axins.plot(tB[mask], TB[mask], color='steelblue', lw=1.6)
        axins.axvline(cross, color='k', ls='--', alpha=0.5)
        axins.set_xlim(0.05, 0.5)
        axins.set_title("crossover zoom", fontsize=9)
        axins.tick_params(labelsize=7)
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("granular temperature $T$")
    axes[0].set_title("(a) Mpemba in granular free cooling")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Panel (b): a_2(t)
    axes[1].plot(tA, aA, color='crimson',   lw=2, label='$a_2$ — A')
    axes[1].plot(tB, aB, color='steelblue', lw=2, label='$a_2$ — B')
    axes[1].axhline(0, color='gray', lw=0.5)
    a2_HCS = aA[-1]  # both converge to a_2^HCS
    axes[1].axhline(a2_HCS, color='green', ls=':', lw=1.5,
                    label=f'$a_2^{{HCS}} \\approx {a2_HCS:.4f}$')
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("excess kurtosis $a_2$")
    axes[1].set_title("(b) $a_2(t)$: both converge to homogeneous cooling state")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Mpemba effect — Lasanta analytic moment equations\n"
                 "(reproduces Fig. 4 of Lasanta et al., PRL 119, 148001 (2017))",
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=140)
    print(f"saved: {out}")
    if cross:
        print(f"Mpemba crossover at t* = {cross:.3f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
