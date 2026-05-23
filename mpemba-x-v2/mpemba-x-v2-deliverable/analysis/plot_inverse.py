#!/usr/bin/env python3
"""Visualize anomalous heating (inverse Mpemba effect, Kumar-Chetrite 2022).

Reads results/langevin_inverse/distances.csv with columns:
    t, D_L1_cold, D_L1_cool, D_KL_cold, D_KL_cool
"""
import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(path):
    with open(path) as f:
        rdr = csv.reader(f); next(rdr)
        rows = [list(map(float, r)) for r in rdr]
    return np.array(rows)

def main(results_dir="results/langevin_inverse",
         out=None, T_cold=0.05, T_cool=0.3, T_bath=1.0):
    if out is None: out = os.path.join(results_dir, "inverse_summary.png")
    p = os.path.join(results_dir, "distances.csv")
    if not os.path.exists(p):
        print(f"no distances.csv in {results_dir}", file=sys.stderr); return 1
    arr = load(p)
    t = arr[:, 0]
    L1_cold = arr[:, 1]; L1_cool = arr[:, 2]
    KL_cold = arr[:, 3]; KL_cool = arr[:, 4]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, y_cold, y_cool, name in [(axes[0], L1_cold, L1_cool, "L_1"),
                                       (axes[1], KL_cold, KL_cool, "KL")]:
        ax.semilogy(t, y_cold, color='steelblue', lw=2,
                    label=f"cold init ($T_0={T_cold}$, $T_b={T_bath}$)")
        ax.semilogy(t, y_cool, color='orange', lw=2,
                    label=f"cool init ($T_0={T_cool}$, $T_b={T_bath}$)")
        # Look for inverse crossover: cold dips below cool while heating
        cross = None
        for i in range(1, len(t)):
            if y_cold[i-1] > y_cool[i-1] and y_cold[i] < y_cool[i]:
                cross = t[i]; break
        if cross is not None:
            ax.axvline(cross, color='k', ls='--', alpha=0.5)
            ax.text(cross*1.05, max(y_cold[0], y_cool[0])*0.5,
                    f"inverse ME\n$t^* \\approx {cross:.3f}$",
                    fontsize=10)
            ax.set_title(f"({name}) inverse Mpemba: cold beats cool")
        else:
            ax.set_title(f"({name}) no inverse crossover (effect is weaker)")
        ax.set_xlabel("time")
        ax.set_ylabel(f"$D_{{{name}}}(t)$")
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')

    fig.suptitle("Inverse Mpemba: anomalous HEATING in a colloidal system\n"
                 "(Kumar, Chetrite & Bechhoefer, PNAS 119, e2118484119 (2022))",
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out, dpi=140)
    print(f"saved {out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
