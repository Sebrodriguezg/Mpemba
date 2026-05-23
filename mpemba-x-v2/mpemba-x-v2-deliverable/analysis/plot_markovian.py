#!/usr/bin/env python3
"""Plot Markovian Mpemba effect results (Lu & Raz 2017).

Reads results/markovian/three_state_Tinit_*.csv and plots:
  * D_e (entropic Lu-Raz distance) vs t  for multiple T_init
  * inset showing the crossover region

Run from the project root after compilation:
  ./build/mpemba_markovian ...  (produces CSVs)
  python3 analysis/plot_markovian.py
"""
import os
import glob
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_csv(path):
    times, dL1, dKL, de = [], [], [], []
    with open(path) as f:
        r = csv.reader(f); next(r)
        for row in r:
            times.append(float(row[0]))
            dL1.append(float(row[1]))
            dKL.append(float(row[2]))
            de.append(float(row[3]))
    return np.array(times), np.array(dL1), np.array(dKL), np.array(de)

def main(results_dir="results/markovian", out="results/markovian/plot_markovian.png"):
    files = sorted(glob.glob(os.path.join(results_dir, "three_state_Tinit_*.csv")))
    if not files:
        print(f"no files in {results_dir}", file=sys.stderr); return 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap("plasma")

    # Sort by T_init to color from cold (blue) to hot (red)
    Ts = []
    for f in files:
        T = float(f.split("Tinit_")[1].rstrip(".csv"))
        Ts.append((T, f))
    Ts.sort()
    Tmin, Tmax = Ts[0][0], Ts[-1][0]

    crossings = []
    Thot_data = None
    Tcold_data = None

    for T, f in Ts:
        t, dL1, dKL, de = load_csv(f)
        # Color: cold = blue, hot = red
        c = cmap((T - Tmin) / max(Tmax - Tmin, 1e-9))
        for ax, y, lab in [(axes[0], de, "$D_e$"), (axes[1], dL1, "$D_{L_1}$")]:
            ax.plot(t, y, color=c, lw=1.5, label=f"$T_{{init}}={T}$")
        if T == Tmax:    Thot_data = (t, de)
        if T == Tmin:    Tcold_data = (t, de)

    for ax, ylabel in [(axes[0], "$D_e$ (Lu-Raz entropic)"),
                       (axes[1], "$D_{L_1}$")]:
        ax.set_xlabel("time")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.set_title(ylabel + " vs $t$")
        ax.legend(fontsize=7, loc="best", ncol=2)
        ax.grid(True, alpha=0.3)

    # Annotate crossover on D_e panel
    if Thot_data is not None and Tcold_data is not None:
        t, deh = Thot_data
        _, dec = Tcold_data
        # Find crossover
        for i in range(1, len(t)):
            if (deh[i-1] > dec[i-1]) and (deh[i] <= dec[i]):
                tc = t[i]
                axes[0].axvline(tc, color='k', linestyle='--', alpha=0.6)
                axes[0].annotate(f"Mpemba crossover\nat $t \\approx {tc:.1f}$",
                                 xy=(tc, deh[i]), xytext=(tc + 5, deh[i] * 2.5),
                                 fontsize=10,
                                 arrowprops=dict(arrowstyle="->", color='k'))
                break

    fig.suptitle("Markovian Mpemba effect — 3-state model (Lu & Raz, PNAS 2017)\n"
                 f"Bath $T_b = 0.05$,  energies $E=\\{{0,0.1,0.7\\}}$,  "
                 f"barriers $B_{{12}}=1.5,B_{{13}}=0.8,B_{{23}}=1.2$",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=140)
    print(f"saved: {out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
