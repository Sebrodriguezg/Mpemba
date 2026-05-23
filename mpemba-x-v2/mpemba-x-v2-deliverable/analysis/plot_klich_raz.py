#!/usr/bin/env python3
"""Plot Klich-Raz Mpemba-index statistics.

Reads results/klich_raz/index_stats_L<L>_Tb<Tb>.csv and a2_sample.csv.
Produces:
  * histogram of the Mpemba index across REM realizations
  * a_2(T) curve for a representative sample, showing zeros = index increments
"""
import os
import csv
import glob
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_index_stats(p):
    idx, cnt, frac = [], [], []
    with open(p) as f:
        r = csv.reader(f); next(r)
        for row in r:
            idx.append(int(float(row[0])))
            cnt.append(int(float(row[1])))
            frac.append(float(row[2]))
    return np.asarray(idx), np.asarray(cnt), np.asarray(frac)

def load_a2_sample(p):
    Ts, a2s = [], []
    with open(p) as f:
        r = csv.reader(f); next(r)
        for row in r:
            Ts.append(float(row[0]))
            a2s.append(float(row[1]))
    return np.asarray(Ts), np.asarray(a2s)

def main(results_dir="results/klich_raz", out="results/klich_raz/plot_klich_raz.png"):
    stat_files = glob.glob(os.path.join(results_dir, "index_stats_*.csv"))
    if not stat_files:
        print(f"no index_stats in {results_dir}", file=sys.stderr); return 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Index histogram ---
    p = stat_files[0]
    base = os.path.basename(p)
    idx, cnt, frac = load_index_stats(p)
    # only show indices that occurred
    mask = cnt > 0
    bars = axes[0].bar(idx[mask], frac[mask], width=0.7,
                       color='steelblue', edgecolor='k', alpha=0.85)
    for b, c, fr in zip(bars, cnt[mask], frac[mask]):
        axes[0].text(b.get_x() + b.get_width() / 2,
                     b.get_height() * 1.04,
                     f"{c}\n({fr*100:.2f}%)",
                     ha='center', fontsize=9)
    axes[0].set_xlabel("Mpemba index $I_M$")
    axes[0].set_ylabel("fraction of realizations")
    axes[0].set_title(f"(a) Mpemba index histogram\n({base})")
    axes[0].set_yscale("log")
    axes[0].set_xticks(idx[mask])
    axes[0].grid(True, alpha=0.3, axis='y')

    # --- a_2(T) sample trace ---
    sample = os.path.join(results_dir, "a2_sample.csv")
    if os.path.exists(sample):
        T, a2 = load_a2_sample(sample)
        axes[1].plot(T, a2, 'k-', lw=1.6)
        axes[1].axhline(0, color='red', ls='--', lw=1)
        # mark zero crossings
        n_zeros = 0
        for i in range(1, len(a2)):
            if (a2[i-1] > 0) != (a2[i] > 0):
                T_zero = T[i-1] + (T[i] - T[i-1]) * abs(a2[i-1]) / (abs(a2[i-1]) + abs(a2[i]))
                axes[1].axvline(T_zero, color='orange', ls=':', alpha=0.7)
                n_zeros += 1
        axes[1].set_xscale("log")
        axes[1].set_xlabel("initial temperature $T$")
        axes[1].set_ylabel("$a_2(T)$ — overlap with slowest mode")
        axes[1].set_title(f"(b) $a_2(T)$ for one REM realization — {n_zeros} zero(s)")
        axes[1].grid(True, alpha=0.3)

    fig.suptitle("Strong Mpemba effect — index statistics\n"
                 "(Klich, Raz, Hirschberg & Vucelja, PRX 9, 021060 (2019))",
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=140)
    print(f"saved: {out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
