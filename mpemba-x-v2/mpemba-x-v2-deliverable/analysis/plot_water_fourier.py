#!/usr/bin/env python3
"""Visualize the water Fourier Mpemba experiment.

Reads results/water_*/field_hot.csv and field_cold.csv (heatmap data),
and timeseries_*.csv (averaged quantities).

Produces:
  * heatmap T(x, t) for hot and cold preparations
  * time evolution of T_mean, mass, enthalpy
  * RMS distance to bath: cooling curves on log scale + crossover detection
  * combined Mpemba diagnostic figure
"""
import os, sys, csv, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def load_field(path):
    with open(path) as f:
        rdr = csv.reader(f)
        header = next(rdr)
        xs = np.array([float(x) for x in header[1:]])
        rows = [list(map(float, row)) for row in rdr]
    arr = np.array(rows)
    ts = arr[:, 0]
    T_xt = arr[:, 1:]
    return ts, xs, T_xt

def load_ts(path):
    with open(path) as f:
        rdr = csv.reader(f)
        header = next(rdr)
        rows = [list(map(float, row)) for row in rdr]
    return header, np.array(rows)

def find_crossover(t, y_hot, y_cold):
    for i in range(1, len(t)):
        if y_hot[i-1] > y_cold[i-1] and y_hot[i] <= y_cold[i]:
            return t[i]
    return None

def main(results_dir="results/water_demo", out_png=None):
    if out_png is None:
        out_png = os.path.join(results_dir, "water_mpemba_summary.png")

    t_hot, x_hot, T_hot = load_field(os.path.join(results_dir, "field_hot.csv"))
    t_cold, x_cold, T_cold = load_field(os.path.join(results_dir, "field_cold.csv"))
    head_h, ts_h = load_ts(os.path.join(results_dir, "timeseries_hot.csv"))
    head_c, ts_c = load_ts(os.path.join(results_dir, "timeseries_cold.csv"))

    # Find column indices
    col = {name: i for i, name in enumerate(head_h)}
    t_col = col["t"]
    T_mean = col["T_mean_C"]
    mass = col["mass_kg"]
    enthalpy = col["enthalpy_J"]
    rms = col["rms_dist_to_bath_C"]

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.30)

    # --- Heatmap T(x, t) hot ---
    ax = fig.add_subplot(gs[0, 0])
    im = ax.pcolormesh(x_hot * 100, t_hot, T_hot, cmap='hot', shading='auto')
    plt.colorbar(im, ax=ax, label='T (°C)')
    ax.set_xlabel("position $x$ (cm)")
    ax.set_ylabel("time (s)")
    ax.set_title("(a) Hot sample $T(x,t)$")
    ax.invert_yaxis()

    # --- Heatmap T(x, t) cold ---
    ax = fig.add_subplot(gs[0, 1])
    im = ax.pcolormesh(x_cold * 100, t_cold, T_cold, cmap='hot', shading='auto')
    plt.colorbar(im, ax=ax, label='T (°C)')
    ax.set_xlabel("position $x$ (cm)")
    ax.set_ylabel("time (s)")
    ax.set_title("(b) Cold sample $T(x,t)$")
    ax.invert_yaxis()

    # --- Profile snapshots at selected times ---
    ax = fig.add_subplot(gs[0, 2])
    snapshot_idx = [0, len(t_hot)//10, len(t_hot)//4, len(t_hot)//2, -1]
    cmap_h = plt.cm.Reds
    cmap_c = plt.cm.Blues
    for k, idx in enumerate(snapshot_idx):
        alpha = (k+1) / len(snapshot_idx)
        ax.plot(x_hot*100, T_hot[idx], color=cmap_h(0.4 + 0.6*alpha),
                lw=1.6, label=f'hot t={t_hot[idx]:.0f}s' if k == 0 or k == len(snapshot_idx)-1 else None)
        ax.plot(x_cold*100, T_cold[idx], color=cmap_c(0.4 + 0.6*alpha),
                lw=1.6, ls='--', label=f'cold t={t_cold[idx]:.0f}s' if k == 0 or k == len(snapshot_idx)-1 else None)
    ax.set_xlabel("position $x$ (cm)")
    ax.set_ylabel("T (°C)")
    ax.set_title("(c) Profile snapshots\n(darker = later)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Mean T(t) ---
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(ts_h[:, t_col], ts_h[:, T_mean], color='crimson',  lw=2,
            label=f'hot ($T_0={ts_h[0, T_mean]:.0f}$°C)')
    ax.plot(ts_c[:, t_col], ts_c[:, T_mean], color='steelblue', lw=2,
            label=f'cold ($T_0={ts_c[0, T_mean]:.0f}$°C)')
    ax.set_xlabel("time (s)")
    ax.set_ylabel("$\\langle T \\rangle$ (°C)")
    ax.set_title("(d) Spatial-mean temperature")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Mass conservation check ---
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(ts_h[:, t_col], ts_h[:, mass]*1000, color='crimson',  lw=2, label='hot')
    ax.plot(ts_c[:, t_col], ts_c[:, mass]*1000, color='steelblue', lw=2, label='cold')
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mass (g)")
    ax.set_title("(e) Mass conservation\n(varies via $\\rho(T)$)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Enthalpy ---
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(ts_h[:, t_col], ts_h[:, enthalpy]/1000, color='crimson',  lw=2, label='hot')
    ax.plot(ts_c[:, t_col], ts_c[:, enthalpy]/1000, color='steelblue', lw=2, label='cold')
    ax.set_xlabel("time (s)")
    ax.set_ylabel("enthalpy (kJ)")
    ax.set_title("(f) Total enthalpy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- RMS distance to bath: cooling curve in log scale ---
    ax = fig.add_subplot(gs[2, :2])
    ax.semilogy(ts_h[:, t_col], ts_h[:, rms], color='crimson',  lw=2,
                label=f'hot init ($T_0={ts_h[0, T_mean]:.0f}$°C)')
    ax.semilogy(ts_c[:, t_col], ts_c[:, rms], color='steelblue', lw=2,
                label=f'cold init ($T_0={ts_c[0, T_mean]:.0f}$°C)')
    tcross = find_crossover(ts_h[:, t_col], ts_h[:, rms], ts_c[:, rms])
    if tcross is not None:
        ax.axvline(tcross, color='k', ls='--', alpha=0.6)
        ax.text(tcross*1.05, 0.5*ts_h[ts_h.shape[0]//2, rms],
                f"Mpemba crossover\n$t^* \\approx {tcross:.0f}$ s",
                fontsize=11)
        title = "(g) RMS distance to bath -- MPEMBA crossover detected"
    else:
        title = "(g) RMS distance to bath (no crossover; classical cooling)"
    ax.set_xlabel("time (s)")
    ax.set_ylabel("$\\sqrt{\\langle (T - T_{bath})^2 \\rangle}$ (°C)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # --- Energy lost ratio (Burridge's diagnostic) ---
    ax = fig.add_subplot(gs[2, 2])
    H_hot_init  = ts_h[0, enthalpy]
    H_cold_init = ts_c[0, enthalpy]
    H_hot  = ts_h[:, enthalpy] - ts_h[-1, enthalpy]
    H_cold = ts_c[:, enthalpy] - ts_c[-1, enthalpy]
    # Burridge-Linden Mpemba line: heat flux ratio vs enthalpy ratio
    dE_hot  = H_hot_init  - ts_h[-1, enthalpy]
    dE_cold = H_cold_init - ts_c[-1, enthalpy]
    ratio = dE_hot / dE_cold if dE_cold != 0 else 0
    ax.bar(['hot $\\Delta E$', 'cold $\\Delta E$'], [dE_hot/1000, dE_cold/1000],
           color=['crimson', 'steelblue'])
    ax.set_ylabel("enthalpy drop (kJ)")
    ax.set_title(f"(h) Enthalpy difference\nratio $\\Delta E_H/\\Delta E_C = {ratio:.2f}$")
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle("Mpemba effect in water (Fourier + skin supersolidity + H-bond memory)\n"
                 "Zhang et al., Phys. Chem. Chem. Phys. 16, 22995 (2014)",
                 fontsize=13, fontweight='bold')
    fig.savefig(out_png, dpi=140)
    print(f"saved {out_png}")
    if tcross is not None:
        print(f"  Mpemba crossover at t* = {tcross:.1f} s")
    return 0

if __name__ == "__main__":
    rd = sys.argv[1] if len(sys.argv) > 1 else "results/water_demo"
    sys.exit(main(rd))
