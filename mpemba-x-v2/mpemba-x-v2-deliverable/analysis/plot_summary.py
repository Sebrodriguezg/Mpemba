#!/usr/bin/env python3
"""Master summary figure -- the Mpemba effect across 9 theoretical frameworks.

Reads results from all modules and produces a 3x3 grid:
  (a) Markovian (Lu-Raz)
  (b) Klich-Raz Mpemba index
  (c) Lasanta granular analytic
  (d) Langevin colloid (Kumar-Bechhoefer)
  (e) Inverse Mpemba (Kumar-Chetrite)
  (f) Quantum Lindblad QFI (Carollo + Chattopadhyay)
  (g) Water Fourier (Zhang)
  (h) Water MD (Jin-Goddard)
  (i) Thermomajorization universal diagnostic (Vu-Hayakawa)
"""
import os, sys, csv, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_csv(p, has_header=True):
    with open(p) as f:
        rdr = csv.reader(f)
        if has_header: next(rdr)
        return np.array([list(map(float, r)) for r in rdr])

def main(root="results", out="results/summary_mpemba_x_v2.png"):
    fig, axes = plt.subplots(3, 3, figsize=(17, 14))

    # ====== (a) Markovian ======
    ax = axes[0, 0]
    files = sorted(glob.glob(os.path.join(root, "markovian", "three_state_Tinit_*.csv")))
    Ts = []
    for f in files:
        T = float(f.split("Tinit_")[1].rstrip(".csv"))
        Ts.append((T, f))
    Ts.sort()
    Tmin, Tmax = Ts[0][0], Ts[-1][0]
    cmap = plt.get_cmap("plasma")
    for T, f in Ts:
        arr = load_csv(f)
        c = cmap((T - Tmin) / max(1e-9, Tmax - Tmin))
        ax.plot(arr[:, 0], arr[:, 3], color=c, lw=1.4, label=f"$T_0={T}$")
    ax.set_yscale("log")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$D_e(t)$")
    ax.set_title("(a) Markovian (Lu & Raz, PNAS 2017)\n3-state Arrhenius system")
    ax.axvline(22.4, color='k', ls='--', alpha=0.4)
    ax.text(24, 1.5, "ME\ncrossover", fontsize=9)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # ====== (b) Klich-Raz ======
    ax = axes[0, 1]
    p = os.path.join(root, "klich_raz", "a2_sample.csv")
    if os.path.exists(p):
        arr = load_csv(p)
        ax.plot(arr[:, 0], arr[:, 1], 'k-', lw=2)
        ax.axhline(0, color='red', ls='--', lw=1)
        for i in range(1, len(arr)):
            if (arr[i-1, 1] > 0) != (arr[i, 1] > 0):
                Tz = arr[i-1, 0] + (arr[i, 0] - arr[i-1, 0]) * abs(arr[i-1, 1]) / (abs(arr[i-1, 1]) + abs(arr[i, 1]))
                ax.axvline(Tz, color='orange', ls=':', lw=2)
                ax.text(Tz*1.5, max(arr[:, 1])*0.5, f"sign change\n$T^*\\approx{Tz:.2f}$", fontsize=9)
                break
    ax.set_xscale("log")
    ax.set_xlabel("initial $T$")
    ax.set_ylabel("$a_2(T)$")
    ax.set_title("(b) Strong Mpemba index\n(Klich, Raz et al. PRX 2019)")
    ax.grid(True, alpha=0.3)

    # ====== (c) Granular analytic ======
    ax = axes[0, 2]
    fA = os.path.join(root, "granular_analytic", "T_a2_A.csv")
    fB = os.path.join(root, "granular_analytic", "T_a2_B.csv")
    if os.path.exists(fA) and os.path.exists(fB):
        a = load_csv(fA); b = load_csv(fB)
        ax.plot(a[:, 0], a[:, 1], color='crimson',   lw=2,
                label='$A$: $T=1, a_2=+0.5$')
        ax.plot(b[:, 0], b[:, 1], color='steelblue', lw=2,
                label='$B$: $T=0.99, a_2=-0.35$')
        for i in range(1, len(a)):
            if a[i-1, 1] > b[i-1, 1] and a[i, 1] <= b[i, 1]:
                ax.axvline(a[i, 0], color='k', ls='--', alpha=0.5)
                ax.text(a[i, 0]*1.1, 0.7,
                        f"$t^*\\approx{a[i, 0]:.2f}$", fontsize=9); break
    ax.set_xlabel("time")
    ax.set_ylabel("granular $T$")
    ax.set_title("(c) Granular fluid (Lasanta et al. PRL 2017)\nanalytic moment ODE")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ====== (d) Langevin forward ======
    ax = axes[1, 0]
    p = os.path.join(root, "langevin", "distances.csv")
    if os.path.exists(p):
        arr = load_csv(p)
        ax.semilogy(arr[:, 0], arr[:, 1], color='crimson',   lw=2, label='$T_h=1000$')
        ax.semilogy(arr[:, 0], arr[:, 2], color='orange',    lw=2, label='$T_w=12$')
        ax.semilogy(arr[:, 0], arr[:, 3], color='steelblue', lw=2, label='$T_c=1$')
        for i in range(1, len(arr)):
            if arr[i-1, 1] > arr[i-1, 2] and arr[i, 1] <= arr[i, 2]:
                ax.axvline(arr[i, 0], color='k', ls='--', alpha=0.5)
                ax.text(arr[i, 0]*1.5, 1.0,
                        f"$t^*\\approx{arr[i, 0]:.3f}$", fontsize=9); break
    ax.set_xlabel("time")
    ax.set_ylabel(r"$D_{L_1}(t)$")
    ax.set_title("(d) Langevin colloid\n(Kumar & Bechhoefer, Nature 2020)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # ====== (e) Inverse Mpemba ======
    ax = axes[1, 1]
    p = os.path.join(root, "langevin_inverse", "distances.csv")
    if os.path.exists(p):
        arr = load_csv(p)
        ax.semilogy(arr[:, 0], arr[:, 1], color='steelblue', lw=2, label='$T_{cold}$ init')
        ax.semilogy(arr[:, 0], arr[:, 2], color='orange',    lw=2, label='$T_{cool}$ init')
    ax.set_xlabel("time")
    ax.set_ylabel(r"$D_{L_1}(t)$")
    ax.set_title("(e) Anomalous heating (Kumar, Chetrite,\nBechhoefer PNAS 2022)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # ====== (f) Quantum Lindblad ======
    ax = axes[1, 2]
    qfiles = sorted(glob.glob(os.path.join(root, "quantum_lindblad", "qubit_Tinit_*.csv")))
    Ts_q = []
    for f in qfiles:
        m = re.search(r'qubit_Tinit_([\d.]+?)\.csv', f)
        Ts_q.append((float(m.group(1)), f))
    Ts_q.sort()
    if Ts_q:
        Tmin_q, Tmax_q = Ts_q[0][0], Ts_q[-1][0]
        cmap2 = plt.get_cmap("plasma")
        for T, f in Ts_q:
            arr = load_csv(f)
            c = cmap2((np.log10(T) - np.log10(Tmin_q)) /
                      max(1e-9, np.log10(Tmax_q) - np.log10(Tmin_q)))
            ax.plot(arr[:, 0], arr[:, 2], color=c, lw=1.6, label=f"$T_0={T}$")
        ax.axhline(arr[-1, 2], color='k', ls='--', alpha=0.5,
                   label=f"eq. = {arr[-1, 2]:.3f}")
    ax.set_xlabel("time")
    ax.set_ylabel("$F_T(t)$")
    ax.set_title("(f) Metrological Mpemba\n(Carollo 2021 + Chattopadhyay 2026)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ====== (g) Water Fourier ======
    ax = axes[2, 0]
    for sub in ["water_demo", "water_fourier"]:
        ph = os.path.join(root, sub, "timeseries_hot.csv")
        pc = os.path.join(root, sub, "timeseries_cold.csv")
        if os.path.exists(ph) and os.path.exists(pc):
            arr_h = load_csv(ph); arr_c = load_csv(pc)
            ax.semilogy(arr_h[:, 0], arr_h[:, 6], color='crimson',
                        lw=2, label=f'hot ($T_0={arr_h[0,1]:.0f}$°C)')
            ax.semilogy(arr_c[:, 0], arr_c[:, 6], color='steelblue',
                        lw=2, label=f'cold ($T_0={arr_c[0,1]:.0f}$°C)')
            for i in range(1, len(arr_h)):
                if arr_h[i-1, 6] > arr_c[i-1, 6] and arr_h[i, 6] <= arr_c[i, 6]:
                    ax.axvline(arr_h[i, 0], color='k', ls='--', alpha=0.5)
                    ax.text(arr_h[i, 0]*1.05, 1.0,
                            f"$t^*\\approx{arr_h[i, 0]:.0f}$ s", fontsize=9); break
            break
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"$\sqrt{\langle(T-T_b)^2\rangle}$ (°C)")
    ax.set_title("(g) Water Fourier (Zhang PCCP 2014)\nfreezer experiment")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # ====== (h) Water MD ======
    ax = axes[2, 1]
    md_files = sorted(glob.glob(os.path.join(root, "water_md", "md_Tinit_*.csv")))
    cmap3 = plt.get_cmap("coolwarm_r")
    if md_files:
        Tmin_K = min(float(re.search(r'Tinit_([\d.]+)K', f).group(1)) for f in md_files)
        Tmax_K = max(float(re.search(r'Tinit_([\d.]+)K', f).group(1)) for f in md_files)
        for f in md_files:
            T = float(re.search(r'Tinit_([\d.]+)K', f).group(1))
            arr = load_csv(f)
            c = cmap3((T - Tmin_K) / max(1e-9, Tmax_K - Tmin_K))
            ax.plot(arr[:, 1], arr[:, 4], color=c, lw=1.6,
                    label=f"$T_0={T:.0f}$ K")
    ax.set_xlabel("time (reduced units)")
    ax.set_ylabel("coordination number")
    ax.set_title("(h) Water MD (Jin & Goddard JPCC 2015)\ncoarse-grained mW potential")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ====== (i) Thermomajorization (universal) ======
    ax = axes[2, 2]
    p = os.path.join(root, "markovian", "thermomajorization_diagnostic.csv")
    if os.path.exists(p):
        arr = load_csv(p)
        ax.plot(arr[:, 0], arr[:, 1], color='red',  lw=1.8,
                label='max gap (hot-cold)')
        ax.plot(arr[:, 0], arr[:, 2], color='blue', lw=1.8,
                label='min gap (hot-cold)')
        ax.axhline(0, color='k', lw=0.5)
        # Highlight where curves cross (effect ambiguous in some metrics)
        cross = arr[:, 4]
        for i in range(1, len(arr)):
            if cross[i] == 1 and cross[i-1] == 0:
                ax.axvline(arr[i, 0], color='orange', ls=':', alpha=0.6)
                break
    ax.set_xlabel("time")
    ax.set_ylabel("thermomaj gap")
    ax.set_title("(i) Universal diagnostic\n(Vu & Hayakawa PRL 2025)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Mpemba-X v2 — the Mpemba effect across nine theoretical frameworks\n"
                 "from classical Markov dynamics to quantum thermometry and the thermomajorization unification",
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)
    print(f"saved {out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
