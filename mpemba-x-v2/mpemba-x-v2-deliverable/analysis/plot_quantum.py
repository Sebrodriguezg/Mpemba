#!/usr/bin/env python3
"""Visualize the quantum Lindblad Mpemba effect with QFI tracking.

Shows population p_1(t), trace distance to bath, and QFI for each T_init.
"""
import os, sys, csv, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_csv(path):
    with open(path) as f:
        rdr = csv.reader(f)
        head = next(rdr)
        rows = [list(map(float, r)) for r in rdr]
    return head, np.array(rows)

def main(results_dir="results/quantum_lindblad",
         out=None):
    if out is None: out = os.path.join(results_dir, "quantum_summary.png")
    files = sorted(glob.glob(os.path.join(results_dir, "qubit_Tinit_*.csv")))
    if not files:
        print(f"no qubit files in {results_dir}", file=sys.stderr); return 1
    T_inits = []
    data = {}
    for f in files:
        m = re.search(r'qubit_Tinit_([\d.]+?)\.csv', f)
        T = float(m.group(1))
        head, arr = load_csv(f)
        T_inits.append(T)
        data[T] = (head, arr)
    T_inits.sort()

    Tmin, Tmax = T_inits[0], T_inits[-1]
    cmap = plt.get_cmap("plasma")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- p_1(t) ---
    for T in T_inits:
        head, arr = data[T]
        c = cmap((np.log10(T) - np.log10(Tmin)) /
                 max(1e-9, np.log10(Tmax) - np.log10(Tmin)))
        axes[0].plot(arr[:, 0], arr[:, 1], color=c, lw=1.6, label=f"$T_0={T}$")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("$p_1(t)$")
    axes[0].set_title("(a) Excited-state population")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # --- Trace distance ---
    for T in T_inits:
        head, arr = data[T]
        c = cmap((np.log10(T) - np.log10(Tmin)) /
                 max(1e-9, np.log10(Tmax) - np.log10(Tmin)))
        axes[1].semilogy(arr[:, 0], arr[:, 3], color=c, lw=1.6, label=f"$T_0={T}$")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel(r"$|p_1(t) - p_1^{eq}(T_b)|$")
    axes[1].set_title("(b) Trace distance to bath")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, which='both')

    # --- QFI ---
    eq_qfi = None
    for T in T_inits:
        head, arr = data[T]
        c = cmap((np.log10(T) - np.log10(Tmin)) /
                 max(1e-9, np.log10(Tmax) - np.log10(Tmin)))
        axes[2].plot(arr[:, 0], arr[:, 2], color=c, lw=1.6, label=f"$T_0={T}$")
        eq_qfi = arr[-1, 2]
    if eq_qfi is not None:
        axes[2].axhline(eq_qfi, color='k', ls='--', alpha=0.5,
                        label=f"eq. QFI = {eq_qfi:.3f}")
    axes[2].set_xlabel("time")
    axes[2].set_ylabel("$F_T(t)$  (quantum Fisher information)")
    axes[2].set_title("(c) Metrological Mpemba: QFI > equilibrium\n"
                      "for nonequilibrium initializations")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Quantum Mpemba in a Davies qubit (Carollo 2021) + metrological QFI gain\n"
                 "(Chattopadhyay, Santos, Misra arXiv:2601.05046, 2026)",
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out, dpi=140)
    print(f"saved {out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
