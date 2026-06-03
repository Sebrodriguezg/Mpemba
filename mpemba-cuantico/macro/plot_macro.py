#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_macro.py -- figura del entregable 8: del Mpemba macro clasico al cuantico.

Muestra la MISMA firma (cruce de curvas de relajacion) en tres dominios usando
los datos experimentales/numericos estandarizados del repositorio:
  (a) coloide clasico en trampa optica (Kumar 2022, PNAS)
  (b) circuito cuantico aleatorio (Turkeshi 2024) -- asimetria de entrelazamiento
y, como puente, (c) la simulacion cuantica propia (Ising disipativo, este trabajo).
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DATA = os.path.join(HERE, "..", "..", "mpemba_data", "standardized")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))

# (a) coloide clasico (Kumar) -- distancia al equilibrio D(t)
k = pd.read_csv(os.path.join(DATA, "kumar2022_colloid.csv"))
cmap = plt.get_cmap("coolwarm")
T0s = sorted(k["protocol_value"].unique())
for T0 in T0s:
    g = k[k["protocol_value"] == T0].sort_values("x")
    c = cmap((np.log10(T0) - np.log10(min(T0s))) / (np.log10(max(T0s)) - np.log10(min(T0s))))
    ax[0].plot(g["x"], g["obs"], color=c, lw=1.4, label=f"$T_0$={T0:g}")
ax[0].set_xlabel("tiempo (ms)"); ax[0].set_ylabel("distancia al equilibrio $D$")
ax[0].set_title("(a) Coloide clasico (Kumar 2022)\nexperimental, trampa optica")
ax[0].legend(fontsize=6, ncol=2); ax[0].grid(alpha=0.3)

# (b) cuantico (Turkeshi) -- asimetria de entrelazamiento
t = pd.read_csv(os.path.join(DATA, "turkeshi2024_quantum.csv"))
for th in sorted(t["protocol_value"].unique()):
    g = t[t["protocol_value"] == th].sort_values("x")
    ax[1].plot(g["x"], g["obs"], lw=1.6, label=rf"$\theta$={th:g}")
ax[1].set_xlabel("pasos del circuito"); ax[1].set_ylabel(r"asimetria de entrelazamiento $\Delta\tilde S_2$")
ax[1].set_title("(b) Circuito cuantico (Turkeshi 2024)\nmuchos cuerpos, numerico")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

# (c) puente: simulacion cuantica propia (Mpemba fuerte espectral, context)
sys.path.insert(0, os.path.join(HERE, "..", "context", "python"))
try:
    import models, qmpe
    H, Ls, info = models.lambda_three_level(omega=1.0, gamma1=1.5, gamma2=0.3, T=0.6)
    sp = qmpe.Spectrum(H, Ls)
    rss = sp.rho_ss
    real_idx = [kk for kk in range(1, sp.d * sp.d) if abs(sp.eigvals[kk].imag) < 1e-6]
    ks, kf = real_idx[0], real_idx[-1]
    def herm(kk):
        r = sp.r_ops[kk]; r = 0.5 * (r + r.conj().T); r -= np.trace(r) / sp.d * np.eye(sp.d)
        return r / np.sqrt(np.real(np.trace(r @ r)))
    ev = np.linalg.eigvalsh(rss).min()
    rf, rs = herm(kf), herm(ks)
    cf = 0.85 * ev / max(np.abs(np.linalg.eigvalsh(rf)).max(), 1e-9)
    cs = 0.85 * ev / max(np.abs(np.linalg.eigvalsh(rs)).max(), 1e-9)
    rho_hot, rho_warm = rss + cf * rf, rss + 0.5 * cs * rs
    times = np.linspace(0, 6, 300)
    Dh = [qmpe.trace_distance(r, rss) for r in qmpe.evolve_spectral(sp, rho_hot, times)]
    Dw = [qmpe.trace_distance(r, rss) for r in qmpe.evolve_spectral(sp, rho_warm, times)]
    ax[2].semilogy(times, Dh, "r-", lw=1.8, label="caliente (modo rapido)")
    ax[2].semilogy(times, Dw, "b-", lw=1.8, label="templado (modo lento)")
    ax[2].set_xlabel("t"); ax[2].set_ylabel(r"$D_{tr}(\rho_t\|\rho_{ss})$")
    ax[2].set_title("(c) Simulacion cuantica propia\nMpemba fuerte (geometria espectral)")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3, which="both")
except Exception as e:
    ax[2].text(0.5, 0.5, f"(c) no disponible:\n{e}", ha="center", va="center", transform=ax[2].transAxes)

fig.suptitle("La firma universal del efecto Mpemba: cruce de curvas de relajacion, "
             "del macro clasico al cuantico", fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(RES, "fig_macro_universal.png"), dpi=140)
print("-> results/fig_macro_universal.png")
