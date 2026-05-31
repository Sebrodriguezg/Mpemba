#!/usr/bin/env python3
"""Grafica los datos estandarizados de los 3 papers para observar el efecto Mpemba.

Genera dos figuras en mpemba_data/standardized/:
  - mpemba_comparison.png : 3 paneles (un paper por panel) con las curvas de
        relajación. El efecto Mpemba = cruce de curvas: una condición inicial
        más alejada del equilibrio lo alcanza antes que otra más cercana.
  - mpemba_water_diagnostic.png : diagnóstico experimental del agua (Hallstadius):
        tiempo en alcanzar -5 °C para la muestra caliente vs la fría de cada
        ensayo "DifferentTemp". Cuando caliente < fría => efecto Mpemba.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    return pd.read_csv(os.path.join(HERE, name))


# --------------------------------------------------------------------------
def comparison_figure():
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))

    # ---- (a) Kumar 2022 : D(t) para 7 temperaturas iniciales ----
    ax = axes[0]
    k = load("kumar2022_colloid.csv")
    T0s = sorted(k.protocol_value.unique())
    cmap = plt.get_cmap("coolwarm")
    lo, hi = np.log10(min(T0s)), np.log10(max(T0s))
    for T0 in T0s:
        s = k[k.protocol_value == T0].sort_values("x")
        c = cmap((np.log10(T0) - lo) / (hi - lo))
        ax.plot(s.x, s.obs, color=c, lw=1.5, label=f"$T_0={T0:g}$")
    ax.set_yscale("log")
    ax.set_xlabel("tiempo  [ms]")
    ax.set_ylabel(r"distancia al equilibrio  $\mathcal{D}(t)$")
    ax.set_title("(a) Kumar et al. 2022, PNAS\nColoide en trampa óptica (clásico/exp.)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.25)

    # ---- (b) Hallstadius 2020 : enfriamiento caliente vs frío (un ensayo) ----
    ax = axes[1]
    w = load("hallstadius2020_water.csv")
    blk = "DistilledWater-DifferentTemp-16 (60ml)"
    sub = w[w.protocol.str.startswith(blk)]
    for surf, color in (("rough", "#c0392b"), ("smooth", "#2471a3")):
        s = sub[sub.protocol.str.endswith(surf)].sort_values("x")
        if not len(s):
            continue
        Ti = s.obs.iloc[0]
        tag = "caliente" if Ti > 45 else "fría"
        ax.plot(s.x / 60.0, s.obs, color=color, lw=1.4,
                label=f"{surf} ({tag}, $T_0={Ti:.0f}$°C)")
    ax.axhline(0, color="k", ls=":", lw=1, alpha=0.7)
    ax.axhline(-5, color="gray", ls="--", lw=1, alpha=0.6)
    ax.text(ax.get_xlim()[1]*0.55, -4.2, "umbral de congelación", fontsize=8, color="gray")
    ax.set_xlabel("tiempo  [min]")
    ax.set_ylabel("temperatura  [°C]")
    ax.set_title("(b) Hallstadius & Burridge 2020, RSPA\nAgua: caliente vs fría (clásico/exp.)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # ---- (c) Turkeshi 2024 : asimetría de entrelazamiento ΔS̃₂(t) ----
    ax = axes[2]
    q = load("turkeshi2024_quantum.csv")
    thetas = sorted(q.protocol_value.unique())
    cmap = plt.get_cmap("plasma")
    for i, th in enumerate(thetas):
        s = q[q.protocol_value == th].sort_values("x")
        c = cmap(i / max(1, len(thetas) - 1))
        ax.plot(s.x, s.obs, color=c, lw=1.8, label=rf"$\theta={th:g}$")
    ax.set_xscale("log")
    ax.set_xlabel("tiempo  [pasos]")
    ax.set_ylabel(r"asimetría de entrelazamiento  $\Delta\tilde{S}_2(t)=$ EAQ$-$EA")
    ax.set_title("(c) Turkeshi et al. 2024, arXiv\nCircuito cuántico aleatorio (cuántico/num.)")
    ax.annotate(r"$\theta=1.2$ parte más asimétrico"+"\npero cruza por debajo (Mpemba)",
                xy=(120, 0.2), xytext=(3, 0.45), fontsize=7.5, color="darkred",
                arrowprops=dict(arrowstyle="->", color="darkred", lw=1))
    ax.legend(fontsize=8, title="ángulo de inclinación")
    ax.grid(True, which="both", alpha=0.25)

    fig.suptitle("Efecto Mpemba a través de tres dominios — datos estandarizados",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    out = os.path.join(HERE, "mpemba_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("guardado:", out)


# --------------------------------------------------------------------------
def water_diagnostic():
    """Para cada ensayo 'DifferentTemp': tiempo en llegar a -5°C, caliente vs fría."""
    w = load("hallstadius2020_water.csv")
    THR = -5.0

    def t_thr(s):
        s = s.sort_values("x")
        below = s[s.obs <= THR]
        return below.x.iloc[0] / 60.0 if len(below) else np.nan

    blocks = sorted({p.rsplit("|", 1)[0] for p in w.protocol.unique()
                     if "DiffTemp" in p or "DifferentTemp" in p})
    labels, hot_t, cold_t = [], [], []
    for blk in blocks:
        sub = w[w.protocol.str.startswith(blk)]
        series = []
        for surf in ("smooth", "rough"):
            s = sub[sub.protocol.str.endswith(surf)]
            if len(s):
                series.append((s.obs.sort_index().iloc[0], t_thr(s)))
        if len(series) < 2:
            continue
        series.sort()                       # por temperatura inicial
        cold, hot = series[0], series[-1]
        if np.isnan(hot[1]) or np.isnan(cold[1]):
            continue
        labels.append(blk.replace("DistilledWater-", "").replace(" (60ml)", ""))
        hot_t.append(hot[1])
        cold_t.append(cold[1])

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - 0.2, hot_t, 0.4, label="muestra CALIENTE", color="#c0392b")
    ax.bar(x + 0.2, cold_t, 0.4, label="muestra FRÍA", color="#2471a3")
    for i, (h, c) in enumerate(zip(hot_t, cold_t)):
        if h < c:                            # Mpemba: la caliente congela antes
            ax.annotate("Mpemba", (i, max(h, c)), ha="center", va="bottom",
                        fontsize=8, color="green", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("tiempo en alcanzar  −5 °C  [min]")
    ax.set_title("Diagnóstico Mpemba experimental (agua, Hallstadius & Burridge 2020)\n"
                 "Barra caliente más baja que la fría ⇒ la caliente congela primero (efecto Mpemba)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(HERE, "mpemba_water_diagnostic.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("guardado:", out)


if __name__ == "__main__":
    comparison_figure()
    water_diagnostic()
