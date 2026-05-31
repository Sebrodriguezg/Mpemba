#!/usr/bin/env python3
"""Aísla y estandariza los datos más relevantes del efecto Mpemba de los 3 papers.

De cada repositorio se extrae la observable que mejor muestra el efecto Mpemba:
una "distancia al estado estacionario" que decae en el tiempo a partir de varias
condiciones iniciales (el parámetro de protocolo). El efecto Mpemba se manifiesta
como el CRUCE de esas curvas: una condición inicial que parte más lejos del
equilibrio lo alcanza antes que otra que partía más cerca.

Fuentes:
  1. Kumar et al. 2022, PNAS  -> coloide óptico (clásico/experimental)
       Hoja 'Fig5': D(t) [distancia al equilibrio] vs t para 7 temperaturas T0.
  2. Hallstadius & Burridge 2020, Proc. R. Soc. A -> congelación de agua (clásico/exp.)
       Hoja 'Test': T(°C) vs t, superficie lisa vs rugosa, 21 ensayos.
  3. Turkeshi et al. 2024, arXiv -> circuitos cuánticos aleatorios (cuántico/numérico)
       'TN_TF_N512NA16.csv': asimetría de entrelazamiento EA(t) para 4 ángulos theta.

Esquema 'tidy' común de salida (un archivo por paper + uno combinado):
  source, system, domain, protocol, protocol_value, x_name, x, obs_name, obs
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # .../mpemba_data
OUT = HERE                            # .../mpemba_data/standardized

COLS = ["source", "system", "domain", "protocol",
        "protocol_value", "x_name", "x", "obs_name", "obs"]


# --------------------------------------------------------------------------
def kumar():
    """Kumar et al. 2022 PNAS, hoja Fig5: D(t) para 7 temperaturas iniciales."""
    f = os.path.join(ROOT, "Kumar_et_al_2022_PNAS", "Excel FIles_Graphical Data.xlsx")
    df = pd.read_excel(f, sheet_name="Fig5")
    tcol = df.columns[0]                      # 'Time (ms)'
    recs = []
    for col in df.columns[1:]:                # 'T=1e-5', ...
        T0 = float(col.replace("T=", ""))
        sub = df[[tcol, col]].dropna()
        for t, d in zip(sub[tcol], sub[col]):
            recs.append(dict(source="Kumar2022_PNAS", system="colloid_optical_trap",
                             domain="classical", protocol="T0_initial_temp",
                             protocol_value=T0, x_name="time_ms", x=float(t),
                             obs_name="distance_to_equilibrium_D", obs=float(d)))
    return pd.DataFrame(recs, columns=COLS)


# --------------------------------------------------------------------------
def hallstadius():
    """Hallstadius & Burridge 2020, hoja Test: curvas de enfriamiento del agua.

    83 columnas en bloques de 4: [Time(s), Smooth(°C), Rough(°C), vacío].
    Cada bloque es un ensayo. Se genera una serie por (ensayo, superficie).
    """
    f = os.path.join(ROOT, "Hallstadius_&_Burridge _2020", "rspa20190829_si_001.xlsx")
    raw = pd.read_excel(f, sheet_name="Test", header=None)
    titles = raw.iloc[0]
    sub_hdr = raw.iloc[1]
    data = raw.iloc[2:].reset_index(drop=True)
    recs = []
    for c in range(0, raw.shape[1], 4):
        title = titles[c]
        if not isinstance(title, str):
            continue
        title = title.strip()
        time = pd.to_numeric(data[c], errors="coerce")
        for off, default_surf in ((1, "smooth"), (2, "rough")):
            if c + off >= raw.shape[1]:
                continue
            hdr = sub_hdr[c + off]
            surf = default_surf
            if isinstance(hdr, str) and "rough" in hdr.lower():
                surf = "rough"
            elif isinstance(hdr, str) and "smooth" in hdr.lower():
                surf = "smooth"
            temp = pd.to_numeric(data[c + off], errors="coerce")
            m = time.notna() & temp.notna()
            for t, T in zip(time[m], temp[m]):
                recs.append(dict(source="Hallstadius2020_RSPA", system="water_freezing",
                                 domain="classical", protocol=f"{title}|{surf}",
                                 protocol_value=np.nan, x_name="time_s", x=float(t),
                                 obs_name="temperature_C", obs=float(T)))
    return pd.DataFrame(recs, columns=COLS)


# --------------------------------------------------------------------------
def turkeshi():
    """Turkeshi et al. 2024, TN_TF_N512NA16.csv: asimetría de entrelazamiento.

    La asimetría de entrelazamiento de Rényi-2 que mide la ruptura de simetría es
    ΔS̃₂ = EAQ − EA (ver data_analysis.ipynb del repositorio, panel TF). Decae
    hacia 0 = restauración de la simetría. El efecto Mpemba cuántico aparece como
    el cruce: el estado más inclinado (theta grande) parte más asimétrico pero
    restaura la simetría más rápido y cruza por debajo de los menos inclinados.
    """
    f = os.path.join(ROOT, "Turkeshi_et_al_2024_arXiv", "deployment_mpemba",
                     "data", "TN_TF_N512NA16.csv")
    df = pd.read_csv(f)                       # cols: idx,time,EA,EAQ,th
    df["dS2"] = df["EAQ"] - df["EA"]
    recs = []
    for _, r in df.iterrows():
        recs.append(dict(source="Turkeshi2024_arXiv", system="random_circuit_TF",
                         domain="quantum", protocol="theta_tilt_angle",
                         protocol_value=float(r["th"]), x_name="time_steps",
                         x=float(r["time"]), obs_name="entanglement_asymmetry_dS2",
                         obs=float(r["dS2"])))
    return pd.DataFrame(recs, columns=COLS)


# --------------------------------------------------------------------------
def main():
    builders = {"kumar2022_colloid": kumar,
                "hallstadius2020_water": hallstadius,
                "turkeshi2024_quantum": turkeshi}
    all_df = []
    for name, fn in builders.items():
        df = fn()
        path = os.path.join(OUT, name + ".csv")
        df.to_csv(path, index=False)
        n_series = df.groupby(["source", "protocol", "protocol_value"], dropna=False).ngroups
        print(f"{name:24s} -> {len(df):7d} filas, {n_series:3d} series  ({path})")
        all_df.append(df)
    combined = pd.concat(all_df, ignore_index=True)
    combined.to_csv(os.path.join(OUT, "mpemba_all_standardized.csv"), index=False)
    print(f"{'COMBINADO':24s} -> {len(combined):7d} filas  (mpemba_all_standardized.csv)")


if __name__ == "__main__":
    main()
