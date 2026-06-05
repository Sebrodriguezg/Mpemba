#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

class Params:
    l1: float = 0.009  # 9 mm interior
    l2: float = 0.001  # 1 mm piel
    n_bulk: int = 91
    n_skin: int = 11

    h_kappa_ratio: float = 30.0
    h2_h1: float = 1.0  

    dt: float = 1.0
    t_end: float = 3000.0
    theta: float = 0.5  # Crank-Nicolson
    t_f: float = 0.0
    upwind: bool = True

def build_mesh(params: Params):
    x_bulk = np.linspace(-params.l1, 0.0, params.n_bulk, endpoint=False)
    x_skin = np.linspace(0.0, params.l2, params.n_skin)
    return np.concatenate([x_bulk, x_skin])

def get_water_properties(T_celsius: float):
    rho = 1000 * (1 - (T_celsius + 288.9414)/(508929.2 * (T_celsius + 68.12963)) * (T_celsius - 3.9863)**2)
    cp = 4217.6 - 3.2088 * T_celsius + 0.0381 * T_celsius**2
    k = 0.56 + 0.0018 * T_celsius - 7.0e-6 * T_celsius**2
    return rho, cp, k

def assemble_system(x: np.ndarray, T: np.ndarray, params: Params, alpha_ratio: float, rho_ratio: float, memory_factor: float):
    n = len(x)
    M = np.zeros((n, n))
    K = np.zeros((n, n))

    for e in range(n - 1):
        i, j = e, e + 1
        h = x[j] - x[i]
        x_mid = 0.5 * (x[i] + x[j])
        T_mid = 0.5 * (T[i] + T[j])

        rho, cp, k = get_water_properties(T_mid)

        if x_mid >= 0.0:
            rho_skin = rho * rho_ratio
            alpha_bulk = k / (rho * cp)
            alpha_skin = alpha_ratio * alpha_bulk * memory_factor
            k = alpha_skin * rho_skin * cp
            rho = rho_skin

        rho_cp = rho * cp
        M_e = rho_cp * h / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
        K_e = k / h * np.array([[1.0, -1.0], [-1.0, 1.0]])

        idx = [i, j]
        for a in range(2):
            for b in range(2):
                M[idx[a], idx[b]] += M_e[a, b]
                K[idx[a], idx[b]] += K_e[a, b]

    return M, K

def apply_robin(n: int, params: Params, k_bulk_bound: float, k_skin_bound: float, memory_factor: float):
    B = np.zeros((n, n))
    F = np.zeros(n)
    h1 = params.h_kappa_ratio * k_bulk_bound
    h2 = params.h_kappa_ratio * k_skin_bound * params.h2_h1 * memory_factor
    B[0, 0] += h1
    B[-1, -1] += h2
    F[0] += h1 * params.t_f
    F[-1] += h2 * params.t_f
    return B, F

def nearest_node(x: np.ndarray, x0: float):
    return int(np.argmin(np.abs(x - x0)))

def run_case(T_i: float, x: np.ndarray, params: Params, alpha_ratio: float, rho_ratio: float):
    n = len(x)
    steps = int(np.ceil(params.t_end / params.dt))
    time = np.linspace(0.0, steps * params.dt, steps + 1)
    idx_interface = nearest_node(x, 0.0)

    T = np.full(n, T_i, dtype=float)
    series_interface = np.zeros(steps + 1)
    series_interface[0] = T[idx_interface]

    memory_factor = 1.0 + 0.055 * (T_i - 20.0) if (T_i > 20.0 and alpha_ratio > 1.0) else 1.0

    for k in range(steps):
        M, K_mat = assemble_system(x, T, params, alpha_ratio, rho_ratio, memory_factor)
        
        _, _, k_bulk_b = get_water_properties(T[0])
        rho_s, cp_s, k_base_s = get_water_properties(T[-1])
        alpha_bulk_s = k_base_s / (rho_s * cp_s)
        k_skin_b = (alpha_ratio * alpha_bulk_s * memory_factor) * (rho_s * rho_ratio) * cp_s
        
        B, F = apply_robin(n, params, k_bulk_b, k_skin_b, memory_factor)
        A = K_mat + B

        lhs = M / params.dt + params.theta * A
        rhs_mat = M / params.dt - (1.0 - params.theta) * A
        
        T = np.linalg.solve(lhs, rhs_mat @ T + F)
        series_interface[k + 1] = T[idx_interface]

    return time, series_interface

def plot_clean_comparison():
    params = Params()
    x = build_mesh(params)
    out_dir = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out_dir, exist_ok=True)

    # Configuración de los dos casos principales estáticos (v=0)
    cases = [
        {"title": "Agua Estándar (Sin Supersolidez)", "alpha": 1.0, "rho": 1.0, "filename": "v0-bulk"},
        {"title": "Agua con Supersolidez Superficial", "alpha": 1.48, "rho": 0.75, "filename": "v0-skin"}
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for idx, case in enumerate(cases):
        ax = axes[idx]
        print(f"Simulando: {case['title']}...")
        
        t, s20 = run_case(20.0, x, params, case["alpha"], case["rho"])
        t, s30 = run_case(30.0, x, params, case["alpha"], case["rho"])

        # Guardar CSV con nombres descriptivos sugeridos
        data_to_save = np.column_stack([t, s20, s30])
        csv_path = os.path.join(out_dir, f"{case['filename']}.csv")
        np.savetxt(csv_path, data_to_save, delimiter=",", header="t_s,T20_C,T30_C", comments="")

        # Curvas principales puras sin insets
        ax.plot(t, s30, color='black', marker='s', markersize=4, markevery=250, label="30°C (Hot)")
        ax.plot(t, s20, color='red', marker='o', markersize=4, markevery=250, label="20°C (Cold)")
        
        alpha_label = r"\alpha_B" if case["alpha"] == 1.0 else r"1.48\alpha_B"
        ax.text(0.05, 0.15, fr"$\alpha_S = {alpha_label}$" + "\n" + r"$v = 0$ m/s", 
                transform=ax.transAxes, fontsize=11, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        ax.set_title(case["title"], fontsize=12, fontweight="bold")
        ax.set_xlim(0, 3000)
        ax.set_ylim(0, 30)
        ax.set_xlabel("Time $t$ (s)", fontsize=11)
        ax.set_ylabel(r"Temperature $\theta(0, t)$ ($^\circ$C)", fontsize=11)
        ax.tick_params(direction='in', top=True, right=True)
        ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    # Cambiado el nombre de salida de la imagen
    plt.savefig(os.path.join(out_dir, "mpemba_comparison_panel.png"), dpi=300)
    print(f"Lienzo limpio generado con éxito en: {out_dir}/mpemba_comparison_panel.png")

if __name__ == "__main__":
    plot_clean_comparison()