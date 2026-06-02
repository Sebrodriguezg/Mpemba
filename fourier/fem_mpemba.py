#!/usr/bin/env python3
from dataclasses import dataclass
import os
import numpy as np


@dataclass
class Params:
    # Geometry (m)
    l1: float = 0.01
    l2: float = 0.001
    n_bulk: int = 101
    n_skin: int = 21

    # Bulk material
    rho_b: float = 1000.0
    cp_b: float = 4181.0
    k_b: float = 0.6

    # Skin ratios
    rho_ratio: float = 0.75
    alpha_ratio: float = 1.48

    # Flow and boundary
    v: float = 1.0e-4
    h_kappa_ratio: float = 30.0
    h2_h1: float = 1.0

    # Time
    dt: float = 0.5
    t_end: float = 2000.0
    theta: float = 0.5

    # Temperatures (C)
    t_f: float = 0.0
    t_i_list: tuple = (20.0, 40.0, 60.0, 80.0)

    # Numerics
    upwind: bool = True


def build_mesh(params: Params) -> np.ndarray:
    x_bulk = np.linspace(-params.l1, 0.0, params.n_bulk, endpoint=False)
    x_skin = np.linspace(0.0, params.l2, params.n_skin)
    return np.concatenate([x_bulk, x_skin])


def compute_materials(params: Params) -> dict:
    rho_s = params.rho_ratio * params.rho_b
    cp_s = params.cp_b
    alpha_b = params.k_b / (params.rho_b * params.cp_b)
    alpha_s = params.alpha_ratio * alpha_b
    k_s = alpha_s * rho_s * cp_s
    return {
        "rho_b": params.rho_b,
        "cp_b": params.cp_b,
        "k_b": params.k_b,
        "rho_s": rho_s,
        "cp_s": cp_s,
        "k_s": k_s,
    }


def assemble_matrices(x: np.ndarray, params: Params, mats: dict) -> tuple:
    n = len(x)
    M = np.zeros((n, n))
    K = np.zeros((n, n))
    C = np.zeros((n, n))

    for e in range(n - 1):
        i = e
        j = e + 1
        h = x[j] - x[i]
        x_mid = 0.5 * (x[i] + x[j])

        if x_mid < 0.0:
            rho = mats["rho_b"]
            cp = mats["cp_b"]
            k = mats["k_b"]
        else:
            rho = mats["rho_s"]
            cp = mats["cp_s"]
            k = mats["k_s"]

        rho_cp = rho * cp
        k_eff = k
        if params.upwind and params.v != 0.0:
            k_eff += 0.5 * rho_cp * abs(params.v) * h

        M_e = rho_cp * h / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
        K_e = k_eff / h * np.array([[1.0, -1.0], [-1.0, 1.0]])
        C_e = rho_cp * params.v * 0.5 * np.array([[-1.0, 1.0], [-1.0, 1.0]])

        idx = [i, j]
        for a in range(2):
            for b in range(2):
                M[idx[a], idx[b]] += M_e[a, b]
                K[idx[a], idx[b]] += K_e[a, b]
                C[idx[a], idx[b]] += C_e[a, b]

    return M, K, C


def apply_robin(n: int, params: Params, mats: dict) -> tuple:
    B = np.zeros((n, n))
    F = np.zeros(n)

    h1 = params.h_kappa_ratio * mats["k_b"]
    h2 = params.h_kappa_ratio * mats["k_s"] * params.h2_h1

    B[0, 0] += h1
    B[-1, -1] += h2
    F[0] += h1 * params.t_f
    F[-1] += h2 * params.t_f

    return B, F


def nearest_node(x: np.ndarray, x0: float) -> int:
    return int(np.argmin(np.abs(x - x0)))


def run_case(T_i: float, x: np.ndarray, params: Params, M: np.ndarray, A: np.ndarray, F: np.ndarray) -> tuple:
    n = len(x)
    steps = int(np.ceil(params.t_end / params.dt))
    time = np.linspace(0.0, steps * params.dt, steps + 1)

    idx_interface = nearest_node(x, 0.0)
    idx_bulk = nearest_node(x, -0.5 * params.l1)
    idx_surface = n - 1

    T = np.full(n, T_i, dtype=float)
    series = np.zeros(steps + 1)
    delta = np.zeros(steps + 1)

    series[0] = T[idx_interface]
    delta[0] = T[idx_surface] - T[idx_bulk]

    lhs = M / params.dt + params.theta * A
    rhs_mat = M / params.dt - (1.0 - params.theta) * A

    for k in range(steps):
        rhs = rhs_mat @ T + F
        T = np.linalg.solve(lhs, rhs)
        series[k + 1] = T[idx_interface]
        delta[k + 1] = T[idx_surface] - T[idx_bulk]

    return time, series, delta


def save_series(path: str, time: np.ndarray, series: dict) -> None:
    labels = list(series.keys())
    data = np.column_stack([time] + [series[label] for label in labels])
    header = "t," + ",".join(labels)
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def plot_series(path: str, time: np.ndarray, series: dict, y_label: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plots")
        return

    plt.figure(figsize=(6.5, 4.0))
    for label, values in series.items():
        plt.plot(time, values, label=label)
    plt.xlabel("t (s)")
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    params = Params()
    mats = compute_materials(params)
    x = build_mesh(params)

    M, K, C = assemble_matrices(x, params, mats)
    B, F = apply_robin(len(x), params, mats)
    A = K + C + B

    out_dir = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out_dir, exist_ok=True)

    relax = {}
    delta = {}

    for T_i in params.t_i_list:
        time, series, dseries = run_case(T_i, x, params, M, A, F)
        label = f"Ti_{T_i:.0f}C"
        relax[label] = series
        delta[label] = dseries

    save_series(os.path.join(out_dir, "relaxation.csv"), time, relax)
    save_series(os.path.join(out_dir, "delta.csv"), time, delta)

    plot_series(os.path.join(out_dir, "relaxation.png"), time, relax, "T at x=0 (C)")
    plot_series(os.path.join(out_dir, "delta.png"), time, delta, "T(surface) - T(bulk) (C)")

    print("Outputs written to:", out_dir)


if __name__ == "__main__":
    main()
