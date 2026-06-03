#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>

// (Mantén aquí las mismas constantes constexpr de tu serial.cpp)
constexpr int N_NODES = 1e6;
constexpr double DX = 1e-4, DT = 1e-3;
constexpr int MAX_STEPS = 1000; // Reducido a 1k para benchmarking rápido
constexpr int SKIN_NODES = 10;
constexpr double RHO_BULK = 1000.0, CP_BULK = 4184.0, K_BULK = 0.6;
constexpr double RHO_SKIN = RHO_BULK * 0.75, K_SKIN = K_BULK * 1.48;
constexpr double C_MEM = 5.0e7, T_SCALE = 57.2887, D_OH_0 = 1.0046, TAU_HB_BASE = 10.0;
constexpr int Q = 3;
constexpr double W[3] = {2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0};

struct Node {
    double f[Q], f_new[Q], T, d_oh, alpha, tau_lbm, rho_cp;
};

double compute_d_oh_eq(double T) { return D_OH_0 - 2.7912e-5 * std::exp((T + 273.15) / T_SCALE); }
double compute_tau_lbm(double alpha) { return 0.5 + alpha / (((DX * DX) / (DT * DT * 3.0)) * DT); }

int main() {
    std::vector<Node> grid(N_NODES);
    const double T_INIT = 95.0, T_BATH = -18.0;
    double tau_hb = TAU_HB_BASE * std::exp(-T_INIT / T_SCALE);

    for (int i = 0; i < N_NODES; ++i) {
        grid[i].alpha = (i < SKIN_NODES) ? K_SKIN/(RHO_SKIN*CP_BULK) : K_BULK/(RHO_BULK*CP_BULK);
        grid[i].rho_cp = (i < SKIN_NODES) ? RHO_SKIN*CP_BULK : RHO_BULK*CP_BULK;
        grid[i].tau_lbm = compute_tau_lbm(grid[i].alpha);
        grid[i].T = T_INIT;
        grid[i].d_oh = compute_d_oh_eq(T_INIT);
        for (int q = 0; q < Q; ++q) { grid[i].f[q] = grid[i].f_new[q] = W[q] * T_INIT; }
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step <= MAX_STEPS; ++step) {
        #pragma omp parallel for
        for (int i = 0; i < N_NODES; ++i) {
            grid[i].T = grid[i].f[0] + grid[i].f[1] + grid[i].f[2];
            double d_oh_eq = compute_d_oh_eq(grid[i].T);
            double d_oh_dot = -(grid[i].d_oh - d_oh_eq) / tau_hb;
            grid[i].d_oh += d_oh_dot * DT;
            double source_T = (C_MEM / grid[i].rho_cp) * d_oh_dot;

            for (int q = 0; q < Q; ++q) {
                double f_eq = W[q] * grid[i].T;
                grid[i].f_new[q] = grid[i].f[q] - (1.0 / grid[i].tau_lbm) * (grid[i].f[q] - f_eq) + W[q] * source_T * DT;
            }
        }

        #pragma omp parallel for
        for (int i = 1; i < N_NODES - 1; ++i) {
            grid[i].f[0] = grid[i].f_new[0];
            grid[i].f[1] = grid[i - 1].f_new[1];
            grid[i].f[2] = grid[i + 1].f_new[2];
        }

        grid[0].T = T_BATH;
        grid[0].f[1] = W[1] * T_BATH + W[2] * T_BATH - grid[0].f_new[2];
        grid[0].f[0] = W[0] * T_BATH; grid[0].f[2] = grid[1].f_new[2]; 
        grid[N_NODES - 1].f[0] = grid[N_NODES - 1].f_new[0];
        grid[N_NODES - 1].f[2] = grid[N_NODES - 1].f_new[1];
        grid[N_NODES - 1].f[1] = grid[N_NODES - 2].f_new[1];
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "OpenMP," << omp_get_max_threads() << "," << diff.count() << "\n";
    return 0;
}
