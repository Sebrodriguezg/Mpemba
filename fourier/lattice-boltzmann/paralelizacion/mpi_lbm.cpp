#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>

// (Mantén aquí las mismas constantes constexpr que en omp_lbm.cpp)
constexpr int N_NODES = 1e6, MAX_STEPS = 1000, Q = 3;
constexpr double DX = 1e-4, DT = 1e-3, SKIN_NODES = 10;
constexpr double RHO_BULK = 1000.0, CP_BULK = 4184.0, K_BULK = 0.6;
constexpr double RHO_SKIN = RHO_BULK * 0.75, K_SKIN = K_BULK * 1.48;
constexpr double C_MEM = 5.0e7, T_SCALE = 57.2887, D_OH_0 = 1.0046, TAU_HB_BASE = 10.0;
constexpr double W[3] = {2.0/3.0, 1.0/6.0, 1.0/6.0};

struct Node { double f[Q], f_new[Q], T, d_oh, alpha, tau_lbm, rho_cp; };

double compute_d_oh_eq(double T) { return D_OH_0 - 2.7912e-5 * std::exp((T + 273.15) / T_SCALE); }
double compute_tau_lbm(double alpha) { return 0.5 + alpha / (((DX * DX) / (DT * DT * 3.0)) * DT); }

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_N = N_NODES / size;
    std::vector<Node> local_grid(local_N + 2); // +2 por las celdas fantasma

    const double T_INIT = 95.0, T_BATH = -18.0;
    double tau_hb = TAU_HB_BASE * std::exp(-T_INIT / T_SCALE);

    for (int i = 1; i <= local_N; ++i) {
        int global_i = rank * local_N + (i - 1);
        local_grid[i].alpha = (global_i < SKIN_NODES) ? K_SKIN/(RHO_SKIN*CP_BULK) : K_BULK/(RHO_BULK*CP_BULK);
        local_grid[i].rho_cp = (global_i < SKIN_NODES) ? RHO_SKIN*CP_BULK : RHO_BULK*CP_BULK;
        local_grid[i].tau_lbm = compute_tau_lbm(local_grid[i].alpha);
        local_grid[i].T = T_INIT;
        local_grid[i].d_oh = compute_d_oh_eq(T_INIT);
        for(int q=0; q<Q; ++q) local_grid[i].f[q] = local_grid[i].f_new[q] = W[q]*T_INIT;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    for (int step = 0; step <= MAX_STEPS; ++step) {
        for (int i = 1; i <= local_N; ++i) {
            local_grid[i].T = local_grid[i].f[0] + local_grid[i].f[1] + local_grid[i].f[2];
            double d_oh_eq = compute_d_oh_eq(local_grid[i].T);
            double d_oh_dot = -(local_grid[i].d_oh - d_oh_eq) / tau_hb;
            local_grid[i].d_oh += d_oh_dot * DT;
            double source_T = (C_MEM / local_grid[i].rho_cp) * d_oh_dot;

            for (int q = 0; q < Q; ++q) {
                double f_eq = W[q] * local_grid[i].T;
                local_grid[i].f_new[q] = local_grid[i].f[q] - (1.0/local_grid[i].tau_lbm)*(local_grid[i].f[q]-f_eq) + W[q]*source_T*DT;
            }
        }

        // Intercambio de Celdas Fantasma (MPI Communication)
        if (rank > 0) {
            MPI_Send(&local_grid[1].f_new[2], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&local_grid[0].f_new[1], 1, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Recv(&local_grid[local_N + 1].f_new[2], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&local_grid[local_N].f_new[1], 1, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD);
        }

        // Streaming (Propagación)
        for (int i = 1; i <= local_N; ++i) {
            local_grid[i].f[0] = local_grid[i].f_new[0];
            local_grid[i].f[1] = local_grid[i - 1].f_new[1];
            local_grid[i].f[2] = local_grid[i + 1].f_new[2];
        }

        // Fronteras Globales
        if (rank == 0) {
            local_grid[1].T = T_BATH;
            local_grid[1].f[1] = W[1]*T_BATH + W[2]*T_BATH - local_grid[1].f_new[2];
            local_grid[1].f[0] = W[0]*T_BATH; 
            local_grid[1].f[2] = local_grid[2].f_new[2]; 
        }
        if (rank == size - 1) {
            local_grid[local_N].f[0] = local_grid[local_N].f_new[0];
            local_grid[local_N].f[2] = local_grid[local_N].f_new[1];
            local_grid[local_N].f[1] = local_grid[local_N - 1].f_new[1];
        }
    }

    double end_time = MPI_Wtime();
    if (rank == 0) std::cout << "MPI," << size << "," << (end_time - start_time) << "\n";
    
    MPI_Finalize();
    return 0;
}
