#include <iostream>
#include <cmath>

// Constantes físicas
constexpr int N_NODES = 1e6, MAX_STEPS = 1000, Q = 3;
constexpr double DX = 1e-4, DT = 1e-3, SKIN_NODES = 10;
constexpr double RHO_BULK = 1000.0, CP_BULK = 4184.0, K_BULK = 0.6;
constexpr double RHO_SKIN = RHO_BULK * 0.75, K_SKIN = K_BULK * 1.48;
constexpr double C_MEM = 5.0e7, T_SCALE = 57.2887, D_OH_0 = 1.0046, TAU_HB_BASE = 10.0;
__constant__ double d_W[3] = {2.0/3.0, 1.0/6.0, 1.0/6.0};

struct Node { double f[Q], f_new[Q], T, d_oh, alpha, tau_lbm, rho_cp; };

__device__ double d_compute_d_oh_eq(double T) { return D_OH_0 - 2.7912e-5 * exp((T + 273.15) / T_SCALE); }

__global__ void collision_kernel(Node* grid, double tau_hb, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N_NODES) {
        grid[i].T = grid[i].f[0] + grid[i].f[1] + grid[i].f[2];
        double d_oh_eq = d_compute_d_oh_eq(grid[i].T);
        double d_oh_dot = -(grid[i].d_oh - d_oh_eq) / tau_hb;
        grid[i].d_oh += d_oh_dot * dt;
        double source_T = (C_MEM / grid[i].rho_cp) * d_oh_dot;

        for (int q = 0; q < 3; ++q) {
            double f_eq = d_W[q] * grid[i].T;
            grid[i].f_new[q] = grid[i].f[q] - (1.0/grid[i].tau_lbm)*(grid[i].f[q]-f_eq) + d_W[q]*source_T*dt;
        }
    }
}

__global__ void streaming_kernel(Node* grid) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0 && i < N_NODES - 1) {
        grid[i].f[0] = grid[i].f_new[0];
        grid[i].f[1] = grid[i - 1].f_new[1];
        grid[i].f[2] = grid[i + 1].f_new[2];
    }
}

__global__ void boundary_kernel(Node* grid, double T_BATH) {
    // Frontera izquierda
    grid[0].T = T_BATH;
    grid[0].f[1] = d_W[1]*T_BATH + d_W[2]*T_BATH - grid[0].f_new[2];
    grid[0].f[0] = d_W[0]*T_BATH; grid[0].f[2] = grid[1].f_new[2]; 
    // Frontera derecha
    grid[N_NODES - 1].f[0] = grid[N_NODES - 1].f_new[0];
    grid[N_NODES - 1].f[2] = grid[N_NODES - 1].f_new[1];
    grid[N_NODES - 1].f[1] = grid[N_NODES - 2].f_new[1];
}

int main() {
    Node* h_grid = new Node[N_NODES];
    const double T_INIT = 95.0, T_BATH = -18.0;
    double tau_hb = TAU_HB_BASE * std::exp(-T_INIT / T_SCALE);

    for (int i = 0; i < N_NODES; ++i) {
        h_grid[i].alpha = (i < SKIN_NODES) ? K_SKIN/(RHO_SKIN*CP_BULK) : K_BULK/(RHO_BULK*CP_BULK);
        h_grid[i].rho_cp = (i < SKIN_NODES) ? RHO_SKIN*CP_BULK : RHO_BULK*CP_BULK;
        double cs2 = (DX * DX) / (DT * DT * 3.0);
        h_grid[i].tau_lbm = 0.5 + h_grid[i].alpha / (cs2 * DT);
        h_grid[i].T = T_INIT;
        h_grid[i].d_oh = D_OH_0 - 2.7912e-5 * exp((T_INIT + 273.15) / T_SCALE);
        h_grid[i].f[0]=h_grid[i].f_new[0]= (2.0/3.0)*T_INIT;
        h_grid[i].f[1]=h_grid[i].f_new[1]=h_grid[i].f[2]=h_grid[i].f_new[2]= (1.0/6.0)*T_INIT;
    }

    Node* d_grid;
    cudaMalloc(&d_grid, N_NODES * sizeof(Node));
    cudaMemcpy(d_grid, h_grid, N_NODES * sizeof(Node), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N_NODES + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int step = 0; step <= MAX_STEPS; ++step) {
        collision_kernel<<<blocks, threads>>>(d_grid, tau_hb, DT);
        streaming_kernel<<<blocks, threads>>>(d_grid);
        boundary_kernel<<<1, 1>>>(d_grid, T_BATH);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "CUDA,GPU," << milliseconds / 1000.0 << "\n";

    cudaFree(d_grid);
    delete[] h_grid;
    return 0;
}
