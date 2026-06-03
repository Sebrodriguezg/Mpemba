/**
 * @file cuda_lbm.cu
 * @brief Simulador LBM D1Q3 - Efecto Mpemba (Difusividad Dinámica Acoplada)
 * Corregido para acoplamiento de estado (Zhang) y fronteras térmicas.
 */

#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>

constexpr int N_NODES = 1000;
constexpr double DX = 1e-4;
constexpr double DT = 1e-2;
constexpr int MAX_STEPS = 250000;  
constexpr int OUTPUT_FREQ = 2000;  
constexpr int SKIN_NODES = 50;

// Constantes termodinámicas base
constexpr double RHO_BULK = 1000.0, CP_BULK = 4184.0, K_BULK = 0.6;
constexpr double RHO_SKIN = RHO_BULK * 0.75, K_SKIN = K_BULK * 1.48;
constexpr double T_SCALE = 57.2887, D_OH_0 = 1.0046;
__constant__ double d_W[3] = {2.0/3.0, 1.0/6.0, 1.0/6.0};

// Factor fenomenológico ampliado. Se requiere un valor alto para superar 
// el calor sensible masivo en un modelo 1D puramente conductivo.
constexpr double BETA_DIFFUSION = 15000.0; 


struct Node { double f[3], f_new[3], T, d_oh, alpha_base, rho_cp; };

__device__ double d_compute_d_oh_eq(double T) { 
    return D_OH_0 - 2.7912e-5 * exp((T + 273.15) / T_SCALE); 
}

__global__ void collision_kernel(Node* grid, double tau_hb, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N_NODES) {
        // 1. Calcular variables macroscópicas
        grid[i].T = grid[i].f[0] + grid[i].f[1] + grid[i].f[2];
        
        // 2. Dinámica de relajación del enlace O:H-O
        double d_oh_eq = d_compute_d_oh_eq(grid[i].T);
        double d_oh_dot = -(grid[i].d_oh - d_oh_eq) / tau_hb;
        grid[i].d_oh += d_oh_dot * dt;
        
        // 3. NÚCLEO DEL EFECTO MPEMBA (Zhang 2014) - Corrección de Estado
        // Se evalúa la deformación respecto al baño térmico (-18°C)
        double d_oh_cold = 1.0046 - 2.7912e-5 * exp((-18.0 + 273.15) / T_SCALE);
        
        // La difusividad escala con la compresión retenida (memoria) del enlace
        double alpha_dynamic = grid[i].alpha_base * (1.0 + BETA_DIFFUSION * ((d_oh_cold - grid[i].d_oh) / d_oh_cold));

        // 4. Mapeo a Lattice Boltzmann (Tiempo de relajación dinámico)
        double cs2 = (DX * DX) / (dt * dt * 3.0);
        double tau_lbm_dynamic = 0.5 + alpha_dynamic / (cs2 * dt);

        // 5. Operador de Colisión BGK
        for (int q = 0; q < 3; ++q) {
            double f_eq = d_W[q] * grid[i].T;
            grid[i].f_new[q] = grid[i].f[q] - (1.0 / tau_lbm_dynamic) * (grid[i].f[q] - f_eq);
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
    // Frontera Izquierda (Dirichlet térmico exacto vía Anti-Bounce-Back)
    grid[0].T = T_BATH;
    grid[0].f[2] = grid[1].f_new[2]; // Streaming explícito entrante
    grid[0].f[0] = d_W[0] * T_BATH;  // Equilibrio local
    grid[0].f[1] = (1.0 / 3.0) * T_BATH - grid[0].f[2]; // Rebote de poblaciones
    
    // Frontera Derecha (Condición de simetría / Adiabática)
    grid[N_NODES - 1].f[0] = grid[N_NODES - 1].f_new[0];
    grid[N_NODES - 1].f[1] = grid[N_NODES - 2].f_new[1];
    grid[N_NODES - 1].f[2] = grid[N_NODES - 1].f[1]; // Rebote especular
}

int main(int argc, char* argv[]) {
    double T_INIT = 95.0; 
    if (argc > 1) T_INIT = std::stod(argv[1]);
    
    std::string filename = "datos-cuda_T" + std::to_string(static_cast<int>(T_INIT)) + ".csv";
    const double T_BATH = -18.0;

    // Fórmula exacta empírica de Zhang (Fig 6a) convertida a segundos
    double tau_hb = 474.0 * (std::exp(-(T_INIT - 129.9) / 47.5) - 1.0);

    Node* h_grid = new Node[N_NODES];
    for (int i = 0; i < N_NODES; ++i) {
        h_grid[i].alpha_base = (i < SKIN_NODES) ? K_SKIN/(RHO_SKIN*CP_BULK) : K_BULK/(RHO_BULK*CP_BULK);
        h_grid[i].rho_cp = (i < SKIN_NODES) ? RHO_SKIN*CP_BULK : RHO_BULK*CP_BULK;
        
        h_grid[i].T = T_INIT;
        h_grid[i].d_oh = D_OH_0 - 2.7912e-5 * exp((T_INIT + 273.15) / T_SCALE);
        h_grid[i].f[0] = h_grid[i].f_new[0] = (2.0/3.0)*T_INIT;
        h_grid[i].f[1] = h_grid[i].f_new[1] = h_grid[i].f[2] = h_grid[i].f_new[2] = (1.0/6.0)*T_INIT;
    }

    Node* d_grid;
    cudaMalloc(&d_grid, N_NODES * sizeof(Node));
    cudaMemcpy(d_grid, h_grid, N_NODES * sizeof(Node), cudaMemcpyHostToDevice);

    std::ofstream out_file(filename);
    out_file << "step,x_pos,T,d_oh\n";

    int threads = 256;
    int blocks = (N_NODES + threads - 1) / threads;

    std::cout << "-> GPU Computando: T_INIT = " << T_INIT << "C (Tau_HB = " << tau_hb << " s)..." << std::flush;

    for (int step = 0; step <= MAX_STEPS; ++step) {
        collision_kernel<<<blocks, threads>>>(d_grid, tau_hb, DT);
        streaming_kernel<<<blocks, threads>>>(d_grid);
        boundary_kernel<<<1, 1>>>(d_grid, T_BATH);

        if (step % OUTPUT_FREQ == 0) {
            cudaMemcpy(h_grid, d_grid, N_NODES * sizeof(Node), cudaMemcpyDeviceToHost);
            for (int i = 0; i < N_NODES; i += 10) { 
                out_file << step << "," << i * DX << "," << h_grid[i].T << "," 
                         << std::fixed << std::setprecision(6) << h_grid[i].d_oh << "\n";
            }
        }
    }

    out_file.close();
    cudaFree(d_grid);
    delete[] h_grid;
    std::cout << " [COMPLETADO]" << std::endl;
    return 0;
}
