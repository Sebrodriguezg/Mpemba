/**
 * @file serial.cpp
 * @brief Simulador LBM D1Q3 Serial para benchmarking del efecto Mpemba
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono> // Reemplaza fstream, iomanip y string

// --- Parámetros Computacionales y Físicos ---
constexpr int N_NODES = 1e6;
constexpr double DX = 1e-4;
constexpr double DT = 1e-3;
constexpr int MAX_STEPS = 1000; // Ajustado a 1k para coincidir con las versiones paralelas
constexpr int SKIN_NODES = 10;

// Constantes termodinámicas
constexpr double RHO_BULK = 1000.0;
constexpr double CP_BULK = 4184.0;
constexpr double K_BULK = 0.6;
constexpr double ALPHA_BULK = K_BULK / (RHO_BULK * CP_BULK);

constexpr double RHO_SKIN = RHO_BULK * 0.75;
constexpr double K_SKIN = K_BULK * 1.48;
constexpr double ALPHA_SKIN = K_SKIN / (RHO_SKIN * CP_BULK);

// Parámetros de Memoria del Enlace H
constexpr double C_MEM = 5.0e7;
constexpr double T_SCALE = 57.2887;
constexpr double D_OH_0 = 1.0046;
constexpr double TAU_HB_BASE = 10.0;

constexpr int Q = 3;
constexpr double W[3] = {2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0};
constexpr int CX[3] = {0, 1, -1};

struct Node {
    double f[Q];
    double f_new[Q];
    double T;
    double d_oh;
    double alpha;
    double tau_lbm;
    double rho_cp;
};

double compute_d_oh_eq(double T) {
    return D_OH_0 - 2.7912e-5 * std::exp((T + 273.15) / T_SCALE);
}

double compute_tau_lbm(double alpha) {
    double cs2 = (DX * DX) / (DT * DT * 3.0);
    return 0.5 + (alpha) / (cs2 * DT);
}

int main(int argc, char* argv[]) {
    std::vector<Node> grid(N_NODES);
    const double T_BATH = -18.0; 
    
    // Configuración dinámica de T_INIT por consola
    double T_INIT = 95.0; // Valor por defecto
    if (argc > 1) {
        T_INIT = std::stod(argv[1]);
    }
    
    double tau_hb = TAU_HB_BASE * std::exp(-(T_INIT) / T_SCALE);

    for (int i = 0; i < N_NODES; ++i) {
        if (i < SKIN_NODES) {
            grid[i].alpha = ALPHA_SKIN;
            grid[i].rho_cp = RHO_SKIN * CP_BULK;
        } else {
            grid[i].alpha = ALPHA_BULK;
            grid[i].rho_cp = RHO_BULK * CP_BULK;
        }
        
        grid[i].tau_lbm = compute_tau_lbm(grid[i].alpha);
        grid[i].T = T_INIT;
        grid[i].d_oh = compute_d_oh_eq(T_INIT);

        for (int q = 0; q < Q; ++q) {
            grid[i].f[q] = W[q] * grid[i].T;
            grid[i].f_new[q] = grid[i].f[q];
        }
    }

    // --- INICIO DEL CRONÓMETRO ---
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step <= MAX_STEPS; ++step) {
        for (int i = 0; i < N_NODES; ++i) {
            grid[i].T = grid[i].f[0] + grid[i].f[1] + grid[i].f[2];
            
            double d_oh_eq = compute_d_oh_eq(grid[i].T);
            double d_oh_dot = -(grid[i].d_oh - d_oh_eq) / tau_hb;
            grid[i].d_oh += d_oh_dot * DT;
            
            double source_T = (C_MEM / grid[i].rho_cp) * d_oh_dot;

            for (int q = 0; q < Q; ++q) {
                double f_eq = W[q] * grid[i].T;
                grid[i].f_new[q] = grid[i].f[q] - (1.0 / grid[i].tau_lbm) * (grid[i].f[q] - f_eq) 
                                 + W[q] * source_T * DT;
            }
        }

        for (int i = 1; i < N_NODES - 1; ++i) {
            grid[i].f[0] = grid[i].f_new[0];
            grid[i].f[1] = grid[i - 1].f_new[1];
            grid[i].f[2] = grid[i + 1].f_new[2];
        }

        grid[0].T = T_BATH;
        grid[0].f[1] = W[1] * T_BATH + W[2] * T_BATH - grid[0].f_new[2];
        grid[0].f[0] = W[0] * T_BATH;
        grid[0].f[2] = grid[1].f_new[2]; 

        grid[N_NODES - 1].f[0] = grid[N_NODES - 1].f_new[0];
        grid[N_NODES - 1].f[2] = grid[N_NODES - 1].f_new[1];
        grid[N_NODES - 1].f[1] = grid[N_NODES - 2].f_new[1];
    }

    // --- FIN DEL CRONÓMETRO ---
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    
    // Imprime en el formato exacto que espera plot_performance.py
    std::cout << "Serial,1," << diff.count() << "\n";
    
    return 0;
}
