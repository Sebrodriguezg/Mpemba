/**
 * @file serial.cpp
 * @brief Simulador LBM D1Q3 Serial para el efecto Mpemba macroscópico (Zhang et al. 2014)
 * @details Resuelve la ecuación de calor 1D acoplada a la relajación térmica del enlace de hidrógeno.
 * @compiler g++ -std=c++17 -O3 mpemba_lbm_1d.cpp -o mpemba_serial
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>

// --- Parámetros Computacionales y Físicos ---
constexpr int N_NODES = 1000;                // Resolución espacial
constexpr double DX = 1e-4;                  // Tamaño de celda [m]
constexpr double DT = 1e-4;                  // Paso de tiempo [s]
constexpr int MAX_STEPS = 1000000;             // Iteraciones temporales totales
constexpr int OUTPUT_FREQ = 500;             // Frecuencia de guardado de datos

constexpr int SKIN_NODES = 10;               // El primer milímetro (piel)

// Constantes termodinámicas (agua estándar vs supersólida)
constexpr double RHO_BULK = 1000.0;          // [kg/m^3]
constexpr double CP_BULK = 4184.0;           // [J/(kg K)]
constexpr double K_BULK = 0.6;               // [W/(m K)]
constexpr double ALPHA_BULK = K_BULK / (RHO_BULK * CP_BULK);

constexpr double RHO_SKIN = RHO_BULK * 0.75; // Densidad reducida
constexpr double K_SKIN = K_BULK * 1.48;     // Alta conductividad en piel
constexpr double ALPHA_SKIN = K_SKIN / (RHO_SKIN * CP_BULK);

// Parámetros de Memoria del Enlace H (Zhang et al.)
constexpr double C_MEM = 5.0e7;              // Constante de acoplamiento térmico-memoria
constexpr double T_SCALE = 57.2887;          // Escala termodinámica para O:H-O
constexpr double D_OH_0 = 1.0046;            // [Angstroms]
constexpr double TAU_HB_BASE = 10.0;         // Constante de tiempo base [s]

// Parámetros de Lattice Boltzmann D1Q3 térmico
constexpr int Q = 3;
constexpr double W[3] = {2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0}; // Pesos
constexpr int CX[3] = {0, 1, -1};                          // Velocidades discretas

// --- Estructura del Nodo LBM ---
struct Node {
    double f[Q];       // Función de distribución actual
    double f_new[Q];   // Función de distribución tras colisión
    double T;          // Temperatura macroscópica local
    double d_oh;       // Memoria local del enlace O:H-O
    double alpha;      // Difusividad térmica local
    double tau_lbm;    // Tiempo de relajación LBM
    double rho_cp;     // Capacidad calorífica volumétrica
};

// --- Funciones Auxiliares ---
double compute_d_oh_eq(double T) {
    // Curva de dilatación/contracción del enlace reportada en el paper (Eq. de memoria)
    return D_OH_0 - 2.7912e-5 * std::exp((T + 273.15) / T_SCALE);
}

double compute_tau_lbm(double alpha) {
    // Mapeo macro-mesoscópico para esquema D1Q3
    // alpha = c_s^2 * (tau_lbm - 0.5) * DT, donde c_s^2 = (DX/DT)^2 / 3
    double cs2 = (DX * DX) / (DT * DT * 3.0);
    return 0.5 + (alpha) / (cs2 * DT);
}

int main() {
    std::vector<Node> grid(N_NODES);
    const double T_INIT = 95.0; // Inicialmente caliente (cambiar para probar efecto Mpemba)
    const double T_BATH = -18.0; // Temperatura del congelador

    // Calcular el tiempo de relajación del enlace H dependiente de T_INIT (Core Mpemba)
    double tau_hb = TAU_HB_BASE * std::exp(-(T_INIT) / T_SCALE);

    // 1. Inicialización
    for (int i = 0; i < N_NODES; ++i) {
        // Asignar propiedades de Piel vs Volumen
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

        // Equilibrio inicial para f_i
        for (int q = 0; q < Q; ++q) {
            grid[i].f[q] = W[q] * grid[i].T;
            grid[i].f_new[q] = grid[i].f[q];
        }
    }

    std::ofstream out_file("datos-serial.csv");
    out_file << "step,x_pos,T,d_oh\n";

    // 2. Bucle Temporal LBM
    for (int step = 0; step <= MAX_STEPS; ++step) {
        
        // --- COLISIÓN Y TÉRMINO FUENTE (Memoria) ---
        for (int i = 0; i < N_NODES; ++i) {
            // Actualizar temperatura local
            grid[i].T = grid[i].f[0] + grid[i].f[1] + grid[i].f[2];
            
            // Relajación de la memoria del enlace H (Euler explícito)
            double d_oh_eq = compute_d_oh_eq(grid[i].T);
            double d_oh_dot = -(grid[i].d_oh - d_oh_eq) / tau_hb;
            grid[i].d_oh += d_oh_dot * DT;
            
            // Término fuente inyectado como tasa de calor (q_mem)
            double source_T = (C_MEM / grid[i].rho_cp) * d_oh_dot;

            // Operador de Colisión BGK
            for (int q = 0; q < Q; ++q) {
                double f_eq = W[q] * grid[i].T;
                grid[i].f_new[q] = grid[i].f[q] - (1.0 / grid[i].tau_lbm) * (grid[i].f[q] - f_eq) 
                                 + W[q] * source_T * DT;
            }
        }

        // --- STREAMING (Propagación) ---
        for (int i = 1; i < N_NODES - 1; ++i) {
            grid[i].f[0] = grid[i].f_new[0];         // Reposo
            grid[i].f[1] = grid[i - 1].f_new[1];     // Deriva derecha
            grid[i].f[2] = grid[i + 1].f_new[2];     // Deriva izquierda
        }

        // --- CONDICIONES DE FRONTERA ---
        // Nodo 0: Frontera Dirichlet simplificada en la piel (en contacto con baño frío)
        // (Podemos refinar a una condición de Robin más adelante)
        grid[0].T = T_BATH;
        grid[0].f[1] = W[1] * T_BATH + W[2] * T_BATH - grid[0].f_new[2]; // Anti-bounce-back
        grid[0].f[0] = W[0] * T_BATH;
        grid[0].f[2] = grid[1].f_new[2]; 

        // Nodo N-1: Adiabático (Bounce-back térmico)
        grid[N_NODES - 1].f[0] = grid[N_NODES - 1].f_new[0];
        grid[N_NODES - 1].f[2] = grid[N_NODES - 1].f_new[1]; // Reflejo
        grid[N_NODES - 1].f[1] = grid[N_NODES - 2].f_new[1];

        // --- ESCRITURA DE DATOS ---
        if (step % OUTPUT_FREQ == 0) {
            for (int i = 0; i < N_NODES; i += 10) { // Guardar 1 de cada 10 nodos para no saturar el CSV
                out_file << step << "," << i * DX << "," << grid[i].T << "," << std::fixed << std::setprecision(6) << grid[i].d_oh << "\n";
            }
        }
    }

    out_file.close();
    std::cout << "Simulación LBM 1D finalizada con éxito. Datos guardados en CSV." << std::endl;
    return 0;
}
