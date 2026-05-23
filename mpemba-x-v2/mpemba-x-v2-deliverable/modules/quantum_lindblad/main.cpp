// modules/quantum_lindblad/main.cpp
//
// MODULE: Quantum Mpemba effect through Lindblad dissipative dynamics.
//
// REFERENCES
//   F. Carollo, A. Lasanta, I. Lesanovsky, "Exponentially Accelerated Approach
//   to Stationarity in Markovian Open Quantum Systems through the Mpemba
//   Effect", PRL 127, 060401 (2021).  [carollo2021.pdf]
//
//   J. Zhang et al., "Observation of quantum strong Mpemba effect", Nature
//   Comm. 16, 301 (2025).  [in updated bibliography]
//
//   P. Chattopadhyay, J. F. G. Santos, A. Misra, "Anomaly to Resource: The
//   Mpemba Effect in Quantum Thermometry", arXiv 2601.05046 (2026).
//
// PHYSICAL MODEL
// --------------
// Two-level quantum probe (qubit) with energy gap omega_0 coupled to a
// thermal bosonic bath at temperature T. Dynamics governed by a Davies-type
// generalized amplitude damping master equation
//
//   d rho/dt = gamma * (n_bar + 1) D[sigma_-] rho
//            + gamma * n_bar       D[sigma_+] rho
//
// where  n_bar(T) = 1 / (exp(omega_0/T) - 1)  is Bose-Einstein occupation,
//        D[L] rho = L rho L^dagger - (1/2){L^dagger L, rho}
//        sigma_+- are raising/lowering operators.
//
// In the energy basis, coherences decouple from populations under this
// dynamics. The diagonal of rho follows a 2x2 classical master equation on
// p_0(t), p_1(t) = 1 - p_0(t):
//
//   d p_1/dt = gamma * n_bar       * p_0
//            - gamma * (n_bar + 1) * p_1
//
// Equilibrium population: p_1^eq(T) = n_bar(T) / (1 + 2 n_bar(T)) ... no wait,
//   the steady state of the 2-level Davies equation is the Gibbs state
//   p_1^eq(T) = exp(-omega_0/T) / Z(T) = 1 / (e^{omega_0/T} + 1)
//
// MPEMBA & METROLOGY
// ------------------
// We track:
//   1. Population p_1(t) for several initial temperatures.
//   2. Crossings: a Mpemba effect occurs if a "hot" init lands closer to
//      p_1^eq(T_bath) before a "warm" init at some finite t > 0.
//   3. QUANTUM FISHER INFORMATION (QFI) for temperature estimation
//      (Chattopadhyay et al. 2026):
//
//         F_T(t) = [d p_1(t) / dT]^2 / [p_1(t) (1 - p_1(t))]
//
//      The metrological Mpemba effect: hotter-init QFI > cold init QFI > eq QFI
//      at some t* > 0.
//
// PARALLELIZATION
// ---------------
// MPI distributes initial temperatures across ranks. OpenMP not needed for
// 2-level system (state is 2D); but the partial derivative dp_1/dT is
// computed by central differences, which doubles the number of ODE
// integrations per T_init (parallelized over ranks).

#include <mpi.h>
#include <omp.h>

#include "../../core/config_parser.hpp"
#include "../../core/csv_io.hpp"

#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>

using namespace mpemba;

// =========================================================================
// 2-level Davies dynamics
// =========================================================================
struct Qubit {
    double omega0;     // energy gap (natural units, kB = hbar = 1)
    double gamma;      // system-bath coupling rate
    double T_bath;
};

inline double n_bose(double omega, double T) {
    if (omega/T > 700.0) return 0.0;   // avoid overflow
    return 1.0 / (std::exp(omega / T) - 1.0);
}

inline double p1_eq(double omega, double T) {
    // Gibbs population of the excited state for two-level system
    if (omega/T > 700.0) return 0.0;
    return 1.0 / (std::exp(omega / T) + 1.0);
}

inline double dp1_dT(double omega, double T, double eps = 1e-4) {
    return (p1_eq(omega, T + eps) - p1_eq(omega, T - eps)) / (2.0 * eps);
}

// d p_1/dt = gamma * n_bar * (1 - p_1) - gamma * (n_bar + 1) * p_1
//         = gamma * [n_bar - (1 + 2 n_bar) * p_1]
inline double drift_p1(const Qubit& q, double p_1) {
    double nb = n_bose(q.omega0, q.T_bath);
    return q.gamma * (nb - (1.0 + 2.0 * nb) * p_1);
}

// RK4 step for the scalar ODE
inline double rk4_p1(const Qubit& q, double p_1, double dt) {
    double k1 = drift_p1(q, p_1);
    double k2 = drift_p1(q, p_1 + 0.5 * dt * k1);
    double k3 = drift_p1(q, p_1 + 0.5 * dt * k2);
    double k4 = drift_p1(q, p_1 + dt * k3);
    return p_1 + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0;
}

// =========================================================================
// Quench from T_init at the bath T_bath. Track p_1(t), QFI(t).
// =========================================================================
struct QuantumResult {
    std::vector<double> times;
    std::vector<double> p_1;
    std::vector<double> qfi;
    std::vector<double> trace_distance;  // |p_1(t) - p_1^eq(T_bath)|
};

QuantumResult run_quench_qubit(const Qubit& q, double T_init,
                               double t_max, double dt, int log_every) {
    QuantumResult res;
    double p_1 = p1_eq(q.omega0, T_init);   // initial Gibbs state at T_init

    int n_steps = (int)std::ceil(t_max / dt);
    double p_1_eq_bath = p1_eq(q.omega0, q.T_bath);

    // To compute QFI we also evolve a "shadow" trajectory at T_bath + eps
    // and one at T_bath - eps to get dp_1/dT(t).
    double eps = 1e-3 * q.T_bath;
    Qubit q_plus  = q; q_plus.T_bath  = q.T_bath + eps;
    Qubit q_minus = q; q_minus.T_bath = q.T_bath - eps;
    double p_1_plus  = p_1;
    double p_1_minus = p_1;

    for (int step = 0; step <= n_steps; ++step) {
        double t = step * dt;
        if (step % log_every == 0) {
            // QFI at finite time
            double dpdT = (p_1_plus - p_1_minus) / (2.0 * eps);
            double denom = std::max(p_1 * (1.0 - p_1), 1e-15);
            double F_T = dpdT * dpdT / denom;
            res.times.push_back(t);
            res.p_1.push_back(p_1);
            res.qfi.push_back(F_T);
            res.trace_distance.push_back(std::fabs(p_1 - p_1_eq_bath));
        }
        if (step < n_steps) {
            p_1       = rk4_p1(q,        p_1,       dt);
            p_1_plus  = rk4_p1(q_plus,   p_1_plus,  dt);
            p_1_minus = rk4_p1(q_minus,  p_1_minus, dt);
        }
    }
    return res;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) std::cerr << "usage: " << argv[0] << " config.ini\n";
        MPI_Finalize(); return 1;
    }
    Config cfg;
    try { cfg.load(argv[1]); }
    catch (const std::exception& e) {
        if (rank == 0) std::cerr << "Config error: " << e.what() << "\n";
        MPI_Finalize(); return 1;
    }

    Qubit q;
    q.omega0  = cfg.require_double("model", "omega0");
    q.gamma   = cfg.require_double("model", "gamma");
    q.T_bath  = cfg.require_double("system", "T_bath");

    double t_max   = cfg.require_double("integration", "t_max");
    double dt      = cfg.require_double("integration", "dt");
    int log_every  = cfg.get_int       ("integration", "log_every", 20);
    std::vector<double> T_inits = cfg.get_double_list("quench", "T_inits");
    std::string out_dir = cfg.get_string("io", "output_dir",
                                         "results/quantum_lindblad");

    if (T_inits.empty()) {
        if (rank == 0) std::cerr << "[quench] T_inits required\n";
        MPI_Finalize(); return 1;
    }

    if (rank == 0) {
        std::cout << "=== Quantum Lindblad Mpemba (Carollo 2021 + Chattopadhyay 2026) ===\n";
        std::cout << "  omega0 = " << q.omega0 << "  gamma = " << q.gamma
                  << "  T_bath = " << q.T_bath << "\n";
        std::cout << "  T_inits = [";
        for (double T : T_inits) std::cout << " " << T;
        std::cout << " ]\n";
        std::cout << "  p_1^eq(T_bath) = " << p1_eq(q.omega0, q.T_bath) << "\n";
        std::cout << "  Equilibrium QFI = "
                  << std::pow(dp1_dT(q.omega0, q.T_bath), 2)
                     / (p1_eq(q.omega0, q.T_bath) * (1 - p1_eq(q.omega0, q.T_bath)))
                  << "\n";
    }

    // Distribute T_inits across MPI ranks
    int nT = (int)T_inits.size();
    int per_rank = (nT + size - 1) / size;
    int t_lo = std::min(rank * per_rank, nT);
    int t_hi = std::min(t_lo + per_rank, nT);

    auto t_start = std::chrono::steady_clock::now();
    for (int ti = t_lo; ti < t_hi; ++ti) {
        double T_init = T_inits[ti];
        QuantumResult res = run_quench_qubit(q, T_init, t_max, dt, log_every);
        std::ostringstream oss;
        oss << out_dir << "/qubit_Tinit_" << T_init << ".csv";
        CSVWriter csv(oss.str());
        csv.header({"t", "p_1", "QFI", "trace_distance"});
        for (std::size_t k = 0; k < res.times.size(); ++k) {
            csv.row({res.times[k], res.p_1[k], res.qfi[k], res.trace_distance[k]});
        }
        std::cout << "  rank " << rank << " T_init=" << T_init
                  << " final QFI = " << res.qfi.back()
                  << "  final tr_dist = " << res.trace_distance.back() << "\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_end = std::chrono::steady_clock::now();
    if (rank == 0) {
        double elapsed = std::chrono::duration<double>(t_end - t_start).count();
        std::cout << "[quantum_lindblad] done in " << elapsed << " s\n";
    }
    MPI_Finalize();
    return 0;
}
