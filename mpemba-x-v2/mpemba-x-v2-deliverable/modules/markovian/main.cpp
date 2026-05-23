// modules/markovian/main.cpp
//
// MODULE 1 (v2): Markovian Mpemba effect (Lu & Raz, PNAS 2017).
//
// Same physics as v1: master equation dp/dt = R(T_b) p for a 3-state Arrhenius
// system or a 1D antiferromagnetic Ising chain, integrated with RK4.
// Three distance metrics tracked simultaneously (L1, KL, entropic).
//
// NEW IN V2:
//   - Configuration file driven (no more long CLI flags).
//   - Thermomajorization diagnostic at each logged time step.
//
// USAGE:  mpirun -np K ./mpemba_markovian config.ini
//
// References: lu2017.pdf, luraz2017nonequilibriumthermodynamicsofthemarkovianmpembaeffectanditsinverse.pdf

#include <mpi.h>
#include <omp.h>

#include "../../core/config_parser.hpp"
#include "../../core/distance_metrics.hpp"
#include "../../core/thermomajorization.hpp"
#include "../../core/csv_io.hpp"

#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <fstream>

using namespace mpemba;

// =========================================================================
// Generic Markov-rate machinery
// =========================================================================
struct DenseMarkov {
    int n = 0;
    std::vector<double> R;          // n*n, row-major
    std::vector<double> energies;   // n
};

DenseMarkov build_dense_markov(const std::vector<double>& energies,
                               const std::vector<double>& B,
                               double T_b, double Gamma = 1.0) {
    int n = (int)energies.size();
    DenseMarkov M;
    M.n = n;
    M.R.assign((std::size_t)n * n, 0.0);
    M.energies = energies;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            double Bij = B[(std::size_t)i * n + j];
            double Ej  = energies[j];
            M.R[(std::size_t)i * n + j] = Gamma * std::exp(-(Bij - Ej) / T_b);
        }
    }
    for (int j = 0; j < n; ++j) {
        double s = 0.0;
        for (int i = 0; i < n; ++i) if (i != j) s += M.R[(std::size_t)i * n + j];
        M.R[(std::size_t)j * n + j] = -s;
    }
    return M;
}

void dense_matvec(const DenseMarkov& M, const double* x, double* y) {
    int n = M.n;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        const double* row = &M.R[(std::size_t)i * n];
        for (int j = 0; j < n; ++j) s += row[j] * x[j];
        y[i] = s;
    }
}

void rk4_step_dense(const DenseMarkov& M, std::vector<double>& p, double dt) {
    int n = M.n;
    std::vector<double> k1(n), k2(n), k3(n), k4(n), tmp(n);
    dense_matvec(M, p.data(), k1.data());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) tmp[i] = p[i] + 0.5 * dt * k1[i];
    dense_matvec(M, tmp.data(), k2.data());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) tmp[i] = p[i] + 0.5 * dt * k2[i];
    dense_matvec(M, tmp.data(), k3.data());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) tmp[i] = p[i] + dt * k3[i];
    dense_matvec(M, tmp.data(), k4.data());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        p[i] += dt * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]) / 6.0;
    }
    double s = 0.0;
    for (int i = 0; i < n; ++i) { if (p[i] < 0) p[i] = 0.0; s += p[i]; }
    if (s > 0) for (int i = 0; i < n; ++i) p[i] /= s;
}

// =========================================================================
// System builders
// =========================================================================
DenseMarkov system_three_state(double T_b) {
    // Lu-Raz paper: E = {0, 0.1, 0.7}, B12=1.5, B13=0.8, B23=1.2.
    std::vector<double> E = {0.0, 0.1, 0.7};
    std::vector<double> B(9, 0.0);
    auto set = [&](int i, int j, double v) {
        B[(std::size_t)i * 3 + j] = v;
        B[(std::size_t)j * 3 + i] = v;
    };
    set(0,1,1.5); set(0,2,0.8); set(1,2,1.2);
    return build_dense_markov(E, B, T_b);
}

struct IsingChain {
    int N; double J, h;
    std::vector<double> energies;
};

IsingChain build_ising_chain(int N, double J, double h) {
    IsingChain ic; ic.N = N; ic.J = J; ic.h = h;
    long n_states = 1L << N;
    ic.energies.resize((std::size_t)n_states);
    auto bit_to_spin = [](int b) { return b ? +1 : -1; };
    int s_left = 1, s_right = 1;
    for (long st = 0; st < n_states; ++st) {
        double E = 0.0;
        int sk_prev = s_left;
        for (int k = 0; k < N; ++k) {
            int sk = bit_to_spin((st >> k) & 1);
            E += -J * (sk_prev * sk);
            E += -h * sk;
            sk_prev = sk;
        }
        E += -J * (sk_prev * s_right);
        ic.energies[(std::size_t)st] = E;
    }
    return ic;
}

DenseMarkov build_ising_markov(const IsingChain& ic, double T_b, double Gamma = 1.0) {
    long n_states = 1L << ic.N;
    DenseMarkov M;
    M.n = (int)n_states;
    M.R.assign((std::size_t)n_states * (std::size_t)n_states, 0.0);
    M.energies = ic.energies;
    #pragma omp parallel for schedule(static)
    for (long j = 0; j < n_states; ++j) {
        double diag = 0.0;
        for (int k = 0; k < ic.N; ++k) {
            long i = j ^ (1L << k);
            double Ei = ic.energies[(std::size_t)i];
            double Ej = ic.energies[(std::size_t)j];
            double rate = Gamma / (1.0 + std::exp((Ei - Ej) / T_b));
            M.R[(std::size_t)i * n_states + (std::size_t)j] = rate;
            diag += rate;
        }
        M.R[(std::size_t)j * n_states + (std::size_t)j] = -diag;
    }
    return M;
}

// =========================================================================
// Quench routine with thermomajorization tracking
// =========================================================================
void run_quench(const DenseMarkov& M_bath, double T_init, double T_bath,
                double t_max, double dt, int log_every,
                const std::string& csv_path,
                std::vector<std::vector<double>>* trajectory_p = nullptr,
                std::vector<double>* trajectory_t = nullptr) {
    int n = M_bath.n;
    std::vector<double> p(n), pi_bath(n);
    boltzmann(M_bath.energies.data(), T_init, p.data(),       n);
    boltzmann(M_bath.energies.data(), T_bath, pi_bath.data(), n);

    CSVWriter csv(csv_path);
    csv.header({"t", "D_L1", "D_KL", "D_e"});

    int n_steps = (int)std::ceil(t_max / dt);
    for (int step = 0; step <= n_steps; ++step) {
        double t = step * dt;
        if (step % log_every == 0) {
            double dL1 = distance_L1(p.data(), pi_bath.data(), n);
            double dKL = distance_KL(p.data(), pi_bath.data(), n);
            double de  = distance_entropic_LuRaz(p.data(), pi_bath.data(),
                                                 M_bath.energies.data(), T_bath, n);
            csv.row({t, dL1, dKL, de});
            if (trajectory_p && trajectory_t) {
                trajectory_p->push_back(p);
                trajectory_t->push_back(t);
            }
        }
        if (step < n_steps) rk4_step_dense(M_bath, p, dt);
    }
}

// =========================================================================
// Main
// =========================================================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0)
            std::cerr << "usage: " << argv[0] << " config.ini\n";
        MPI_Finalize();
        return 1;
    }

    Config cfg;
    try { cfg.load(argv[1]); }
    catch (const std::exception& e) {
        if (rank == 0) std::cerr << "Config error: " << e.what() << "\n";
        MPI_Finalize();
        return 1;
    }

    std::string system_name = cfg.get_string("system", "name", "three_state");
    double T_bath           = cfg.require_double("system", "T_bath");
    int    N_ising          = cfg.get_int   ("system", "N_ising", 8);
    double J_ising          = cfg.get_double("system", "J", -2.0);
    double h_ising          = cfg.get_double("system", "h", 0.2);

    double t_max            = cfg.require_double("integration", "t_max");
    double dt               = cfg.require_double("integration", "dt");
    int    log_every        = cfg.get_int   ("integration", "log_every", 20);

    std::vector<double> T_inits = cfg.get_double_list("quench", "T_inits");
    if (T_inits.empty()) {
        if (rank == 0) std::cerr << "No T_inits specified in [quench]\n";
        MPI_Finalize();
        return 1;
    }
    bool save_thermomaj     = cfg.get_bool("thermomajorization", "enable", true);
    int  thermomaj_logstep  = cfg.get_int ("thermomajorization", "log_every", 50);

    std::string out_dir = cfg.get_string("io", "output_dir", "results/markovian");

    if (rank == 0) {
        std::cout << "=== Markovian Mpemba (Lu & Raz 2017) ===\n";
        std::cout << "  system     = " << system_name << "\n";
        std::cout << "  T_bath     = " << T_bath << "\n";
        std::cout << "  T_inits    = [";
        for (double T : T_inits) std::cout << " " << T;
        std::cout << " ]\n";
        std::cout << "  t_max      = " << t_max << ",  dt = " << dt << "\n";
        std::cout << "  ranks      = " << size << ",  threads/rank = "
                  << omp_get_max_threads() << "\n";
    }

    // Build the Markov matrix (depends only on T_bath)
    DenseMarkov M;
    if (system_name == "three_state") {
        M = system_three_state(T_bath);
    } else if (system_name == "ising") {
        if (N_ising > 12) {
            if (rank == 0) std::cerr << "N_ising > 12 not supported.\n";
            MPI_Finalize(); return 1;
        }
        IsingChain ic = build_ising_chain(N_ising, J_ising, h_ising);
        M = build_ising_markov(ic, T_bath);
    } else {
        if (rank == 0) std::cerr << "Unknown system: " << system_name << "\n";
        MPI_Finalize(); return 1;
    }

    // Distribute T_inits across ranks
    int nT = (int)T_inits.size();
    int per_rank = (nT + size - 1) / size;
    int t_lo = std::min(rank * per_rank, nT);
    int t_hi = std::min(t_lo + per_rank, nT);

    auto t_start = std::chrono::steady_clock::now();

    // For each T_init we keep its distribution trajectory if thermomaj is enabled
    std::vector<std::vector<std::vector<double>>> all_p(t_hi - t_lo);
    std::vector<std::vector<double>>              all_t(t_hi - t_lo);

    for (int ti = t_lo; ti < t_hi; ++ti) {
        double T_init = T_inits[ti];
        std::ostringstream oss;
        oss << out_dir << "/" << system_name << "_Tinit_" << T_init << ".csv";
        std::cout << "  rank " << rank << " -> T_init=" << T_init
                  << " -> " << oss.str() << "\n";
        if (save_thermomaj) {
            run_quench(M, T_init, T_bath, t_max, dt, log_every, oss.str(),
                       &all_p[ti - t_lo], &all_t[ti - t_lo]);
        } else {
            run_quench(M, T_init, T_bath, t_max, dt, log_every, oss.str());
        }
    }

    // ---- Thermomajorization: compare hottest vs coldest at each logged time ----
    if (save_thermomaj && rank == 0) {
        // Find which ranks have the hottest and coldest indices
        int idx_hot  = nT - 1;
        int idx_cold = 0;
        // For simplicity, in this driver we recompute on rank 0:
        // re-run two quenches locally to get the trajectories.
        std::vector<std::vector<double>> p_hot, p_cold;
        std::vector<double> t_hot, t_cold;
        run_quench(M, T_inits[idx_hot],  T_bath, t_max, dt, log_every,
                   out_dir + "/_thermomaj_hot.csv", &p_hot, &t_hot);
        run_quench(M, T_inits[idx_cold], T_bath, t_max, dt, log_every,
                   out_dir + "/_thermomaj_cold.csv", &p_cold, &t_cold);
        CSVWriter csvtm(out_dir + "/thermomajorization_diagnostic.csv");
        csvtm.header({"t", "max_gap_hot_minus_cold", "min_gap_hot_minus_cold",
                      "hot_dominates_cold", "curves_cross"});
        int every = std::max(1, thermomaj_logstep / std::max(1, log_every));
        for (std::size_t k = 0; k < t_hot.size(); k += every) {
            if (k >= p_cold.size()) break;
            auto cmp = thermomajorizes(p_hot[k], p_cold[k], M.energies, T_bath);
            csvtm.row({t_hot[k], cmp.max_gap, cmp.min_gap,
                       cmp.dominates_weak ? 1.0 : 0.0,
                       cmp.curves_cross ? 1.0 : 0.0});
        }
        std::cout << "[Markovian] thermomajorization diagnostic written.\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_end = std::chrono::steady_clock::now();
    if (rank == 0) {
        double elapsed = std::chrono::duration<double>(t_end - t_start).count();
        std::cout << "[Markovian] done in " << elapsed << " s\n";
    }
    MPI_Finalize();
    return 0;
}
