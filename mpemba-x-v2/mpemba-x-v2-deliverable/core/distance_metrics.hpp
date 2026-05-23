// distance_metrics.hpp
// Distance-from-equilibrium functionals used to *define* the Mpemba effect.
//
// Three distances, three independent tests for the ME.
// If the effect appears in only one, it is an artifact of the metric (Burridge-Linden critique).
//
// Implemented:
//   - L1 distance         (Kumar & Bechhoefer 2020, Methods)
//   - Kullback-Leibler    (Lu & Raz 2017, alternative)
//   - Entropic Lu-Raz     (Eq. 5 of Lu & Raz 2017)
//
// All three satisfy the three properties of a "good" distance for the ME
// (monotonic decrease in time, monotonic increase with |T_init - T_bath|, convexity).
#pragma once
#include <vector>
#include <cmath>
#include <cstddef>
#include <algorithm>

namespace mpemba {

// L1 distance between two probability distributions of size n.
// D_L1 = sum_i |p_i - pi_i|
inline double distance_L1(const double* p, const double* pi_eq, std::size_t n) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        s += std::fabs(p[i] - pi_eq[i]);
    }
    return s;
}

// Kullback-Leibler divergence D_KL(p || pi_eq) = sum p_i * log(p_i / pi_eq_i)
// Adds an epsilon floor to avoid log(0).
inline double distance_KL(const double* p, const double* pi_eq, std::size_t n,
                          double eps = 1e-300) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double pi = std::max(p[i],     eps);
        double qi = std::max(pi_eq[i], eps);
        s += pi * (std::log(pi) - std::log(qi));
    }
    return s;
}

// Entropic distance of Lu & Raz (2017), Eq. (5):
//   D_e[p; T_b] = sum_i ( E_i * (p_i - pi_eq_i) / T_b + p_i ln p_i - pi_eq_i ln pi_eq_i )
// where the energies E_i and bath temperature T_b are required.
// Units: kB = 1.
inline double distance_entropic_LuRaz(const double* p, const double* pi_eq,
                                      const double* energies, double T_bath,
                                      std::size_t n, double eps = 1e-300) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double pi   = std::max(p[i],     eps);
        double pieq = std::max(pi_eq[i], eps);
        double dp   = p[i] - pi_eq[i];
        s += energies[i] * dp / T_bath
           + p[i]    * std::log(pi)
           - pi_eq[i]* std::log(pieq);
    }
    return s;
}

// Continuous (binned) L1 distance for distributions on a discretized 1D state space.
// dx is the bin width.
inline double distance_L1_continuous(const std::vector<double>& p,
                                     const std::vector<double>& pi_eq,
                                     double dx) {
    double s = 0.0;
    const std::size_t n = std::min(p.size(), pi_eq.size());
    for (std::size_t i = 0; i < n; ++i) {
        s += std::fabs(p[i] - pi_eq[i]);
    }
    return s * dx;
}

// Boltzmann distribution from energies at temperature T (kB=1).
// Outputs into pi[] (must be allocated, size n).
inline void boltzmann(const double* energies, double T, double* pi_eq, std::size_t n) {
    double Emin = energies[0];
    for (std::size_t i = 1; i < n; ++i) {
        if (energies[i] < Emin) Emin = energies[i];
    }
    double Z = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        pi_eq[i] = std::exp(-(energies[i] - Emin) / T);
        Z       += pi_eq[i];
    }
    for (std::size_t i = 0; i < n; ++i) {
        pi_eq[i] /= Z;
    }
}

} // namespace mpemba
