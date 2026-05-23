// modules/thermomajorization/main.cpp
//
// MODULE: Thermomajorization diagnostic (universal Mpemba detector).
//
// REFERENCE
//   Tan Van Vu & Hisao Hayakawa, "Thermomajorization Mpemba effect",
//   Phys. Rev. Lett. 134, 107101 (2025).
//   arXiv:2502.00123 (full extended version with correlations)
//
// WHAT THIS DOES
// --------------
// Reads two probability-distribution trajectories from CSV files (one per
// preparation; columns t, p_0, p_1, ..., p_{n-1}) and the corresponding
// energy levels. For each common time t, computes the thermomajorization
// curves of both distributions relative to the bath Gibbs distribution and
// determines:
//
//   * Does p_hot thermomajorize p_warm at this time? -> Mpemba is certified
//     for ALL monotone metrics simultaneously.
//   * Do the curves cross? -> ambiguous, Mpemba is metric-dependent.
//   * Max gap and min gap (signed): quantifies the "strength" of the effect.
//
// This module is meant to be run as a POST-PROCESSING step on the output
// of other modules (markovian, klich_raz, langevin).
//
// USAGE: ./mpemba_thermomajorization config.ini
//
// Config requires:
//   [bath] T_bath = ...
//   [files]
//     energies      = path/to/energies.csv    ; one column with energies
//     p_hot_trajectory = path/to/p_hot.csv    ; rows: t, p_0, p_1, ...
//     p_warm_trajectory = path/to/p_warm.csv
//   [io] output_dir = ...

#include <mpi.h>
#include <omp.h>

#include "../../core/config_parser.hpp"
#include "../../core/thermomajorization.hpp"
#include "../../core/csv_io.hpp"

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <stdexcept>

using namespace mpemba;

// Load a CSV with columns: t, p_0, p_1, ..., p_{n-1}.
// Returns the time vector and a vector of distributions (one per row).
std::pair<std::vector<double>, std::vector<std::vector<double>>>
load_trajectory_csv(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) throw std::runtime_error("Cannot open " + path);
    std::string line;
    std::getline(ifs, line);  // header
    std::vector<double> times;
    std::vector<std::vector<double>> dists;
    while (std::getline(ifs, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<double> row;
        while (std::getline(ss, token, ',')) {
            row.push_back(std::stod(token));
        }
        if (row.empty()) continue;
        times.push_back(row[0]);
        dists.emplace_back(row.begin() + 1, row.end());
    }
    return {times, dists};
}

// Load energies (one number per line)
std::vector<double> load_energies(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) throw std::runtime_error("Cannot open " + path);
    std::vector<double> out;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        out.push_back(std::stod(line));
    }
    return out;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    (void)size;

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

    double T_bath = cfg.require_double("bath", "T_bath");
    std::string energies_path = cfg.require_string("files", "energies");
    std::string hot_path      = cfg.require_string("files", "p_hot_trajectory");
    std::string warm_path     = cfg.require_string("files", "p_warm_trajectory");
    std::string out_dir       = cfg.get_string("io", "output_dir",
                                               "results/thermomajorization");

    if (rank != 0) { MPI_Finalize(); return 0; }   // run only on rank 0

    std::cout << "=== Thermomajorization diagnostic (Vu & Hayakawa 2025) ===\n";
    std::cout << "  T_bath  = " << T_bath << "\n";
    std::cout << "  energies = " << energies_path << "\n";
    std::cout << "  hot      = " << hot_path << "\n";
    std::cout << "  warm     = " << warm_path << "\n";

    std::vector<double> energies = load_energies(energies_path);
    auto [t_hot,  p_hot]  = load_trajectory_csv(hot_path);
    auto [t_warm, p_warm] = load_trajectory_csv(warm_path);

    std::cout << "  loaded " << p_hot.size() << " hot snapshots, "
              << p_warm.size() << " warm snapshots, "
              << energies.size() << " energy levels\n";

    CSVWriter csv(out_dir + "/thermomaj_certification.csv");
    csv.header({"t", "max_gap_hot_minus_warm", "min_gap_hot_minus_warm",
                "hot_dominates_warm", "curves_cross",
                "first_crossover_certified_t"});
    double t_certified = -1.0;
    std::size_t N = std::min(p_hot.size(), p_warm.size());
    for (std::size_t k = 0; k < N; ++k) {
        auto cmp = thermomajorizes(p_warm[k], p_hot[k], energies, T_bath);
        // We test whether warm "thermomajorizes" hot (i.e., warm is further
        // from equilibrium). Mpemba effect = at some time, hot becomes
        // closer to equilibrium than warm in the thermomaj sense, which
        // means warm thermomajorizes hot AND they don't cross.
        bool mpemba_certified = cmp.dominates_strict;
        if (mpemba_certified && t_certified < 0)
            t_certified = t_hot[k];
        csv.row({t_hot[k], cmp.max_gap, cmp.min_gap,
                 mpemba_certified ? 1.0 : 0.0,
                 cmp.curves_cross ? 1.0 : 0.0,
                 t_certified});
    }
    if (t_certified > 0) {
        std::cout << "\n*** Mpemba effect CERTIFIED at t = " << t_certified
                  << " (for ALL monotone metrics simultaneously) ***\n";
    } else {
        std::cout << "\n[thermomaj] No metric-universal Mpemba certification "
                  << "found in the data.\n";
        std::cout << "  (the effect may still be present in specific metrics)\n";
    }

    MPI_Finalize();
    return 0;
}
