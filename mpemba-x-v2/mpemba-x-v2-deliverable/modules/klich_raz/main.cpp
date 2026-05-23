// modules/klich_raz/main.cpp
//
// MODULE 2: Strong Mpemba effect and the Mpemba INDEX
//           (Klich, Raz, Hirschberg & Vucelja, Phys. Rev. X 9, 021060 (2019)).
//
// We implement the Random Energy Model with random barriers exactly as in
// Section V of the paper:
//
//   * L energy levels E_i are IID Gaussian with zero mean, variance sigma_E^2 * ln^2(L)
//     (Eq. 59 of the paper).
//   * Barriers B_ij are IID truncated Gaussian (non-negative), variance sigma_B^2 * ln^2(L).
//   * The Markov matrix is given by an Arrhenius form, with the convention
//     R(T_b) symmetrized through F^{1/2} as in Eq. (4) of the paper.
//
// For each realization of (E, B):
//   1. Build the symmetrized rate matrix \tilde R(T_b).
//   2. Diagonalize it with Jacobi rotations -> get eigenvalues lambda_k and
//      eigenvectors f_k. The slowest decaying mode is f_2 (lambda_1=0 trivially).
//   3. For a grid of initial temperatures, compute the overlap coefficient
//      a_2(T) (Eq. 11 of the paper, with the equivalent form Eq. 35).
//   4. Count sign changes of a_2(T) on T > T_b -> Mpemba index for the DIRECT effect.
//      Sign at T -> inf gives the PARITY of the index (Eq. 15).
//
// Ensemble averaging over many realizations is parallelized with MPI:
// each rank handles a slice of the total realizations. Within a rank, OpenMP
// parallelizes the grid of initial temperatures.
//
// Output:
//   results/klich_raz/index_stats_L<L>_Tb<Tb>.csv
//   results/klich_raz/a2_sample.csv   (a single sample trace, for plotting)

#include <mpi.h>
#include <omp.h>

#include "../../core/config_parser.hpp"
#include "../../core/philox_rng.hpp"
#include "../../core/jacobi_eigen.hpp"
#include "../../core/distance_metrics.hpp"
#include "../../core/csv_io.hpp"

#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <sstream>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <chrono>

using namespace mpemba;

// Build energies E_i ~ Gauss(0, sigma_E * log(L))
std::vector<double> sample_energies(int L, double sigma_E, PhiloxRNG& rng) {
    std::vector<double> E(L);
    double sd = sigma_E * std::log((double)L);
    for (int i = 0; i < L; ++i) E[i] = rng.normal(0.0, sd);
    return E;
}

// Build barriers B[i*L+j] IID truncated Gaussian (B >= 0), symmetric, zero diagonal.
std::vector<double> sample_barriers(int L, double sigma_B, PhiloxRNG& rng) {
    std::vector<double> B((std::size_t)L * L, 0.0);
    double sd = sigma_B * std::log((double)L);
    for (int i = 0; i < L; ++i) {
        for (int j = i + 1; j < L; ++j) {
            double b;
            do { b = rng.normal(0.0, sd); } while (b < 0.0);
            B[(std::size_t)i * L + j] = b;
            B[(std::size_t)j * L + i] = b;
        }
    }
    return B;
}

// Build the symmetrized rate matrix \tilde R at bath temperature T_b.
//   R_ij = Gamma exp( - (B_ij - E_j) / T_b )    (i != j)
//   R_ii = - sum_{k!=i} R_ki
//   pi_i = e^{-E_i/T_b} / Z
//   F = diag(pi_i)
//   \tilde R = F^{-1/2} R F^{1/2}    (this is symmetric by detailed balance)
//
// Returns the symmetrized matrix (row-major), the equilibrium pi, and the
// associated "f" eigenvector basis transformation parameters.
struct SymRateBuild {
    std::vector<double> Rtilde;     // L*L symmetric
    std::vector<double> pi_eq;       // L
    std::vector<double> sqrt_pi;     // L (F^{1/2})
};
SymRateBuild build_symmetrized_rate(const std::vector<double>& E,
                                    const std::vector<double>& B,
                                    int L, double T_b, double Gamma = 1.0) {
    SymRateBuild out;
    out.Rtilde.assign((std::size_t)L * L, 0.0);
    out.pi_eq.assign(L, 0.0);
    out.sqrt_pi.assign(L, 0.0);

    // pi_i = e^{-E_i/T_b}/Z. Shift by minimum for numerical safety.
    double Emin = *std::min_element(E.begin(), E.end());
    double Z = 0.0;
    for (int i = 0; i < L; ++i) {
        out.pi_eq[i] = std::exp(-(E[i] - Emin) / T_b);
        Z += out.pi_eq[i];
    }
    for (int i = 0; i < L; ++i) {
        out.pi_eq[i]  /= Z;
        out.sqrt_pi[i] = std::sqrt(out.pi_eq[i]);
    }

    // R_ij (i != j) = Gamma * exp(-(B_ij - E_j)/T_b). Then symmetrize:
    //   Rtilde_ij = R_ij * sqrt(pi_j) / sqrt(pi_i)
    // For i != j this is symmetric in (i,j) by detailed balance.
    for (int i = 0; i < L; ++i) {
        double diag = 0.0;
        for (int j = 0; j < L; ++j) {
            if (i == j) continue;
            double Bij = B[(std::size_t)i * L + j];
            double Rij = Gamma * std::exp(-(Bij - E[j]) / T_b);
            out.Rtilde[(std::size_t)i * L + j] =
                Rij * out.sqrt_pi[j] / std::max(out.sqrt_pi[i], 1e-300);
            // Accumulate -R_ki contribution to column-sum for j-th column's diagonal
            // (handled below by symmetric diagonal correction)
        }
    }
    // Diagonal of R: -sum_{k!=i} R_ki.
    // Then Rtilde_ii = R_ii (since F^{-1/2} R_ii F^{1/2} = R_ii on the diagonal).
    for (int i = 0; i < L; ++i) {
        double col_sum = 0.0;
        for (int k = 0; k < L; ++k) {
            if (k == i) continue;
            double Bki = B[(std::size_t)k * L + i];
            double Rki = Gamma * std::exp(-(Bki - E[i]) / T_b);
            col_sum += Rki;
        }
        out.Rtilde[(std::size_t)i * L + i] = -col_sum;
    }
    // Symmetrize numerically to clean tiny asymmetries from floating point.
    for (int i = 0; i < L; ++i) {
        for (int j = i + 1; j < L; ++j) {
            double a = out.Rtilde[(std::size_t)i * L + j];
            double b = out.Rtilde[(std::size_t)j * L + i];
            double m = 0.5 * (a + b);
            out.Rtilde[(std::size_t)i * L + j] = m;
            out.Rtilde[(std::size_t)j * L + i] = m;
        }
    }
    return out;
}

// Given the symmetrized rate matrix already diagonalized to f-vectors and lambdas,
// compute a_2(T) for the second-slowest mode using Eq. (35) of Klich-Raz:
//
//   a_2(T) = sum_i (f2)_i * e^{-(beta - beta_b/2) E_i} / Z(T)
//
// where Z(T) = sum_j exp(-E_j / T) is the partition function of the system at T.
// Result normalized by ||f2||^2 (== 1 for unit eigenvectors).
double compute_a2(const std::vector<double>& f2,
                  const std::vector<double>& E, int L, double T, double T_b) {
    double beta   = 1.0 / T;
    double beta_b = 1.0 / T_b;
    // Z(T)
    double Emin = *std::min_element(E.begin(), E.end());
    double ZT = 0.0;
    for (int i = 0; i < L; ++i) ZT += std::exp(-(E[i] - Emin) / T);
    double a2 = 0.0;
    for (int i = 0; i < L; ++i) {
        double w = std::exp(-(beta - 0.5 * beta_b) * (E[i] - Emin));
        a2 += f2[i] * w;
    }
    return a2 / ZT;
}

// For each realization, find the Mpemba index = number of sign changes of
// a_2(T) on T in (T_b, T_max). We sample on a log-spaced grid.
struct IndexResult {
    int index;            // number of zeros found
    int parity;           // +1 if last sign matches T->inf limit prediction, else -1
    double a2_at_inf;     // value of a_2 at the largest T (proxy for infinity)
    std::vector<double> Ts;
    std::vector<double> a2s;
};
IndexResult compute_mpemba_index(const std::vector<double>& f2,
                                 const std::vector<double>& E, int L,
                                 double T_b, double T_max, int nT) {
    IndexResult out;
    out.Ts.resize(nT);
    out.a2s.resize(nT);
    double logT_lo = std::log(T_b * 1.05);
    double logT_hi = std::log(T_max);
    for (int k = 0; k < nT; ++k) {
        double T = std::exp(logT_lo + (logT_hi - logT_lo) * k / (nT - 1));
        out.Ts[k]  = T;
        out.a2s[k] = compute_a2(f2, E, L, T, T_b);
    }
    int count = 0;
    for (int k = 1; k < nT; ++k) {
        if ((out.a2s[k - 1] > 0) != (out.a2s[k] > 0)) ++count;
    }
    out.index = count;
    out.a2_at_inf = out.a2s.back();
    out.parity    = (out.a2_at_inf > 0) ? +1 : -1;
    return out;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) std::cerr << "usage: " << argv[0] << " config.ini\n";
        MPI_Finalize();
        return 1;
    }
    Config cfg;
    try { cfg.load(argv[1]); }
    catch (const std::exception& e) {
        if (rank == 0) std::cerr << "Config error: " << e.what() << "\n";
        MPI_Finalize(); return 1;
    }

    int      L          = cfg.get_int   ("model",      "L",         10);
    int      n_realiz   = cfg.get_int   ("ensemble",   "n_realiz",  2000);
    double   T_b        = cfg.require_double("model",  "T_bath");
    double   sigma_E    = cfg.get_double("model",      "sigma_E",   1.0);
    double   sigma_B    = cfg.get_double("model",      "sigma_B",   5.0);
    double   T_max      = cfg.get_double("scan",       "T_max",     1000.0);
    int      nT_grid    = cfg.get_int   ("scan",       "n_T_grid",  200);
    uint64_t seed       = (uint64_t)cfg.get_double("ensemble", "seed", 12648430.0);
    std::string out_dir = cfg.get_string("io",         "output_dir", "results/klich_raz");

    if (rank == 0) {
        std::cout << "[KlichRaz] L=" << L
                  << "  N_realiz=" << n_realiz
                  << "  T_b=" << T_b
                  << "  sigma_E=" << sigma_E
                  << "  sigma_B=" << sigma_B
                  << "  ranks=" << size
                  << "  threads/rank=" << omp_get_max_threads() << "\n";
    }

    // Distribute realizations across ranks
    int per_rank = (n_realiz + size - 1) / size;
    int r_lo = std::min(rank * per_rank, n_realiz);
    int r_hi = std::min(r_lo + per_rank, n_realiz);

    // Aggregate counters: index histogram for direct effect on T_b < T_critical
    // Following the paper Fig. 10: we histogram index up to some cap.
    constexpr int INDEX_CAP = 8;
    std::vector<long long> index_hist_local(INDEX_CAP + 1, 0);
    std::vector<long long> parity_pos_local(INDEX_CAP + 1, 0);
    long long total_local = 0;

    // Save a single representative sample (the first realization on rank 0)
    bool save_sample = (rank == 0);

    auto t_start = std::chrono::steady_clock::now();

    // OpenMP parallelize over realizations within a rank.
    // Each thread builds its own L*L matrix and does Jacobi.
    #pragma omp parallel
    {
        std::vector<long long> index_hist_t(INDEX_CAP + 1, 0);
        std::vector<long long> parity_pos_t(INDEX_CAP + 1, 0);
        long long total_t = 0;
        #pragma omp for schedule(dynamic, 4)
        for (int r_idx = r_lo; r_idx < r_hi; ++r_idx) {
            uint64_t stream = seed
                              + (uint64_t)rank * 1000003ULL
                              + (uint64_t)r_idx;
            PhiloxRNG rng = make_rng(rank, omp_get_thread_num(),
                                     stream);
            // Generate energies and barriers
            auto E = sample_energies(L, sigma_E, rng);
            auto B = sample_barriers(L, sigma_B, rng);

            // Build symmetrized rate matrix
            SymRateBuild S = build_symmetrized_rate(E, B, L, T_b);

            // Diagonalize
            std::vector<double> evals, evecs;
            jacobi_eigen(S.Rtilde, L, evals, evecs);

            // Eigenvalues sorted ascending. lambda_0 should be 0 (or numerically near 0).
            // lambda_1 is the slowest decaying mode (most negative -> smallest in magnitude
            // among the non-zero ones, but jacobi_eigen sorts ascending; for relaxation
            // matrix eigenvalues are <= 0, so after ascending sort the largest in
            // absolute value first. We want the eigenvector corresponding to the
            // *second* eigenvalue (next to zero).)
            //
            // In our convention 'evals' is ascending. The zero eigenvalue is the largest
            // (closest to zero). So lambda_2 (slowest non-trivial) = evals[L-2].
            int idx_zero = L - 1;     // closest to zero
            int idx_slow = L - 2;     // slowest non-trivial
            std::vector<double> f2(L);
            for (int i = 0; i < L; ++i) f2[i] = evecs[(std::size_t)i * L + idx_slow];

            // Compute Mpemba index on a temperature grid.
            IndexResult ir = compute_mpemba_index(f2, E, L, T_b, T_max, nT_grid);
            int idx = std::min(ir.index, INDEX_CAP);
            index_hist_t[idx]++;
            if (ir.parity > 0) parity_pos_t[idx]++;
            total_t++;

            // Save a sample with non-zero index if we find one
            // (skip the typical index-0 case which is uninformative).
            if (save_sample && ir.index > 0) {
                #pragma omp critical(save_sample)
                {
                    if (save_sample) {  // re-check inside critical
                        save_sample = false;
                        CSVWriter csv(out_dir + "/a2_sample.csv");
                        csv.header({"T", "a2"});
                        for (int k = 0; k < nT_grid; ++k) {
                            csv.row({ir.Ts[k], ir.a2s[k]});
                        }
                        std::ofstream ofs(out_dir + "/eigenvalues_sample.txt");
                        ofs << "# lambdas (ascending order)\n";
                        for (int i = 0; i < L; ++i) ofs << evals[i] << "\n";
                        ofs << "# index found = " << ir.index << "\n";
                    }
                }
            }
        }
        #pragma omp critical(reduce)
        {
            for (int i = 0; i <= INDEX_CAP; ++i) {
                index_hist_local[i] += index_hist_t[i];
                parity_pos_local[i] += parity_pos_t[i];
            }
            total_local += total_t;
        }
    }

    // MPI reduce
    std::vector<long long> index_hist_global(INDEX_CAP + 1, 0);
    std::vector<long long> parity_pos_global(INDEX_CAP + 1, 0);
    long long total_global = 0;
    MPI_Reduce(index_hist_local.data(), index_hist_global.data(), INDEX_CAP + 1,
               MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(parity_pos_local.data(), parity_pos_global.data(), INDEX_CAP + 1,
               MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_local, &total_global, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        auto t_end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t_end - t_start).count();
        std::ostringstream oss;
        oss << out_dir << "/index_stats_L" << L << "_Tb" << T_b << ".csv";
        CSVWriter csv(oss.str());
        csv.header({"mpemba_index", "count", "frac", "frac_parity_pos"});
        for (int i = 0; i <= INDEX_CAP; ++i) {
            double frac = total_global ? (double)index_hist_global[i] / total_global : 0.0;
            double par_frac = index_hist_global[i] ?
                              (double)parity_pos_global[i] / index_hist_global[i] : 0.0;
            csv.row({(double)i, (double)index_hist_global[i], frac, par_frac});
        }
        std::cout << "[KlichRaz] total realizations = " << total_global
                  << "  elapsed = " << elapsed << " s\n";
        std::cout << "[KlichRaz] index histogram:\n";
        for (int i = 0; i <= INDEX_CAP; ++i) {
            if (index_hist_global[i] > 0)
                std::cout << "  idx=" << i << "  count=" << index_hist_global[i]
                          << "  frac=" << (double)index_hist_global[i]/total_global
                          << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
