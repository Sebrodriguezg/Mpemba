// modules/langevin_inverse/main.cpp
//
// MODULE: Anomalous HEATING - the INVERSE Mpemba effect.
//
// REFERENCE
//   Avinash Kumar, Raphaël Chétrite, John Bechhoefer, "Anomalous heating in
//   a colloidal system", PNAS 119, e2118484119 (2022).
//   [kumaretal2022anomalousheatinginacolloidalsystem.pdf in project]
//
// PHYSICAL MODEL
// --------------
// Overdamped Langevin in 1D tilted double-well potential (same as forward
// Mpemba module). Difference: instead of cooling, we HEAT.
//
// Protocol:
//   * Bath at HOT temperature T_b.
//   * Prepare two initial conditions at COLD temperatures
//        T_cold < T_cool < T_b
//   * Both ensembles relax UPWARD in temperature.
//
// The INVERSE Mpemba effect occurs when the colder initial sample
// (T_cold) reaches the bath equilibrium FASTER than the cooler one (T_cool).
// Kumar-Chétrite (2022) confirmed this experimentally.
//
// Note: This effect is generically WEAKER than the forward one because
// entropic effects in the tilted potential favor downward (cooling)
// transitions over upward (heating). To observe it cleanly, one needs
// a finely tuned potential and a sharp temperature contrast.

#include <mpi.h>
#include <omp.h>

#include "../../core/config_parser.hpp"
#include "../../core/philox_rng.hpp"
#include "../../core/distance_metrics.hpp"
#include "../../core/csv_io.hpp"

#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <numeric>

using namespace mpemba;

// -----------------------------------------------------------------------
// Tilted double-well potential of Kumar-Chétrite 2022 Fig. 2:
//   U(x) = U0 * (x^4/4 - x^2/2) + a*x  with x in [xmin, xmax]
//   Asymmetry parameter alpha = |xmax/xmin| ~ 2
// -----------------------------------------------------------------------
struct TiltedDoubleWell {
    double U0   = 4.0;
    double a    = 1.2;     // tilt
    double xmin = -1.0;
    double xmax = 2.0;
    double wall_slope = 80.0;  // soft wall outside [xmin, xmax]

    inline double potential(double x) const {
        if (x < xmin) return wall_slope * (xmin - x) + potential_core(xmin);
        if (x > xmax) return wall_slope * (x - xmax) + potential_core(xmax);
        return potential_core(x);
    }
    inline double potential_core(double x) const {
        return U0 * (x*x*x*x / 4.0 - x*x / 2.0) + a * x;
    }
    inline double force(double x) const {
        if (x < xmin) return  wall_slope;
        if (x > xmax) return -wall_slope;
        return -(U0 * (x*x*x - x) + a);
    }
};

struct EquilDistribution {
    std::vector<double> x_grid, x_centers, pdf, cdf;
    double dx;
    void build(const TiltedDoubleWell& pot, double T, int n_bins) {
        x_grid.resize(n_bins + 1);
        x_centers.resize(n_bins);
        pdf.assign(n_bins, 0.0);
        cdf.assign(n_bins + 1, 0.0);
        dx = (pot.xmax - pot.xmin) / n_bins;
        for (int i = 0; i <= n_bins; ++i) x_grid[i] = pot.xmin + i * dx;
        for (int i = 0; i < n_bins; ++i)
            x_centers[i] = pot.xmin + (i + 0.5) * dx;
        double Umin = pot.potential_core(x_centers[0]);
        for (int i = 1; i < n_bins; ++i)
            Umin = std::min(Umin, pot.potential_core(x_centers[i]));
        double Z = 0.0;
        for (int i = 0; i < n_bins; ++i) {
            pdf[i] = std::exp(-(pot.potential_core(x_centers[i]) - Umin) / T);
            Z += pdf[i];
        }
        Z *= dx;
        for (int i = 0; i < n_bins; ++i) pdf[i] /= Z;
        cdf[0] = 0.0;
        for (int i = 0; i < n_bins; ++i) cdf[i + 1] = cdf[i] + pdf[i] * dx;
        double cmax = cdf.back();
        if (cmax > 0) for (auto& c : cdf) c /= cmax;
    }
    double sample(PhiloxRNG& rng) const {
        double u = rng.uniform();
        int lo = 0, hi = (int)cdf.size() - 1;
        while (lo + 1 < hi) {
            int mid = (lo + hi) / 2;
            if (cdf[mid] <= u) lo = mid; else hi = mid;
        }
        double cl = cdf[lo], cr = cdf[lo + 1];
        double xl = x_grid[lo], xr = x_grid[lo + 1];
        if (cr - cl < 1e-300) return xl;
        return xl + (xr - xl) * (u - cl) / (cr - cl);
    }
};

struct HeatingResult {
    std::vector<double> times;
    std::vector<double> D_L1;
    std::vector<double> D_KL;
};

HeatingResult run_heating(const TiltedDoubleWell& pot,
                          const EquilDistribution& init_dist,
                          const EquilDistribution& bath_dist,
                          double T_bath, double t_max, double dt,
                          int N_traj_local, int n_bins,
                          int n_steps_per_log,
                          int mpi_rank, uint64_t seed) {
    HeatingResult res;
    int n_steps = (int)std::ceil(t_max / dt);
    int n_logs  = n_steps / n_steps_per_log + 1;
    res.times.resize(n_logs);
    res.D_L1.resize(n_logs);
    res.D_KL.resize(n_logs);

    std::vector<double> X(N_traj_local);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        PhiloxRNG rng = make_rng(mpi_rank, tid, seed ^ 0xABCDEF1234567890ULL);
        rng.counter += (uint64_t)tid * 1000003ULL;
        #pragma omp for schedule(static)
        for (int i = 0; i < N_traj_local; ++i) {
            X[i] = init_dist.sample(rng);
        }
    }

    std::vector<long long> hist_local(n_bins, 0);
    std::vector<long long> hist_global(n_bins, 0);
    std::vector<double>    pxt(n_bins);

    auto histogram = [&]() {
        std::fill(hist_local.begin(), hist_local.end(), 0LL);
        int nthreads = omp_get_max_threads();
        std::vector<std::vector<long long>> hist_t(nthreads,
                                                   std::vector<long long>(n_bins, 0));
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto& h = hist_t[tid];
            #pragma omp for schedule(static)
            for (int i = 0; i < N_traj_local; ++i) {
                double x = X[i];
                int b = (int)std::floor((x - pot.xmin) / init_dist.dx);
                if (b >= 0 && b < n_bins) h[b]++;
            }
        }
        for (int t = 0; t < nthreads; ++t)
            for (int b = 0; b < n_bins; ++b) hist_local[b] += hist_t[t][b];
        MPI_Allreduce(hist_local.data(), hist_global.data(), n_bins,
                      MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        long long total = std::accumulate(hist_global.begin(), hist_global.end(), 0LL);
        double inv_total_dx = (total > 0) ? 1.0 / (total * init_dist.dx) : 0.0;
        for (int b = 0; b < n_bins; ++b) pxt[b] = hist_global[b] * inv_total_dx;
        double dL1 = distance_L1_continuous(pxt, bath_dist.pdf, init_dist.dx);
        double dKL = 0.0;
        const double eps = 1e-12;
        for (int b = 0; b < n_bins; ++b) {
            double p = std::max(pxt[b], eps);
            double q = std::max(bath_dist.pdf[b], eps);
            dKL += p * (std::log(p) - std::log(q));
        }
        dKL *= init_dist.dx;
        return std::make_pair(dL1, dKL);
    };

    int log_idx = 0;
    {
        auto [dL1, dKL] = histogram();
        res.times[log_idx] = 0.0;
        res.D_L1[log_idx] = dL1;
        res.D_KL[log_idx] = dKL;
        log_idx++;
    }

    double sqrt_2T_dt = std::sqrt(2.0 * T_bath * dt);
    int nthreads = omp_get_max_threads();
    std::vector<PhiloxRNG> rngs;
    for (int tid = 0; tid < nthreads; ++tid) {
        PhiloxRNG r = make_rng(mpi_rank, tid, seed ^ 0xFEEDFACEBADC0FFEULL);
        r.counter += (uint64_t)tid * 31337ULL;
        rngs.push_back(r);
    }

    for (int step = 1; step <= n_steps; ++step) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto& rng = rngs[tid];
            #pragma omp for schedule(static)
            for (int i = 0; i < N_traj_local; ++i) {
                double xi = X[i];
                double F  = pot.force(xi);
                double eta = rng.normal();
                xi += F * dt + sqrt_2T_dt * eta;
                X[i] = xi;
            }
        }
        if (step % n_steps_per_log == 0 && log_idx < n_logs) {
            auto [dL1, dKL] = histogram();
            res.times[log_idx] = step * dt;
            res.D_L1[log_idx] = dL1;
            res.D_KL[log_idx] = dKL;
            log_idx++;
        }
    }
    res.times.resize(log_idx);
    res.D_L1.resize(log_idx);
    res.D_KL.resize(log_idx);
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

    int    N_traj_total   = cfg.get_int   ("ensemble", "n_trajectories", 200000);
    double T_cold         = cfg.require_double("quench", "T_cold");
    double T_cool         = cfg.require_double("quench", "T_cool");
    double T_bath         = cfg.require_double("quench", "T_bath");
    double t_max          = cfg.require_double("integration", "t_max");
    double dt             = cfg.require_double("integration", "dt");
    int    n_bins         = cfg.get_int   ("integration", "n_bins", 200);
    int    n_steps_per_log = cfg.get_int  ("integration", "n_steps_per_log", 50);
    uint64_t seed         = (uint64_t)cfg.get_double("ensemble", "seed", 0xC01DBEEFULL);
    std::string out_dir   = cfg.get_string("io", "output_dir",
                                           "results/langevin_inverse");

    TiltedDoubleWell pot;
    pot.U0   = cfg.require_double("potential", "U0");
    pot.a    = cfg.require_double("potential", "tilt_a");
    pot.xmin = cfg.get_double    ("potential", "xmin", -1.0);
    pot.xmax = cfg.get_double    ("potential", "xmax",  2.0);

    if (rank == 0) {
        std::cout << "=== Inverse Mpemba (anomalous heating, Kumar-Chetrite 2022) ===\n";
        std::cout << "  N_traj=" << N_traj_total
                  << "  T_cold=" << T_cold
                  << "  T_cool=" << T_cool
                  << "  T_bath=" << T_bath << "\n";
        std::cout << "  U0=" << pot.U0 << "  tilt=" << pot.a << "\n";
        std::cout << "  Note: T_cold < T_cool < T_bath required.\n";
    }

    if (!(T_cold < T_cool && T_cool < T_bath)) {
        if (rank == 0) std::cerr << "ERROR: need T_cold < T_cool < T_bath\n";
        MPI_Finalize(); return 1;
    }

    int N_traj_local = N_traj_total / size + (rank < (N_traj_total % size) ? 1 : 0);

    EquilDistribution bath_dist, cold_dist, cool_dist;
    bath_dist.build(pot, T_bath, n_bins);
    cold_dist.build(pot, T_cold, n_bins);
    cool_dist.build(pot, T_cool, n_bins);

    auto t_start = std::chrono::steady_clock::now();
    if (rank == 0) std::cout << "  running heating from T_cold = " << T_cold << "\n";
    auto res_cold = run_heating(pot, cold_dist, bath_dist, T_bath, t_max, dt,
                                N_traj_local, n_bins, n_steps_per_log, rank, seed + 11);
    if (rank == 0) std::cout << "  running heating from T_cool = " << T_cool << "\n";
    auto res_cool = run_heating(pot, cool_dist, bath_dist, T_bath, t_max, dt,
                                N_traj_local, n_bins, n_steps_per_log, rank, seed + 13);

    auto t_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    if (rank == 0) {
        std::cout << "[Langevin-inverse] done in " << elapsed << " s\n";
        CSVWriter csv(out_dir + "/distances.csv");
        csv.header({"t", "D_L1_cold", "D_L1_cool", "D_KL_cold", "D_KL_cool"});
        std::size_t Ntimes = res_cold.times.size();
        for (std::size_t k = 0; k < Ntimes; ++k) {
            csv.row({res_cold.times[k], res_cold.D_L1[k], res_cool.D_L1[k],
                     res_cold.D_KL[k], res_cool.D_KL[k]});
        }
        // Crossover check: inverse ME = cold drops below cool
        bool found = false;
        for (std::size_t k = 1; k < Ntimes; ++k) {
            if (res_cold.D_L1[k] < res_cool.D_L1[k]) {
                std::cout << "  *** INVERSE Mpemba crossover at t = "
                          << res_cold.times[k]
                          << "  (D_L1_cold=" << res_cold.D_L1[k]
                          << " < D_L1_cool=" << res_cool.D_L1[k] << ")\n";
                found = true; break;
            }
        }
        if (!found)
            std::cout << "  no inverse crossover observed (effect is generically weaker).\n";
    }
    MPI_Finalize();
    return 0;
}
