// modules/langevin/main.cpp
//
// MODULE 4: Mpemba effect in a colloidal system (Kumar & Bechhoefer,
//           Nature 584, 64 (2020)).
//
// This is the cleanest experimental demonstration of the ME, so it is also
// where simulation makes the effect most evident.
//
// We simulate an overdamped Langevin equation in a 1D ASYMMETRIC DOUBLE-WELL
// potential U(x). The asymmetry creates a metastable shallower well and a
// deeper ground-state well; the basin volumes (in state space) do not match
// the Boltzmann weights of the two macrostates at the bath temperature, which
// is exactly the geometric condition for the ME (Fig. 1b of the paper).
//
// Equation (overdamped, in units where kB = 1, gamma = 1):
//     dx/dt = -U'(x) + xi(t)
//     <xi(t) xi(t')> = 2 T_bath delta(t - t')
//
// Discretized with Euler-Maruyama:
//     x_{n+1} = x_n - U'(x_n) * dt + sqrt(2 T_bath dt) * eta_n,  eta ~ N(0,1)
//
// Protocol (matches Kumar-Bechhoefer):
//   1. Define three initial temperatures: T_h > T_w > T_c = T_bath.
//   2. For each initial T_init in {T_h, T_w, T_c}, sample N_traj initial
//      positions from the equilibrium Boltzmann distribution
//          pi(x; T_init) = exp(-U(x)/T_init) / Z(T_init)
//      using inverse-CDF sampling on a fine grid.
//   3. Quench: evolve all trajectories under the Langevin equation with
//      T_bath = T_c. Compute the distance D_L1(p(x, t), pi(x; T_bath)) at
//      regular intervals by binning trajectory positions.
//   4. The Mpemba effect appears if D_L1 from T_h falls below D_L1 from T_w
//      at some t > 0.
//
// Parallelization:
//   * MPI: each rank handles N_traj/size trajectories.
//   * OpenMP: trajectories within a rank.
//   * Final binning is reduced across ranks with MPI_Allreduce.
//
// References (from the project): kumar2020.pdf, lu2017.pdf

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

// --------------------------------------------------------------------------
// Asymmetric double-well potential.
//   U(x) = U0 * [ (x/L)^4 - 2 (x/L)^2 ] + h * (x/L)
//
// With h != 0 the potential is asymmetric: the well at negative x is the
// ground state (lower), and the one at positive x is metastable.
//
// Derivative:
//   U'(x) = U0 * [4 (x/L)^3 - 4 (x/L)] / L + h / L
// --------------------------------------------------------------------------
struct DoubleWell {
    double U0 = 6.0;     // depth scale
    double L  = 1.0;     // well separation scale
    double h  = 0.8;     // asymmetry
    double xmin = -3.0;
    double xmax =  3.0;

    inline double potential(double x) const {
        double u = x / L;
        double u2 = u * u;
        return U0 * (u2 * u2 - 2.0 * u2) + h * u;
    }
    inline double force(double x) const {
        double u = x / L;
        double u3 = u * u * u;
        return -(U0 * (4.0 * u3 - 4.0 * u) / L + h / L);
    }
};

// --------------------------------------------------------------------------
// Tabulate equilibrium pi(x; T) on a uniform grid and build the inverse-CDF.
// --------------------------------------------------------------------------
struct EquilDistribution {
    std::vector<double> x_grid;        // size n_bins+1 (cell edges)
    std::vector<double> x_centers;     // size n_bins
    std::vector<double> pdf;           // normalized so that sum(pdf)*dx = 1
    std::vector<double> cdf;           // size n_bins+1, cdf[0]=0, cdf[n]=1
    double dx;

    void build(const DoubleWell& pot, double T, int n_bins) {
        x_grid.resize(n_bins + 1);
        x_centers.resize(n_bins);
        pdf.assign(n_bins, 0.0);
        cdf.assign(n_bins + 1, 0.0);
        dx = (pot.xmax - pot.xmin) / n_bins;
        for (int i = 0; i <= n_bins; ++i) x_grid[i] = pot.xmin + i * dx;
        for (int i = 0; i < n_bins; ++i) x_centers[i] = pot.xmin + (i + 0.5) * dx;

        // Find U_min for numerical safety
        double Umin = pot.potential(x_centers[0]);
        for (int i = 1; i < n_bins; ++i) {
            double u = pot.potential(x_centers[i]);
            if (u < Umin) Umin = u;
        }
        double Z = 0.0;
        for (int i = 0; i < n_bins; ++i) {
            pdf[i] = std::exp(-(pot.potential(x_centers[i]) - Umin) / T);
            Z += pdf[i];
        }
        Z *= dx;
        for (int i = 0; i < n_bins; ++i) pdf[i] /= Z;
        cdf[0] = 0.0;
        for (int i = 0; i < n_bins; ++i) cdf[i + 1] = cdf[i] + pdf[i] * dx;
        // Numerical: ensure cdf[end] = 1
        double cmax = cdf.back();
        if (cmax > 0) for (auto& c : cdf) c /= cmax;
    }

    // Sample a position by inverse-CDF (linear interpolation between bins).
    double sample(PhiloxRNG& rng) const {
        double u = rng.uniform();
        // Binary search for cdf[k] <= u < cdf[k+1]
        int lo = 0, hi = (int)cdf.size() - 1;
        while (lo + 1 < hi) {
            int mid = (lo + hi) / 2;
            if (cdf[mid] <= u) lo = mid;
            else               hi = mid;
        }
        // Linear interp inside bin lo
        double cl = cdf[lo], cr = cdf[lo + 1];
        double xl = x_grid[lo], xr = x_grid[lo + 1];
        if (cr - cl < 1e-300) return xl;
        return xl + (xr - xl) * (u - cl) / (cr - cl);
    }
};

// --------------------------------------------------------------------------
// Run an ensemble of overdamped Langevin trajectories starting from pi(x; T_init)
// and evolving under temperature T_bath. Bin positions at each logging time
// to estimate p(x, t) and compute D_L1 against pi(x; T_bath).
// --------------------------------------------------------------------------
struct LangevinResult {
    std::vector<double> times;
    std::vector<double> D_L1;
    std::vector<double> D_KL;
};

LangevinResult run_quench(const DoubleWell& pot,
                          const EquilDistribution& init_dist,
                          const EquilDistribution& bath_dist,
                          double T_bath, double t_max, double dt,
                          int N_traj_local, int n_bins,
                          int log_every, int n_steps_per_log,
                          int mpi_rank, uint64_t seed) {
    LangevinResult res;
    // Times: log every n_steps_per_log * dt
    int n_steps = (int)std::ceil(t_max / dt);
    int n_logs  = n_steps / n_steps_per_log + 1;
    res.times.resize(n_logs);
    res.D_L1.resize(n_logs);
    res.D_KL.resize(n_logs);

    // Allocate trajectory array
    std::vector<double> X(N_traj_local);

    // Initialize positions from initial distribution
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

    // Buffers for histogram
    std::vector<long long> hist_local(n_bins, 0);
    std::vector<long long> hist_global(n_bins, 0);
    std::vector<double>    pxt(n_bins);

    auto histogram = [&]() {
        std::fill(hist_local.begin(), hist_local.end(), 0LL);
        // Reduction with private histograms per thread
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
        // KL divergence with binned distributions
        double dKL = 0.0;
        const double eps = 1e-12;
        for (int b = 0; b < n_bins; ++b) {
            double p = std::max(pxt[b],            eps);
            double q = std::max(bath_dist.pdf[b],  eps);
            dKL += p * (std::log(p) - std::log(q));
        }
        dKL *= init_dist.dx;
        return std::make_pair(dL1, dKL);
    };

    int log_idx = 0;
    // log step 0
    {
        auto [dL1, dKL] = histogram();
        res.times[log_idx] = 0.0;
        res.D_L1[log_idx]  = dL1;
        res.D_KL[log_idx]  = dKL;
        log_idx++;
    }

    double sqrt_2T_dt = std::sqrt(2.0 * T_bath * dt);

    // Each thread keeps its own RNG to generate Gaussians
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
                // Soft reflective boundaries (avoid escape - very rare for U0 large)
                if (xi < pot.xmin) xi = pot.xmin + (pot.xmin - xi) * 0.5;
                if (xi > pot.xmax) xi = pot.xmax - (xi - pot.xmax) * 0.5;
                X[i] = xi;
            }
        }
        if (step % n_steps_per_log == 0 && log_idx < n_logs) {
            auto [dL1, dKL] = histogram();
            res.times[log_idx] = step * dt;
            res.D_L1[log_idx]  = dL1;
            res.D_KL[log_idx]  = dKL;
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

    int    N_traj_total = cfg.get_int   ("ensemble",   "n_trajectories", 200000);
    double T_h          = cfg.require_double("quench", "T_hot");
    double T_w          = cfg.require_double("quench", "T_warm");
    double T_c          = cfg.require_double("quench", "T_bath");
    double t_max        = cfg.require_double("integration", "t_max");
    double dt           = cfg.require_double("integration", "dt");
    int    n_bins       = cfg.get_int   ("integration", "n_bins", 200);
    int    log_every    = 20; // unused
    int    n_steps_per_log = cfg.get_int("integration", "n_steps_per_log", 50);
    uint64_t seed       = (uint64_t)cfg.get_double("ensemble", "seed", 195935983.0);
    std::string out_dir = cfg.get_string("io", "output_dir", "results/langevin");

    DoubleWell pot;
    pot.U0   = cfg.require_double("potential", "U0");
    pot.h    = cfg.require_double("potential", "h");
    pot.L    = cfg.get_double    ("potential", "L",     1.0);
    pot.xmin = cfg.get_double    ("potential", "xmin", -3.0);
    pot.xmax = cfg.get_double    ("potential", "xmax",  3.0);

    if (rank == 0) {
        std::cout << "[Langevin] N_traj=" << N_traj_total
                  << "  T_h=" << T_h << "  T_w=" << T_w << "  T_c(=T_bath)=" << T_c
                  << "\n           U0=" << pot.U0 << "  h=" << pot.h
                  << "  t_max=" << t_max << "  dt=" << dt
                  << "\n           ranks=" << size
                  << "  threads/rank=" << omp_get_max_threads() << "\n";
    }

    int N_traj_local = N_traj_total / size + (rank < (N_traj_total % size) ? 1 : 0);

    // Build bath distribution (target equilibrium)
    EquilDistribution bath_dist;
    bath_dist.build(pot, T_c, n_bins);

    // Build initial distributions at the three temperatures
    EquilDistribution init_h, init_w, init_c;
    init_h.build(pot, T_h, n_bins);
    init_w.build(pot, T_w, n_bins);
    init_c.build(pot, T_c, n_bins);

    auto t_start = std::chrono::steady_clock::now();

    if (rank == 0) std::cout << "  running quench from T_h = " << T_h << "\n";
    auto res_h = run_quench(pot, init_h, bath_dist, T_c, t_max, dt,
                            N_traj_local, n_bins, log_every,
                            n_steps_per_log, rank, seed + 1);
    if (rank == 0) std::cout << "  running quench from T_w = " << T_w << "\n";
    auto res_w = run_quench(pot, init_w, bath_dist, T_c, t_max, dt,
                            N_traj_local, n_bins, log_every,
                            n_steps_per_log, rank, seed + 2);
    if (rank == 0) std::cout << "  running quench from T_c = " << T_c << "\n";
    auto res_c = run_quench(pot, init_c, bath_dist, T_c, t_max, dt,
                            N_traj_local, n_bins, log_every,
                            n_steps_per_log, rank, seed + 3);

    auto t_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    if (rank == 0) {
        std::cout << "[Langevin] done in " << elapsed << " s\n";
        // Write CSV (combined, with all three columns)
        CSVWriter csv(out_dir + "/distances.csv");
        csv.header({"t", "D_L1_h", "D_L1_w", "D_L1_c",
                         "D_KL_h", "D_KL_w", "D_KL_c"});
        std::size_t Ntimes = res_h.times.size();
        for (std::size_t k = 0; k < Ntimes; ++k) {
            csv.row({res_h.times[k],
                     res_h.D_L1[k], res_w.D_L1[k], res_c.D_L1[k],
                     res_h.D_KL[k], res_w.D_KL[k], res_c.D_KL[k]});
        }
        // Detect Mpemba: find first t such that D_L1_h < D_L1_w (both > equilibrium)
        std::cout << "\n[Langevin] Mpemba diagnostic:\n";
        std::cout << "  D_L1 at t=0: hot=" << res_h.D_L1[0]
                  << " warm=" << res_w.D_L1[0] << " cold=" << res_c.D_L1[0] << "\n";
        bool found_cross = false;
        for (std::size_t k = 1; k < Ntimes; ++k) {
            if (res_h.D_L1[k] < res_w.D_L1[k]) {
                std::cout << "  *** Mpemba crossover at t = " << res_h.times[k]
                          << "  (D_L1_h=" << res_h.D_L1[k]
                          << " <  D_L1_w=" << res_w.D_L1[k] << ")\n";
                found_cross = true;
                break;
            }
        }
        if (!found_cross)
            std::cout << "  no crossover in observed window.\n";
    }
    MPI_Finalize();
    return 0;
}
