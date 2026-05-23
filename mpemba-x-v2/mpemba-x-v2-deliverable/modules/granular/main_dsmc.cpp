// modules/granular/main.cpp
//
// MODULE 3: Mpemba effect in a granular fluid (Lasanta, Vega Reyes, Prados, Santos,
//           Phys. Rev. Lett. 119, 148001 (2017)).
//
// We implement a homogeneous Direct Simulation Monte Carlo (DSMC) of N smooth
// inelastic hard spheres in 3D with coefficient of restitution alpha and an
// optional white-noise thermostat (variance chi).
//
// Following the paper:
//   * Granular temperature T(t) = (m/3N) sum_i |v_i|^2  (with m=1, kB=1)
//   * Excess kurtosis a_2(t) = (3/5) * <v^4> / <v^2>^2 - 1   in 3D
//     (we use the standard definition; a_2 = 0 for a Maxwellian).
//   * Initial conditions: two samples (A, B) prepared with
//          T_A > T_B   but    a_{2,A} > a_{2,B} (sufficiently)
//     The condition (Lasanta Eq. 9):
//          Delta T_0 / Delta a_{2,0}  <  (2/3) * mu_2^(1) / (lambda_a - mu_2^HCS)
//     enables the Mpemba effect: the initially HOTTER sample (A) cools below
//     the initially COLDER one (B) at finite time.
//
// Two scenarios are simulated:
//   (1) Freely cooling granular gas (no thermostat, chi=0).
//   (2) Uniformly heated granular gas (white-noise thermostat at variance chi).
//
// Initial velocity preparation: we draw from a Maxwell-Boltzmann scaled to
// target temperature T0 and then re-shape into a *generalized Gaussian* family
// to give a prescribed kurtosis a_2. The simplest way is a power-deformation
//      v_i := v_i^MB * |v_i^MB / v_rms|^q
// and binary-search q so the empirical kurtosis matches a_2_target.
//
// Parallelization:
//   * MPI: independent ensemble copies of the (A, B) pair across ranks.
//   * OpenMP: per-step collision loop, velocity updates and moment calculations.
//
// Output:
//   results/granular/T_evolution_<label>.csv
//   results/granular/a2_evolution_<label>.csv
//   results/granular/summary.csv  (cross-over times)
//
// References (from the project): lasanta2017.pdf

#include <mpi.h>
#include <omp.h>

#include "../../core/philox_rng.hpp"
#include "../../core/csv_io.hpp"

#include <vector>
#include <array>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <limits>

using namespace mpemba;

struct GranularGas {
    int N;
    std::vector<double> vx, vy, vz;  // velocities

    explicit GranularGas(int N_) : N(N_), vx(N_), vy(N_), vz(N_) {}

    // Temperature T = (1/3N) sum |v|^2 (mass = 1)
    double temperature() const {
        double s = 0.0;
        #pragma omp parallel for reduction(+:s) schedule(static)
        for (int i = 0; i < N; ++i) s += vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i];
        return s / (3.0 * N);
    }

    // <v^2> and <v^4> across all particles
    void second_fourth_moments(double& m2, double& m4) const {
        double s2 = 0.0, s4 = 0.0;
        #pragma omp parallel for reduction(+:s2,s4) schedule(static)
        for (int i = 0; i < N; ++i) {
            double v2 = vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i];
            s2 += v2;
            s4 += v2 * v2;
        }
        m2 = s2 / N;
        m4 = s4 / N;
    }

    // Excess kurtosis in 3D. For a Maxwell-Boltzmann distribution
    // <v^4>/<v^2>^2 = 5/3 in 3D, so we define a_2 = (3/5)*<v^4>/<v^2>^2 - 1.
    double excess_kurtosis() const {
        double m2, m4;
        second_fourth_moments(m2, m4);
        if (m2 < 1e-300) return 0.0;
        return (3.0 / 5.0) * m4 / (m2 * m2) - 1.0;
    }

    // Mean velocity (should be zero throughout for a homogeneous gas)
    void center_of_mass(double& cmx, double& cmy, double& cmz) const {
        double sx = 0, sy = 0, sz = 0;
        #pragma omp parallel for reduction(+:sx,sy,sz) schedule(static)
        for (int i = 0; i < N; ++i) { sx += vx[i]; sy += vy[i]; sz += vz[i]; }
        cmx = sx / N; cmy = sy / N; cmz = sz / N;
    }

    void subtract_com() {
        double cmx, cmy, cmz;
        center_of_mass(cmx, cmy, cmz);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) { vx[i] -= cmx; vy[i] -= cmy; vz[i] -= cmz; }
    }

    // Rescale velocities so that T -> T_target
    void rescale_to_T(double T_target) {
        double T_curr = temperature();
        if (T_curr <= 0) return;
        double s = std::sqrt(T_target / T_curr);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) { vx[i] *= s; vy[i] *= s; vz[i] *= s; }
    }
};

// --------------------------------------------------------------------------
// Initial conditions: prepare velocities with target temperature T0 and
// target excess kurtosis a2_target.
//
// We use a deformation v -> sign(v) * |v|^q applied componentwise to a Maxwell
// sample, then re-center, rescale to T0. We binary-search q in [0.5, 1.6].
// --------------------------------------------------------------------------
void init_velocities(GranularGas& gas, double T0, double a2_target,
                     int mpi_rank, uint64_t seed_base) {
    int N = gas.N;
    // Step 1: draw Maxwell-Boltzmann velocities at T0 (mass=1, kB=1)
    #pragma omp parallel
    {
        PhiloxRNG rng = make_rng(mpi_rank, omp_get_thread_num(),
                                 seed_base ^ 0xA1F2B3C4D5E6F7A8ULL);
        // jump rng forward by some amount per thread to ensure disjoint streams
        rng.counter += (uint64_t)omp_get_thread_num() * 1000003ULL;
        #pragma omp for schedule(static)
        for (int i = 0; i < N; ++i) {
            double sigma = std::sqrt(T0);
            gas.vx[i] = sigma * rng.normal();
            gas.vy[i] = sigma * rng.normal();
            gas.vz[i] = sigma * rng.normal();
        }
    }
    gas.subtract_com();
    gas.rescale_to_T(T0);

    if (std::fabs(a2_target) < 1e-6) {
        // Maxwellian already has a_2 = 0 within statistical noise.
        return;
    }

    // Step 2: apply deformation v_i := sign(v_i) * |v_i|^q  to get target a_2
    // (positive q > 1 fattens tails -> increases a_2; q < 1 thins them.)
    std::vector<double> vx_orig = gas.vx;
    std::vector<double> vy_orig = gas.vy;
    std::vector<double> vz_orig = gas.vz;

    auto apply_q = [&](double q) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < gas.N; ++i) {
            double ax = std::pow(std::fabs(vx_orig[i]), q);
            double ay = std::pow(std::fabs(vy_orig[i]), q);
            double az = std::pow(std::fabs(vz_orig[i]), q);
            gas.vx[i] = (vx_orig[i] >= 0 ? 1 : -1) * ax;
            gas.vy[i] = (vy_orig[i] >= 0 ? 1 : -1) * ay;
            gas.vz[i] = (vz_orig[i] >= 0 ? 1 : -1) * az;
        }
        gas.subtract_com();
        gas.rescale_to_T(T0);
    };

    // Binary search q to hit a2_target
    double q_lo = 0.55, q_hi = 1.7;
    // a_2 monotonically increases with q (heavier tails)
    apply_q(q_lo);
    double a2_lo = gas.excess_kurtosis();
    apply_q(q_hi);
    double a2_hi = gas.excess_kurtosis();
    if (a2_target < a2_lo || a2_target > a2_hi) {
        // Clamp - cannot achieve. Use boundary.
        apply_q(a2_target < a2_lo ? q_lo : q_hi);
        return;
    }
    for (int it = 0; it < 60; ++it) {
        double q_mid = 0.5 * (q_lo + q_hi);
        apply_q(q_mid);
        double a2_mid = gas.excess_kurtosis();
        if (a2_mid < a2_target) { q_lo = q_mid; a2_lo = a2_mid; }
        else                    { q_hi = q_mid; a2_hi = a2_mid; }
        if (std::fabs(a2_mid - a2_target) < 1e-3) break;
    }
}

// --------------------------------------------------------------------------
// One DSMC step (Bird's NTC scheme, homogeneous).
//
// dt is the physical time step. We attempt N_pairs collision pairs, each pair
// taken from random particles. The number of attempts uses the standard
// homogeneous DSMC formula:
//     N_att = (N * (N-1) / 2) * v_rel_max * sigma * dt / V_cell
// For our purposes we work in dimensionless units where the collision frequency
// is set by an effective rate "nu_collisions". We follow the simplified
// homogeneous rule of choosing N_att proportional to N * sqrt(T) (so the
// per-particle collision rate is ~ sqrt(T)) and accept with v_rel / v_rel_ref.
// --------------------------------------------------------------------------
void dsmc_step(GranularGas& gas, double alpha, double dt,
               double nu0,        // collision rate prefactor
               double& v_rel_ref,
               int mpi_rank, uint64_t seed_base, uint64_t step_idx) {
    int N = gas.N;
    double T = gas.temperature();
    if (T <= 0) return;

    // Per-step collision attempt count (homogeneous DSMC, NTC).
    // Mean per-particle collision rate ~ nu0 * sqrt(T)
    // Number of attempts ~ 0.5 * N * nu0 * sqrt(T) * dt  (each attempt picks one pair).
    double mean_attempts = 0.5 * N * nu0 * std::sqrt(T) * dt;
    int N_att = (int)std::floor(mean_attempts);
    double frac = mean_attempts - N_att;

    // Update v_rel_ref slowly so acceptance ratios stay reasonable
    double sample_v_rel_max = 0.0;

    // Each thread keeps its own RNG and a small log of pending updates.
    // To avoid races (two pairs sharing a particle), we batch updates and
    // apply them serially per thread. Practical workaround for moderate N.
    int nthreads = omp_get_max_threads();
    std::vector<std::vector<std::array<double, 7>>> updates(nthreads);  // {idx_i, idx_j, dvx_i, dvy_i, dvz_i, ...}

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        PhiloxRNG rng = make_rng(mpi_rank, tid,
                                 seed_base ^ (0xDEADBEEFULL + step_idx * 0x9E3779B97F4A7C15ULL));
        rng.counter += (uint64_t)tid * 1000003ULL + step_idx;

        double local_v_rel_max = 0.0;

        // Each thread handles a slice of attempts
        int my_start = (tid * N_att) / nthreads;
        int my_end   = ((tid + 1) * N_att) / nthreads;
        for (int k = my_start; k < my_end; ++k) {
            int i = (int)rng.randint((uint64_t)N);
            int j = (int)rng.randint((uint64_t)N);
            if (i == j) j = (j + 1) % N;
            double rx = gas.vx[i] - gas.vx[j];
            double ry = gas.vy[i] - gas.vy[j];
            double rz = gas.vz[i] - gas.vz[j];
            double vrel = std::sqrt(rx*rx + ry*ry + rz*rz);
            if (vrel > local_v_rel_max) local_v_rel_max = vrel;

            // Acceptance probability vrel / v_rel_ref (NTC)
            double accept = (v_rel_ref > 0) ? (vrel / v_rel_ref) : 1.0;
            if (rng.uniform() > accept) continue;

            // Random unit vector for collision normal (uniform on sphere)
            double u1 = 2.0 * rng.uniform() - 1.0;
            double u2 = 2.0 * 3.141592653589793 * rng.uniform();
            double s  = std::sqrt(std::max(0.0, 1.0 - u1*u1));
            double nx = s * std::cos(u2);
            double ny = s * std::sin(u2);
            double nz = u1;

            // Inelastic collision (smooth hard sphere)
            // v_i' = v_i - (1+alpha)/2 * ((v_rel . n)) * n
            // v_j' = v_j + (1+alpha)/2 * ((v_rel . n)) * n
            double vrn = rx * nx + ry * ny + rz * nz;
            // Apply only if particles are approaching: vrn > 0 means v_i moves toward j
            // For DSMC we accept regardless (post-collision dynamics).
            double f = 0.5 * (1.0 + alpha) * vrn;
            std::array<double, 7> upd = {
                (double)i, (double)j,
                -f * nx, -f * ny, -f * nz,
                +f * nx, +f * ny
                // We store 7 numbers but need 8; restructure below.
            };
            // Easier: just store 8-tuple via two pushes.
            updates[tid].push_back({(double)i, -f*nx, -f*ny, -f*nz, (double)j, +f*nx, +f*ny});
            // We don't include +f*nz here; pack it in update lower part by re-using slot.
            // To keep this simple/clear, restructure:
        }
        #pragma omp critical(vrm_reduce)
        if (local_v_rel_max > sample_v_rel_max) sample_v_rel_max = local_v_rel_max;
    }

    // Apply updates serially (avoids race conditions on same particle).
    // NOTE: in this simplified scheme the same particle can appear in multiple
    // pairs in a single step; we apply updates in order. For physical accuracy
    // in dense systems this would need finer time-stepping, but for the
    // homogeneous Mpemba demonstration the leading-order kurtosis dynamics
    // is captured.
    for (int t = 0; t < nthreads; ++t) {
        for (auto& upd : updates[t]) {
            int i = (int)upd[0];
            int j = (int)upd[4];
            double dvx_i = upd[1], dvy_i = upd[2], dvz_i = upd[3];
            double dvx_j = upd[5], dvy_j = upd[6];
            // dvz_j wasn't stored; recompute. Take dvz_j = -dvz_i to conserve momentum.
            double dvz_j = -dvz_i;
            gas.vx[i] += dvx_i; gas.vy[i] += dvy_i; gas.vz[i] += dvz_i;
            gas.vx[j] += dvx_j; gas.vy[j] += dvy_j; gas.vz[j] += dvz_j;
        }
    }

    // Adaptive v_rel_ref: relax to sampled max
    if (sample_v_rel_max > v_rel_ref) v_rel_ref = sample_v_rel_max;
    else                              v_rel_ref = 0.95 * v_rel_ref + 0.05 * sample_v_rel_max;
}

// White-noise thermostat (uniform heating): adds independent Gaussian impulses
//   dv = sqrt(2 * chi * dt) * N(0, 1)   per component
void thermostat_kick(GranularGas& gas, double chi, double dt,
                     int mpi_rank, uint64_t seed_base, uint64_t step_idx) {
    if (chi <= 0) return;
    int N = gas.N;
    double sigma = std::sqrt(2.0 * chi * dt);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        PhiloxRNG rng = make_rng(mpi_rank, tid,
                                 seed_base ^ (0xBABECAFE0000ULL + step_idx));
        rng.counter += (uint64_t)tid * 31337ULL + step_idx;
        #pragma omp for schedule(static)
        for (int i = 0; i < N; ++i) {
            gas.vx[i] += sigma * rng.normal();
            gas.vy[i] += sigma * rng.normal();
            gas.vz[i] += sigma * rng.normal();
        }
    }
}

// --------------------------------------------------------------------------
// Driver
// --------------------------------------------------------------------------
struct GranularRun {
    std::string label;
    double T0;
    double a2_0;
};

void run_one(const GranularRun& cfg, int N, double alpha, double chi,
             double t_max, double dt, int log_every, double nu0,
             const std::string& out_dir, int mpi_rank, uint64_t seed_base) {
    GranularGas gas(N);
    init_velocities(gas, cfg.T0, cfg.a2_0, mpi_rank, seed_base);

    std::ostringstream osT, osA;
    osT << out_dir << "/T_evolution_" << cfg.label << "_rank" << mpi_rank << ".csv";
    osA << out_dir << "/a2_evolution_" << cfg.label << "_rank" << mpi_rank << ".csv";
    CSVWriter csvT(osT.str()), csvA(osA.str());
    csvT.header({"t", "T"});
    csvA.header({"t", "a2"});

    int n_steps = (int)std::ceil(t_max / dt);
    double v_rel_ref = 4.0 * std::sqrt(cfg.T0);  // initial guess

    for (int step = 0; step <= n_steps; ++step) {
        double t = step * dt;
        if (step % log_every == 0) {
            csvT.row({t, gas.temperature()});
            csvA.row({t, gas.excess_kurtosis()});
        }
        if (step < n_steps) {
            dsmc_step(gas, alpha, dt, nu0, v_rel_ref,
                      mpi_rank, seed_base, (uint64_t)step);
            thermostat_kick(gas, chi, dt,
                            mpi_rank, seed_base, (uint64_t)step);
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Default parameters (Lasanta 2017, Figs. 3-4 regime)
    int    N        = 20000;
    double alpha    = 0.9;
    double chi      = 0.0;            // 0 = free cooling
    double t_max    = 8.0;
    double dt       = 0.005;
    int    log_every = 4;
    double nu0      = 5.0;            // collision rate prefactor (dimensionless)
    uint64_t seed   = 0xC0FFEEULL;
    std::string out_dir = "results/granular";
    std::string scenario = "free_cooling";  // or "heated"

    // Two preparations A (hot but more out-of-equilibrium, larger a_2)
    // and B (cooler but closer to a Maxwellian).
    GranularRun A{"hot_A",  1.00, 0.50};
    GranularRun B{"cold_B", 0.99, -0.35};

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next  = [&](double& v) { v = std::stod(argv[++i]); };
        auto nexti = [&](int& v) { v = std::stoi(argv[++i]); };
        auto nexts = [&](std::string& v) { v = argv[++i]; };
        if      (a == "--N")        nexti(N);
        else if (a == "--alpha")    next(alpha);
        else if (a == "--chi")      next(chi);
        else if (a == "--tmax")     next(t_max);
        else if (a == "--dt")       next(dt);
        else if (a == "--log-every") nexti(log_every);
        else if (a == "--nu0")      next(nu0);
        else if (a == "--seed") { uint64_t s = std::stoull(argv[++i]); seed = s; }
        else if (a == "--out")      nexts(out_dir);
        else if (a == "--scenario") nexts(scenario);
        else if (a == "--TA")       next(A.T0);
        else if (a == "--TB")       next(B.T0);
        else if (a == "--a2A")      next(A.a2_0);
        else if (a == "--a2B")      next(B.a2_0);
    }
    if (scenario == "heated" && chi == 0.0) chi = 0.5;

    if (rank == 0) {
        std::cout << "[Granular] N=" << N << "  alpha=" << alpha
                  << "  chi=" << chi << "  scenario=" << scenario
                  << "\n           T_A=" << A.T0 << "  a2_A=" << A.a2_0
                  << "  T_B=" << B.T0 << "  a2_B=" << B.a2_0
                  << "  ranks=" << size << "  threads=" << omp_get_max_threads() << "\n";
    }

    auto t_start = std::chrono::steady_clock::now();

    // Even ranks do A, odd ranks do B. Each rank produces an independent
    // realization (ensemble copy). Combining offline.
    GranularRun cfg = (rank % 2 == 0) ? A : B;
    uint64_t my_seed = seed + (uint64_t)rank * 1000003ULL;
    run_one(cfg, N, alpha, chi, t_max, dt, log_every, nu0,
            out_dir, rank, my_seed);

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_end = std::chrono::steady_clock::now();
    if (rank == 0) {
        double elapsed = std::chrono::duration<double>(t_end - t_start).count();
        std::cout << "[Granular] done in " << elapsed << " s\n";
    }
    MPI_Finalize();
    return 0;
}
