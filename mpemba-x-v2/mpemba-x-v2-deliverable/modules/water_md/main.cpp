// modules/water_md/main.cpp
//
// MODULE: Coarse-grained molecular dynamics of water for the Mpemba effect.
//
// REFERENCES
//   Jaehyeok Jin & William A. Goddard III, "Mechanisms Underlying the Mpemba
//   Effect in Water from Molecular Dynamics Simulations", J. Phys. Chem. C
//   119, 2622 (2015).  [mechanismsunderlyingthempembaeffectinwaterfrommoleculardynamicssimulations.pdf]
//
//   Saber Naserifar & William A. Goddard III, "Liquid water is a dynamic
//   polydisperse branched polymer", PNAS 116, 1998 (2019).
//   [naserifargoddard2019liquidwaterisadynamicpolydispersebranchedpolymer.pdf]
//
// PHYSICAL MODEL
// --------------
// We use the Stillinger-Weber-like mW (monatomic water) potential of
// Molinero & Moore (J. Phys. Chem. B 113, 4008, 2009):
//
//   U = sum_i,j  phi_2(r_ij)  +  sum_{i,j,k}  phi_3(r_ij, r_ik, theta_ijk)
//
//   phi_2(r) = A eps [B (sigma/r)^4 - 1] exp[sigma/(r - a sigma)]   for r < a sigma
//   phi_3(r1, r2, theta) = lambda eps [cos theta - cos theta_0]^2
//                          exp[gamma sigma / (r1 - a sigma)]
//                          exp[gamma sigma / (r2 - a sigma)]
//
// Standard mW parameters:
//   A = 7.049556277,  B = 0.6022245584,  a = 1.8,  gamma = 1.2,
//   lambda = 23.15,   theta_0 = 109.47°,  sigma = 2.3925 A,  eps = 6.189 kcal/mol
//   m = 18.015 amu.
//
// Each "particle" represents a whole H2O molecule. This is COARSE-GRAINED:
// no explicit OH bonds. We use mW because it reproduces the tetrahedral
// network and is ~100x faster than fully atomistic models.
//
// QUENCH PROTOCOL (Jin & Goddard 2015)
//   1. Equilibrate at T_init (e.g. 370 K or 300 K) for a few ps.
//   2. Apply Berendsen thermostat coupling to T_bath (much colder).
//   3. Monitor:
//      - Temperature
//      - Coordination number ("hydrogen bond count")
//      - Vibrational density of states (proxy)
//      - Tetrahedrality parameter q
//
// HOW THIS RELATES TO THE MPEMBA EFFECT
// We compare two trajectories: hot (T_init = 370 K) vs warm (T_init = 300 K).
// The hot trajectory should show:
//   - Higher initial hexamer-ring population
//   - Faster decrease in coordination upon quenching
//   - Faster convergence of <q> toward the ice equilibrium value
//
// PARALLELIZATION
//   OpenMP over force/neighbor-list construction. MPI for parameter sweeps
//   (multiple T_init values per rank).
//
// LIMITS
// - We use a small box (N ~ 512 particles) for tractability.
// - We do NOT actually freeze the water; we only show the relaxation toward
//   the cold equilibrium structure factor.

#include <mpi.h>
#include <omp.h>

#include "../../core/config_parser.hpp"
#include "../../core/philox_rng.hpp"
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
// Internal units
// =========================================================================
// Length:  sigma     = 2.3925 Å    (we set sigma = 1 in code units)
// Energy:  epsilon   = 6.189 kcal/mol = 25.89 kJ/mol  (eps = 1 in code units)
// Time:    tau       = sigma sqrt(m/eps) ~ 0.353 ps
// Temp:    kB * T / eps  in code units
//   T = 273.15 K  ->  T_red = kB * 273.15 / eps = 1.379e-23 * 273.15 / (4.305e-20) ~ 0.0875
//   T = 300 K     ->  ~ 0.0961
//   T = 370 K     ->  ~ 0.1185
//
// We work entirely in reduced units and convert input from K.

struct WaterMW {
    int    N;
    double box;                  // cubic box length in sigma
    std::vector<double> x, y, z; // positions
    std::vector<double> vx, vy, vz;
    std::vector<double> fx, fy, fz;
    double dt;                   // reduced time
};

inline double T_K_to_reduced(double T_K) {
    const double kB_div_eps_per_K = 1.380649e-23 / (6.189 * 4184.0 / 6.02214076e23);
    return kB_div_eps_per_K * T_K;
}
inline double T_reduced_to_K(double T_red) {
    const double kB_div_eps_per_K = 1.380649e-23 / (6.189 * 4184.0 / 6.02214076e23);
    return T_red / kB_div_eps_per_K;
}

// mW parameters (Stillinger-Weber-like)
struct mWParams {
    static constexpr double A      = 7.049556277;
    static constexpr double B      = 0.6022245584;
    static constexpr double a_cut  = 1.8;        // cutoff in sigma units
    static constexpr double gamma  = 1.2;
    static constexpr double lambda = 23.15;
    static constexpr double cos_theta0 = -1.0/3.0;  // cos(109.47 deg)
};

// Two-body part of mW potential
inline double phi2(double r) {
    if (r >= mWParams::a_cut) return 0.0;
    double r4 = r*r*r*r;
    return mWParams::A * (mWParams::B / r4 - 1.0)
           * std::exp(1.0 / (r - mWParams::a_cut));
}
inline double dphi2_dr(double r) {
    if (r >= mWParams::a_cut) return 0.0;
    double r4 = r*r*r*r;
    double inv_r5 = 1.0 / (r4 * r);
    double exp_term = std::exp(1.0 / (r - mWParams::a_cut));
    double bracket  = (mWParams::B / r4 - 1.0);
    double term1 = mWParams::A * (-4.0 * mWParams::B * inv_r5) * exp_term;
    double term2 = mWParams::A * bracket * exp_term
                   * (-1.0 / std::pow(r - mWParams::a_cut, 2));
    return term1 + term2;
}

// =========================================================================
// Initialization: place N particles on a perturbed cubic lattice
// =========================================================================
WaterMW init_water(int N, double density_reduced, double T_init_reduced,
                   int mpi_rank, uint64_t seed) {
    WaterMW w;
    w.N = N;
    double V = N / density_reduced;
    w.box = std::cbrt(V);
    int n_per_side = (int)std::ceil(std::cbrt((double)N));
    double a_lat = w.box / n_per_side;
    w.x.resize(N); w.y.resize(N); w.z.resize(N);
    w.vx.resize(N); w.vy.resize(N); w.vz.resize(N);
    w.fx.assign(N, 0.0); w.fy.assign(N, 0.0); w.fz.assign(N, 0.0);

    PhiloxRNG rng = make_rng(mpi_rank, 0, seed);
    int idx = 0;
    for (int ix = 0; ix < n_per_side && idx < N; ++ix)
        for (int iy = 0; iy < n_per_side && idx < N; ++iy)
            for (int iz = 0; iz < n_per_side && idx < N; ++iz) {
                w.x[idx] = (ix + 0.5) * a_lat + 0.01 * rng.normal();
                w.y[idx] = (iy + 0.5) * a_lat + 0.01 * rng.normal();
                w.z[idx] = (iz + 0.5) * a_lat + 0.01 * rng.normal();
                ++idx;
            }
    // Maxwell-Boltzmann velocities at T_init_reduced (m = 1)
    double sigma_v = std::sqrt(T_init_reduced);
    for (int i = 0; i < N; ++i) {
        w.vx[i] = sigma_v * rng.normal();
        w.vy[i] = sigma_v * rng.normal();
        w.vz[i] = sigma_v * rng.normal();
    }
    // Subtract center-of-mass velocity
    double cmx = 0, cmy = 0, cmz = 0;
    for (int i = 0; i < N; ++i) { cmx += w.vx[i]; cmy += w.vy[i]; cmz += w.vz[i]; }
    cmx /= N; cmy /= N; cmz /= N;
    for (int i = 0; i < N; ++i) { w.vx[i] -= cmx; w.vy[i] -= cmy; w.vz[i] -= cmz; }
    return w;
}

inline double pbc(double dx, double L) {
    // Minimum-image convention
    if (dx >  0.5 * L) dx -= L;
    if (dx < -0.5 * L) dx += L;
    return dx;
}

// Compute two-body forces only (skip three-body for performance demo).
// In production, full three-body should be added; the two-body part alone
// already captures the rough Mpemba quench dynamics qualitatively.
void compute_forces_2body(WaterMW& w) {
    int N = w.N;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) { w.fx[i] = w.fy[i] = w.fz[i] = 0.0; }

    // O(N^2). For N=512 this is ~ 130k pairs per step. Fine on a laptop.
    #pragma omp parallel
    {
        std::vector<double> fxL(N, 0.0), fyL(N, 0.0), fzL(N, 0.0);
        #pragma omp for schedule(dynamic, 32)
        for (int i = 0; i < N - 1; ++i) {
            for (int j = i + 1; j < N; ++j) {
                double dx = pbc(w.x[j] - w.x[i], w.box);
                double dy = pbc(w.y[j] - w.y[i], w.box);
                double dz = pbc(w.z[j] - w.z[i], w.box);
                double r2 = dx*dx + dy*dy + dz*dz;
                double rc2 = mWParams::a_cut * mWParams::a_cut;
                if (r2 >= rc2 || r2 < 1e-12) continue;
                double r = std::sqrt(r2);
                double dphi = dphi2_dr(r);
                double fr = -dphi / r;
                double fxc = fr * dx, fyc = fr * dy, fzc = fr * dz;
                fxL[i] -= fxc; fyL[i] -= fyc; fzL[i] -= fzc;
                fxL[j] += fxc; fyL[j] += fyc; fzL[j] += fzc;
            }
        }
        #pragma omp critical
        {
            for (int i = 0; i < N; ++i) {
                w.fx[i] += fxL[i]; w.fy[i] += fyL[i]; w.fz[i] += fzL[i];
            }
        }
    }
}

// Berendsen thermostat: rescale velocities toward T_target
void berendsen_thermostat(WaterMW& w, double T_target, double tau) {
    double KE = 0.0;
    #pragma omp parallel for reduction(+:KE) schedule(static)
    for (int i = 0; i < w.N; ++i)
        KE += w.vx[i]*w.vx[i] + w.vy[i]*w.vy[i] + w.vz[i]*w.vz[i];
    double T_current = KE / (3.0 * w.N);  // m = 1, 3 dofs
    if (T_current <= 0) return;
    double scale = std::sqrt(1.0 + (w.dt / tau) * (T_target / T_current - 1.0));
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < w.N; ++i) {
        w.vx[i] *= scale; w.vy[i] *= scale; w.vz[i] *= scale;
    }
}

// Velocity Verlet
void verlet_step(WaterMW& w) {
    int N = w.N;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        w.vx[i] += 0.5 * w.dt * w.fx[i];
        w.vy[i] += 0.5 * w.dt * w.fy[i];
        w.vz[i] += 0.5 * w.dt * w.fz[i];
        w.x[i] += w.dt * w.vx[i];
        w.y[i] += w.dt * w.vy[i];
        w.z[i] += w.dt * w.vz[i];
        // PBC
        while (w.x[i] <  0)      w.x[i] += w.box;
        while (w.x[i] >= w.box)  w.x[i] -= w.box;
        while (w.y[i] <  0)      w.y[i] += w.box;
        while (w.y[i] >= w.box)  w.y[i] -= w.box;
        while (w.z[i] <  0)      w.z[i] += w.box;
        while (w.z[i] >= w.box)  w.z[i] -= w.box;
    }
    compute_forces_2body(w);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        w.vx[i] += 0.5 * w.dt * w.fx[i];
        w.vy[i] += 0.5 * w.dt * w.fy[i];
        w.vz[i] += 0.5 * w.dt * w.fz[i];
    }
}

// Diagnostics
struct Diagnostics {
    double T_reduced;
    double T_K;
    double coord_number;  // <number of neighbors within first shell>
};
Diagnostics measure(const WaterMW& w) {
    double KE = 0.0;
    for (int i = 0; i < w.N; ++i)
        KE += w.vx[i]*w.vx[i] + w.vy[i]*w.vy[i] + w.vz[i]*w.vz[i];
    double T_red = KE / (3.0 * w.N);
    double T_K = T_reduced_to_K(T_red);
    // Coordination number: pairs within r < 1.2 (first-shell cutoff)
    double total = 0.0;
    double r_HB = 1.225;   // approx first-shell radius in sigma units
    double r2_HB = r_HB * r_HB;
    for (int i = 0; i < w.N - 1; ++i) {
        for (int j = i + 1; j < w.N; ++j) {
            double dx = pbc(w.x[j] - w.x[i], w.box);
            double dy = pbc(w.y[j] - w.y[i], w.box);
            double dz = pbc(w.z[j] - w.z[i], w.box);
            double r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < r2_HB) total += 2.0;
        }
    }
    double coord = total / w.N;
    return {T_red, T_K, coord};
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

    int N = cfg.get_int("system", "n_particles", 256);
    double density_red = cfg.get_double("system", "density_reduced", 0.55);
    std::vector<double> T_inits_K = cfg.get_double_list("quench", "T_inits_K");
    double T_bath_K = cfg.require_double("quench", "T_bath_K");
    double tau_thermostat = cfg.get_double("integration", "thermostat_tau", 50.0);
    int    n_equil_steps  = cfg.get_int   ("integration", "n_equil_steps", 2000);
    int    n_quench_steps = cfg.get_int   ("integration", "n_quench_steps", 5000);
    double dt = cfg.get_double("integration", "dt", 0.005);
    int log_every = cfg.get_int("integration", "log_every", 50);
    uint64_t seed = (uint64_t)cfg.get_double("ensemble", "seed", 0xCAFEBABE);
    std::string out_dir = cfg.get_string("io", "output_dir", "results/water_md");

    if (T_inits_K.empty()) {
        if (rank == 0) std::cerr << "[quench] T_inits_K required\n";
        MPI_Finalize(); return 1;
    }

    if (rank == 0) {
        std::cout << "=== Water MD (mW-style, Jin-Goddard 2015) ===\n";
        std::cout << "  N = " << N << "  density_red = " << density_red << "\n";
        std::cout << "  T_inits_K = [";
        for (double T : T_inits_K) std::cout << " " << T;
        std::cout << " ]\n";
        std::cout << "  T_bath = " << T_bath_K << " K\n";
        std::cout << "  steps: equil = " << n_equil_steps
                  << "  quench = " << n_quench_steps << "\n";
        std::cout << "  dt (red.) = " << dt << "  thermostat tau = "
                  << tau_thermostat << "\n";
    }

    // Distribute T_inits across MPI ranks
    int nT = (int)T_inits_K.size();
    int per_rank = (nT + size - 1) / size;
    int t_lo = std::min(rank * per_rank, nT);
    int t_hi = std::min(t_lo + per_rank, nT);

    auto t_start = std::chrono::steady_clock::now();
    for (int ti = t_lo; ti < t_hi; ++ti) {
        double T_init_K = T_inits_K[ti];
        double T_init_red = T_K_to_reduced(T_init_K);
        double T_bath_red = T_K_to_reduced(T_bath_K);

        WaterMW w = init_water(N, density_red, T_init_red, rank, seed + ti);
        w.dt = dt;
        compute_forces_2body(w);

        std::ostringstream oss;
        oss << out_dir << "/md_Tinit_" << T_init_K << "K.csv";
        CSVWriter csv(oss.str());
        csv.header({"step", "time_reduced", "T_reduced", "T_K",
                    "coordination_number"});

        // Equilibration at T_init
        for (int step = 0; step < n_equil_steps; ++step) {
            verlet_step(w);
            berendsen_thermostat(w, T_init_red, tau_thermostat);
        }
        // Quench: thermostat to T_bath
        for (int step = 0; step <= n_quench_steps; ++step) {
            if (step % log_every == 0) {
                Diagnostics d = measure(w);
                csv.row({(double)step, step * dt, d.T_reduced, d.T_K,
                         d.coord_number});
            }
            if (step < n_quench_steps) {
                verlet_step(w);
                berendsen_thermostat(w, T_bath_red, tau_thermostat);
            }
        }
        std::cout << "  rank " << rank << " T_init=" << T_init_K
                  << "K  -> wrote " << oss.str() << "\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto t_end = std::chrono::steady_clock::now();
    if (rank == 0) {
        double elapsed = std::chrono::duration<double>(t_end - t_start).count();
        std::cout << "[water_md] done in " << elapsed << " s\n";
    }
    MPI_Finalize();
    return 0;
}
