// modules/water_fourier/main.cpp
//
// MODULE: Macroscopic water Mpemba effect via the Fourier thermal-fluid
// equation with hydrogen-bond memory and water-skin supersolidity.
//
// REFERENCE
//   Xi Zhang et al., "Hydrogen-bond memory and water-skin supersolidity
//   resolving the Mpemba paradox", PCCP 16, 22995 (2014). [c4cp03669g.pdf]
//
// PHYSICAL MODEL
// --------------
// 1D heat equation on a column of water of length L_tube (m), discretized
// into N finite-volume cells with temperature theta_i (Celsius):
//
//   rho_i Cp_i d(theta_i)/dt
//     = (1/dx) * [ q_{i-1/2}  -  q_{i+1/2} ]            (diffusion)
//       - rho_i Cp_i v ( theta_i - theta_{i-1} ) / dx    (convection upwind)
//       + q_mem,i                                        (H-bond memory)
//
// where the diffusive flux at the face (i, i+1/2) is
//   q_{i+1/2} = - kappa_face * (theta_{i+1} - theta_i) / dx
// with kappa_face = harmonic mean of kappa_i and kappa_{i+1}.
//
// BOUNDARY CONDITIONS
//   x = 0:  Robin BC, heat flux into cell 0 = h_drain * (T_bath - T_0).
//   x = L:  adiabatic (closed top).
//
// HYDROGEN-BOND MEMORY  (Zhang Eq. 2)
//   d(d_OH)/dt = -(d_OH - d_OH_eq(theta)) / tau_HB(theta_initial)
//   q_mem      = C_mem * d(d_OH)/dt
//
// SKIN SUPERSOLIDITY (Zhang Fig. 3)
//   In the skin region x < skin_thickness:
//      rho   *= rho_skin_over_bulk   (~ 0.75)
//      kappa *= kappa_skin_factor    (~ 1.33)
//
// NUMERICAL SCHEME
//   Explicit Euler with sub-stepping at the diffusive CFL limit.
//   Stable, simple, parallelized with OpenMP.

#include <mpi.h>
#include <omp.h>

#include "../../core/config_parser.hpp"
#include "../../core/csv_io.hpp"

#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>

using namespace mpemba;

// =========================================================================
// Material properties (SI, theta in Celsius)
// =========================================================================
inline double rho_water(double theta) {
    double a0 = 999.83952, a1 = 16.945176, a2 = -7.9870401e-3;
    double a3 = -46.170461e-6, a4 = 105.56302e-9, a5 = -280.54253e-12;
    double t = theta;
    return (a0 + t*(a1 + t*(a2 + t*(a3 + t*(a4 + t*a5))))) /
           (1.0 + 16.879850e-3 * t);
}
inline double Cp_water(double theta)   { return 4179.0 + 0.5 * (theta - 30.0); }
inline double kappa_water(double theta){ return 0.598 + 0.0016 * (theta - 20.0); }
inline double d_OH_equilibrium(double theta) { return 1.0 + 0.002 * theta; }
inline double tau_HB(double theta_initial) {
    return 600.0 * std::exp(-(theta_initial - 20.0) / 60.0);
}

// =========================================================================
// State
// =========================================================================
struct WaterColumn {
    int N;
    double L_tube, dx;
    double cross_section_m2;
    double skin_thickness;
    std::vector<double> theta;
    std::vector<double> d_OH;
    std::vector<double> x_centers;
    std::vector<bool>   is_skin;
    double theta_bath, theta_initial;
    double h_drain;
    double C_mem;
    double rho_skin_ratio, kappa_skin_factor;
    double conv_velocity;
};

inline double local_rho(const WaterColumn& w, int i) {
    double r = rho_water(w.theta[i]);
    if (w.is_skin[i]) r *= w.rho_skin_ratio;
    return r;
}
inline double local_kappa(const WaterColumn& w, int i) {
    double k = kappa_water(w.theta[i]);
    if (w.is_skin[i]) k *= w.kappa_skin_factor;
    return k;
}

WaterColumn make_column(int N, double L_tube, double cross_section_m2,
                        double skin_thickness,
                        double theta_initial, double theta_bath,
                        double h_drain, double C_mem,
                        double rho_skin_ratio, double kappa_skin_factor,
                        double conv_velocity) {
    WaterColumn w;
    w.N = N; w.L_tube = L_tube; w.dx = L_tube / N;
    w.cross_section_m2 = cross_section_m2;
    w.skin_thickness = skin_thickness;
    w.theta.assign(N, theta_initial);
    w.d_OH.assign(N, d_OH_equilibrium(theta_initial));
    w.x_centers.assign(N, 0.0);
    w.is_skin.assign(N, false);
    for (int i = 0; i < N; ++i) {
        w.x_centers[i] = (i + 0.5) * w.dx;
        if (w.x_centers[i] < skin_thickness) w.is_skin[i] = true;
    }
    w.theta_bath        = theta_bath;
    w.theta_initial     = theta_initial;
    w.h_drain           = h_drain;
    w.C_mem             = C_mem;
    w.rho_skin_ratio    = rho_skin_ratio;
    w.kappa_skin_factor = kappa_skin_factor;
    w.conv_velocity     = conv_velocity;
    return w;
}

double max_dt_cfl(const WaterColumn& w) {
    double alpha_max = 0.0;
    for (int i = 0; i < w.N; ++i) {
        double a = local_kappa(w, i) / (local_rho(w, i) * Cp_water(w.theta[i]));
        if (a > alpha_max) alpha_max = a;
    }
    // Robin BC stability:
    //   dt <= rho Cp dx^2 / (2 (kappa + h dx))
    // We use a conservative safety factor of 0.25.
    double k0     = local_kappa(w, 0);
    double rhoCp0 = local_rho(w, 0) * Cp_water(w.theta[0]);
    double dt_bc  = 0.25 * rhoCp0 * w.dx * w.dx
                    / std::max(k0 + w.h_drain * w.dx, 1e-15);
    double dt_diff = 0.25 * w.dx * w.dx / std::max(alpha_max, 1e-15);
    return std::min(dt_bc, dt_diff);
}

// Explicit-Euler micro-step (dt must be CFL-stable)
void micro_step(WaterColumn& w, double dt) {
    int N = w.N;
    double dx = w.dx;
    double tau = tau_HB(w.theta_initial);
    std::vector<double> theta_new(N), d_OH_new(N);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        // Flux into cell from the left side
        double q_in_left;
        if (i == 0) {
            q_in_left = w.h_drain * (w.theta_bath - w.theta[i]);
        } else {
            double k_i  = local_kappa(w, i);
            double k_im = local_kappa(w, i - 1);
            double kf   = 2.0 * k_i * k_im / (k_i + k_im + 1e-30);
            q_in_left = -kf * (w.theta[i] - w.theta[i - 1]) / dx;
        }
        // Flux out of cell to the right side
        double q_out_right;
        if (i == N - 1) {
            q_out_right = 0.0;     // adiabatic
        } else {
            double k_i  = local_kappa(w, i);
            double k_ip = local_kappa(w, i + 1);
            double kf   = 2.0 * k_i * k_ip / (k_i + k_ip + 1e-30);
            q_out_right = -kf * (w.theta[i + 1] - w.theta[i]) / dx;
        }
        double net_per_vol = (q_in_left - q_out_right) / dx;

        // Convection (upwind toward +x)
        double conv_term = 0.0;
        if (w.conv_velocity > 0.0 && i > 0) {
            conv_term = -w.conv_velocity * (w.theta[i] - w.theta[i - 1]) / dx;
        }

        // H-bond memory release
        double dOH_eq = d_OH_equilibrium(w.theta[i]);
        double ddOH   = -(w.d_OH[i] - dOH_eq) / tau;
        double q_mem  = w.C_mem * ddOH;

        double rhoCp = local_rho(w, i) * Cp_water(w.theta[i]);
        theta_new[i] = w.theta[i]
                     + dt * (net_per_vol / rhoCp + conv_term + q_mem / rhoCp);
        d_OH_new[i]  = w.d_OH[i] + dt * ddOH;
    }
    w.theta.swap(theta_new);
    w.d_OH.swap(d_OH_new);
}

void integrate(WaterColumn& w, double dt_outer) {
    double dt_cfl = max_dt_cfl(w);
    int n_sub = std::max(1, (int)std::ceil(dt_outer / dt_cfl));
    double dt = dt_outer / n_sub;
    for (int s = 0; s < n_sub; ++s) micro_step(w, dt);
}

struct Snapshot {
    double t;
    double T_mean, T_skin, T_bulk;
    double enthalpy_J, mass_kg;
    double rms_dist_to_bath;
};
Snapshot summarize(const WaterColumn& w, double t) {
    Snapshot s; s.t = t;
    double sT = 0, sTs = 0, sTb = 0;
    int ns = 0, nb = 0;
    double e = 0, m = 0, d2 = 0;
    double dV = w.dx * w.cross_section_m2;
    for (int i = 0; i < w.N; ++i) {
        sT += w.theta[i];
        if (w.is_skin[i]) { sTs += w.theta[i]; ++ns; }
        else              { sTb += w.theta[i]; ++nb; }
        double rho = local_rho(w, i);
        e += rho * Cp_water(w.theta[i]) * w.theta[i] * dV;
        m += rho * dV;
        d2 += std::pow(w.theta[i] - w.theta_bath, 2);
    }
    s.T_mean = sT / w.N;
    s.T_skin = (ns > 0) ? sTs / ns : 0.0;
    s.T_bulk = (nb > 0) ? sTb / nb : 0.0;
    s.enthalpy_J = e;
    s.mass_kg = m;
    s.rms_dist_to_bath = std::sqrt(d2 / w.N);
    return s;
}

void run_one(WaterColumn& w, double t_max, double dt_log,
             const std::string& tag, const std::string& out_dir) {
    CSVWriter csv_ts(out_dir + "/timeseries_" + tag + ".csv");
    csv_ts.header({"t", "T_mean_C", "T_skin_C", "T_bulk_C",
                   "enthalpy_J", "mass_kg", "rms_dist_to_bath_C"});
    std::ofstream field_ofs(out_dir + "/field_" + tag + ".csv");
    field_ofs.precision(7);
    field_ofs << "t";
    for (int i = 0; i < w.N; ++i) field_ofs << "," << w.x_centers[i];
    field_ofs << "\n";

    int n_logs = (int)std::ceil(t_max / dt_log);
    double t = 0.0;
    for (int k = 0; k <= n_logs; ++k) {
        Snapshot s = summarize(w, t);
        csv_ts.row({s.t, s.T_mean, s.T_skin, s.T_bulk,
                    s.enthalpy_J, s.mass_kg, s.rms_dist_to_bath});
        field_ofs << t;
        for (int i = 0; i < w.N; ++i) field_ofs << "," << w.theta[i];
        field_ofs << "\n";
        if (k < n_logs) {
            integrate(w, dt_log);
            t += dt_log;
        }
    }
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

    double mass_g          = cfg.require_double("water", "mass_g");
    double tube_length_m   = cfg.require_double("water", "tube_length_m");
    double skin_thick_m    = cfg.get_double    ("water", "skin_thickness_m", 1e-3);
    int    N_grid          = cfg.get_int       ("simulation", "n_grid", 200);

    double rho0 = 999.84;
    double V    = (mass_g / 1000.0) / rho0;
    double A    = V / tube_length_m;

    double T_hot   = cfg.require_double("preparation_hot",  "T_initial_C");
    double T_cold  = cfg.require_double("preparation_cold", "T_initial_C");
    double T_bath  = cfg.require_double("bath", "T_bath_C");
    double h_drain = cfg.get_double    ("bath", "h_drain_W_m2_K", 100.0);

    double rho_skin_ratio    = cfg.get_double("zhang_model", "rho_skin_over_bulk", 0.75);
    double kappa_skin_factor = cfg.get_double("zhang_model", "kappa_skin_factor",  1.333);
    double C_mem             = cfg.get_double("zhang_model", "C_mem_J_per_m3",     0.0);
    double conv_velocity     = cfg.get_double("zhang_model", "conv_velocity_m_s",  0.0);

    double t_max  = cfg.require_double("simulation", "t_max_s");
    double dt_log = cfg.require_double("simulation", "dt_log_s");

    std::string out_dir = cfg.get_string("io", "output_dir", "results/water_fourier");

    if (rank == 0) {
        std::cout << "===========================================================\n";
        std::cout << "  Water Fourier Mpemba (Zhang et al. PCCP 2014)\n";
        std::cout << "===========================================================\n";
        std::cout << "  Mass         = " << mass_g << " g\n";
        std::cout << "  Tube length  = " << tube_length_m << " m\n";
        std::cout << "  Cross sect.  = " << A*1e4 << " cm^2 (derived)\n";
        std::cout << "  Skin thick.  = " << skin_thick_m << " m\n";
        std::cout << "  N grid cells = " << N_grid << "\n";
        std::cout << "  T_hot init   = " << T_hot  << " C\n";
        std::cout << "  T_cold init  = " << T_cold << " C\n";
        std::cout << "  T_bath       = " << T_bath << " C\n";
        std::cout << "  h_drain      = " << h_drain << " W/m^2/K\n";
        std::cout << "  rho_skin/B   = " << rho_skin_ratio << "\n";
        std::cout << "  kappa_skin   = " << kappa_skin_factor << "\n";
        std::cout << "  C_mem        = " << C_mem << " J/m^3\n";
        std::cout << "  conv vel.    = " << conv_velocity << " m/s\n";
        std::cout << "  t_max        = " << t_max << " s\n";
        std::cout << "  dt_log       = " << dt_log << " s\n";
        std::cout << "  MPI ranks    = " << size << " threads = "
                  << omp_get_max_threads() << "\n";
    }

    auto t_start = std::chrono::steady_clock::now();

    auto run_for = [&](double T_init, const std::string& tag) {
        WaterColumn w = make_column(N_grid, tube_length_m, A, skin_thick_m,
                                    T_init, T_bath, h_drain, C_mem,
                                    rho_skin_ratio, kappa_skin_factor, conv_velocity);
        if (rank == 0)
            std::cout << "  [" << tag << "] T_init=" << T_init
                      << " tau_HB=" << tau_HB(T_init)
                      << " dt_CFL=" << max_dt_cfl(w) << " s\n";
        run_one(w, t_max, dt_log, tag, out_dir);
    };

    if (size >= 2) {
        if (rank == 0) run_for(T_hot,  "hot");
        if (rank == 1) run_for(T_cold, "cold");
    } else {
        run_for(T_hot,  "hot");
        run_for(T_cold, "cold");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_end = std::chrono::steady_clock::now();
    if (rank == 0) {
        double elapsed = std::chrono::duration<double>(t_end - t_start).count();
        std::cout << "[water_fourier] done in " << elapsed << " s\n";
    }
    MPI_Finalize();
    return 0;
}
