// rk4_evolution.cpp  --  Entregable T2(a): evolucion temporal por INTEGRACION DIRECTA.
//
// Metodo: integra d rho/dt = L[rho] con Runge-Kutta clasico de 4o orden (RK4),
// aplicando el Liouvilliano matrix-free (sin formar la matriz d^2 x d^2).
// El hotspot es el producto de matrices densas en apply_lindblad (lindblad.hpp),
// paralelizado con OpenMP. El mismo binario, con distinto OMP_NUM_THREADS, da
// las curvas serial vs paralelo.
//
// Salidas:
//   - stdout: linea de benchmark  N,threads,n_steps,walltime_s,dhs_final
//   - opcional: CSV de la curva D_HS(t)  (para la figura fisica)
//
// Uso:  OMP_NUM_THREADS=k ./rk4_evolution config.ini [curve.csv]

#include <omp.h>
#include <iostream>
#include <iomanip>
#include "../../common/lindblad.hpp"

// un paso RK4 de la ecuacion maestra
static Mat rk4_step(const Mat& H, const std::vector<Mat>& Ls, const std::vector<Mat>& Lds,
                    const std::vector<Mat>& LdL, const Mat& rho, double dt, int d) {
    Mat k1 = apply_lindblad(H, Ls, Lds, LdL, rho, d);
    Mat t2 = rho; axpy(t2, k1, 0.5 * dt); Mat k2 = apply_lindblad(H, Ls, Lds, LdL, t2, d);
    Mat t3 = rho; axpy(t3, k2, 0.5 * dt); Mat k3 = apply_lindblad(H, Ls, Lds, LdL, t3, d);
    Mat t4 = rho; axpy(t4, k3, dt);       Mat k4 = apply_lindblad(H, Ls, Lds, LdL, t4, d);
    Mat out = rho;
    for (int i = 0; i < d * d; ++i) out[i] += dt * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]) / 6.0;
    return out;
}

int main(int argc, char** argv) {
    Config cfg;
    if (argc > 1) cfg.load(argv[1]);
    int    N     = cfg.geti("model.N", 7);
    double J     = cfg.getd("model.J", 1.0);
    double h     = cfg.getd("model.h", 0.5);
    double gamma = cfg.getd("model.gamma", 0.4);
    double Tbath = cfg.getd("model.T", 0.8);
    double T0    = cfg.getd("run.T0", 5.0);
    double t_max = cfg.getd("run.t_max", 6.0);
    double dt    = cfg.getd("run.dt", 2e-3);
    int    log_every = cfg.geti("run.log_every", 25);
    std::string curve = (argc > 2) ? argv[2] : "";

    Mat H; std::vector<Mat> Ls; int d;
    build_ising(N, J, h, gamma, Tbath, H, Ls, d);
    std::vector<Mat> Lds, LdL; precompute(Ls, d, Lds, LdL);

    // Referencia de relajacion = estado de Gibbs del bano (objetivo termico del
    // disipador), calculado por tiempo imaginario (barato). Preparacion inicial:
    // estado de Gibbs a la temperatura T0 (mas caliente).
    Mat rho_ss = gibbs_state(H, Tbath, d);
    Mat rho    = gibbs_state(H, T0, d);

    int n_steps = (int)std::ceil(t_max / dt);
    std::ofstream out;
    if (!curve.empty()) { out.open(curve); out << "t,D_HS\n"; }

    double t0 = now_s();
    for (int s = 0; s <= n_steps; ++s) {
        if (!curve.empty() && s % log_every == 0) out << s * dt << "," << d_hs(rho, rho_ss, d) << "\n";
        if (s < n_steps) rho = rk4_step(H, Ls, Lds, LdL, rho, dt, d);
    }
    double wall = now_s() - t0;

    double dhs_final = d_hs(rho, rho_ss, d);
    int threads = omp_get_max_threads();
    // linea de benchmark (parseable): N,threads,n_steps,walltime_s,dhs_final
    std::cout << std::setprecision(10)
              << N << "," << threads << "," << n_steps << "," << wall << "," << dhs_final << "\n";
    return 0;
}
