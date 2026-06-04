// trajectories_mpi.cpp -- Entregable T3(a): muchos cuerpos por TRAYECTORIAS CUANTICAS.
//
// Metodo: Monte Carlo wavefunction (quantum jumps). En vez de propagar el
// operador densidad rho (dimension d^2), se propagan M vectores de estado puros
// |psi> en C^d y se promedia rho(t) = E[|psi><psi|]. Reduce la memoria de d^2 a
// d y, sobre todo, es EMBARRASSINGLY PARALLEL: cada trayectoria es independiente.
//
// Paralelizacion: MPI reparte las M trayectorias entre rangos; una unica
// reduccion (MPI_Reduce) al final promedia. Escalado casi ideal (contraste con
// el OpenMP sublineal de T2). La precision mejora como M^{-1/2}.
//
// Algoritmo por trayectoria (1 matvec/paso usando H_eff = H - (i/2) sum L†L):
//   psi' = psi - i dt (H_eff psi);  norm2 = ||psi'||^2;  dp = 1 - norm2
//   si eps~U(0,1) < dp:  SALTO (elegir canal mu ~ ||L_mu psi||^2; psi = L_mu psi / ||.||)
//   si no:               psi = psi' / sqrt(norm2)
//
// Estado inicial: estado puro |0> (base computacional). Referencia de relajacion
// rho_ss = Gibbs del bano. Salida benchmark: N,ranks,M,wall_s,dhs_final
//
// Uso: mpirun -np P ./trajectories_mpi config.ini [curve.csv]

#include <mpi.h>
#include <vector>
#include <complex>
#include <random>
#include <iostream>
#include <iomanip>
#include "../../../T2/common/lindblad.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Config cfg; if (argc > 1) cfg.load(argv[1]);
    int    N     = cfg.geti("model.N", 8);
    double J     = cfg.getd("model.J", 1.0);
    double h     = cfg.getd("model.h", 0.5);
    double gamma = cfg.getd("model.gamma", 0.4);
    double Tbath = cfg.getd("model.T", 0.8);
    double t_max = cfg.getd("run.t_max", 4.0);
    double dt    = cfg.getd("run.dt", 0.02);
    int    M     = cfg.geti("run.M", 1000);
    int    log_every = cfg.geti("run.log_every", 10);
    long   seed  = cfg.geti("run.seed", 12345);
    std::string curve = (argc > 2) ? argv[2] : "";

    Mat H; std::vector<Mat> Ls; int d;
    build_ising(N, J, h, gamma, Tbath, H, Ls, d);
    // Gamma = sum L†L ; H_eff = H - (i/2) Gamma
    std::vector<Mat> Lds, LdL; precompute(Ls, d, Lds, LdL);
    Mat Heff = H;
    for (auto& g : LdL) for (int i = 0; i < d * d; ++i) Heff[i] += cd(0, -0.5) * g[i];
    // referencia = estado estacionario VERDADERO (nucleo de L), no Gibbs del bano.
    // Lo calcula SOLO el rango 0 (OpenMP) y lo difunde: evita que los P rangos lo
    // recomputen redundantemente y se sobre-suscriban los nucleos.
    Mat rho_ss(d * d, cd(0, 0));
    if (rank == 0) rho_ss = steady_state(H, Ls, Lds, LdL, d);
    MPI_Bcast(reinterpret_cast<double*>(rho_ss.data()), 2 * d * d, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int n_steps = (int)std::ceil(t_max / dt);
    std::vector<int> log_steps;
    for (int s = 0; s <= n_steps; s += log_every) log_steps.push_back(s);
    int nlog = (int)log_steps.size();

    // reparto de trayectorias entre rangos
    int per = M / size, rem = M % size;
    int my_M = per + (rank < rem ? 1 : 0);
    int my_start = rank * per + std::min(rank, rem);

    // acumulador local: nlog matrices d x d (aplanadas)
    std::vector<cd> acc((size_t)nlog * d * d, cd(0, 0));

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int tr = 0; tr < my_M; ++tr) {
        std::mt19937_64 rng(seed + (uint64_t)(my_start + tr));
        std::uniform_real_distribution<double> U(0.0, 1.0);
        std::vector<cd> psi(d, cd(0, 0)); psi[0] = 1.0;   // |0>
        int li = 0;
        for (int s = 0; s <= n_steps; ++s) {
            if (li < nlog && s == log_steps[li]) {        // acumular |psi><psi|
                cd* A = &acc[(size_t)li * d * d];
                for (int a = 0; a < d; ++a) { cd pa = psi[a];
                    for (int b = 0; b < d; ++b) A[a * d + b] += pa * std::conj(psi[b]); }
                ++li;
            }
            if (s == n_steps) break;
            // paso determinista RK4 de  dpsi/dt = -i H_eff psi  (no unitario; la
            // norma decae). 4 matvecs/paso -> elimina el sesgo O(dt) del Euler.
            auto f = [&](const std::vector<cd>& v) {
                std::vector<cd> Hv = matvec(Heff, v, d), r(d);
                for (int a = 0; a < d; ++a) r[a] = cd(0, -1) * Hv[a];
                return r;
            };
            std::vector<cd> k1 = f(psi), tmp(d);
            for (int a = 0; a < d; ++a) tmp[a] = psi[a] + 0.5 * dt * k1[a];
            std::vector<cd> k2 = f(tmp);
            for (int a = 0; a < d; ++a) tmp[a] = psi[a] + 0.5 * dt * k2[a];
            std::vector<cd> k3 = f(tmp);
            for (int a = 0; a < d; ++a) tmp[a] = psi[a] + dt * k3[a];
            std::vector<cd> k4 = f(tmp);
            std::vector<cd> psip(d);
            double norm2 = 0;
            for (int a = 0; a < d; ++a) {
                psip[a] = psi[a] + dt * (k1[a] + 2.0 * k2[a] + 2.0 * k3[a] + k4[a]) / 6.0;
                norm2 += std::norm(psip[a]);
            }
            double dp = 1.0 - norm2;
            if (U(rng) < dp) {                            // SALTO
                std::vector<double> w(Ls.size());
                double wsum = 0;
                for (size_t mu = 0; mu < Ls.size(); ++mu) {
                    std::vector<cd> Lp = matvec(Ls[mu], psi, d);
                    double nn = 0; for (auto& z : Lp) nn += std::norm(z);
                    w[mu] = nn; wsum += nn;
                }
                double r = U(rng) * wsum, c = 0; size_t mu = 0;
                for (; mu < Ls.size(); ++mu) { c += w[mu]; if (r <= c) break; }
                if (mu >= Ls.size()) mu = Ls.size() - 1;
                std::vector<cd> Lp = matvec(Ls[mu], psi, d);
                double nn = std::sqrt(std::max(1e-300, [&]{ double s2 = 0; for (auto& z : Lp) s2 += std::norm(z); return s2; }()));
                for (int a = 0; a < d; ++a) psi[a] = Lp[a] / nn;
            } else {                                      // paso determinista
                double inv = 1.0 / std::sqrt(norm2);
                for (int a = 0; a < d; ++a) psi[a] = psip[a] * inv;
            }
        }
    }
    double wall_local = MPI_Wtime() - t0;

    // reduccion: sumar acumuladores -> rho(t) en rank 0
    std::vector<cd> glob;
    if (rank == 0) glob.assign(acc.size(), cd(0, 0));
    MPI_Reduce(reinterpret_cast<double*>(acc.data()),
               rank == 0 ? reinterpret_cast<double*>(glob.data()) : nullptr,
               (int)(2 * acc.size()), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    double wall; MPI_Reduce(&wall_local, &wall, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::ofstream out; if (!curve.empty()) { out.open(curve); out << "t,D_HS\n"; }
        double dhs_final = 0;
        for (int li = 0; li < nlog; ++li) {
            Mat rho(d * d);
            for (int i = 0; i < d * d; ++i) rho[i] = glob[(size_t)li * d * d + i] / (double)M;
            double D = d_hs(rho, rho_ss, d);
            if (!curve.empty()) out << log_steps[li] * dt << "," << D << "\n";
            if (li == nlog - 1) dhs_final = D;
        }
        std::cout << std::setprecision(10)
                  << N << "," << size << "," << M << "," << wall << "," << dhs_final << "\n";
    }
    MPI_Finalize();
    return 0;
}
