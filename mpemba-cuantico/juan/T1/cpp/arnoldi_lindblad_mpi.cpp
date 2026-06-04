// arnoldi_lindblad_mpi.cpp
//
// Tarea T1 PARALELA: modos lentos del Liouvilliano por Arnoldi-Lindblad,
// con paralelismo hibrido MPI + OpenMP.
//
// Algoritmo (Minganti & Huybrechts, Quantum 6, 649 (2022)):
//   1. Se construye el subespacio de Krylov del PROPAGADOR P(tau)=e^{tau L},
//      cuya accion se calcula matrix-free integrando dV/dt=L[V] con RK4
//      (kernel apply_lindblad de qmpe_core.hpp).
//   2. El espectro de P y el de L estan ligados por mu_k = e^{tau lambda_k};
//      los MODOS LENTOS de L son los autovalores DOMINANTES en modulo de P,
//      que es justo lo que Arnoldi extrae primero.
//   3. Se proyecta P sobre la base de Krylov (matriz de Hessenberg H_m), se
//      resuelve el pequeno problema de autovalores m x m con LAPACK (zgeev) y
//      se recuperan lambda_k = ln(mu_k)/tau.
//
// PARALELIZACION:
//   * OpenMP: paraleliza el producto de matrices del SpMV (intranodo).
//   * MPI:    distribuye las filas de cada matmul entre rangos y reconstruye
//             con Allgatherv (paralelismo de datos, Seccion 8.1 nivel 1).
//   Arnoldi es secuencialmente acoplado: el paralelismo escalable vive dentro
//   del SpMV, que es el hotspot (n_it * SpMV, Seccion 7.4).
//
// Uso:  mpirun -np K ./arnoldi_lindblad_mpi config.ini

#include "qmpe_core.hpp"
#include <chrono>
#include <vector>

// --- LAPACK zgeev: autovalores/autovectores de matriz general compleja ---
extern "C" void zgeev_(const char* jobvl, const char* jobvr, const int* n,
                       cd* a, const int* lda, cd* w,
                       cd* vl, const int* ldvl, cd* vr, const int* ldvr,
                       cd* work, const int* lwork, double* rwork, int* info);

// Autovalores (w) y autovectores derechos (VR, columnas) de A (n x n, col-major copia)
static void eig_general(const std::vector<cd>& A_rowmajor, int n,
                        std::vector<cd>& w, std::vector<cd>& VR_colmajor) {
    // LAPACK usa column-major: transponer.
    std::vector<cd> A(n * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A[j * n + i] = A_rowmajor[i * n + j];
    w.assign(n, cd(0, 0));
    VR_colmajor.assign(n * n, cd(0, 0));
    std::vector<cd> vl(1);
    std::vector<double> rwork(2 * n);
    int info = 0, ldvl = 1, lwork = -1;
    cd wkopt;
    zgeev_("N", "V", &n, A.data(), &n, w.data(), vl.data(), &ldvl,
           VR_colmajor.data(), &n, &wkopt, &lwork, rwork.data(), &info);
    lwork = (int)wkopt.real();
    std::vector<cd> work(std::max(1, lwork));
    zgeev_("N", "V", &n, A.data(), &n, w.data(), vl.data(), &ldvl,
           VR_colmajor.data(), &n, work.data(), &lwork, rwork.data(), &info);
    if (info != 0 && /*solo rank 0 imprime*/ true) {
        // No abortamos: dejamos que el caller revise NaNs.
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Config cfg;
    std::string cfgpath = (argc > 1) ? argv[1] : "config.ini";
    cfg.load(cfgpath);

    int N        = cfg.geti("ising.N", 4);
    double J     = cfg.getd("ising.J", 1.0);
    double h     = cfg.getd("ising.h", 0.5);
    double gamma = cfg.getd("ising.gamma", 0.4);
    double Tbath = cfg.getd("ising.T", 0.8);
    double tau   = cfg.getd("arnoldi.tau", 0.5);
    double dt    = cfg.getd("arnoldi.dt", 2e-3);
    int m        = cfg.geti("arnoldi.m", 30);
    int restarts = cfg.geti("arnoldi.restarts", 3);
    int kmodes   = cfg.geti("arnoldi.k", 6);

    // ---- modelo ----
    Mat H; std::vector<Mat> Ls; int d;
    build_ising(N, J, h, gamma, Tbath, H, Ls, d);
    std::vector<Mat> Lds, LdL;
    for (auto& L : Ls) { Mat Ld = dagger(L, d); LdL.push_back(matmul(Ld, L, d)); Lds.push_back(Ld); }

    RowDist rd; rd.setup(d, rank, size);

    if (rank == 0) {
        std::cout << "=== T1: Arnoldi-Lindblad (Ising disipativo) ===\n";
        std::cout << "  N=" << N << "  d=2^N=" << d << "  dim(L)=" << d * d << "x" << d * d << "\n";
        std::cout << "  tau=" << tau << " dt=" << dt << " m=" << m
                  << " restarts=" << restarts << " k=" << kmodes << "\n";
        std::cout << "  MPI ranks=" << size << "  OMP threads=" << omp_get_max_threads() << "\n";
    }

    // ---- Arnoldi-Lindblad sobre el propagador P(tau)=e^{tau L} ----
    auto matvec = [&](const Mat& V, long* nsp) {
        return propagator_action(H, Ls, Lds, LdL, V, tau, dt, d, rd, nsp, /*dual=*/false);
    };

    long n_spmv = 0;
    auto t0 = std::chrono::steady_clock::now();

    // semilla: operador hermitico "aleatorio" determinista (mismo en todos los rangos)
    Mat V0(d * d, cd(0, 0));
    {
        unsigned long s = 12345;
        auto nextu = [&]() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return ((s >> 33) & 0xFFFFFFFF) / 4294967296.0 - 0.5; };
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j) V0[IDX(i, j, d)] = cd(nextu(), nextu());
        Mat Vd = dagger(V0, d); for (int i = 0; i < d * d; ++i) V0[i] += Vd[i];
    }

    std::vector<cd> ritz_lambda;   // autovalores lentos de L
    std::vector<Mat> ritz_ops;     // autooperadores de Ritz (derechos)

    for (int rs = 0; rs < std::max(1, restarts); ++rs) {
        // --- iteracion de Arnoldi: base de Krylov de P sobre <.,.>_HS ---
        double beta = hs_norm(V0);
        std::vector<Mat> Q; Q.reserve(m + 1);
        Q.push_back(Mat(d * d));
        for (int i = 0; i < d * d; ++i) Q[0][i] = V0[i] / beta;

        std::vector<cd> Hess((size_t)m * m, cd(0, 0));   // row-major m x m
        int kdim = m;
        for (int j = 0; j < m; ++j) {
            Mat w = matvec(Q[j], &n_spmv);
            for (int i = 0; i <= j; ++i) {
                cd hij = hs_inner(Q[i], w);
                Hess[IDX(i, j, m)] = hij;
                for (int t = 0; t < d * d; ++t) w[t] -= hij * Q[i][t];
            }
            double hjp = hs_norm(w);
            if (j + 1 < m) Hess[IDX(j + 1, j, m)] = hjp;
            if (hjp < 1e-12) { kdim = j + 1; break; }
            Mat q(d * d); for (int t = 0; t < d * d; ++t) q[t] = w[t] / hjp;
            Q.push_back(std::move(q));
        }

        // --- autovalores del Hessenberg (problema pequeno, LAPACK) ---
        std::vector<cd> Hsub((size_t)kdim * kdim);
        for (int i = 0; i < kdim; ++i)
            for (int j = 0; j < kdim; ++j) Hsub[IDX(i, j, kdim)] = Hess[IDX(i, j, m)];
        std::vector<cd> mu, Y;
        eig_general(Hsub, kdim, mu, Y);   // Y col-major: columna c = autovector c

        // ordenar por |mu| descendente (modos lentos primero)
        std::vector<int> ord(kdim); for (int i = 0; i < kdim; ++i) ord[i] = i;
        std::sort(ord.begin(), ord.end(), [&](int a, int b) { return std::abs(mu[a]) > std::abs(mu[b]); });

        ritz_lambda.clear(); ritz_ops.clear();
        int kk = std::min(kmodes, kdim);
        for (int c = 0; c < kk; ++c) {
            int idx = ord[c];
            ritz_lambda.push_back(std::log(mu[idx]) / tau);
            Mat R(d * d, cd(0, 0));
            for (int i = 0; i < (int)Q.size() && i < kdim; ++i) {
                cd y = Y[(size_t)idx * kdim + i];   // col-major
                for (int t = 0; t < d * d; ++t) R[t] += y * Q[i][t];
            }
            ritz_ops.push_back(std::move(R));
        }

        // restart explicito: re-sembrar con la suma de los k modos dominantes
        if (rs + 1 < std::max(1, restarts)) {
            for (int t = 0; t < d * d; ++t) V0[t] = cd(0, 0);
            for (auto& R : ritz_ops) for (int t = 0; t < d * d; ++t) V0[t] += R[t];
            Mat Vd = dagger(V0, d); for (int t = 0; t < d * d; ++t) V0[t] += Vd[t];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    // ---- ordenar lambda por |Re| creciente (estacionario primero) y reportar ----
    if (rank == 0) {
        std::vector<int> ord(ritz_lambda.size());
        for (size_t i = 0; i < ord.size(); ++i) ord[i] = (int)i;
        std::sort(ord.begin(), ord.end(), [&](int a, int b) {
            return std::abs(ritz_lambda[a].real()) < std::abs(ritz_lambda[b].real());
        });
        std::cout << "  --- autovalores lentos de L (lambda_k = ln(mu_k)/tau) ---\n";
        for (size_t i = 0; i < ord.size(); ++i) {
            cd l = ritz_lambda[ord[i]];
            std::cout << "    lambda_" << (i + 1) << " = "
                      << l.real() << (l.imag() >= 0 ? " + " : " - ")
                      << std::abs(l.imag()) << "i\n";
        }
        double gap = ord.size() > 1 ? std::abs(ritz_lambda[ord[1]].real()) : 0.0;
        std::cout << "  gap |Re(lambda_2)| = " << gap
                  << "   (tiempo de relajacion tau_R = " << (gap > 0 ? 1.0 / gap : 0.0) << ")\n";
        std::cout << "  SpMV totales = " << n_spmv << "\n";
        std::cout << "[T1] wall = " << elapsed << " s  (ranks=" << size
                  << ", threads=" << omp_get_max_threads() << ")\n";
        // linea CSV para el benchmark (parsea run_bench.sh)
        std::cout << "BENCH," << N << "," << d << "," << size << ","
                  << omp_get_max_threads() << "," << n_spmv << "," << elapsed << "\n";
    }

    MPI_Finalize();
    return 0;
}
