// qmpe_hpc.cpp
//
// Nucleo HPC del efecto Mpemba cuantico (QMpE) -- acompana a mpemba_cuantico.tex.
//
// Implementa las PRIMITIVAS paralelizables del informe (Secciones 7 y 8):
//   * SpMV 'matrix-free' del Liouvilliano:  L[V] = -i[H,V] + sum_mu D[L_mu]V
//     (Seccion 7.2). Es el hotspot; se paraleliza con OpenMP.
//   * Evolucion temporal por RK4 (tarea T2) sin formar la matriz d^2 x d^2.
//   * Estado de Gibbs por evolucion en tiempo imaginario (sin eigensolver).
//   * Barrido de preparaciones distribuido entre rangos MPI (Seccion 8.1, nivel 1/2).
//   * Distancia de Hilbert-Schmidt D_HS (no requiere diagonalizar) y deteccion
//     del cruce de Mpemba.
//
// Modelo de referencia: cadena de Ising transversa disipativa (Seccion 6.3),
// el caso que EXIGE HPC porque d = 2^N y dim(L) = 4^N.
//
// COSTURAS DE PARALELIZACION (marcadas con  // [PARALELIZAR] ):
//   1. OpenMP sobre el SpMV / multiplicacion de matrices (intranodo).      [hecho]
//   2. MPI sobre el barrido de preparaciones / parametros (internodo).     [hecho]
//   3. Sustituir RK4 denso por matrix-free disperso + Krylov (T2).         [seam]
//   4. Anadir trayectorias cuanticas (T3a, embarrassingly parallel).       [seam]
//   5. Diagnostico espectral T1 via SLEPc/ARPACK (shift-invert sigma=0).   [seam]
//
// Compilacion:  ver CMakeLists.txt (MPI + OpenMP, C++17).
// Uso:          mpirun -np K ./qmpe_hpc config.ini

#include <mpi.h>
#include <omp.h>

#include <vector>
#include <complex>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <algorithm>
#include <chrono>

using cd = std::complex<double>;
using Mat = std::vector<cd>;   // matriz d x d en orden por filas (row-major)

// =====================================================================
//  Algebra de matrices densas (con OpenMP en el producto)
// =====================================================================
struct Dim { int d; };

inline int IDX(int i, int j, int d) { return i * d + j; }

// C = A * B   -- [PARALELIZAR] nivel 3 (intranodo, OpenMP)
Mat matmul(const Mat& A, const Mat& B, int d) {
    Mat C(d * d, cd(0, 0));
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < d; ++i) {
        for (int k = 0; k < d; ++k) {
            cd a = A[IDX(i, k, d)];
            if (a == cd(0, 0)) continue;
            for (int j = 0; j < d; ++j)
                C[IDX(i, j, d)] += a * B[IDX(k, j, d)];
        }
    }
    return C;
}

Mat dagger(const Mat& A, int d) {
    Mat B(d * d);
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j)
            B[IDX(i, j, d)] = std::conj(A[IDX(j, i, d)]);
    return B;
}

void axpy(Mat& Y, const Mat& X, cd a) {              // Y += a*X
    for (size_t i = 0; i < Y.size(); ++i) Y[i] += a * X[i];
}
Mat scaled(const Mat& X, cd a) {
    Mat Y(X.size());
    for (size_t i = 0; i < X.size(); ++i) Y[i] = a * X[i];
    return Y;
}
cd trace(const Mat& A, int d) {
    cd t(0, 0);
    for (int i = 0; i < d; ++i) t += A[IDX(i, i, d)];
    return t;
}

// =====================================================================
//  SpMV matrix-free del Liouvilliano  (Seccion 7.2)
//     L[V] = -i[H,V] + sum_mu ( L_mu V L_mu^dag - 1/2 {L_mu^dag L_mu, V} )
// =====================================================================
Mat apply_lindblad(const Mat& H, const std::vector<Mat>& Ls,
                   const std::vector<Mat>& Lds,       // L_mu^dagger precomputados
                   const std::vector<Mat>& LdL,       // L_mu^dag L_mu precomputados
                   const Mat& V, int d) {
    // parte coherente -i[H,V]
    Mat HV = matmul(H, V, d);
    Mat VH = matmul(V, H, d);
    Mat out(d * d);
    for (int i = 0; i < d * d; ++i) out[i] = cd(0, -1) * (HV[i] - VH[i]);
    // parte disipativa
    for (size_t mu = 0; mu < Ls.size(); ++mu) {
        Mat LV = matmul(Ls[mu], V, d);
        Mat LVLd = matmul(LV, Lds[mu], d);            // L V L^dag
        Mat AV = matmul(LdL[mu], V, d);               // (L^dag L) V
        Mat VA = matmul(V, LdL[mu], d);               // V (L^dag L)
        for (int i = 0; i < d * d; ++i)
            out[i] += LVLd[i] - 0.5 * (AV[i] + VA[i]);
    }
    return out;
}

// =====================================================================
//  Paso RK4 de la ecuacion maestra  (tarea T2)
// =====================================================================
Mat rk4_step(const Mat& H, const std::vector<Mat>& Ls, const std::vector<Mat>& Lds,
             const std::vector<Mat>& LdL, const Mat& rho, double dt, int d) {
    Mat k1 = apply_lindblad(H, Ls, Lds, LdL, rho, d);
    Mat t2 = rho; axpy(t2, k1, 0.5 * dt);
    Mat k2 = apply_lindblad(H, Ls, Lds, LdL, t2, d);
    Mat t3 = rho; axpy(t3, k2, 0.5 * dt);
    Mat k3 = apply_lindblad(H, Ls, Lds, LdL, t3, d);
    Mat t4 = rho; axpy(t4, k3, dt);
    Mat k4 = apply_lindblad(H, Ls, Lds, LdL, t4, d);
    Mat out = rho;
    for (int i = 0; i < d * d; ++i)
        out[i] += dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
    return out;
}

// =====================================================================
//  Estado de Gibbs por evolucion en tiempo imaginario (sin eigensolver)
//     dG/dbeta = -1/2 {H, G},  G(0)=I  =>  G(beta)=e^{-beta H};  rho=G/Tr(G)
// =====================================================================
Mat gibbs_state(const Mat& H, double T, int d) {
    double beta = 1.0 / T;
    int n = std::max(200, (int)(beta * 50));
    double dtau = beta / n;
    Mat G(d * d, cd(0, 0));
    for (int i = 0; i < d; ++i) G[IDX(i, i, d)] = 1.0;       // identidad
    auto f = [&](const Mat& X) {                              // -1/2{H,X}
        Mat HX = matmul(H, X, d), XH = matmul(X, H, d);
        Mat r(d * d);
        for (int i = 0; i < d * d; ++i) r[i] = -0.5 * (HX[i] + XH[i]);
        return r;
    };
    for (int s = 0; s < n; ++s) {
        Mat k1 = f(G);
        Mat a2 = G; axpy(a2, k1, 0.5 * dtau); Mat k2 = f(a2);
        Mat a3 = G; axpy(a3, k2, 0.5 * dtau); Mat k3 = f(a3);
        Mat a4 = G; axpy(a4, k3, dtau);       Mat k4 = f(a4);
        for (int i = 0; i < d * d; ++i)
            G[i] += dtau * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
    }
    cd Z = trace(G, d);
    return scaled(G, 1.0 / Z);
}

// =====================================================================
//  Distancia de Hilbert-Schmidt  D_HS = sqrt(Tr[(rho-rss)^2])
//  (no requiere diagonalizar; valida como diagnostico de Mpemba)
// =====================================================================
double d_hs(const Mat& rho, const Mat& rss, int d) {
    Mat diff(d * d);
    for (int i = 0; i < d * d; ++i) diff[i] = rho[i] - rss[i];
    Mat sq = matmul(diff, diff, d);
    return std::sqrt(std::max(0.0, trace(sq, d).real()));
}

// =====================================================================
//  Modelo: cadena de Ising transversa disipativa  (Seccion 6.3)
// =====================================================================
Mat kron(const Mat& A, int da, const Mat& B, int db) {
    int d = da * db;
    Mat C(d * d, cd(0, 0));
    for (int ia = 0; ia < da; ++ia)
        for (int ja = 0; ja < da; ++ja)
            for (int ib = 0; ib < db; ++ib)
                for (int jb = 0; jb < db; ++jb)
                    C[IDX(ia * db + ib, ja * db + jb, d)] =
                        A[IDX(ia, ja, da)] * B[IDX(ib, jb, db)];
    return C;
}

Mat op_at(const Mat& op2, int site, int N) {
    Mat I2 = {1, 0, 0, 1};
    Mat result = (site == 0) ? op2 : I2;
    int dim = 2;
    for (int k = 1; k < N; ++k) {
        const Mat& next = (k == site) ? op2 : I2;
        result = kron(result, dim, next, 2);
        dim *= 2;
    }
    return result;
}

double n_bose(double w, double T) {
    if (w / T > 700) return 0.0;
    return 1.0 / (std::exp(w / T) - 1.0);
}

void build_ising(int N, double J, double h, double gamma, double T,
                 Mat& H, std::vector<Mat>& Ls, int& d) {
    d = 1 << N;
    Mat SX = {0, 1, 1, 0};
    Mat SZ = {1, 0, 0, -1};
    Mat SM = {0, 1, 0, 0};   // |0><1| baja
    Mat SP = {0, 0, 1, 0};   // |1><0| sube
    H.assign(d * d, cd(0, 0));
    for (int i = 0; i < N; ++i) {           // -h sum X_i
        Mat xi = op_at(SX, i, N);
        axpy(H, xi, -h);
    }
    for (int i = 0; i < N - 1; ++i) {       // -J sum Z_i Z_{i+1}
        Mat zz = matmul(op_at(SZ, i, N), op_at(SZ, i + 1, N), d);
        axpy(H, zz, -J);
    }
    double nb = n_bose(2 * h > 0 ? 2 * h : 1.0, T);
    Ls.clear();
    for (int i = 0; i < N; ++i) {
        Mat sm = op_at(SM, i, N); for (auto& x : sm) x *= std::sqrt(gamma * (nb + 1));
        Mat sp = op_at(SP, i, N); for (auto& x : sp) x *= std::sqrt(gamma * nb);
        Ls.push_back(sm);
        Ls.push_back(sp);
    }
}

// =====================================================================
//  Lector INI minimo
// =====================================================================
struct Config {
    std::map<std::string, std::string> kv;
    void load(const std::string& path) {
        std::ifstream f(path);
        std::string line, section;
        while (std::getline(f, line)) {
            auto c = line.find(';'); if (c != std::string::npos) line = line.substr(0, c);
            std::string s; for (char ch : line) if (!std::isspace((unsigned char)ch)) s += ch;
            if (s.empty()) continue;
            if (s.front() == '[') { section = s.substr(1, s.find(']') - 1); continue; }
            auto eq = s.find('='); if (eq == std::string::npos) continue;
            kv[section + "." + s.substr(0, eq)] = s.substr(eq + 1);
        }
    }
    double getd(const std::string& k, double def) {
        auto it = kv.find(k); return it == kv.end() ? def : std::stod(it->second);
    }
    int geti(const std::string& k, int def) {
        auto it = kv.find(k); return it == kv.end() ? def : std::stoi(it->second);
    }
    std::vector<double> getlist(const std::string& k) {
        std::vector<double> v; auto it = kv.find(k); if (it == kv.end()) return v;
        std::stringstream ss(it->second); std::string tok;
        while (std::getline(ss, tok, ',')) if (!tok.empty()) v.push_back(std::stod(tok));
        return v;
    }
};

// =====================================================================
//  main
// =====================================================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Config cfg;
    std::string cfgpath = (argc > 1) ? argv[1] : "config.ini";
    cfg.load(cfgpath);

    int N        = cfg.geti("ising.N", 3);
    double J     = cfg.getd("ising.J", 1.0);
    double h     = cfg.getd("ising.h", 0.5);
    double gamma = cfg.getd("ising.gamma", 0.4);
    double Tbath = cfg.getd("ising.T", 0.8);
    double t_max = cfg.getd("run.t_max", 8.0);
    double dt    = cfg.getd("run.dt", 2e-3);
    int log_every= cfg.geti("run.log_every", 25);
    std::vector<double> T0s = cfg.getlist("run.T0_list");
    if (T0s.empty()) T0s = {0.3, 0.6, 1.0, 2.0, 5.0, 20.0};
    std::string outdir = "results";
    {   auto it = cfg.kv.find("io.output_dir"); if (it != cfg.kv.end()) outdir = it->second; }

    // ---- construir modelo ----
    Mat H; std::vector<Mat> Ls; int d;
    build_ising(N, J, h, gamma, Tbath, H, Ls, d);
    std::vector<Mat> Lds, LdL;
    for (auto& L : Ls) { Mat Ld = dagger(L, d); LdL.push_back(matmul(Ld, L, d)); Lds.push_back(Ld); }

    if (rank == 0) {
        std::cout << "=== QMpE HPC (Ising disipativo) ===\n";
        std::cout << "  N=" << N << "  d=2^N=" << d << "  dim(L)=" << d*d << "x" << d*d << "\n";
        std::cout << "  J=" << J << " h=" << h << " gamma=" << gamma << " T_bath=" << Tbath << "\n";
        std::cout << "  preparaciones T0 = ["; for (double t : T0s) std::cout << " " << t; std::cout << " ]\n";
        std::cout << "  ranks MPI=" << size << "  OMP threads=" << omp_get_max_threads() << "\n";
    }

    // ---- estado estacionario: evolucion larga del estado maximamente mixto ----
    Mat rho_ss(d * d, cd(0, 0));
    for (int i = 0; i < d; ++i) rho_ss[IDX(i, i, d)] = 1.0 / d;
    {
        double Tss = std::max(t_max * 3.0, 30.0);
        int nss = (int)(Tss / dt);
        for (int s = 0; s < nss; ++s) rho_ss = rk4_step(H, Ls, Lds, LdL, rho_ss, dt, d);
    }

    // ---- barrido de preparaciones distribuido entre rangos MPI ----
    // [PARALELIZAR] nivel 2 (internodo): cada rango procesa un subconjunto de T0.
    int nT = (int)T0s.size();
    int per = (nT + size - 1) / size;
    int lo = std::min(rank * per, nT), hi = std::min(lo + per, nT);

    int n_log = 0;
    auto t_start = std::chrono::steady_clock::now();
    for (int ti = lo; ti < hi; ++ti) {
        double T0 = T0s[ti];
        Mat rho = gibbs_state(H, T0, d);          // preparacion inicial
        int n_steps = (int)std::ceil(t_max / dt);
        std::ostringstream fn; fn << outdir << "/qmpe_T0_" << T0 << ".csv";
        std::ofstream out(fn.str());
        out << "t,D_HS\n";
        for (int s = 0; s <= n_steps; ++s) {
            if (s % log_every == 0) out << s * dt << "," << d_hs(rho, rho_ss, d) << "\n";
            if (s < n_steps) rho = rk4_step(H, Ls, Lds, LdL, rho, dt, d);
        }
        n_log++;
        std::cout << "  rank " << rank << " T0=" << T0
                  << " D_HS(0)->fin escrito en " << fn.str() << "\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_end = std::chrono::steady_clock::now();
    if (rank == 0) {
        double el = std::chrono::duration<double>(t_end - t_start).count();
        std::cout << "[qmpe_hpc] hecho en " << el << " s\n";
        std::cout << "  -> analizar cruces con  python ../python/plot_hpc.py " << outdir << "\n";
    }
    MPI_Finalize();
    return 0;
}
