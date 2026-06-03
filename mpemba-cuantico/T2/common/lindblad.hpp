// lindblad.hpp
//
// Infraestructura compartida para los entregables de evolucion temporal (T2 a/b)
// y trayectorias (T3) del efecto Mpemba cuantico.
//
// Contiene:
//   * tipo matriz densa compleja (row-major) y algebra basica
//   * producto de matrices con OpenMP  (matmul)  -> hotspot paralelizable
//   * SpMV matrix-free del Liouvilliano  apply_lindblad  (ec. GKSL)
//   * modelo fisico comun: cadena de Ising transversa disipativa (build_ising)
//   * estado de Gibbs por evolucion en tiempo imaginario (gibbs_state)
//   * distancia de Hilbert-Schmidt (d_hs) y utilidades de tiempo / config
//
// hbar = kB = 1. La paralelizacion OpenMP esta en matmul(): el mismo binario,
// ejecutado con distinto OMP_NUM_THREADS, da las curvas serial vs paralelo.
#pragma once
#include <vector>
#include <complex>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <chrono>
#include <cctype>

using cd  = std::complex<double>;
using Mat = std::vector<cd>;          // matriz d x d, orden por filas

inline int IDX(int i, int j, int d) { return i * d + j; }

// ---------------------------------------------------------------------
//  Producto de matrices C = A*B  -- PARALELIZADO con OpenMP (orden i-k-j)
// ---------------------------------------------------------------------
inline Mat matmul(const Mat& A, const Mat& B, int d) {
    Mat C(d * d, cd(0, 0));
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < d; ++i) {
        for (int k = 0; k < d; ++k) {
            cd a = A[IDX(i, k, d)];
            if (a == cd(0, 0)) continue;
            const cd* brow = &B[IDX(k, 0, d)];
            cd* crow = &C[IDX(i, 0, d)];
            for (int j = 0; j < d; ++j) crow[j] += a * brow[j];
        }
    }
    return C;
}

// producto matriz-vector y = A x  (para trayectorias: estado |psi> en C^d)
inline std::vector<cd> matvec(const Mat& A, const std::vector<cd>& x, int d) {
    std::vector<cd> y(d, cd(0, 0));
    for (int i = 0; i < d; ++i) {
        cd s(0, 0); const cd* row = &A[IDX(i, 0, d)];
        for (int j = 0; j < d; ++j) s += row[j] * x[j];
        y[i] = s;
    }
    return y;
}

inline Mat dagger(const Mat& A, int d) {
    Mat B(d * d);
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j)
            B[IDX(i, j, d)] = std::conj(A[IDX(j, i, d)]);
    return B;
}
inline void axpy(Mat& Y, const Mat& X, cd a) {
    for (size_t i = 0; i < Y.size(); ++i) Y[i] += a * X[i];
}
inline Mat scaled(const Mat& X, cd a) {
    Mat Y(X.size());
    for (size_t i = 0; i < X.size(); ++i) Y[i] = a * X[i];
    return Y;
}
inline cd trace(const Mat& A, int d) {
    cd t(0, 0);
    for (int i = 0; i < d; ++i) t += A[IDX(i, i, d)];
    return t;
}

// ---------------------------------------------------------------------
//  SpMV matrix-free del Liouvilliano
//     L[V] = -i[H,V] + sum_mu ( L_mu V L_mu^dag - 1/2 {L_mu^dag L_mu, V} )
//  (Lds = L_mu^dag, LdL = L_mu^dag L_mu precomputados)
// ---------------------------------------------------------------------
inline Mat apply_lindblad(const Mat& H, const std::vector<Mat>& Ls,
                          const std::vector<Mat>& Lds, const std::vector<Mat>& LdL,
                          const Mat& V, int d) {
    Mat HV = matmul(H, V, d), VH = matmul(V, H, d);
    Mat out(d * d);
    for (int i = 0; i < d * d; ++i) out[i] = cd(0, -1) * (HV[i] - VH[i]);
    for (size_t mu = 0; mu < Ls.size(); ++mu) {
        Mat LVLd = matmul(matmul(Ls[mu], V, d), Lds[mu], d);
        Mat AV = matmul(LdL[mu], V, d), VA = matmul(V, LdL[mu], d);
        for (int i = 0; i < d * d; ++i) out[i] += LVLd[i] - 0.5 * (AV[i] + VA[i]);
    }
    return out;
}

// precomputa daggers y L^dag L una sola vez
inline void precompute(const std::vector<Mat>& Ls, int d,
                       std::vector<Mat>& Lds, std::vector<Mat>& LdL) {
    Lds.clear(); LdL.clear();
    for (const auto& L : Ls) { Mat Ld = dagger(L, d); LdL.push_back(matmul(Ld, L, d)); Lds.push_back(Ld); }
}

// ---------------------------------------------------------------------
//  Modelo: cadena de Ising transversa disipativa
//     H = -J sum Z_i Z_{i+1} - h sum X_i ; canales locales sqrt(g) sigma-_i, +
// ---------------------------------------------------------------------
inline Mat kron(const Mat& A, int da, const Mat& B, int db) {
    int d = da * db;
    Mat C(d * d, cd(0, 0));
    for (int ia = 0; ia < da; ++ia)
        for (int ja = 0; ja < da; ++ja)
            for (int ib = 0; ib < db; ++ib)
                for (int jb = 0; jb < db; ++jb)
                    C[IDX(ia * db + ib, ja * db + jb, d)] = A[IDX(ia, ja, da)] * B[IDX(ib, jb, db)];
    return C;
}
inline Mat op_at(const Mat& op2, int site, int N) {
    Mat I2 = {1, 0, 0, 1};
    Mat result = (site == 0) ? op2 : I2;
    int dim = 2;
    for (int k = 1; k < N; ++k) { result = kron(result, dim, (k == site) ? op2 : I2, 2); dim *= 2; }
    return result;
}
inline double n_bose(double w, double T) {
    if (w / T > 700) return 0.0;
    return 1.0 / (std::exp(w / T) - 1.0);
}
inline void build_ising(int N, double J, double h, double gamma, double T,
                        Mat& H, std::vector<Mat>& Ls, int& d) {
    d = 1 << N;
    Mat SX = {0, 1, 1, 0}, SZ = {1, 0, 0, -1};
    Mat SM = {0, 1, 0, 0}, SP = {0, 0, 1, 0};   // baja / sube
    H.assign(d * d, cd(0, 0));
    for (int i = 0; i < N; ++i)     { Mat xi = op_at(SX, i, N); axpy(H, xi, -h); }
    for (int i = 0; i < N - 1; ++i) { Mat zz = matmul(op_at(SZ, i, N), op_at(SZ, i + 1, N), d); axpy(H, zz, -J); }
    double nb = n_bose(2 * h > 0 ? 2 * h : 1.0, T);
    Ls.clear();
    for (int i = 0; i < N; ++i) {
        Mat sm = op_at(SM, i, N); for (auto& x : sm) x *= std::sqrt(gamma * (nb + 1));
        Mat sp = op_at(SP, i, N); for (auto& x : sp) x *= std::sqrt(gamma * nb);
        Ls.push_back(sm); Ls.push_back(sp);
    }
}

// estado de Gibbs por tiempo imaginario: dG/dbeta = -1/2{H,G}, rho = G/Tr(G)
inline Mat gibbs_state(const Mat& H, double T, int d) {
    double beta = 1.0 / T;
    int n = std::max(200, (int)(beta * 50));
    double dtau = beta / n;
    Mat G(d * d, cd(0, 0));
    for (int i = 0; i < d; ++i) G[IDX(i, i, d)] = 1.0;
    auto f = [&](const Mat& X) { Mat HX = matmul(H, X, d), XH = matmul(X, H, d); Mat r(d * d);
                                 for (int i = 0; i < d * d; ++i) r[i] = -0.5 * (HX[i] + XH[i]); return r; };
    for (int s = 0; s < n; ++s) {
        Mat k1 = f(G);
        Mat a2 = G; axpy(a2, k1, 0.5 * dtau); Mat k2 = f(a2);
        Mat a3 = G; axpy(a3, k2, 0.5 * dtau); Mat k3 = f(a3);
        Mat a4 = G; axpy(a4, k3, dtau);       Mat k4 = f(a4);
        for (int i = 0; i < d * d; ++i) G[i] += dtau * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]) / 6.0;
    }
    return scaled(G, 1.0 / trace(G, d));
}

inline double d_hs(const Mat& rho, const Mat& rss, int d) {
    Mat diff(d * d);
    for (int i = 0; i < d * d; ++i) diff[i] = rho[i] - rss[i];
    return std::sqrt(std::max(0.0, trace(matmul(diff, diff, d), d).real()));
}

// reloj de pared
inline double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// lector INI minimo
struct Config {
    std::map<std::string, std::string> kv;
    void load(const std::string& path) {
        std::ifstream f(path); std::string line, section;
        while (std::getline(f, line)) {
            auto c = line.find(';'); if (c != std::string::npos) line = line.substr(0, c);
            std::string s; for (char ch : line) if (!std::isspace((unsigned char)ch)) s += ch;
            if (s.empty()) continue;
            if (s.front() == '[') { section = s.substr(1, s.find(']') - 1); continue; }
            auto eq = s.find('='); if (eq == std::string::npos) continue;
            kv[section + "." + s.substr(0, eq)] = s.substr(eq + 1);
        }
    }
    double getd(const std::string& k, double def) { auto it = kv.find(k); return it==kv.end()?def:std::stod(it->second); }
    int    geti(const std::string& k, int def)    { auto it = kv.find(k); return it==kv.end()?def:std::stoi(it->second); }
    std::string gets(const std::string& k, const std::string& def) { auto it = kv.find(k); return it==kv.end()?def:it->second; }
};
