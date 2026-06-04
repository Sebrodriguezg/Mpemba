// qmpe_core.hpp
//
// Nucleo compartido de la tarea T1 (Arnoldi-Lindblad) en C++.
//
// Contiene:
//   * Algebra de matrices densas d x d (row-major) con OpenMP.
//   * matmul DISTRIBUIDO: cada rango MPI calcula un bloque de filas y se
//     reconstruye la matriz completa con MPI_Allgatherv  (paralelismo de datos
//     nivel-1 del informe, Seccion 8.1).
//   * SpMV matrix-free del Liouvilliano  L[V] = -i[H,V] + sum_mu D[L_mu]V
//     (Seccion 7.2): la primitiva dominante de T1.
//   * Estado de Gibbs por tiempo imaginario y constructor de Ising disipativo.
//   * Lector INI minimo.
//
// La idea de paralelizacion de T1: Arnoldi es SECUENCIALMENTE acoplado (cada
// vector de Krylov depende del anterior), de modo que el unico paralelismo
// escalable vive DENTRO del SpMV. Aqui el coste dominante es el producto de
// matrices O(d^3); se distribuye por filas entre rangos (MPI) y se paraleliza
// con hilos (OpenMP) en cada rango.

#pragma once
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
#include <cctype>

using cd = std::complex<double>;
using Mat = std::vector<cd>;   // matriz d x d en orden por filas (row-major)

inline int IDX(int i, int j, int d) { return i * d + j; }

// =====================================================================
//  Reparto de filas entre rangos MPI (bloques contiguos)
// =====================================================================
struct RowDist {
    int d = 0, rank = 0, size = 1, lo = 0, hi = 0;
    std::vector<int> counts;   // # de cd que aporta cada rango (filas*d)
    std::vector<int> displs;   // desplazamientos en cd

    void setup(int d_, int rank_, int size_) {
        d = d_; rank = rank_; size = size_;
        int per = (d + size - 1) / size;
        lo = std::min(rank * per, d);
        hi = std::min(lo + per, d);
        counts.assign(size, 0);
        displs.assign(size, 0);
        for (int r = 0; r < size; ++r) {
            int rlo = std::min(r * per, d);
            int rhi = std::min(rlo + per, d);
            counts[r] = (rhi - rlo) * d;
            displs[r] = rlo * d;
        }
    }
};

// =====================================================================
//  Algebra densa
// =====================================================================
// C = A * B   --  cada rango calcula sus filas [lo,hi); OpenMP dentro.
// Tras el calculo, MPI_Allgatherv reconstruye C completa en todos los rangos.
inline Mat matmul_dist(const Mat& A, const Mat& B, int d, const RowDist& rd) {
    Mat C(d * d, cd(0, 0));
    #pragma omp parallel for schedule(static)
    for (int i = rd.lo; i < rd.hi; ++i) {
        for (int k = 0; k < d; ++k) {
            cd a = A[IDX(i, k, d)];
            if (a == cd(0, 0)) continue;
            const cd* Brow = &B[IDX(k, 0, d)];
            cd* Crow = &C[IDX(i, 0, d)];
            for (int j = 0; j < d; ++j) Crow[j] += a * Brow[j];
        }
    }
    if (rd.size > 1) {
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       C.data(), rd.counts.data(), rd.displs.data(),
                       MPI_C_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    }
    return C;
}

// matmul local (no distribuido), para matrices pequenas / constantes
inline Mat matmul(const Mat& A, const Mat& B, int d) {
    Mat C(d * d, cd(0, 0));
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < d; ++i)
        for (int k = 0; k < d; ++k) {
            cd a = A[IDX(i, k, d)];
            if (a == cd(0, 0)) continue;
            for (int j = 0; j < d; ++j) C[IDX(i, j, d)] += a * B[IDX(k, j, d)];
        }
    return C;
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
inline cd trace(const Mat& A, int d) {
    cd t(0, 0);
    for (int i = 0; i < d; ++i) t += A[IDX(i, i, d)];
    return t;
}
// producto interno de Hilbert-Schmidt <A,B> = Tr(A^dag B) = sum conj(A)*B
inline cd hs_inner(const Mat& A, const Mat& B) {
    cd s(0, 0);
    for (size_t i = 0; i < A.size(); ++i) s += std::conj(A[i]) * B[i];
    return s;
}
inline double hs_norm(const Mat& A) { return std::sqrt(std::max(0.0, hs_inner(A, A).real())); }

// =====================================================================
//  SpMV matrix-free del Liouvilliano  (la primitiva dominante de T1)
//     L[V]    = -i[H,V] + sum_mu ( L_mu V L_mu^dag - 1/2 {L_mu^dag L_mu, V} )
//     L^dag[V]= +i[H,V] + sum_mu ( L_mu^dag V L_mu - 1/2 {L_mu^dag L_mu, V} )
// =====================================================================
inline Mat apply_lindblad(const Mat& H, const std::vector<Mat>& Ls,
                          const std::vector<Mat>& Lds, const std::vector<Mat>& LdL,
                          const Mat& V, int d, const RowDist& rd, bool dual = false) {
    Mat HV = matmul_dist(H, V, d, rd);
    Mat VH = matmul_dist(V, H, d, rd);
    Mat out(d * d);
    cd sign = dual ? cd(0, 1) : cd(0, -1);     // +i[H,V] (dual) o -i[H,V]
    for (int i = 0; i < d * d; ++i) out[i] = sign * (HV[i] - VH[i]);
    for (size_t mu = 0; mu < Ls.size(); ++mu) {
        const Mat& A  = dual ? Lds[mu] : Ls[mu];
        const Mat& Ad = dual ? Ls[mu] : Lds[mu];
        Mat AV  = matmul_dist(A, V, d, rd);
        Mat AVAd = matmul_dist(AV, Ad, d, rd);     // A V A^dag
        Mat BV  = matmul_dist(LdL[mu], V, d, rd);  // (L^dag L) V
        Mat VB  = matmul_dist(V, LdL[mu], d, rd);  // V (L^dag L)
        for (int i = 0; i < d * d; ++i)
            out[i] += AVAd[i] - 0.5 * (BV[i] + VB[i]);
    }
    return out;
}

// =====================================================================
//  Accion del propagador  e^{tau L}[V]  via RK4 matrix-free.
//  Devuelve el resultado; acumula el numero de SpMV en *n_spmv.
// =====================================================================
inline Mat propagator_action(const Mat& H, const std::vector<Mat>& Ls,
                             const std::vector<Mat>& Lds, const std::vector<Mat>& LdL,
                             const Mat& V, double tau, double dt, int d,
                             const RowDist& rd, long* n_spmv, bool dual = false) {
    int n_steps = std::max(1, (int)std::ceil(tau / dt));
    double h = tau / n_steps;
    Mat X = V;
    for (int s = 0; s < n_steps; ++s) {
        Mat k1 = apply_lindblad(H, Ls, Lds, LdL, X, d, rd, dual);
        Mat t2 = X; axpy(t2, k1, 0.5 * h); Mat k2 = apply_lindblad(H, Ls, Lds, LdL, t2, d, rd, dual);
        Mat t3 = X; axpy(t3, k2, 0.5 * h); Mat k3 = apply_lindblad(H, Ls, Lds, LdL, t3, d, rd, dual);
        Mat t4 = X; axpy(t4, k3, h);       Mat k4 = apply_lindblad(H, Ls, Lds, LdL, t4, d, rd, dual);
        for (int i = 0; i < d * d; ++i)
            X[i] += h * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
        if (n_spmv) *n_spmv += 4;
    }
    return X;
}

// =====================================================================
//  Estado de Gibbs por tiempo imaginario:  G(beta)=e^{-beta H}, rho=G/Tr(G)
// =====================================================================
inline Mat gibbs_state(const Mat& H, double T, int d, const RowDist& rd) {
    double beta = 1.0 / T;
    int n = std::max(200, (int)(beta * 50));
    double dtau = beta / n;
    Mat G(d * d, cd(0, 0));
    for (int i = 0; i < d; ++i) G[IDX(i, i, d)] = 1.0;
    auto f = [&](const Mat& X) {
        Mat HX = matmul_dist(H, X, d, rd), XH = matmul_dist(X, H, d, rd);
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
    Mat R(d * d);
    for (int i = 0; i < d * d; ++i) R[i] = G[i] / Z;
    return R;
}

// =====================================================================
//  Modelo: cadena de Ising transversa disipativa  (Seccion 6.3)
// =====================================================================
inline Mat kron(const Mat& A, int da, const Mat& B, int db) {
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

inline Mat op_at(const Mat& op2, int site, int N) {
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

inline double n_bose(double w, double T) {
    if (w / T > 700) return 0.0;
    return 1.0 / (std::exp(w / T) - 1.0);
}

inline void build_ising(int N, double J, double h, double gamma, double T,
                        Mat& H, std::vector<Mat>& Ls, int& d) {
    d = 1 << N;
    Mat SX = {0, 1, 1, 0};
    Mat SZ = {1, 0, 0, -1};
    Mat SM = {0, 1, 0, 0};
    Mat SP = {0, 0, 1, 0};
    H.assign(d * d, cd(0, 0));
    for (int i = 0; i < N; ++i) { Mat xi = op_at(SX, i, N); axpy(H, xi, -h); }
    for (int i = 0; i < N - 1; ++i) {
        Mat zz = matmul(op_at(SZ, i, N), op_at(SZ, i + 1, N), d);
        axpy(H, zz, -J);
    }
    double nb = n_bose(2 * h > 0 ? 2 * h : 1.0, T);
    Ls.clear();
    for (int i = 0; i < N; ++i) {
        Mat sm = op_at(SM, i, N); for (auto& x : sm) x *= std::sqrt(gamma * (nb + 1));
        Mat sp = op_at(SP, i, N); for (auto& x : sp) x *= std::sqrt(gamma * nb);
        Ls.push_back(sm); Ls.push_back(sp);
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
};
