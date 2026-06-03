// krylov_evolution.cpp -- Entregable T2(b): evolucion temporal por ACCION DEL EXPONENCIAL.
//
// Metodo: en vez de integrar paso a paso, se aproxima la accion del exponencial
// del Liouvilliano  rho(t+tau) = e^{tau L}[rho(t)]  proyectando en un subespacio
// de Krylov de dimension m construido con Arnoldi sobre la accion matrix-free
// L[.] (lindblad.hpp). Para tau dentro del radio de convergencia, basta m<<d^2
// para precision de maquina, y un solo bloque Krylov avanza un paso de tiempo
// GRANDE (donde RK4 necesitaria muchos pasos pequenos).
//
//   e^{tau L} v  ~=  ||v|| * Q_m * exp(tau H_m) * e_1
//
// con Q_m la base ortonormal de Krylov (operadores d x d) y H_m la Hessenberg
// m x m de Arnoldi. exp(tau H_m) se calcula con scaling-and-squaring + Taylor.
//
// El hotspot paralelizable (OpenMP) es la accion L[.] (matmul), igual que en (a).
// Salida benchmark: N,threads,m,tau,n_chunks,L_applies,wall_s,dhs_final
//
// Uso: OMP_NUM_THREADS=k ./krylov_evolution config.ini [curve.csv]

#include <omp.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "../../common/lindblad.hpp"

// producto interno de Frobenius <A,B> = sum conj(A_i) B_i  (= Tr(A^dag B))
static cd fro_dot(const Mat& A, const Mat& B) {
    cd s(0, 0); for (size_t i = 0; i < A.size(); ++i) s += std::conj(A[i]) * B[i]; return s;
}
static double fro_norm(const Mat& A) { return std::sqrt(fro_dot(A, A).real()); }

// ---- exponencial de matriz pequena (m x m, row-major) por scaling+Taylor ----
static std::vector<cd> sm_matmul(const std::vector<cd>& A, const std::vector<cd>& B, int m) {
    std::vector<cd> C(m * m, cd(0, 0));
    for (int i = 0; i < m; ++i) for (int k = 0; k < m; ++k) { cd a = A[i*m+k];
        if (a == cd(0,0)) continue; for (int j = 0; j < m; ++j) C[i*m+j] += a * B[k*m+j]; }
    return C;
}
static std::vector<cd> small_expm(std::vector<cd> A, int m) {
    // norma 1 para elegir el escalado
    double n1 = 0; for (int j = 0; j < m; ++j) { double c = 0; for (int i = 0; i < m; ++i) c += std::abs(A[i*m+j]); n1 = std::max(n1, c); }
    int s = std::max(0, (int)std::ceil(std::log2(std::max(1.0, n1))));
    double scale = std::pow(2.0, -s);
    for (auto& x : A) x *= scale;
    std::vector<cd> E(m * m, cd(0, 0)), term(m * m, cd(0, 0));
    for (int i = 0; i < m; ++i) { E[i*m+i] = 1.0; term[i*m+i] = 1.0; }   // I
    for (int k = 1; k <= 18; ++k) {                                       // Taylor
        term = sm_matmul(term, A, m);
        cd inv = 1.0 / (double)k; for (auto& x : term) x *= inv;
        for (int i = 0; i < m * m; ++i) E[i] += term[i];
    }
    for (int q = 0; q < s; ++q) E = sm_matmul(E, E, m);                   // squaring
    return E;
}

// ---- un bloque Krylov: avanza rho un tiempo tau. Devuelve nuevo rho. ----
static Mat krylov_step(const Mat& H, const std::vector<Mat>& Ls, const std::vector<Mat>& Lds,
                       const std::vector<Mat>& LdL, const Mat& rho, double tau, int m, int d,
                       long& L_applies) {
    double beta = fro_norm(rho);
    if (beta < 1e-300) return rho;
    std::vector<Mat> Q; Q.reserve(m + 1);
    Q.push_back(scaled(rho, 1.0 / beta));
    std::vector<cd> Hess((m + 1) * m, cd(0, 0));   // Hessenberg (m+1) x m
    int mm = m;
    for (int j = 0; j < m; ++j) {
        Mat w = apply_lindblad(H, Ls, Lds, LdL, Q[j], d); ++L_applies;
        for (int i = 0; i <= j; ++i) { cd hij = fro_dot(Q[i], w); Hess[i*m+j] = hij; axpy(w, Q[i], -hij); }
        double hj = fro_norm(w);
        Hess[(j+1)*m+j] = hj;
        if (hj < 1e-12) { mm = j + 1; break; }        // convergencia anticipada
        Q.push_back(scaled(w, 1.0 / hj));
    }
    // H_m (mm x mm) y exp(tau H_m)
    std::vector<cd> Hm(mm * mm, cd(0, 0));
    for (int i = 0; i < mm; ++i) for (int j = 0; j < mm; ++j) Hm[i*mm+j] = Hess[i*m+j];
    for (auto& x : Hm) x *= tau;
    std::vector<cd> expH = small_expm(Hm, mm);
    // rho_new = beta * sum_i expH[i,0] Q[i]
    Mat out(d * d, cd(0, 0));
    for (int i = 0; i < mm; ++i) axpy(out, Q[i], beta * expH[i*mm+0]);
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
    double tau   = cfg.getd("run.tau", 0.5);   // paso de tiempo (grande) por bloque Krylov
    int    m     = cfg.geti("run.m", 30);      // dimension del subespacio de Krylov
    std::string curve = (argc > 2) ? argv[2] : "";

    Mat H; std::vector<Mat> Ls; int d;
    build_ising(N, J, h, gamma, Tbath, H, Ls, d);
    std::vector<Mat> Lds, LdL; precompute(Ls, d, Lds, LdL);

    Mat rho_ss = gibbs_state(H, Tbath, d);
    Mat rho    = gibbs_state(H, T0, d);

    int n_chunks = (int)std::ceil(t_max / tau);
    long L_applies = 0;
    std::ofstream out;
    if (!curve.empty()) { out.open(curve); out << "t,D_HS\n"; out << 0.0 << "," << d_hs(rho, rho_ss, d) << "\n"; }

    double t0 = now_s();
    for (int c = 0; c < n_chunks; ++c) {
        rho = krylov_step(H, Ls, Lds, LdL, rho, tau, m, d, L_applies);
        if (!curve.empty()) out << (c + 1) * tau << "," << d_hs(rho, rho_ss, d) << "\n";
    }
    double wall = now_s() - t0;

    double dhs_final = d_hs(rho, rho_ss, d);
    int threads = omp_get_max_threads();
    std::cout << std::setprecision(10)
              << N << "," << threads << "," << m << "," << tau << "," << n_chunks << ","
              << L_applies << "," << wall << "," << dhs_final << "\n";
    return 0;
}
