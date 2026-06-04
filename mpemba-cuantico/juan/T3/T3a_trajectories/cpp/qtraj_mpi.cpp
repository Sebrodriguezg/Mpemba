// qtraj_mpi.cpp
//
// Tarea T3a PARALELA: trayectorias cuanticas (Monte Carlo wavefunction) para la
// cadena de Ising transversa disipativa, con paralelismo hibrido MPI + OpenMP.
//
// Clave de muchos cuerpos: se propaga el VECTOR DE ESTADO |psi> in C^d
// (d = 2^N) en vez del operador densidad rho in C^{d x d} (dim 4^N). La memoria
// pasa de 4^N a 2^N, lo que permite alcanzar N inviables para la ecuacion
// maestra. Todas las acciones (H_eff, saltos) son MATRIX-FREE y locales: cuestan
// O(N * d), sin almacenar ninguna matriz.
//
// PARALELIZACION (embarrassingly parallel, Seccion 8.1 nivel 2):
//   * Cada trayectoria es INDEPENDIENTE -> se reparten las M trayectorias entre
//     rangos MPI y, dentro de cada rango, entre hilos OpenMP.
//   * La unica comunicacion es la reduccion final del observable promediado
//     (MPI_Reduce). Escalado casi ideal.
//
// Observable: densidad de excitacion  n_exc(t) = (1/N) sum_i <n_i>(t), promediada
// sobre trayectorias. Se simulan varias preparaciones (product states con
// distinta probabilidad de excitacion p0) para buscar el cruce de Mpemba.
//
// Uso:  mpirun -np K ./qtraj_mpi config.ini

#include <mpi.h>
#include <omp.h>
#include <vector>
#include <complex>
#include <cmath>
#include <random>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <algorithm>
#include <chrono>
#include <cctype>

using cd = std::complex<double>;
using Vec = std::vector<cd>;

// ---------- lector INI minimo ----------
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
    double getd(const std::string& k, double def) { auto it = kv.find(k); return it == kv.end() ? def : std::stod(it->second); }
    int geti(const std::string& k, int def) { auto it = kv.find(k); return it == kv.end() ? def : std::stoi(it->second); }
    std::vector<double> getlist(const std::string& k) {
        std::vector<double> v; auto it = kv.find(k); if (it == kv.end()) return v;
        std::stringstream ss(it->second); std::string tok;
        while (std::getline(ss, tok, ',')) if (!tok.empty()) v.push_back(std::stod(tok));
        return v;
    }
};

// ---------- parametros del modelo Ising ----------
struct Model {
    int N, d;
    double J, h, gamma, nbar;     // nbar = ocupacion de Bose del bano
    double gm, gp;                // tasas: gamma*(nbar+1) (bajada), gamma*nbar (subida)
};

static inline int bit(int s, int i) { return (s >> i) & 1; }

// Parte diagonal de H:  -J sum Z_i Z_{i+1}   (Z|0>=+1, Z|1>=-1)
static double H_diag(const Model& M, int s) {
    double e = 0.0;
    for (int i = 0; i < M.N - 1; ++i) {
        int zi = bit(s, i) ? -1 : 1;
        int zj = bit(s, i + 1) ? -1 : 1;
        e += -M.J * zi * zj;
    }
    return e;
}

// Parte diagonal de  sum_mu L_mu^dag L_mu  (todas diagonales):
//   sigma_-^i : L^dag L = |1><1|_i  (peso gm * n_i)
//   sigma_+^i : L^dag L = |0><0|_i  (peso gp * (1-n_i))
static double LdL_diag(const Model& M, int s) {
    double v = 0.0;
    for (int i = 0; i < M.N; ++i) {
        int ni = bit(s, i);
        v += M.gm * ni + M.gp * (1 - ni);
    }
    return v;
}

// H_eff |psi> = (H_diag - i/2 LdL_diag) |psi>  - h sum_i X_i |psi>     (matrix-free)
static void apply_heff(const Model& M, const Vec& psi, Vec& out) {
    int d = M.d;
    #pragma omp parallel for schedule(static)
    for (int s = 0; s < d; ++s) {
        cd acc = (H_diag(M, s) + cd(0, -0.5) * LdL_diag(M, s)) * psi[s];
        for (int i = 0; i < M.N; ++i)         // -h sum X_i : conecta s con s^(1<<i)
            acc += -M.h * psi[s ^ (1 << i)];
        out[s] = acc;
    }
}

// norma^2 de un vector
static double norm2(const Vec& v) {
    double s = 0.0;
    #pragma omp parallel for reduction(+:s) schedule(static)
    for (int i = 0; i < (int)v.size(); ++i) s += std::norm(v[i]);
    return s;
}

// densidad de excitacion <n> = (1/N) sum_i <n_i> para un estado normalizado
static double excitation_density(const Model& M, const Vec& psi) {
    double tot = 0.0;
    #pragma omp parallel for reduction(+:tot) schedule(static)
    for (int s = 0; s < M.d; ++s) {
        double p = std::norm(psi[s]);
        int cnt = 0; for (int i = 0; i < M.N; ++i) cnt += bit(s, i);
        tot += p * cnt;
    }
    return tot / M.N;
}

// product state inicial: cada sitio (sqrt(1-p0)|0> + sqrt(p0)|1>)
static Vec product_state(const Model& M, double p0) {
    Vec psi(M.d, cd(0, 0));
    double a = std::sqrt(1 - p0), b = std::sqrt(p0);
    for (int s = 0; s < M.d; ++s) {
        cd amp = 1.0;
        for (int i = 0; i < M.N; ++i) amp *= bit(s, i) ? b : a;
        psi[s] = amp;
    }
    return psi;
}

// Una trayectoria: registra n_exc en los pasos log_idx. RNG propio (seed unico).
static void one_trajectory(const Model& M, const Vec& psi0, double t_max, double dt,
                           const std::vector<int>& log_idx, uint64_t seed,
                           std::vector<double>& out /* len(log_idx) */) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    int d = M.d;
    Vec psi = psi0, psd(d);
    int n_steps = (int)std::ceil(t_max / dt);
    size_t lj = 0;
    for (int step = 0; step <= n_steps; ++step) {
        if (lj < log_idx.size() && step == log_idx[lj]) {
            out[lj] += excitation_density(M, psi);
            ++lj;
        }
        if (step == n_steps) break;
        apply_heff(M, psi, psd);                         // psd = H_eff psi
        double nrm2 = 0.0;
        for (int s = 0; s < d; ++s) { psd[s] = psi[s] - cd(0, 1) * dt * psd[s]; nrm2 += std::norm(psd[s]); }
        double dp = 1.0 - nrm2;
        if (U(rng) < dp) {                               // SALTO: elegir canal
            // pesos: gm*n_i (bajada), gp*(1-n_i) (subida), por sitio
            std::vector<double> w(2 * M.N, 0.0); double wsum = 0.0;
            for (int i = 0; i < M.N; ++i) {
                double wm = 0.0, wp = 0.0;
                for (int s = 0; s < d; ++s) {
                    double p = std::norm(psi[s]);
                    if (bit(s, i)) wm += p; else wp += p;
                }
                w[2 * i] = M.gm * wm; w[2 * i + 1] = M.gp * wp;
                wsum += w[2 * i] + w[2 * i + 1];
            }
            double r = U(rng) * wsum, c = 0.0; int ch = 0;
            for (int k = 0; k < 2 * M.N; ++k) { c += w[k]; if (r <= c) { ch = k; break; } }
            int site = ch / 2; bool lower = (ch % 2 == 0);
            Vec phi(d, cd(0, 0));
            for (int s = 0; s < d; ++s) {
                if (lower && bit(s, site)) phi[s ^ (1 << site)] = std::sqrt(M.gm) * psi[s];
                else if (!lower && !bit(s, site)) phi[s ^ (1 << site)] = std::sqrt(M.gp) * psi[s];
            }
            double pn = std::sqrt(norm2(phi));
            for (int s = 0; s < d; ++s) psi[s] = phi[s] / pn;
        } else {
            double inv = 1.0 / std::sqrt(nrm2);
            for (int s = 0; s < d; ++s) psi[s] = psd[s] * inv;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Config cfg; cfg.load(argc > 1 ? argv[1] : "config.ini");
    Model M;
    M.N = cfg.geti("ising.N", 8);
    M.J = cfg.getd("ising.J", 1.0);
    M.h = cfg.getd("ising.h", 0.5);
    M.gamma = cfg.getd("ising.gamma", 0.4);
    double Tbath = cfg.getd("ising.T", 0.8);
    double w0 = 2 * M.h > 0 ? 2 * M.h : 1.0;
    M.nbar = (w0 / Tbath > 700) ? 0.0 : 1.0 / (std::exp(w0 / Tbath) - 1.0);
    M.gm = M.gamma * (M.nbar + 1); M.gp = M.gamma * M.nbar;
    M.d = 1 << M.N;

    double t_max = cfg.getd("run.t_max", 6.0);
    double dt = cfg.getd("run.dt", 5e-3);
    int log_every = cfg.geti("run.log_every", 40);
    int Mtraj = cfg.geti("run.M", 2000);
    std::vector<double> p0s = cfg.getlist("run.p0_list");
    if (p0s.empty()) p0s = {0.1, 0.5, 0.9};
    std::string outdir = "results";
    { auto it = cfg.kv.find("io.output_dir"); if (it != cfg.kv.end()) outdir = it->second; }

    int n_steps = (int)std::ceil(t_max / dt);
    std::vector<int> log_idx;
    for (int s = 0; s <= n_steps; s += log_every) log_idx.push_back(s);
    int nlog = (int)log_idx.size();

    if (rank == 0) {
        std::cout << "=== T3a: trayectorias cuanticas (Ising disipativo) ===\n";
        std::cout << "  N=" << M.N << "  d=2^N=" << M.d
                  << "  (densidad rho seria 4^N=" << (1L << (2 * M.N)) << ")\n";
        std::cout << "  M=" << Mtraj << " trayectorias  t_max=" << t_max << " dt=" << dt << "\n";
        std::cout << "  MPI ranks=" << size << "  OMP threads=" << omp_get_max_threads() << "\n";
    }

    // reparto de las M trayectorias entre rangos
    int per = (Mtraj + size - 1) / size;
    int lo = std::min(rank * per, Mtraj), hi = std::min(lo + per, Mtraj);
    int my_M = hi - lo;

    auto t0 = std::chrono::steady_clock::now();

    for (double p0 : p0s) {
        Vec psi0 = product_state(M, p0);
        std::vector<double> local(nlog, 0.0);
        // OpenMP sobre las trayectorias locales (cada una con su acumulador y RNG)
        #pragma omp parallel
        {
            std::vector<double> tloc(nlog, 0.0);
            #pragma omp for schedule(dynamic, 1) nowait
            for (int m = 0; m < my_M; ++m) {
                uint64_t seed = 0x9E3779B97F4A7C15ULL * (uint64_t)(lo + m)
                              + 0xD1B54A32D192ED03ULL * (uint64_t)rank + 1;
                one_trajectory(M, psi0, t_max, dt, log_idx, seed, tloc);
            }
            #pragma omp critical
            for (int j = 0; j < nlog; ++j) local[j] += tloc[j];
        }
        // reduccion MPI del observable + division por M total
        std::vector<double> global(nlog, 0.0);
        MPI_Reduce(local.data(), global.data(), nlog, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            for (int j = 0; j < nlog; ++j) global[j] /= Mtraj;
            std::ostringstream fn; fn << outdir << "/t3a_p0_" << p0 << ".csv";
            std::ofstream out(fn.str());
            out << "t,n_exc\n";
            for (int j = 0; j < nlog; ++j) out << log_idx[j] * dt << "," << global[j] << "\n";
            std::cout << "  p0=" << p0 << " -> " << fn.str()
                      << "  (n_exc: " << global.front() << " -> " << global.back() << ")\n";
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t1 = std::chrono::steady_clock::now();
    double el = std::chrono::duration<double>(t1 - t0).count();
    if (rank == 0) {
        std::cout << "[T3a] wall = " << el << " s\n";
        std::cout << "BENCH," << M.N << "," << M.d << "," << size << ","
                  << omp_get_max_threads() << "," << Mtraj << "," << el << "\n";
    }
    MPI_Finalize();
    return 0;
}
