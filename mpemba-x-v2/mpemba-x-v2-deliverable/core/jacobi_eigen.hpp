// jacobi_eigen.hpp
// Symmetric eigenvalue decomposition via classical Jacobi rotations.
//
// Why this and not LAPACK?
//   - Zero dependencies. Header-only.
//   - For matrices up to ~200x200 it is robust and fast enough for our use cases
//     (Klich-Raz REM, 3-state Markov, small Ising chains via projected R).
//   - For large sparse problems we use power iteration with deflation elsewhere.
//
// Reference: Golub & Van Loan, Matrix Computations, Algorithm 8.4.3.
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstddef>

namespace mpemba {

// Diagonalize a symmetric n x n matrix A (row-major) in-place.
// On output:
//   - 'eigenvalues' contains the eigenvalues (size n)
//   - 'eigenvectors' contains the eigenvectors as columns (row-major n*n)
// Returns the number of sweeps performed.
inline int jacobi_eigen(std::vector<double>& A, int n,
                        std::vector<double>& eigenvalues,
                        std::vector<double>& eigenvectors,
                        double tol = 1e-12, int max_sweeps = 200) {
    eigenvectors.assign((std::size_t)n * n, 0.0);
    for (int i = 0; i < n; ++i) eigenvectors[(std::size_t)i * n + i] = 1.0;

    auto idx = [n](int i, int j) { return (std::size_t)i * n + j; };

    int sweep;
    for (sweep = 0; sweep < max_sweeps; ++sweep) {
        // Compute off-diagonal Frobenius norm
        double off = 0.0;
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q)
                off += A[idx(p, q)] * A[idx(p, q)];
        if (std::sqrt(off) < tol) break;

        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                double apq = A[idx(p, q)];
                if (std::fabs(apq) < 1e-300) continue;

                double app = A[idx(p, p)];
                double aqq = A[idx(q, q)];
                double theta = (aqq - app) / (2.0 * apq);
                double t;
                if (std::fabs(theta) > 1e16) {
                    t = 1.0 / (2.0 * theta);
                } else {
                    t = (theta >= 0 ? 1.0 : -1.0) /
                        (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                }
                double c = 1.0 / std::sqrt(t * t + 1.0);
                double s = t * c;

                A[idx(p, p)] = app - t * apq;
                A[idx(q, q)] = aqq + t * apq;
                A[idx(p, q)] = 0.0;
                A[idx(q, p)] = 0.0;

                for (int r = 0; r < n; ++r) {
                    if (r != p && r != q) {
                        double arp = A[idx(r, p)];
                        double arq = A[idx(r, q)];
                        A[idx(r, p)] = c * arp - s * arq;
                        A[idx(p, r)] = A[idx(r, p)];
                        A[idx(r, q)] = s * arp + c * arq;
                        A[idx(q, r)] = A[idx(r, q)];
                    }
                    double vrp = eigenvectors[idx(r, p)];
                    double vrq = eigenvectors[idx(r, q)];
                    eigenvectors[idx(r, p)] = c * vrp - s * vrq;
                    eigenvectors[idx(r, q)] = s * vrp + c * vrq;
                }
            }
        }
    }

    eigenvalues.resize(n);
    for (int i = 0; i < n; ++i) eigenvalues[i] = A[idx(i, i)];

    // Sort eigenpairs in ascending order of eigenvalue (so eig[0] is smallest in magnitude
    // for negative-semidefinite operators, which is what we want for relaxation generators).
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return eigenvalues[a] < eigenvalues[b]; });

    std::vector<double> evals_sorted(n);
    std::vector<double> evecs_sorted((std::size_t)n * n);
    for (int i = 0; i < n; ++i) {
        evals_sorted[i] = eigenvalues[order[i]];
        for (int r = 0; r < n; ++r) {
            evecs_sorted[idx(r, i)] = eigenvectors[idx(r, order[i])];
        }
    }
    eigenvalues = std::move(evals_sorted);
    eigenvectors = std::move(evecs_sorted);

    return sweep;
}

} // namespace mpemba
