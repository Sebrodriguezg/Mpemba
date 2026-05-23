// thermomajorization.hpp
//
// THERMOMAJORIZATION - the universal diagnostic of the Mpemba effect.
//
// Reference: Tan Van Vu & Hisao Hayakawa, "Thermomajorization Mpemba effect",
//            Phys. Rev. Lett. 134, 107101 (2025).
//
// WHY THIS MATTERS
// ----------------
// Most Mpemba diagnostics depend on the choice of "distance to equilibrium"
// (L1, KL, entropic, ...). Different choices can give contradictory results:
// the effect appears in one metric and not in another (Burridge & Linden 2016).
// Vu & Hayakawa proved that the THERMOMAJORIZATION preorder is equivalent to
// requiring monotonic decrease under EVERY monotone-under-Gibbs-preserving-maps
// distance simultaneously. So if p_h thermomajorizes p_w at every time t > 0
// (i.e., p_h ≻_T p_w), the ME is unambiguous.
//
// HOW IT IS COMPUTED
// ------------------
// Given a probability vector p and the bath Gibbs distribution gamma at T_b,
// define the "beta-ordered" sequence: sort the indices i by gamma_i (ascending).
// The thermomajorization curve is the piecewise-linear function joining the
// points
//          (X_k, Y_k) = ( sum_{j<=k} gamma_{pi(j)},  sum_{j<=k} p_{pi(j)} )
// where pi is the beta-ordering permutation.
//
// p1 thermomajorizes p2 (p1 ≻_{T_b} p2) iff the curve of p1 lies entirely
// at or above the curve of p2 on [0,1].
//
// USAGE
//   ThermomajorizationCurve c1 = build_thermomaj_curve(p1, energies, T_b);
//   ThermomajorizationCurve c2 = build_thermomaj_curve(p2, energies, T_b);
//   ThermomajComparison cmp = compare_thermomaj(c1, c2);
//   if (cmp.dominates_strict) // p1 is strictly further from equilibrium
//
// EQUIVALENCE WITH OTHER METRICS
//   Theorem (Lostaglio, Korzekwa 2022; Vu-Hayakawa 2025):
//   p ≻_{T_b} q  ⇔  D_f(p || gamma) >= D_f(q || gamma)  for ALL Gibbs-
//                   contractive f-divergences (KL, total variation, alpha-
//                   Rényi for alpha > 0, etc.).
//
#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstddef>

namespace mpemba {

struct ThermomajorizationCurve {
    // Curve is given by (X_k, Y_k) for k = 0,1,...,n with X_0 = Y_0 = 0
    // and X_n = Y_n = 1 (assuming p and gamma are normalized).
    std::vector<double> X;
    std::vector<double> Y;
};

// Build the thermomajorization (Lorenz) curve of distribution p with respect
// to the bath Gibbs distribution gamma(T_b). 'energies' is the list of state
// energies. kB = 1 throughout.
inline ThermomajorizationCurve build_thermomaj_curve(const std::vector<double>& p,
                                                    const std::vector<double>& energies,
                                                    double T_b) {
    const std::size_t n = p.size();
    // Compute gamma (Boltzmann distribution at T_b)
    std::vector<double> gamma(n);
    double Emin = *std::min_element(energies.begin(), energies.end());
    double Z = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        gamma[i] = std::exp(-(energies[i] - Emin) / T_b);
        Z += gamma[i];
    }
    for (std::size_t i = 0; i < n; ++i) gamma[i] /= Z;

    // Beta-ordering: sort indices by p_i / gamma_i descending.
    // The thermomajorization curve uses this order so the curve is
    // concave (Lostaglio 2022). With this ordering, p ≻_T q
    // iff Y(p)_k >= Y(q)_k at the breakpoints, AS LONG AS the curves
    // are upper concave envelopes -- which they are after the sort.
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) {
                  // Sort descending by p/gamma
                  double ra = (gamma[a] > 1e-300) ? p[a] / gamma[a]
                                                  : std::numeric_limits<double>::infinity();
                  double rb = (gamma[b] > 1e-300) ? p[b] / gamma[b]
                                                  : std::numeric_limits<double>::infinity();
                  return ra > rb;
              });

    ThermomajorizationCurve c;
    c.X.reserve(n + 1);
    c.Y.reserve(n + 1);
    c.X.push_back(0.0);
    c.Y.push_back(0.0);
    double sx = 0.0, sy = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
        sx += gamma[order[k]];
        sy += p[order[k]];
        c.X.push_back(sx);
        c.Y.push_back(sy);
    }
    return c;
}

// Evaluate a thermomajorization curve at a value x in [0, 1] by linear
// interpolation between the breakpoints.
inline double evaluate_curve(const ThermomajorizationCurve& c, double x) {
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;
    // Binary search for X[k] <= x < X[k+1]
    std::size_t lo = 0, hi = c.X.size() - 1;
    while (lo + 1 < hi) {
        std::size_t mid = (lo + hi) / 2;
        if (c.X[mid] <= x) lo = mid; else hi = mid;
    }
    double xl = c.X[lo], xr = c.X[lo + 1];
    double yl = c.Y[lo], yr = c.Y[lo + 1];
    if (xr - xl < 1e-300) return yl;
    return yl + (yr - yl) * (x - xl) / (xr - xl);
}

struct ThermomajComparison {
    // For c1 vs c2 (does c1 dominate c2, i.e. c1's curve is above c2's?)
    bool dominates_weak     = false;   // Y1(x) >= Y2(x) for all x (with tolerance)
    bool dominates_strict   = false;   // dominates_weak AND strictly somewhere
    bool curves_cross       = false;   // there exist points where c1>c2 and others where c2>c1
    double max_gap          = 0.0;     // max_x [Y1(x) - Y2(x)]
    double min_gap          = 0.0;     // min_x [Y1(x) - Y2(x)]
};

// Compare two thermomajorization curves on a fine grid.
inline ThermomajComparison compare_thermomaj(const ThermomajorizationCurve& c1,
                                             const ThermomajorizationCurve& c2,
                                             int n_grid = 4096,
                                             double tol = 1e-9) {
    ThermomajComparison out;
    bool any_pos = false;
    bool any_neg = false;
    double mx = -std::numeric_limits<double>::infinity();
    double mn =  std::numeric_limits<double>::infinity();
    for (int i = 0; i <= n_grid; ++i) {
        double x = (double)i / n_grid;
        double y1 = evaluate_curve(c1, x);
        double y2 = evaluate_curve(c2, x);
        double d  = y1 - y2;
        if (d > mx) mx = d;
        if (d < mn) mn = d;
        if (d > tol)  any_pos = true;
        if (d < -tol) any_neg = true;
    }
    out.max_gap = mx;
    out.min_gap = mn;
    out.dominates_weak   = (!any_neg);
    out.dominates_strict = (any_pos && !any_neg);
    out.curves_cross     = (any_pos && any_neg);
    return out;
}

// Convenience: given two distributions p1, p2 and a bath, return whether
// p1 thermomajorizes p2.
inline ThermomajComparison thermomajorizes(const std::vector<double>& p1,
                                           const std::vector<double>& p2,
                                           const std::vector<double>& energies,
                                           double T_b) {
    auto c1 = build_thermomaj_curve(p1, energies, T_b);
    auto c2 = build_thermomaj_curve(p2, energies, T_b);
    return compare_thermomaj(c1, c2);
}

} // namespace mpemba
