// philox_rng.hpp
// Counter-based pseudo-random number generator (simplified Philox-like).
//
// Why a counter-based RNG?
//   - Reproducible across OpenMP/MPI: stream(thread_id, mpi_rank, counter) is deterministic.
//   - No state to share between threads; each call is a pure function of (key, counter).
//   - Avoids the classic mistake of seeding std::mt19937 in every thread
//     with time(), which collapses streams.
//
// This is a *minimal* implementation good enough for Monte Carlo / Langevin work,
// inspired by Salmon et al. (2011) "Parallel Random Numbers: As Easy as 1, 2, 3".
// It is NOT cryptographic.
#pragma once
#include <cstdint>
#include <cmath>
#include <array>

namespace mpemba {

struct PhiloxRNG {
    // 64-bit key (set per (mpi_rank, thread_id, run_id))
    uint64_t key;
    // 64-bit counter (incremented per call)
    uint64_t counter;

    explicit PhiloxRNG(uint64_t k = 0xC0FFEEULL, uint64_t c = 0) : key(k), counter(c) {}

    // Bijective integer hash (splitmix64) used as the round function.
    static inline uint64_t mix(uint64_t z) noexcept {
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        return z;
    }

    // Produce one uniform uint64 from the current counter.
    inline uint64_t next_u64() noexcept {
        uint64_t x = mix(counter ^ key);
        x = mix(x + 0x9E3779B97F4A7C15ULL);  // golden ratio constant
        counter += 1;
        return x;
    }

    // Uniform double in [0, 1) with 53 bits of mantissa.
    inline double uniform() noexcept {
        return (next_u64() >> 11) * (1.0 / 9007199254740992.0);  // 2^53
    }

    // Uniform double in (a, b)
    inline double uniform(double a, double b) noexcept {
        return a + (b - a) * uniform();
    }

    // Standard normal via Box-Muller. Both samples consumed; the second is cached.
    inline double normal() noexcept {
        if (have_cached_normal_) {
            have_cached_normal_ = false;
            return cached_normal_;
        }
        double u1, u2;
        do { u1 = uniform(); } while (u1 < 1e-300);
        u2 = uniform();
        double r   = std::sqrt(-2.0 * std::log(u1));
        double phi = 6.283185307179586 * u2;  // 2*pi
        double z0  = r * std::cos(phi);
        double z1  = r * std::sin(phi);
        cached_normal_       = z1;
        have_cached_normal_  = true;
        return z0;
    }

    // Normal with mean mu and std sigma
    inline double normal(double mu, double sigma) noexcept {
        return mu + sigma * normal();
    }

    // Integer in [0, n)
    inline uint64_t randint(uint64_t n) noexcept {
        // Lemire's debiased method, simplified
        uint64_t x   = next_u64();
        __uint128_t m = (__uint128_t)x * (__uint128_t)n;
        return (uint64_t)(m >> 64);
    }

private:
    double cached_normal_      = 0.0;
    bool   have_cached_normal_ = false;
};

// Helper: build a per-stream key from (mpi_rank, omp_thread, run_id).
// Use this once at thread entry to get an independent stream.
inline PhiloxRNG make_rng(int mpi_rank, int thread_id, uint64_t run_id) {
    uint64_t k = 0x9E3779B97F4A7C15ULL;
    k ^= (uint64_t)mpi_rank * 0xD1B54A32D192ED03ULL;
    k ^= ((uint64_t)thread_id + 1) * 0xA4093822299F31D0ULL;
    k ^= run_id * 0xDA942042E4DD58B5ULL;
    return PhiloxRNG(PhiloxRNG::mix(k), 0);
}

} // namespace mpemba
