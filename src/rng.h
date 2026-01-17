/*
 * rng.h - Deterministic RNG for reproducible benchmarks
 *
 * Uses splitmix64 for seeding and xoshiro256** for generation.
 * All state is explicit - no global state.
 */

#ifndef RNG_H
#define RNG_H

#include <stdint.h>

/* splitmix64 - used for seeding xoshiro from a single 64-bit seed */
static inline uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

/* xoshiro256** state */
typedef struct {
    uint64_t s[4];
} rng_state_t;

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

/* xoshiro256** - fast, high-quality PRNG */
static inline uint64_t rng_next(rng_state_t *rng) {
    const uint64_t result = rotl(rng->s[1] * 5, 7) * 9;
    const uint64_t t = rng->s[1] << 17;

    rng->s[2] ^= rng->s[0];
    rng->s[3] ^= rng->s[1];
    rng->s[1] ^= rng->s[2];
    rng->s[0] ^= rng->s[3];

    rng->s[2] ^= t;
    rng->s[3] = rotl(rng->s[3], 45);

    return result;
}

/* Initialize RNG from a single 64-bit seed using splitmix64 */
static inline void rng_seed(rng_state_t *rng, uint64_t seed) {
    rng->s[0] = splitmix64(&seed);
    rng->s[1] = splitmix64(&seed);
    rng->s[2] = splitmix64(&seed);
    rng->s[3] = splitmix64(&seed);
}

/* Generate uniform random in [0, n) using rejection sampling to avoid bias */
static inline uint64_t rng_uniform(rng_state_t *rng, uint64_t n) {
    if (n == 0) return 0;
    uint64_t threshold = -n % n;  /* 2^64 mod n */
    for (;;) {
        uint64_t r = rng_next(rng);
        if (r >= threshold) {
            return r % n;
        }
    }
}

/* Fisher-Yates shuffle - generates a random permutation in-place */
static inline void rng_shuffle(rng_state_t *rng, int32_t *arr, uint64_t n) {
    for (uint64_t i = n - 1; i > 0; i--) {
        uint64_t j = rng_uniform(rng, i + 1);
        int32_t tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

/* Derive a seed for a specific (N, trial) pair from master seed */
static inline uint64_t derive_seed(uint64_t master_seed, uint64_t n, uint64_t trial) {
    uint64_t state = master_seed;
    state ^= n * 0x517cc1b727220a95ULL;
    state ^= trial * 0x2545f4914f6cdd1dULL;
    return splitmix64(&state);
}

#endif /* RNG_H */
