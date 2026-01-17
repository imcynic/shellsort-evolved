/*
 * gaps_baselines.h - Baseline gap sequences for benchmarking
 *
 * All sequences from CLAUDE.md section 5:
 * - Ciura (with 2.25x extension)
 * - Tokuda
 * - Lee (2021) gamma-sequence
 * - Skean et al (2023)
 * - Extended Ciura (machoota) with 1750
 * - Sedgewick 1986
 */

#ifndef GAPS_BASELINES_H
#define GAPS_BASELINES_H

#include "shellsort.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

/*
 * Generate Ciura sequence with 2.25x extension for gaps beyond 701.
 * Base: 1, 4, 10, 23, 57, 132, 301, 701
 * Extension: h_k = floor(2.25 * h_{k-1})
 */
static inline void gaps_ciura(gap_sequence_t *seq, uint64_t max_gap) {
    static const uint64_t ciura_base[] = {1, 4, 10, 23, 57, 132, 301, 701};
    static const size_t ciura_base_len = 8;

    strncpy(seq->name, "Ciura", sizeof(seq->name) - 1);
    seq->num_gaps = 0;

    /* Copy base sequence */
    for (size_t i = 0; i < ciura_base_len && ciura_base[i] <= max_gap; i++) {
        seq->gaps[seq->num_gaps++] = ciura_base[i];
    }

    /* Extend with 2.25x multiplier */
    while (seq->num_gaps < MAX_GAPS) {
        uint64_t next = (uint64_t)(seq->gaps[seq->num_gaps - 1] * 2.25);
        if (next > max_gap || next <= seq->gaps[seq->num_gaps - 1]) break;
        seq->gaps[seq->num_gaps++] = next;
    }
}

/*
 * Extended Ciura (machoota) - Ciura base + 1750 (per OEIS A102549)
 * Then 2.25x extension beyond that.
 */
static inline void gaps_ciura_extended(gap_sequence_t *seq, uint64_t max_gap) {
    static const uint64_t ciura_ext_base[] = {1, 4, 10, 23, 57, 132, 301, 701, 1750};
    static const size_t ciura_ext_base_len = 9;

    strncpy(seq->name, "Ciura-Extended", sizeof(seq->name) - 1);
    seq->num_gaps = 0;

    /* Copy base sequence */
    for (size_t i = 0; i < ciura_ext_base_len && ciura_ext_base[i] <= max_gap; i++) {
        seq->gaps[seq->num_gaps++] = ciura_ext_base[i];
    }

    /* Extend with 2.25x multiplier */
    while (seq->num_gaps < MAX_GAPS) {
        uint64_t next = (uint64_t)(seq->gaps[seq->num_gaps - 1] * 2.25);
        if (next > max_gap || next <= seq->gaps[seq->num_gaps - 1]) break;
        seq->gaps[seq->num_gaps++] = next;
    }
}

/*
 * Tokuda sequence: h_k = ceil((9^k - 4^k) / (5 * 4^(k-1)))
 * For k = 1, 2, 3, ...
 */
static inline void gaps_tokuda(gap_sequence_t *seq, uint64_t max_gap) {
    strncpy(seq->name, "Tokuda", sizeof(seq->name) - 1);
    seq->num_gaps = 0;

    for (int k = 1; seq->num_gaps < MAX_GAPS; k++) {
        /* h_k = ceil((9^k - 4^k) / (5 * 4^(k-1))) */
        double num = pow(9.0, k) - pow(4.0, k);
        double den = 5.0 * pow(4.0, k - 1);
        uint64_t gap = (uint64_t)ceil(num / den);

        if (gap > max_gap) break;

        /* Ensure strictly increasing */
        if (seq->num_gaps > 0 && gap <= seq->gaps[seq->num_gaps - 1]) {
            gap = seq->gaps[seq->num_gaps - 1] + 1;
            if (gap > max_gap) break;
        }

        seq->gaps[seq->num_gaps++] = gap;
    }
}

/*
 * Lee (2021) gamma-sequence: h_k = floor((gamma^k - 1) / (gamma - 1))
 * gamma = 2.243609061420001
 */
static inline void gaps_lee(gap_sequence_t *seq, uint64_t max_gap) {
    static const double GAMMA = 2.243609061420001;

    strncpy(seq->name, "Lee-2021", sizeof(seq->name) - 1);
    seq->num_gaps = 0;

    for (int k = 1; seq->num_gaps < MAX_GAPS; k++) {
        /* h_k = floor((gamma^k - 1) / (gamma - 1)) */
        double gap_d = (pow(GAMMA, k) - 1.0) / (GAMMA - 1.0);
        uint64_t gap = (uint64_t)floor(gap_d);

        if (gap > max_gap) break;

        /* Ensure strictly increasing */
        if (seq->num_gaps > 0 && gap <= seq->gaps[seq->num_gaps - 1]) {
            gap = seq->gaps[seq->num_gaps - 1] + 1;
            if (gap > max_gap) break;
        }

        seq->gaps[seq->num_gaps++] = gap;
    }
}

/*
 * Skean et al (2023): h_k = floor(4.0816 * 8.5714^(k/2.2449))
 * For k = 0, 1, 2, ...
 * NOTE: Formula produces gaps starting at 4, so we prepend 1 to satisfy
 * the requirement that gap sequences must contain 1.
 */
static inline void gaps_skean(gap_sequence_t *seq, uint64_t max_gap) {
    static const double A = 4.0816;
    static const double B = 8.5714;
    static const double C = 2.2449;

    strncpy(seq->name, "Skean-2023", sizeof(seq->name) - 1);
    seq->num_gaps = 0;

    /* Skean formula produces first gap = 4, so prepend 1 */
    seq->gaps[seq->num_gaps++] = 1;

    for (int k = 0; seq->num_gaps < MAX_GAPS; k++) {
        /* h_k = floor(A * B^(k/C)) */
        double gap_d = A * pow(B, (double)k / C);
        uint64_t gap = (uint64_t)floor(gap_d);

        if (gap > max_gap) break;

        /* Ensure strictly increasing */
        if (gap <= seq->gaps[seq->num_gaps - 1]) {
            continue;  /* Skip gaps that would violate strict increase */
        }

        seq->gaps[seq->num_gaps++] = gap;
    }
}

/*
 * Sedgewick 1986: h_k = 4^k + 3*2^(k-1) + 1 for k >= 1, with h_0 = 1
 * Produces: 1, 8, 23, 77, 281, 1073, 4193, ...
 */
static inline void gaps_sedgewick86(gap_sequence_t *seq, uint64_t max_gap) {
    strncpy(seq->name, "Sedgewick-1986", sizeof(seq->name) - 1);
    seq->num_gaps = 0;

    /* h_0 = 1 */
    seq->gaps[seq->num_gaps++] = 1;

    for (int k = 1; seq->num_gaps < MAX_GAPS; k++) {
        /* h_k = 4^k + 3*2^(k-1) + 1 */
        uint64_t gap = ((uint64_t)1 << (2 * k)) + 3 * ((uint64_t)1 << (k - 1)) + 1;

        if (gap > max_gap) break;
        seq->gaps[seq->num_gaps++] = gap;
    }
}

/*
 * Generate ratio-based sequence: h_1 = 1, h_{k+1} = ceil(h_k * ratio)
 * Used for candidate search.
 */
static inline void gaps_ratio(gap_sequence_t *seq, double ratio, uint64_t max_gap, const char *name) {
    if (name) {
        strncpy(seq->name, name, sizeof(seq->name) - 1);
    } else {
        snprintf(seq->name, sizeof(seq->name), "Ratio-%.6f", ratio);
    }
    seq->num_gaps = 0;

    uint64_t gap = 1;
    while (seq->num_gaps < MAX_GAPS && gap <= max_gap) {
        seq->gaps[seq->num_gaps++] = gap;
        uint64_t next = (uint64_t)ceil(gap * ratio);
        if (next <= gap) next = gap + 1;  /* Ensure strictly increasing */
        gap = next;
    }
}

/*
 * Two-phase "split ratio" sequence:
 * Use ratio r1 until threshold, then switch to r2.
 */
static inline void gaps_split_ratio(gap_sequence_t *seq, double r1, double r2,
                                    uint64_t threshold, uint64_t max_gap, const char *name) {
    if (name) {
        strncpy(seq->name, name, sizeof(seq->name) - 1);
    } else {
        snprintf(seq->name, sizeof(seq->name), "Split-%.3f-%.3f@%lu", r1, r2, (unsigned long)threshold);
    }
    seq->num_gaps = 0;

    uint64_t gap = 1;
    while (seq->num_gaps < MAX_GAPS && gap <= max_gap) {
        seq->gaps[seq->num_gaps++] = gap;
        double ratio = (gap < threshold) ? r1 : r2;
        uint64_t next = (uint64_t)ceil(gap * ratio);
        if (next <= gap) next = gap + 1;
        gap = next;
    }
}

/*
 * Evolved sequence from genetic search (+0.54% improvement over Ciura)
 * Validated across ALL sizes: N=1M, 2M, 4M, 8M
 * Run 4, seed 0x1337C0DE, generation 186
 */
static inline void gaps_evolved(gap_sequence_t *seq, uint64_t max_gap) {
    static const uint64_t evolved_gaps[] = {
        1, 4, 10, 23, 57, 132, 301, 701,
        1577, 3524, 7705, 17961, 40056, 94681, 199137, 460316, 1035711, 3236462
    };
    static const size_t evolved_len = 18;

    strncpy(seq->name, "Evolved", sizeof(seq->name) - 1);
    seq->num_gaps = 0;

    for (size_t i = 0; i < evolved_len && evolved_gaps[i] <= max_gap; i++) {
        seq->gaps[seq->num_gaps++] = evolved_gaps[i];
    }

    /* Extend with 2.25x if needed */
    while (seq->num_gaps < MAX_GAPS) {
        uint64_t next = (uint64_t)(seq->gaps[seq->num_gaps - 1] * 2.25);
        if (next > max_gap || next <= seq->gaps[seq->num_gaps - 1]) break;
        seq->gaps[seq->num_gaps++] = next;
    }
}

/* Number of baseline sequences */
#define NUM_BASELINES 6

/* Generate all baseline sequences into an array */
static inline void gaps_all_baselines(gap_sequence_t seqs[NUM_BASELINES], uint64_t max_gap) {
    gaps_ciura(&seqs[0], max_gap);
    gaps_ciura_extended(&seqs[1], max_gap);
    gaps_tokuda(&seqs[2], max_gap);
    gaps_lee(&seqs[3], max_gap);
    gaps_skean(&seqs[4], max_gap);
    gaps_sedgewick86(&seqs[5], max_gap);
    /* gaps_evolved removed - let search find its own path */
}

#endif /* GAPS_BASELINES_H */
