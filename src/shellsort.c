/*
 * shellsort.c - Shellsort implementation with comparison counting
 */

#include "shellsort.h"
#include <stdio.h>
#include <string.h>

uint64_t shellsort(int32_t *arr, size_t n, const gap_sequence_t *seq) {
    uint64_t comparisons = 0;

    /* Apply gaps in descending order (seq stores them ascending) */
    for (size_t g = seq->num_gaps; g > 0; g--) {
        uint64_t gap = seq->gaps[g - 1];

        /* Skip gaps >= n */
        if (gap >= n) continue;

        /* Gapped insertion sort */
        for (size_t i = gap; i < n; i++) {
            int32_t temp = arr[i];
            size_t j = i;

            /*
             * Inner loop: count comparison each time we evaluate arr[j-gap] > temp
             * Per CLAUDE.md 3.2: count ONE comparison per evaluation of the
             * data comparison, regardless of outcome.
             */
            while (j >= gap) {
                comparisons++;  /* Count this comparison */
                if (arr[j - gap] > temp) {
                    arr[j] = arr[j - gap];
                    j -= gap;
                } else {
                    break;
                }
            }
            arr[j] = temp;
        }
    }

    return comparisons;
}

sort_stats_t shellsort_stats(int32_t *arr, size_t n, const gap_sequence_t *seq) {
    sort_stats_t stats = {0, 0};

    /* Apply gaps in descending order (seq stores them ascending) */
    for (size_t g = seq->num_gaps; g > 0; g--) {
        uint64_t gap = seq->gaps[g - 1];

        /* Skip gaps >= n */
        if (gap >= n) continue;

        /* Gapped insertion sort */
        for (size_t i = gap; i < n; i++) {
            int32_t temp = arr[i];
            size_t j = i;

            while (j >= gap) {
                stats.comparisons++;  /* Count comparison */
                if (arr[j - gap] > temp) {
                    arr[j] = arr[j - gap];
                    stats.moves++;    /* Count shift */
                    j -= gap;
                } else {
                    break;
                }
            }
            arr[j] = temp;
            stats.moves++;  /* Count final placement */
        }
    }

    return stats;
}

int gap_sequence_valid(const gap_sequence_t *seq, char *reason, size_t reason_len) {
    if (seq->num_gaps == 0) {
        if (reason) snprintf(reason, reason_len, "Empty sequence");
        return 0;
    }

    /* Must contain 1 as first gap (ascending order) */
    if (seq->gaps[0] != 1) {
        if (reason) snprintf(reason, reason_len, "First gap must be 1, got %lu",
                            (unsigned long)seq->gaps[0]);
        return 0;
    }

    /* Check strictly increasing and positive */
    for (size_t i = 0; i < seq->num_gaps; i++) {
        if (seq->gaps[i] == 0) {
            if (reason) snprintf(reason, reason_len, "Gap %zu is zero", i);
            return 0;
        }

        if (i > 0 && seq->gaps[i] <= seq->gaps[i-1]) {
            if (reason) snprintf(reason, reason_len,
                "Not strictly increasing: gaps[%zu]=%lu <= gaps[%zu]=%lu",
                i, (unsigned long)seq->gaps[i],
                i-1, (unsigned long)seq->gaps[i-1]);
            return 0;
        }
    }

    return 1;
}

void gap_sequence_print(const gap_sequence_t *seq) {
    printf("%s: [", seq->name);
    for (size_t i = 0; i < seq->num_gaps; i++) {
        printf("%lu", (unsigned long)seq->gaps[i]);
        if (i < seq->num_gaps - 1) printf(", ");
    }
    printf("] (%zu gaps)\n", seq->num_gaps);
}

void gap_sequence_copy(gap_sequence_t *dst, const gap_sequence_t *src) {
    memcpy(dst, src, sizeof(gap_sequence_t));
}
