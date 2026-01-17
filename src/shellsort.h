/*
 * shellsort.h - Shellsort with comparison counting
 *
 * Comparison counting follows CLAUDE.md section 3.2:
 * Count ONE comparison each time A[j-gap] > temp is evaluated.
 * Do NOT count loop bounds, swaps, indexing, etc.
 */

#ifndef SHELLSORT_H
#define SHELLSORT_H

#include <stdint.h>
#include <stddef.h>

/* Maximum gaps we support in a sequence */
#define MAX_GAPS 64

/* Gap sequence structure */
typedef struct {
    char name[64];           /* Sequence identifier */
    uint64_t gaps[MAX_GAPS]; /* Gaps in ASCENDING order (1 first) */
    size_t num_gaps;         /* Number of gaps */
} gap_sequence_t;

/* Sort statistics structure */
typedef struct {
    uint64_t comparisons;    /* Data comparisons (A[j-gap] > temp) */
    uint64_t moves;          /* Element moves/assignments */
} sort_stats_t;

/*
 * Sort array using Shellsort with given gap sequence.
 *
 * Parameters:
 *   arr      - array to sort (modified in place)
 *   n        - number of elements
 *   seq      - gap sequence (gaps stored ascending, applied descending)
 *
 * Returns:
 *   Number of data comparisons (A[j-gap] > temp evaluations)
 */
uint64_t shellsort(int32_t *arr, size_t n, const gap_sequence_t *seq);

/*
 * Sort array and return detailed statistics (comparisons + moves).
 *
 * Moves are counted as:
 *   - Each arr[j] = arr[j-gap] in the inner loop (shift)
 *   - The final arr[j] = temp (placement)
 *
 * Returns stats struct with both comparisons and moves.
 */
sort_stats_t shellsort_stats(int32_t *arr, size_t n, const gap_sequence_t *seq);

/*
 * Validate a gap sequence according to CLAUDE.md section 3.3:
 * - Strictly increasing (when stored ascending)
 * - Contains 1
 * - All gaps positive
 * - No duplicates
 *
 * Returns 1 if valid, 0 if invalid.
 * If invalid and reason != NULL, writes explanation to reason.
 */
int gap_sequence_valid(const gap_sequence_t *seq, char *reason, size_t reason_len);

/*
 * Print gap sequence to stdout for debugging.
 */
void gap_sequence_print(const gap_sequence_t *seq);

/*
 * Copy gap sequence.
 */
void gap_sequence_copy(gap_sequence_t *dst, const gap_sequence_t *src);

#endif /* SHELLSORT_H */
