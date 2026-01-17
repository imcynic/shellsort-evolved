/*
 * gaps_candidate.h - Best evolved gap sequence
 *
 * Discovered via evolutionary search: +0.54% improvement over Ciura
 * across ALL sizes (1M, 2M, 4M, 8M).
 */

#ifndef GAPS_CANDIDATE_H
#define GAPS_CANDIDATE_H

#include "shellsort.h"
#include <string.h>

/*
 * Best evolved sequence from genetic search (Run 4, seed 0x1337C0DE).
 * +0.5373% improvement over Ciura on N=1M,2M,4M,8M aggregate.
 */
static inline void gaps_evolved(gap_sequence_t *seq, uint64_t max_gap) {
    static const uint64_t evolved_gaps[] = {
        1, 4, 10, 23, 57, 132, 301, 701,
        1577, 3524, 7705, 17961, 40056, 94681, 199137, 460316,
        1035711, 3236462
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

#endif /* GAPS_CANDIDATE_H */
