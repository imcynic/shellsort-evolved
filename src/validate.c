#define _GNU_SOURCE
/*
 * validate.c - Validate evolved sequence on holdout sizes
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <errno.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "rng.h"
#include "shellsort.h"
#include "gaps_baselines.h"

#define MAGIC 0x5045524D47454E31ULL

typedef struct {
    uint64_t N;
    uint64_t trials;
    int32_t *data;
} perm_dataset_t;

static int load_dataset(const char *dir, uint64_t N, perm_dataset_t *ds) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/perm_%lu.bin", dir, (unsigned long)N);
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    uint64_t magic;
    fread(&magic, sizeof(magic), 1, f);
    if (magic != MAGIC) { fclose(f); return -1; }

    fread(&ds->N, sizeof(ds->N), 1, f);
    fread(&ds->trials, sizeof(ds->trials), 1, f);
    uint64_t seed;
    fread(&seed, sizeof(seed), 1, f);

    size_t total = ds->trials * ds->N;
    ds->data = malloc(total * sizeof(int32_t));
    if (!ds->data) { fclose(f); return -1; }
    fread(ds->data, sizeof(int32_t), total, f);
    fclose(f);
    return 0;
}

static double evaluate(const perm_dataset_t *ds, const gap_sequence_t *seq, int threads) {
    uint64_t N = ds->N;
    uint64_t trials = ds->trials;
    uint64_t total = 0;

    #pragma omp parallel for schedule(static) num_threads(threads) reduction(+:total)
    for (uint64_t t = 0; t < trials; t++) {
        int32_t *arr = malloc(N * sizeof(int32_t));
        if (!arr) continue;
        memcpy(arr, &ds->data[t * N], N * sizeof(int32_t));
        total += shellsort(arr, N, seq);
        free(arr);
    }
    return (double)total / (double)trials;
}

int main(int argc, char **argv) {
    const char *perms_dir = "results/perms";
    int threads = 16;

    if (argc > 1) perms_dir = argv[1];
    if (argc > 2) threads = atoi(argv[2]);

#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif

    /* All target sizes */
    uint64_t sizes[] = {1000000, 2000000, 4000000, 8000000};
    int num_sizes = 4;

    printf("Validating Evolved Sequence on All Sizes (Full Trials)\n");
    printf("=======================================================\n\n");

    printf("%-12s | %-16s | %-16s | %-10s\n", "N", "Ciura", "Evolved", "Diff %");
    printf("-------------|------------------|------------------|------------\n");

    double ciura_total = 0, evolved_total = 0;

    for (int i = 0; i < num_sizes; i++) {
        uint64_t N = sizes[i];

        perm_dataset_t ds;
        if (load_dataset(perms_dir, N, &ds) < 0) {
            printf("Failed to load N=%lu\n", (unsigned long)N);
            continue;
        }

        /* Generate sequences for this N */
        gap_sequence_t ciura, evolved;
        gaps_ciura(&ciura, N);
        gaps_evolved(&evolved, N);

        double ciura_mean = evaluate(&ds, &ciura, threads);
        double evolved_mean = evaluate(&ds, &evolved, threads);

        double diff_pct = (ciura_mean - evolved_mean) / ciura_mean * 100.0;

        printf("%-12lu | %16.2f | %16.2f | %+9.4f%%\n",
               (unsigned long)N, ciura_mean, evolved_mean, diff_pct);

        ciura_total += ciura_mean;
        evolved_total += evolved_mean;

        free(ds.data);
    }

    printf("-------------|------------------|------------------|------------\n");
    double total_diff = (ciura_total - evolved_total) / ciura_total * 100.0;
    printf("%-12s | %16.2f | %16.2f | %+9.4f%%\n", "TOTAL", ciura_total, evolved_total, total_diff);

    printf("\n");
    if (total_diff > 0) {
        printf("*** Evolved sequence is %.4f%% BETTER on holdout sizes ***\n", total_diff);
    } else {
        printf("*** Evolved sequence is %.4f%% WORSE on holdout sizes ***\n", -total_diff);
    }

    return 0;
}
