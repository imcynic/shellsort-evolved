/*
 * all_baselines_bench.c - Benchmark evolved vs ALL baselines
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

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
    fread(ds->data, sizeof(int32_t), total, f);
    fclose(f);
    return 0;
}

static double benchmark(const perm_dataset_t *ds, const gap_sequence_t *seq, int threads) {
    uint64_t total = 0;
    #pragma omp parallel for schedule(static) num_threads(threads) reduction(+:total)
    for (uint64_t t = 0; t < ds->trials; t++) {
        int32_t *arr = malloc(ds->N * sizeof(int32_t));
        memcpy(arr, &ds->data[t * ds->N], ds->N * sizeof(int32_t));
        total += shellsort(arr, ds->N, seq);
        free(arr);
    }
    return (double)total / ds->trials;
}

int main(int argc, char **argv) {
    const char *perms_dir = "results/perms";
    int threads = 16;
    if (argc > 1) perms_dir = argv[1];
    if (argc > 2) threads = atoi(argv[2]);
    
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    
    uint64_t sizes[] = {1000000, 2000000, 4000000, 8000000};
    int num_sizes = 4;
    
    printf("ALL BASELINES COMPARISON\n");
    printf("========================\n\n");
    
    /* Results storage */
    double results[7][4];  /* 7 sequences x 4 sizes */
    const char *names[] = {"Ciura", "Ciura-Ext", "Tokuda", "Lee-2021", "Skean-2023", "Sedgewick-86", "EVOLVED"};
    
    for (int s = 0; s < num_sizes; s++) {
        perm_dataset_t ds;
        if (load_dataset(perms_dir, sizes[s], &ds) < 0) {
            printf("Failed to load N=%lu\n", sizes[s]);
            continue;
        }
        
        gap_sequence_t seqs[7];
        gaps_ciura(&seqs[0], sizes[s]);
        gaps_ciura_extended(&seqs[1], sizes[s]);
        gaps_tokuda(&seqs[2], sizes[s]);
        gaps_lee(&seqs[3], sizes[s]);
        gaps_skean(&seqs[4], sizes[s]);
        gaps_sedgewick86(&seqs[5], sizes[s]);
        gaps_evolved(&seqs[6], sizes[s]);
        
        printf("N = %lu (%lu trials)\n", sizes[s], ds.trials);
        for (int i = 0; i < 7; i++) {
            results[i][s] = benchmark(&ds, &seqs[i], threads);
        }
        /* Print after all benchmarks complete so we have evolved result */
        for (int i = 0; i < 7; i++) {
            double vs_evolved = (results[i][s] - results[6][s]) / results[6][s] * 100.0;
            printf("  %-12s: %14.2f  (vs Evolved: %+.4f%%)\n",
                   names[i], results[i][s], i == 6 ? 0.0 : vs_evolved);
        }
        printf("\n");
        free(ds.data);
    }
    
    /* Summary table */
    printf("\nSUMMARY TABLE (Mean Comparisons)\n");
    printf("================================\n");
    printf("%-12s | %14s | %14s | %14s | %14s | %14s\n", 
           "Sequence", "N=1M", "N=2M", "N=4M", "N=8M", "Total");
    printf("-------------|----------------|----------------|----------------|----------------|----------------\n");
    
    for (int i = 0; i < 7; i++) {
        double total = results[i][0] + results[i][1] + results[i][2] + results[i][3];
        printf("%-12s | %14.0f | %14.0f | %14.0f | %14.0f | %14.0f\n",
               names[i], results[i][0], results[i][1], results[i][2], results[i][3], total);
    }
    
    printf("\n\nIMPROVEMENT OF EVOLVED vs EACH BASELINE\n");
    printf("=======================================\n");
    for (int i = 0; i < 6; i++) {
        double total_baseline = results[i][0] + results[i][1] + results[i][2] + results[i][3];
        double total_evolved = results[6][0] + results[6][1] + results[6][2] + results[6][3];
        double improvement = (total_baseline - total_evolved) / total_baseline * 100.0;
        printf("vs %-12s: %+.4f%%\n", names[i], improvement);
    }
    
    return 0;
}
