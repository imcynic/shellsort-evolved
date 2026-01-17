/*
 * full_bench.c - Comprehensive benchmark with full statistics
 * Outputs per-trial data for statistical analysis
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

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

typedef struct {
    uint64_t *comparisons;  /* per-trial comparisons */
    double *runtimes_us;    /* per-trial runtime in microseconds */
    uint64_t trials;
    double mean_comps;
    double stddev_comps;
    double stderr_comps;
    double ci95_low;
    double ci95_high;
    double mean_runtime;
    double stddev_runtime;
} detailed_stats_t;

static void compute_stats(detailed_stats_t *stats) {
    double sum = 0;
    for (uint64_t i = 0; i < stats->trials; i++) {
        sum += stats->comparisons[i];
    }
    stats->mean_comps = sum / stats->trials;
    
    double var = 0;
    for (uint64_t i = 0; i < stats->trials; i++) {
        double diff = stats->comparisons[i] - stats->mean_comps;
        var += diff * diff;
    }
    stats->stddev_comps = sqrt(var / (stats->trials - 1));
    stats->stderr_comps = stats->stddev_comps / sqrt(stats->trials);
    
    /* 95% CI using t-distribution approximation (t ~ 1.96 for large n) */
    double t_val = 1.96;
    if (stats->trials < 30) t_val = 2.045;  /* t for df=29 */
    if (stats->trials < 20) t_val = 2.093;  /* t for df=19 */
    if (stats->trials < 10) t_val = 2.262;  /* t for df=9 */
    
    stats->ci95_low = stats->mean_comps - t_val * stats->stderr_comps;
    stats->ci95_high = stats->mean_comps + t_val * stats->stderr_comps;
    
    /* Runtime stats */
    sum = 0;
    for (uint64_t i = 0; i < stats->trials; i++) {
        sum += stats->runtimes_us[i];
    }
    stats->mean_runtime = sum / stats->trials;
    
    var = 0;
    for (uint64_t i = 0; i < stats->trials; i++) {
        double diff = stats->runtimes_us[i] - stats->mean_runtime;
        var += diff * diff;
    }
    stats->stddev_runtime = sqrt(var / (stats->trials - 1));
}

static detailed_stats_t benchmark_sequence(const perm_dataset_t *ds, const gap_sequence_t *seq, int threads) {
    detailed_stats_t stats;
    stats.trials = ds->trials;
    stats.comparisons = malloc(ds->trials * sizeof(uint64_t));
    stats.runtimes_us = malloc(ds->trials * sizeof(double));
    
    uint64_t N = ds->N;
    
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (uint64_t t = 0; t < ds->trials; t++) {
        int32_t *arr = malloc(N * sizeof(int32_t));
        memcpy(arr, &ds->data[t * N], N * sizeof(int32_t));
        
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        stats.comparisons[t] = shellsort(arr, N, seq);
        clock_gettime(CLOCK_MONOTONIC, &end);
        
        stats.runtimes_us[t] = (end.tv_sec - start.tv_sec) * 1e6 + 
                               (end.tv_nsec - start.tv_nsec) / 1e3;
        free(arr);
    }
    
    compute_stats(&stats);
    return stats;
}

/* Paired t-test for difference */
static void paired_test(detailed_stats_t *a, detailed_stats_t *b, 
                        double *mean_diff, double *t_stat, double *p_approx) {
    uint64_t n = a->trials < b->trials ? a->trials : b->trials;
    
    double *diffs = malloc(n * sizeof(double));
    double sum_diff = 0;
    for (uint64_t i = 0; i < n; i++) {
        diffs[i] = (double)a->comparisons[i] - (double)b->comparisons[i];
        sum_diff += diffs[i];
    }
    *mean_diff = sum_diff / n;
    
    double var = 0;
    for (uint64_t i = 0; i < n; i++) {
        double d = diffs[i] - *mean_diff;
        var += d * d;
    }
    double se = sqrt(var / (n - 1)) / sqrt(n);
    
    *t_stat = *mean_diff / se;
    
    /* Approximate p-value using normal approximation for large n */
    double z = fabs(*t_stat);
    *p_approx = 2.0 * (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0))));
    
    free(diffs);
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
    
    printf("================================================================================\n");
    printf("COMPREHENSIVE SHELLSORT GAP SEQUENCE BENCHMARK\n");
    printf("================================================================================\n\n");
    
    printf("System Configuration:\n");
    printf("  Threads: %d\n", threads);
    printf("  Permutations directory: %s\n\n", perms_dir);
    
    /* Generate sequences */
    gap_sequence_t ciura, evolved;
    gaps_ciura(&ciura, 10000000);
    gaps_evolved(&evolved, 10000000);
    
    printf("Ciura sequence (%zu gaps): [", ciura.num_gaps);
    for (size_t i = 0; i < ciura.num_gaps; i++) {
        printf("%lu%s", ciura.gaps[i], i < ciura.num_gaps-1 ? ", " : "");
    }
    printf("]\n\n");
    
    printf("Evolved sequence (%zu gaps): [", evolved.num_gaps);
    for (size_t i = 0; i < evolved.num_gaps; i++) {
        printf("%lu%s", evolved.gaps[i], i < evolved.num_gaps-1 ? ", " : "");
    }
    printf("]\n\n");
    
    double total_ciura = 0, total_evolved = 0;
    double total_ciura_var = 0, total_evolved_var = 0;
    
    for (int s = 0; s < num_sizes; s++) {
        uint64_t N = sizes[s];
        
        printf("================================================================================\n");
        printf("N = %lu\n", N);
        printf("================================================================================\n");
        
        perm_dataset_t ds;
        if (load_dataset(perms_dir, N, &ds) < 0) {
            printf("Failed to load dataset for N=%lu\n", N);
            continue;
        }
        
        printf("Trials: %lu\n\n", ds.trials);
        
        /* Benchmark both sequences */
        gap_sequence_t ciura_n, evolved_n;
        gap_sequence_copy(&ciura_n, &ciura);
        gap_sequence_copy(&evolved_n, &evolved);
        
        /* Trim to size */
        while (ciura_n.num_gaps > 0 && ciura_n.gaps[ciura_n.num_gaps-1] >= N)
            ciura_n.num_gaps--;
        while (evolved_n.num_gaps > 0 && evolved_n.gaps[evolved_n.num_gaps-1] >= N)
            evolved_n.num_gaps--;
        
        printf("Ciura gaps used: %zu, Evolved gaps used: %zu\n\n", 
               ciura_n.num_gaps, evolved_n.num_gaps);
        
        detailed_stats_t ciura_stats = benchmark_sequence(&ds, &ciura_n, threads);
        detailed_stats_t evolved_stats = benchmark_sequence(&ds, &evolved_n, threads);
        
        printf("COMPARISON COUNTS:\n");
        printf("%-10s %16s %16s %16s %16s\n", "Sequence", "Mean", "StdDev", "StdErr", "95% CI");
        printf("%-10s %16.2f %16.2f %16.2f [%.2f, %.2f]\n", 
               "Ciura", ciura_stats.mean_comps, ciura_stats.stddev_comps, 
               ciura_stats.stderr_comps, ciura_stats.ci95_low, ciura_stats.ci95_high);
        printf("%-10s %16.2f %16.2f %16.2f [%.2f, %.2f]\n", 
               "Evolved", evolved_stats.mean_comps, evolved_stats.stddev_comps,
               evolved_stats.stderr_comps, evolved_stats.ci95_low, evolved_stats.ci95_high);
        
        double improvement = (ciura_stats.mean_comps - evolved_stats.mean_comps) / 
                             ciura_stats.mean_comps * 100.0;
        printf("\nImprovement: %.4f%%\n", improvement);
        
        /* Paired t-test */
        double mean_diff, t_stat, p_val;
        paired_test(&ciura_stats, &evolved_stats, &mean_diff, &t_stat, &p_val);
        
        printf("\nPAIRED T-TEST (Ciura - Evolved):\n");
        printf("  Mean difference: %.2f comparisons\n", mean_diff);
        printf("  t-statistic: %.4f\n", t_stat);
        printf("  p-value (approx): %.2e\n", p_val);
        printf("  Significant at alpha=0.05: %s\n", p_val < 0.05 ? "YES" : "NO");
        printf("  Significant at alpha=0.01: %s\n", p_val < 0.01 ? "YES" : "NO");
        printf("  Significant at alpha=0.001: %s\n", p_val < 0.001 ? "YES" : "NO");
        
        printf("\nRUNTIME (microseconds):\n");
        printf("%-10s %16s %16s\n", "Sequence", "Mean", "StdDev");
        printf("%-10s %16.2f %16.2f\n", "Ciura", ciura_stats.mean_runtime, ciura_stats.stddev_runtime);
        printf("%-10s %16.2f %16.2f\n", "Evolved", evolved_stats.mean_runtime, evolved_stats.stddev_runtime);
        
        total_ciura += ciura_stats.mean_comps;
        total_evolved += evolved_stats.mean_comps;
        total_ciura_var += ciura_stats.stddev_comps * ciura_stats.stddev_comps;
        total_evolved_var += evolved_stats.stddev_comps * evolved_stats.stddev_comps;
        
        printf("\n");
        
        free(ciura_stats.comparisons);
        free(ciura_stats.runtimes_us);
        free(evolved_stats.comparisons);
        free(evolved_stats.runtimes_us);
        free(ds.data);
    }
    
    printf("================================================================================\n");
    printf("AGGREGATE RESULTS\n");
    printf("================================================================================\n");
    printf("Total Ciura:   %.2f\n", total_ciura);
    printf("Total Evolved: %.2f\n", total_evolved);
    printf("Improvement:   %.4f%%\n", (total_ciura - total_evolved) / total_ciura * 100.0);
    printf("================================================================================\n");
    
    return 0;
}
