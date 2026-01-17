#define _GNU_SOURCE
/*
 * bench.c - Benchmark harness for Shellsort gap sequences
 *
 * Runs all sequences against pre-generated permutation datasets.
 * Uses OpenMP for parallelization over trials.
 *
 * Usage: ./bench --perms <dir> --out <dir> [--threads N]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "rng.h"
#include "shellsort.h"
#include "gaps_baselines.h"

#define MAGIC 0x5045524D47454E31ULL
#define MAX_SIZES 32
#define MAX_SEQUENCES 64

typedef struct {
    char perms_dir[512];
    char out_dir[512];
    int threads;
    uint64_t sizes[MAX_SIZES];
    size_t num_sizes;
} config_t;

typedef struct {
    uint64_t N;
    uint64_t trials;
    uint64_t master_seed;
    int32_t *data;  /* trials * N elements */
} perm_dataset_t;

typedef struct {
    char sequence_name[64];
    uint64_t N;
    uint64_t trials;
    uint64_t total_comparisons;
    double mean_comparisons;
    double stddev;
    double stderr_val;
    double min_comparisons;
    double max_comparisons;
    /* Move statistics */
    uint64_t total_moves;
    double mean_moves;
    double moves_stddev;
    /* Runtime statistics (per-trial) */
    double mean_runtime_us;    /* microseconds per sort */
    double runtime_stddev_us;
    double runtime_stderr_us;
} bench_result_t;

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s --perms <dir> --out <dir> [--threads N] [--sizes n1,n2,...]\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --perms <dir>     Directory containing permutation files\n");
    fprintf(stderr, "  --out <dir>       Output directory for results\n");
    fprintf(stderr, "  --threads N       Number of OpenMP threads (default: all)\n");
    fprintf(stderr, "  --sizes <list>    Comma-separated list of N values to benchmark\n");
    fprintf(stderr, "                    (default: auto-detect from perms dir)\n");
}

static int parse_uint64_list(const char *str, uint64_t *out, size_t max, size_t *count) {
    *count = 0;
    char *copy = strdup(str);
    if (!copy) return -1;

    char *tok = strtok(copy, ",");
    while (tok && *count < max) {
        out[*count] = strtoull(tok, NULL, 0);
        (*count)++;
        tok = strtok(NULL, ",");
    }

    free(copy);
    return 0;
}

static int parse_args(int argc, char **argv, config_t *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->threads = 0;  /* 0 = use all available */

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--perms") == 0 && i + 1 < argc) {
            strncpy(cfg->perms_dir, argv[++i], sizeof(cfg->perms_dir) - 1);
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            strncpy(cfg->out_dir, argv[++i], sizeof(cfg->out_dir) - 1);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            cfg->threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--sizes") == 0 && i + 1 < argc) {
            if (parse_uint64_list(argv[++i], cfg->sizes, MAX_SIZES, &cfg->num_sizes) < 0) {
                fprintf(stderr, "Error: Invalid sizes list\n");
                return -1;
            }
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return -1;
        }
    }

    if (cfg->perms_dir[0] == '\0' || cfg->out_dir[0] == '\0') {
        fprintf(stderr, "Error: --perms and --out are required\n");
        return -1;
    }

    return 0;
}

static int load_dataset(const char *perms_dir, uint64_t N, perm_dataset_t *ds) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/perm_%lu.bin", perms_dir, (unsigned long)N);

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }

    /* Read header */
    uint64_t magic;
    if (fread(&magic, sizeof(magic), 1, f) != 1 || magic != MAGIC) {
        fprintf(stderr, "Error: Invalid magic in %s\n", path);
        fclose(f);
        return -1;
    }

    if (fread(&ds->N, sizeof(ds->N), 1, f) != 1 ||
        fread(&ds->trials, sizeof(ds->trials), 1, f) != 1 ||
        fread(&ds->master_seed, sizeof(ds->master_seed), 1, f) != 1) {
        fprintf(stderr, "Error: Failed to read header from %s\n", path);
        fclose(f);
        return -1;
    }

    if (ds->N != N) {
        fprintf(stderr, "Error: N mismatch in %s (expected %lu, got %lu)\n",
                path, (unsigned long)N, (unsigned long)ds->N);
        fclose(f);
        return -1;
    }

    /* Allocate and read data */
    size_t total = ds->trials * ds->N;
    ds->data = malloc(total * sizeof(int32_t));
    if (!ds->data) {
        fprintf(stderr, "Error: Failed to allocate %zu bytes for dataset\n",
                total * sizeof(int32_t));
        fclose(f);
        return -1;
    }

    if (fread(ds->data, sizeof(int32_t), total, f) != total) {
        fprintf(stderr, "Error: Failed to read data from %s\n", path);
        free(ds->data);
        fclose(f);
        return -1;
    }

    fclose(f);
    return 0;
}

static void free_dataset(perm_dataset_t *ds) {
    free(ds->data);
    ds->data = NULL;
}

static void benchmark_sequence(const perm_dataset_t *ds, const gap_sequence_t *seq,
                               bench_result_t *result, int num_threads) {
    uint64_t N = ds->N;
    uint64_t trials = ds->trials;

    strncpy(result->sequence_name, seq->name, sizeof(result->sequence_name) - 1);
    result->N = N;
    result->trials = trials;

    /* Allocate per-trial statistics */
    uint64_t *comp_counts = malloc(trials * sizeof(uint64_t));
    uint64_t *move_counts = malloc(trials * sizeof(uint64_t));
    double *runtimes_us = malloc(trials * sizeof(double));
    if (!comp_counts || !move_counts || !runtimes_us) {
        fprintf(stderr, "Error: Failed to allocate arrays\n");
        free(comp_counts);
        free(move_counts);
        free(runtimes_us);
        return;
    }

    /* Parallel loop over trials with per-trial timing */
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (uint64_t t = 0; t < trials; t++) {
        /* Copy permutation to local array */
        int32_t *arr = malloc(N * sizeof(int32_t));
        if (!arr) continue;

        memcpy(arr, &ds->data[t * N], N * sizeof(int32_t));

#ifdef _OPENMP
        double t_start = omp_get_wtime();
#else
        clock_t t_start = clock();
#endif

        /* Sort and collect stats */
        sort_stats_t stats = shellsort_stats(arr, N, seq);

#ifdef _OPENMP
        runtimes_us[t] = (omp_get_wtime() - t_start) * 1e6;
#else
        runtimes_us[t] = (double)(clock() - t_start) / CLOCKS_PER_SEC * 1e6;
#endif

        comp_counts[t] = stats.comparisons;
        move_counts[t] = stats.moves;

        free(arr);
    }

    /* Compute comparison statistics */
    result->total_comparisons = 0;
    result->min_comparisons = (double)comp_counts[0];
    result->max_comparisons = (double)comp_counts[0];

    for (uint64_t t = 0; t < trials; t++) {
        result->total_comparisons += comp_counts[t];
        if ((double)comp_counts[t] < result->min_comparisons) result->min_comparisons = (double)comp_counts[t];
        if ((double)comp_counts[t] > result->max_comparisons) result->max_comparisons = (double)comp_counts[t];
    }

    result->mean_comparisons = (double)result->total_comparisons / (double)trials;

    /* Compute comparison stddev */
    double sum_sq = 0;
    for (uint64_t t = 0; t < trials; t++) {
        double diff = (double)comp_counts[t] - result->mean_comparisons;
        sum_sq += diff * diff;
    }
    result->stddev = sqrt(sum_sq / (double)trials);
    result->stderr_val = result->stddev / sqrt((double)trials);

    /* Compute move statistics */
    result->total_moves = 0;
    for (uint64_t t = 0; t < trials; t++) {
        result->total_moves += move_counts[t];
    }
    result->mean_moves = (double)result->total_moves / (double)trials;

    /* Compute move stddev */
    sum_sq = 0;
    for (uint64_t t = 0; t < trials; t++) {
        double diff = (double)move_counts[t] - result->mean_moves;
        sum_sq += diff * diff;
    }
    result->moves_stddev = sqrt(sum_sq / (double)trials);

    /* Compute runtime statistics */
    double runtime_sum = 0;
    for (uint64_t t = 0; t < trials; t++) {
        runtime_sum += runtimes_us[t];
    }
    result->mean_runtime_us = runtime_sum / (double)trials;

    sum_sq = 0;
    for (uint64_t t = 0; t < trials; t++) {
        double diff = runtimes_us[t] - result->mean_runtime_us;
        sum_sq += diff * diff;
    }
    result->runtime_stddev_us = sqrt(sum_sq / (double)trials);
    result->runtime_stderr_us = result->runtime_stddev_us / sqrt((double)trials);

    free(comp_counts);
    free(move_counts);
    free(runtimes_us);
}

static void get_system_info(char *buf, size_t len) {
    struct utsname uts;
    if (uname(&uts) == 0) {
        snprintf(buf, len, "%s %s %s", uts.sysname, uts.release, uts.machine);
    } else {
        strncpy(buf, "unknown", len - 1);
    }
}

static void get_cpu_info(char *buf, size_t len) {
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) {
        strncpy(buf, "unknown", len - 1);
        return;
    }

    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "model name", 10) == 0) {
            char *colon = strchr(line, ':');
            if (colon) {
                colon++;
                while (*colon == ' ' || *colon == '\t') colon++;
                char *nl = strchr(colon, '\n');
                if (nl) *nl = '\0';
                strncpy(buf, colon, len - 1);
                fclose(f);
                return;
            }
        }
    }

    fclose(f);
    strncpy(buf, "unknown", len - 1);
}

int main(int argc, char **argv) {
    config_t cfg;

    if (parse_args(argc, argv, &cfg) < 0) {
        print_usage(argv[0]);
        return 1;
    }

    /* Set up threads */
    int num_threads = cfg.threads;
#ifdef _OPENMP
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    omp_set_num_threads(num_threads);
#else
    num_threads = 1;
#endif

    /* Auto-detect sizes if not specified */
    if (cfg.num_sizes == 0) {
        /* Check for common sizes */
        uint64_t common_sizes[] = {1000, 2000, 10000, 20000, 100000, 200000, 1000000, 2000000};
        for (size_t i = 0; i < sizeof(common_sizes) / sizeof(common_sizes[0]); i++) {
            char path[1024];
            snprintf(path, sizeof(path), "%s/perm_%lu.bin", cfg.perms_dir, (unsigned long)common_sizes[i]);
            if (access(path, F_OK) == 0) {
                cfg.sizes[cfg.num_sizes++] = common_sizes[i];
            }
        }
        if (cfg.num_sizes == 0) {
            fprintf(stderr, "Error: No permutation files found in %s\n", cfg.perms_dir);
            return 1;
        }
    }

    /* Create output directory */
    mkdir(cfg.out_dir, 0755);

    /* Get system info */
    char sys_info[256], cpu_info[256];
    get_system_info(sys_info, sizeof(sys_info));
    get_cpu_info(cpu_info, sizeof(cpu_info));

    /* Get timestamp */
    time_t now = time(NULL);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", localtime(&now));

    /* Open output CSV */
    char csv_path[1024];
    snprintf(csv_path, sizeof(csv_path), "%s/bench_%s.csv", cfg.out_dir, timestamp);
    FILE *csv = fopen(csv_path, "w");
    if (!csv) {
        fprintf(stderr, "Error: Cannot open %s: %s\n", csv_path, strerror(errno));
        return 1;
    }

    /* Write CSV header */
    fprintf(csv, "sequence_name,N,trials,mean_comparisons,comp_stddev,comp_stderr,"
            "mean_moves,moves_stddev,mean_runtime_us,runtime_stddev_us,runtime_stderr_us,"
            "cpu,os,compiler,threads,timestamp\n");

    printf("Shellsort Benchmark\n");
    printf("===================\n");
    printf("System: %s\n", sys_info);
    printf("CPU: %s\n", cpu_info);
    printf("Compiler: %s\n", __VERSION__);
    printf("Threads: %d\n", num_threads);
    printf("Perms dir: %s\n", cfg.perms_dir);
    printf("Output: %s\n", csv_path);
    printf("Sizes: ");
    for (size_t i = 0; i < cfg.num_sizes; i++) {
        printf("%lu", (unsigned long)cfg.sizes[i]);
        if (i < cfg.num_sizes - 1) printf(", ");
    }
    printf("\n\n");

    /* Generate baseline sequences (use max size to determine max gap) */
    uint64_t max_N = 0;
    for (size_t i = 0; i < cfg.num_sizes; i++) {
        if (cfg.sizes[i] > max_N) max_N = cfg.sizes[i];
    }

    gap_sequence_t baselines[NUM_BASELINES];
    gaps_all_baselines(baselines, max_N);

    printf("Baseline sequences:\n");
    for (int i = 0; i < NUM_BASELINES; i++) {
        printf("  ");
        gap_sequence_print(&baselines[i]);
    }
    printf("\n");

    /* Benchmark each size */
    for (size_t si = 0; si < cfg.num_sizes; si++) {
        uint64_t N = cfg.sizes[si];
        printf("=== N = %lu ===\n", (unsigned long)N);

        /* Load dataset */
        perm_dataset_t ds;
        if (load_dataset(cfg.perms_dir, N, &ds) < 0) {
            continue;
        }
        printf("Loaded %lu trials\n", (unsigned long)ds.trials);

        /* Generate sequences for this N */
        gap_sequence_t seqs[NUM_BASELINES];
        gaps_all_baselines(seqs, N);

        /* Benchmark each sequence */
        for (int i = 0; i < NUM_BASELINES; i++) {
            char reason[256];
            if (!gap_sequence_valid(&seqs[i], reason, sizeof(reason))) {
                printf("  [SKIP] %s: %s\n", seqs[i].name, reason);
                continue;
            }

            bench_result_t result;
            benchmark_sequence(&ds, &seqs[i], &result, num_threads);

            printf("  %-16s: comps=%.0f (±%.0f)  moves=%.0f  runtime=%.0f±%.0fμs\n",
                   result.sequence_name,
                   result.mean_comparisons,
                   result.stderr_val,
                   result.mean_moves,
                   result.mean_runtime_us,
                   result.runtime_stderr_us);

            /* Write to CSV */
            fprintf(csv, "%s,%lu,%lu,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,"
                    "\"%s\",\"%s\",\"%s\",%d,%s\n",
                    result.sequence_name,
                    (unsigned long)result.N,
                    (unsigned long)result.trials,
                    result.mean_comparisons,
                    result.stddev,
                    result.stderr_val,
                    result.mean_moves,
                    result.moves_stddev,
                    result.mean_runtime_us,
                    result.runtime_stddev_us,
                    result.runtime_stderr_us,
                    cpu_info,
                    sys_info,
                    __VERSION__,
                    num_threads,
                    timestamp);
        }

        free_dataset(&ds);
        printf("\n");
    }

    fclose(csv);
    printf("Results written to %s\n", csv_path);

    return 0;
}
