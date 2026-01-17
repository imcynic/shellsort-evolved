#define _GNU_SOURCE
/*
 * search.c - Search for optimal Shellsort gap sequences
 *
 * Strategies:
 * 1. Grid search over ratio-based sequences
 * 2. Two-phase split-ratio sequences
 * 3. Local mutations of best-known sequences
 *
 * Uses same benchmark infrastructure as bench.c for consistency.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <errno.h>
#include <sys/stat.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../src/rng.h"
#include "../src/shellsort.h"
#include "../src/gaps_baselines.h"

#define MAGIC 0x5045524D47454E31ULL
#define MAX_SIZES 16
#define MAX_CANDIDATES 1024

typedef struct {
    char perms_dir[512];
    char out_dir[512];
    int threads;
    int verbose;
    uint64_t sizes[MAX_SIZES];
    double weights[MAX_SIZES];
    size_t num_sizes;
} config_t;

typedef struct {
    uint64_t N;
    uint64_t trials;
    uint64_t master_seed;
    int32_t *data;
} perm_dataset_t;

typedef struct {
    gap_sequence_t seq;
    double score;           /* Weighted mean comparisons (lower is better) */
    double scores_by_size[MAX_SIZES];
} candidate_t;

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s --perms <dir> --out <dir> [options]\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --perms <dir>     Directory containing permutation files\n");
    fprintf(stderr, "  --out <dir>       Output directory for results\n");
    fprintf(stderr, "  --threads N       Number of OpenMP threads\n");
    fprintf(stderr, "  --sizes <list>    Comma-separated training sizes (default: 1000,10000,100000)\n");
    fprintf(stderr, "  --weights <list>  Comma-separated weights per size (default: equal)\n");
    fprintf(stderr, "  --verbose         Print more details\n");
}

static int parse_double_list(const char *str, double *out, size_t max, size_t *count) {
    *count = 0;
    char *copy = strdup(str);
    if (!copy) return -1;

    char *tok = strtok(copy, ",");
    while (tok && *count < max) {
        out[*count] = atof(tok);
        (*count)++;
        tok = strtok(NULL, ",");
    }

    free(copy);
    return 0;
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
    cfg->threads = 0;

    /* Defaults */
    cfg->sizes[0] = 1000;
    cfg->sizes[1] = 10000;
    cfg->sizes[2] = 100000;
    cfg->num_sizes = 3;

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
        } else if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
            size_t count;
            if (parse_double_list(argv[++i], cfg->weights, MAX_SIZES, &count) < 0) {
                fprintf(stderr, "Error: Invalid weights list\n");
                return -1;
            }
        } else if (strcmp(argv[i], "--verbose") == 0) {
            cfg->verbose = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            exit(0);
        }
    }

    /* Default weights = equal */
    double weight_sum = 0;
    for (size_t i = 0; i < cfg->num_sizes; i++) {
        if (cfg->weights[i] == 0) cfg->weights[i] = 1.0;
        weight_sum += cfg->weights[i];
    }
    for (size_t i = 0; i < cfg->num_sizes; i++) {
        cfg->weights[i] /= weight_sum;
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

    uint64_t magic;
    if (fread(&magic, sizeof(magic), 1, f) != 1 || magic != MAGIC) {
        fclose(f);
        return -1;
    }

    fread(&ds->N, sizeof(ds->N), 1, f);
    fread(&ds->trials, sizeof(ds->trials), 1, f);
    fread(&ds->master_seed, sizeof(ds->master_seed), 1, f);

    size_t total = ds->trials * ds->N;
    ds->data = malloc(total * sizeof(int32_t));
    if (!ds->data) {
        fclose(f);
        return -1;
    }

    if (fread(ds->data, sizeof(int32_t), total, f) != total) {
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

/*
 * Evaluate a sequence on a dataset.
 * Returns mean comparisons per trial.
 */
static double evaluate_sequence(const perm_dataset_t *ds, const gap_sequence_t *seq,
                                int num_threads) {
    uint64_t N = ds->N;
    uint64_t trials = ds->trials;
    uint64_t total = 0;

    #pragma omp parallel for schedule(static) num_threads(num_threads) reduction(+:total)
    for (uint64_t t = 0; t < trials; t++) {
        int32_t *arr = malloc(N * sizeof(int32_t));
        if (!arr) continue;

        memcpy(arr, &ds->data[t * N], N * sizeof(int32_t));
        total += shellsort(arr, N, seq);
        free(arr);
    }

    return (double)total / (double)trials;
}

/*
 * Evaluate candidate across all training sizes.
 * Returns weighted score.
 */
static double evaluate_candidate(candidate_t *cand, perm_dataset_t *datasets,
                                 const config_t *cfg, int num_threads) {
    double score = 0;

    for (size_t i = 0; i < cfg->num_sizes; i++) {
        /* Generate sequence for this size */
        gap_sequence_t seq;
        gap_sequence_copy(&seq, &cand->seq);

        /* Trim gaps >= N */
        while (seq.num_gaps > 0 && seq.gaps[seq.num_gaps - 1] >= datasets[i].N) {
            seq.num_gaps--;
        }

        char reason[256];
        if (!gap_sequence_valid(&seq, reason, sizeof(reason))) {
            cand->scores_by_size[i] = DBL_MAX;
            return DBL_MAX;
        }

        double mean = evaluate_sequence(&datasets[i], &seq, num_threads);
        cand->scores_by_size[i] = mean;
        score += cfg->weights[i] * mean;
    }

    cand->score = score;
    return score;
}

/*
 * Grid search over single-ratio sequences.
 */
static void search_ratio_grid(candidate_t *best, perm_dataset_t *datasets,
                              const config_t *cfg, int num_threads) {
    printf("=== Ratio Grid Search ===\n");

    /* Search ratios from 2.0 to 3.0 in steps of 0.01 */
    double best_ratio = 0;
    double best_score = DBL_MAX;

    for (double r = 2.0; r <= 3.0; r += 0.01) {
        candidate_t cand;
        uint64_t max_N = cfg->sizes[cfg->num_sizes - 1];
        gaps_ratio(&cand.seq, r, max_N, NULL);

        double score = evaluate_candidate(&cand, datasets, cfg, num_threads);

        if (cfg->verbose) {
            printf("  ratio=%.3f  score=%.2f\n", r, score);
        }

        if (score < best_score) {
            best_score = score;
            best_ratio = r;
            gap_sequence_copy(&best->seq, &cand.seq);
            best->score = score;
            memcpy(best->scores_by_size, cand.scores_by_size, sizeof(best->scores_by_size));
        }
    }

    printf("Best ratio: %.3f (score=%.2f)\n\n", best_ratio, best_score);

    /* Fine-tune around best */
    printf("Fine-tuning...\n");
    for (double r = best_ratio - 0.05; r <= best_ratio + 0.05; r += 0.001) {
        candidate_t cand;
        uint64_t max_N = cfg->sizes[cfg->num_sizes - 1];
        gaps_ratio(&cand.seq, r, max_N, NULL);

        double score = evaluate_candidate(&cand, datasets, cfg, num_threads);

        if (score < best_score) {
            best_score = score;
            best_ratio = r;
            gap_sequence_copy(&best->seq, &cand.seq);
            best->score = score;
            memcpy(best->scores_by_size, cand.scores_by_size, sizeof(best->scores_by_size));
        }
    }

    snprintf(best->seq.name, sizeof(best->seq.name), "Ratio-%.6f", best_ratio);
    printf("Fine-tuned ratio: %.6f (score=%.2f)\n\n", best_ratio, best_score);
}

/*
 * Search split-ratio sequences (two phases).
 */
static void search_split_ratio(candidate_t *best, perm_dataset_t *datasets,
                               const config_t *cfg, int num_threads) {
    printf("=== Split-Ratio Search ===\n");

    double best_r1 = 0, best_r2 = 0;
    uint64_t best_thresh = 0;
    double best_score = DBL_MAX;

    uint64_t max_N = cfg->sizes[cfg->num_sizes - 1];

    /* Coarse search */
    for (double r1 = 2.0; r1 <= 2.8; r1 += 0.1) {
        for (double r2 = 2.0; r2 <= 2.8; r2 += 0.1) {
            for (uint64_t thresh = 10; thresh <= 1000; thresh *= 10) {
                candidate_t cand;
                gaps_split_ratio(&cand.seq, r1, r2, thresh, max_N, NULL);

                double score = evaluate_candidate(&cand, datasets, cfg, num_threads);

                if (score < best_score) {
                    best_score = score;
                    best_r1 = r1;
                    best_r2 = r2;
                    best_thresh = thresh;
                    gap_sequence_copy(&best->seq, &cand.seq);
                    best->score = score;
                    memcpy(best->scores_by_size, cand.scores_by_size, sizeof(best->scores_by_size));
                }
            }
        }
    }

    printf("Coarse: r1=%.2f r2=%.2f thresh=%lu (score=%.2f)\n",
           best_r1, best_r2, (unsigned long)best_thresh, best_score);

    /* Fine search around best */
    for (double r1 = best_r1 - 0.1; r1 <= best_r1 + 0.1; r1 += 0.02) {
        for (double r2 = best_r2 - 0.1; r2 <= best_r2 + 0.1; r2 += 0.02) {
            candidate_t cand;
            gaps_split_ratio(&cand.seq, r1, r2, best_thresh, max_N, NULL);

            double score = evaluate_candidate(&cand, datasets, cfg, num_threads);

            if (score < best_score) {
                best_score = score;
                best_r1 = r1;
                best_r2 = r2;
                gap_sequence_copy(&best->seq, &cand.seq);
                best->score = score;
                memcpy(best->scores_by_size, cand.scores_by_size, sizeof(best->scores_by_size));
            }
        }
    }

    snprintf(best->seq.name, sizeof(best->seq.name),
             "Split-%.4f-%.4f@%lu", best_r1, best_r2, (unsigned long)best_thresh);
    printf("Fine-tuned: r1=%.4f r2=%.4f thresh=%lu (score=%.2f)\n\n",
           best_r1, best_r2, (unsigned long)best_thresh, best_score);
}

int main(int argc, char **argv) {
    config_t cfg;

    if (parse_args(argc, argv, &cfg) < 0) {
        print_usage(argv[0]);
        return 1;
    }

    int num_threads = cfg.threads;
#ifdef _OPENMP
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    omp_set_num_threads(num_threads);
#else
    num_threads = 1;
#endif

    mkdir(cfg.out_dir, 0755);

    printf("Shellsort Gap Sequence Search\n");
    printf("==============================\n");
    printf("Threads: %d\n", num_threads);
    printf("Training sizes: ");
    for (size_t i = 0; i < cfg.num_sizes; i++) {
        printf("%lu (w=%.2f)", (unsigned long)cfg.sizes[i], cfg.weights[i]);
        if (i < cfg.num_sizes - 1) printf(", ");
    }
    printf("\n\n");

    /* Load datasets */
    perm_dataset_t *datasets = malloc(cfg.num_sizes * sizeof(perm_dataset_t));
    if (!datasets) {
        fprintf(stderr, "Error: Failed to allocate datasets array\n");
        return 1;
    }

    for (size_t i = 0; i < cfg.num_sizes; i++) {
        printf("Loading N=%lu...\n", (unsigned long)cfg.sizes[i]);
        if (load_dataset(cfg.perms_dir, cfg.sizes[i], &datasets[i]) < 0) {
            fprintf(stderr, "Failed to load dataset for N=%lu\n", (unsigned long)cfg.sizes[i]);
            return 1;
        }
        printf("  Loaded %lu trials\n", (unsigned long)datasets[i].trials);
    }
    printf("\n");

    /* Evaluate baselines first */
    printf("=== Baseline Evaluation ===\n");
    uint64_t max_N = cfg.sizes[cfg.num_sizes - 1];
    gap_sequence_t baselines[NUM_BASELINES];
    gaps_all_baselines(baselines, max_N);

    candidate_t baseline_results[NUM_BASELINES];
    double best_baseline_score = DBL_MAX;
    int best_baseline_idx = -1;

    for (int i = 0; i < NUM_BASELINES; i++) {
        baseline_results[i].seq = baselines[i];
        double score = evaluate_candidate(&baseline_results[i], datasets, &cfg, num_threads);

        printf("  %-16s: score=%.2f  [", baselines[i].name, score);
        for (size_t j = 0; j < cfg.num_sizes; j++) {
            printf("%.1f", baseline_results[i].scores_by_size[j]);
            if (j < cfg.num_sizes - 1) printf(", ");
        }
        printf("]\n");

        if (score < best_baseline_score) {
            best_baseline_score = score;
            best_baseline_idx = i;
        }
    }
    printf("\nBest baseline: %s (score=%.2f)\n\n", baselines[best_baseline_idx].name, best_baseline_score);

    /* Search for better sequences */
    candidate_t best_ratio, best_split;
    memset(&best_ratio, 0, sizeof(best_ratio));
    memset(&best_split, 0, sizeof(best_split));
    best_ratio.score = DBL_MAX;
    best_split.score = DBL_MAX;

    search_ratio_grid(&best_ratio, datasets, &cfg, num_threads);
    search_split_ratio(&best_split, datasets, &cfg, num_threads);

    /* Summary */
    printf("=== Summary ===\n");
    printf("Best baseline:    %s (score=%.2f)\n", baselines[best_baseline_idx].name, best_baseline_score);
    printf("Best ratio:       %s (score=%.2f)\n", best_ratio.seq.name, best_ratio.score);
    printf("Best split-ratio: %s (score=%.2f)\n", best_split.seq.name, best_split.score);

    /* Determine overall winner */
    candidate_t *winner = &baseline_results[best_baseline_idx];
    if (best_ratio.score < winner->score) winner = &best_ratio;
    if (best_split.score < winner->score) winner = &best_split;

    printf("\n*** WINNER: %s ***\n", winner->seq.name);
    printf("Score: %.2f\n", winner->score);
    printf("Gaps: ");
    gap_sequence_print(&winner->seq);

    double improvement = (best_baseline_score - winner->score) / best_baseline_score * 100.0;
    if (improvement > 0) {
        printf("Improvement over best baseline: %.2f%%\n", improvement);
    } else {
        printf("No improvement over best baseline.\n");
    }

    /* Save results */
    time_t now = time(NULL);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", localtime(&now));

    char results_path[1024];
    snprintf(results_path, sizeof(results_path), "%s/search_%s.txt", cfg.out_dir, timestamp);
    FILE *f = fopen(results_path, "w");
    if (f) {
        fprintf(f, "Shellsort Gap Sequence Search Results\n");
        fprintf(f, "======================================\n\n");

        fprintf(f, "Training sizes: ");
        for (size_t i = 0; i < cfg.num_sizes; i++) {
            fprintf(f, "%lu", (unsigned long)cfg.sizes[i]);
            if (i < cfg.num_sizes - 1) fprintf(f, ", ");
        }
        fprintf(f, "\n\n");

        fprintf(f, "Baselines:\n");
        for (int i = 0; i < NUM_BASELINES; i++) {
            fprintf(f, "  %-16s: score=%.2f\n", baselines[i].name, baseline_results[i].score);
        }

        fprintf(f, "\nBest candidates:\n");
        fprintf(f, "  Ratio:       %s (score=%.2f)\n", best_ratio.seq.name, best_ratio.score);
        fprintf(f, "  Split-ratio: %s (score=%.2f)\n", best_split.seq.name, best_split.score);

        fprintf(f, "\nWINNER: %s\n", winner->seq.name);
        fprintf(f, "Gaps: [");
        for (size_t i = 0; i < winner->seq.num_gaps; i++) {
            fprintf(f, "%lu", (unsigned long)winner->seq.gaps[i]);
            if (i < winner->seq.num_gaps - 1) fprintf(f, ", ");
        }
        fprintf(f, "]\n");

        fclose(f);
        printf("\nResults saved to %s\n", results_path);
    }

    /* Cleanup */
    for (size_t i = 0; i < cfg.num_sizes; i++) {
        free_dataset(&datasets[i]);
    }
    free(datasets);

    return 0;
}
