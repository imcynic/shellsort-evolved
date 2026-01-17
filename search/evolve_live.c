#define _GNU_SOURCE
/*
 * evolve_live.c - Evolutionary search with live status output
 *
 * Writes status to a file every generation for live monitoring.
 * Use: watch -n1 cat results/status.txt
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
#define MAX_GAPS_DISPLAY 20
#define EARLY_TERM_THRESHOLD 0.02   /* 2% worse = skip */
#define EARLY_TERM_FRACTION 0.25    /* Check after 25% of trials */

typedef struct {
    char perms_dir[512];
    char out_dir[512];
    char status_file[512];
    int threads;
    int generations;
    int pop_size;
    int elite_count;
    double mutation_rate;
    uint64_t sizes[MAX_SIZES];
    double weights[MAX_SIZES];
    uint64_t search_trials[MAX_SIZES];  /* Reduced trials for search */
    size_t num_sizes;
    uint64_t rng_seed;
    /* Continuation support */
    uint64_t continue_gaps[MAX_GAPS];
    size_t continue_num_gaps;
} config_t;

typedef struct {
    uint64_t N;
    uint64_t trials;
    int32_t *data;
} perm_dataset_t;

typedef struct {
    gap_sequence_t seq;
    double fitness;
    double scores_by_size[MAX_SIZES];
    int evaluated;
} individual_t;

static rng_state_t global_rng;
static config_t *g_cfg;
static double g_baseline_score;
static time_t g_start_time;
static double g_current_best_fitness;  /* For early termination */

static void write_status(int gen, individual_t *best, individual_t *pop, int pop_size, int plateau) {
    FILE *f = fopen(g_cfg->status_file, "w");
    if (!f) return;

    time_t now = time(NULL);
    int elapsed = (int)(now - g_start_time);

    fprintf(f, "╔══════════════════════════════════════════════════════════════╗\n");
    fprintf(f, "║           SHELLSORT GAP SEQUENCE EVOLUTION                   ║\n");
    fprintf(f, "╠══════════════════════════════════════════════════════════════╣\n");
    fprintf(f, "║ Generation: %4d / %4d    Elapsed: %02d:%02d:%02d                 ║\n",
            gen, g_cfg->generations, elapsed/3600, (elapsed%3600)/60, elapsed%60);
    fprintf(f, "║ Population: %4d          Mutation: %.0f%%    Plateau: %3d/50  ║\n",
            g_cfg->pop_size, g_cfg->mutation_rate * 100, plateau);
    fprintf(f, "╠══════════════════════════════════════════════════════════════╣\n");

    double improvement = (g_baseline_score - best->fitness) / g_baseline_score * 100.0;
    fprintf(f, "║ BEST FITNESS: %14.2f                                 ║\n", best->fitness);
    fprintf(f, "║ BASELINE:     %14.2f (Ciura)                        ║\n", g_baseline_score);
    fprintf(f, "║ IMPROVEMENT:  %+13.4f%%                                  ║\n", improvement);
    fprintf(f, "╠══════════════════════════════════════════════════════════════╣\n");
    fprintf(f, "║ BEST SEQUENCE:                                               ║\n");
    fprintf(f, "║ [");

    int col = 3;
    for (size_t i = 0; i < best->seq.num_gaps && i < MAX_GAPS_DISPLAY; i++) {
        char buf[20];
        int len = snprintf(buf, sizeof(buf), "%lu", (unsigned long)best->seq.gaps[i]);
        if (col + len + 2 > 62) {
            fprintf(f, "\n║  ");
            col = 3;
        }
        fprintf(f, "%s", buf);
        col += len;
        if (i < best->seq.num_gaps - 1 && i < MAX_GAPS_DISPLAY - 1) {
            fprintf(f, ", ");
            col += 2;
        }
    }
    if (best->seq.num_gaps > MAX_GAPS_DISPLAY) fprintf(f, "...");
    fprintf(f, "]\n");

    fprintf(f, "╠══════════════════════════════════════════════════════════════╣\n");
    fprintf(f, "║ SCORES BY SIZE:                                              ║\n");
    for (size_t i = 0; i < g_cfg->num_sizes; i++) {
        fprintf(f, "║   N=%9lu: %14.2f                                ║\n",
                (unsigned long)g_cfg->sizes[i], best->scores_by_size[i]);
    }
    fprintf(f, "╠══════════════════════════════════════════════════════════════╣\n");

    /* Population diversity - show fitness distribution */
    double min_fit = pop[0].fitness;
    double max_fit = pop[pop_size-1].fitness;
    double median_fit = pop[pop_size/2].fitness;
    fprintf(f, "║ POPULATION: best=%.0f  median=%.0f  worst=%.0f    ║\n",
            min_fit, median_fit, max_fit);
    fprintf(f, "╚══════════════════════════════════════════════════════════════╝\n");

    fflush(f);
    fclose(f);
}

/* Rest of the code is similar to evolve.c but with status updates */

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

static void free_dataset(perm_dataset_t *ds) {
    free(ds->data);
    ds->data = NULL;
}

/*
 * Evaluate with early termination and reduced trials.
 * size_idx: which size we're evaluating (to get search_trials limit)
 * early_term_target: if running avg exceeds this by >2% after 25% trials, abort
 * Returns DBL_MAX if early terminated (clearly bad candidate)
 */
static double evaluate_sequence_fast(const perm_dataset_t *ds, const gap_sequence_t *seq,
                                     uint64_t max_trials, double early_term_target) {
    uint64_t N = ds->N;
    uint64_t trials = (max_trials > 0 && max_trials < ds->trials) ? max_trials : ds->trials;
    uint64_t check_point = (uint64_t)(trials * EARLY_TERM_FRACTION);
    if (check_point < 4) check_point = 4;

    uint64_t total = 0;

    /* Run trials sequentially for early termination check */
    for (uint64_t t = 0; t < trials; t++) {
        int32_t *arr = malloc(N * sizeof(int32_t));
        if (!arr) continue;
        memcpy(arr, &ds->data[t * N], N * sizeof(int32_t));
        total += shellsort(arr, N, seq);
        free(arr);

        /* Early termination check at 25% */
        if (t + 1 == check_point && early_term_target > 0) {
            double running_avg = (double)total / (double)(t + 1);
            double threshold = early_term_target * (1.0 + EARLY_TERM_THRESHOLD);
            if (running_avg > threshold) {
                return DBL_MAX;  /* Clearly worse, skip rest */
            }
        }
    }
    return (double)total / (double)trials;
}

/* Full evaluation without early termination (for validation) */
static double evaluate_sequence_full(const perm_dataset_t *ds, const gap_sequence_t *seq, int threads) {
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

/*
 * Fast evaluation with reduced trials and early termination.
 * best_scores_by_size: current best scores per size (for early termination target)
 */
static double evaluate_individual_fast(individual_t *ind, perm_dataset_t *datasets,
                                       const config_t *cfg, double *best_scores_by_size) {
    if (ind->evaluated) return ind->fitness;

    double score = 0;
    for (size_t i = 0; i < cfg->num_sizes; i++) {
        gap_sequence_t seq;
        gap_sequence_copy(&seq, &ind->seq);

        while (seq.num_gaps > 0 && seq.gaps[seq.num_gaps - 1] >= datasets[i].N) {
            seq.num_gaps--;
        }

        char reason[256];
        if (!gap_sequence_valid(&seq, reason, sizeof(reason))) {
            ind->fitness = DBL_MAX;
            ind->evaluated = 1;
            return DBL_MAX;
        }

        /* Use reduced trials and early termination */
        double target = (best_scores_by_size) ? best_scores_by_size[i] : 0;
        double mean = evaluate_sequence_fast(&datasets[i], &seq, cfg->search_trials[i], target);

        if (mean == DBL_MAX) {
            /* Early terminated - clearly bad */
            ind->fitness = DBL_MAX;
            ind->evaluated = 1;
            return DBL_MAX;
        }

        ind->scores_by_size[i] = mean;
        score += cfg->weights[i] * mean;
    }

    ind->fitness = score;
    ind->evaluated = 1;
    return score;
}

/* Full evaluation for baseline and final validation */
static double evaluate_individual_full(individual_t *ind, perm_dataset_t *datasets,
                                       const config_t *cfg, int threads) {
    if (ind->evaluated) return ind->fitness;

    double score = 0;
    for (size_t i = 0; i < cfg->num_sizes; i++) {
        gap_sequence_t seq;
        gap_sequence_copy(&seq, &ind->seq);

        while (seq.num_gaps > 0 && seq.gaps[seq.num_gaps - 1] >= datasets[i].N) {
            seq.num_gaps--;
        }

        char reason[256];
        if (!gap_sequence_valid(&seq, reason, sizeof(reason))) {
            ind->fitness = DBL_MAX;
            ind->evaluated = 1;
            return DBL_MAX;
        }

        double mean = evaluate_sequence_full(&datasets[i], &seq, threads);
        ind->scores_by_size[i] = mean;
        score += cfg->weights[i] * mean;
    }

    ind->fitness = score;
    ind->evaluated = 1;
    return score;
}

static void repair_sequence(gap_sequence_t *seq) {
    if (seq->num_gaps == 0) {
        seq->gaps[0] = 1;
        seq->num_gaps = 1;
        return;
    }

    for (size_t i = 0; i < seq->num_gaps; i++) {
        for (size_t j = i + 1; j < seq->num_gaps; j++) {
            if (seq->gaps[i] > seq->gaps[j]) {
                uint64_t tmp = seq->gaps[i];
                seq->gaps[i] = seq->gaps[j];
                seq->gaps[j] = tmp;
            }
        }
    }

    if (seq->gaps[0] != 1) {
        if (seq->num_gaps < MAX_GAPS) {
            memmove(&seq->gaps[1], &seq->gaps[0], seq->num_gaps * sizeof(uint64_t));
            seq->gaps[0] = 1;
            seq->num_gaps++;
        } else {
            seq->gaps[0] = 1;
        }
    }

    size_t write = 1;
    for (size_t read = 1; read < seq->num_gaps; read++) {
        if (seq->gaps[read] > seq->gaps[write - 1]) {
            seq->gaps[write++] = seq->gaps[read];
        }
    }
    seq->num_gaps = write;
}

static void mutate_insert(gap_sequence_t *seq) {
    if (seq->num_gaps >= MAX_GAPS - 1) return;
    size_t pos = rng_uniform(&global_rng, seq->num_gaps - 1) + 1;
    uint64_t lo = seq->gaps[pos - 1];
    uint64_t hi = seq->gaps[pos];
    if (hi - lo <= 1) return;
    uint64_t new_gap = lo + 1 + rng_uniform(&global_rng, hi - lo - 1);
    memmove(&seq->gaps[pos + 1], &seq->gaps[pos], (seq->num_gaps - pos) * sizeof(uint64_t));
    seq->gaps[pos] = new_gap;
    seq->num_gaps++;
}

static void mutate_delete(gap_sequence_t *seq) {
    if (seq->num_gaps <= 2) return;
    size_t pos = 1 + rng_uniform(&global_rng, seq->num_gaps - 1);
    memmove(&seq->gaps[pos], &seq->gaps[pos + 1], (seq->num_gaps - pos - 1) * sizeof(uint64_t));
    seq->num_gaps--;
}

static void mutate_modify(gap_sequence_t *seq) {
    if (seq->num_gaps <= 1) return;
    size_t pos = 1 + rng_uniform(&global_rng, seq->num_gaps - 1);
    uint64_t lo = seq->gaps[pos - 1];
    uint64_t hi = (pos + 1 < seq->num_gaps) ? seq->gaps[pos + 1] : seq->gaps[pos] * 3;
    if (hi - lo <= 2) return;
    seq->gaps[pos] = lo + 1 + rng_uniform(&global_rng, hi - lo - 2);
}

static void mutate_scale(gap_sequence_t *seq) {
    double factor = 0.9 + (rng_next(&global_rng) % 2001) / 10000.0;
    for (size_t i = 1; i < seq->num_gaps; i++) {
        uint64_t new_val = (uint64_t)(seq->gaps[i] * factor);
        if (new_val <= seq->gaps[i - 1]) new_val = seq->gaps[i - 1] + 1;
        seq->gaps[i] = new_val;
    }
}

static void mutate_perturb(gap_sequence_t *seq) {
    /* Perturb a single gap by a small amount */
    if (seq->num_gaps <= 1) return;
    size_t pos = 1 + rng_uniform(&global_rng, seq->num_gaps - 1);

    int64_t delta = (int64_t)(seq->gaps[pos] * 0.05);  /* up to 5% change */
    if (delta < 1) delta = 1;

    if (rng_uniform(&global_rng, 2) == 0) {
        seq->gaps[pos] += rng_uniform(&global_rng, delta) + 1;
    } else {
        uint64_t sub = rng_uniform(&global_rng, delta) + 1;
        if (seq->gaps[pos] > seq->gaps[pos-1] + sub) {
            seq->gaps[pos] -= sub;
        }
    }
}

static void mutate(gap_sequence_t *seq, double rate) {
    if ((double)rng_uniform(&global_rng, 1000) / 1000.0 >= rate) return;

    int op = rng_uniform(&global_rng, 5);
    switch (op) {
        case 0: mutate_insert(seq); break;
        case 1: mutate_delete(seq); break;
        case 2: mutate_modify(seq); break;
        case 3: mutate_scale(seq); break;
        case 4: mutate_perturb(seq); break;
    }
    repair_sequence(seq);
}

static void crossover(const gap_sequence_t *p1, const gap_sequence_t *p2, gap_sequence_t *child) {
    child->num_gaps = 0;
    size_t i = 0, j = 0;
    uint64_t last = 0;

    while ((i < p1->num_gaps || j < p2->num_gaps) && child->num_gaps < MAX_GAPS) {
        uint64_t g1 = (i < p1->num_gaps) ? p1->gaps[i] : UINT64_MAX;
        uint64_t g2 = (j < p2->num_gaps) ? p2->gaps[j] : UINT64_MAX;

        uint64_t next;
        if (g1 < g2) { next = g1; i++; }
        else if (g2 < g1) { next = g2; j++; }
        else { next = g1; i++; j++; }

        if (next == 1 || rng_uniform(&global_rng, 2) == 0) {
            if (next > last) {
                child->gaps[child->num_gaps++] = next;
                last = next;
            }
        }
    }
    snprintf(child->name, sizeof(child->name), "Evolved");
    repair_sequence(child);
}

static int tournament_select(individual_t *pop, int pop_size, int tournament_size) {
    int best = rng_uniform(&global_rng, pop_size);
    for (int i = 1; i < tournament_size; i++) {
        int challenger = rng_uniform(&global_rng, pop_size);
        if (pop[challenger].fitness < pop[best].fitness) {
            best = challenger;
        }
    }
    return best;
}

static int compare_fitness(const void *a, const void *b) {
    const individual_t *ia = a;
    const individual_t *ib = b;
    if (ia->fitness < ib->fitness) return -1;
    if (ia->fitness > ib->fitness) return 1;
    return 0;
}

static void seed_population(individual_t *pop, int pop_size, uint64_t max_N, const config_t *cfg) {
    int idx = 0;

    /* If continuing from a previous best, seed heavily with that */
    if (cfg->continue_num_gaps > 0) {
        printf("Continuing from provided sequence (%zu gaps)\n", cfg->continue_num_gaps);

        /* First individual: exact copy */
        strncpy(pop[idx].seq.name, "Continued", sizeof(pop[idx].seq.name) - 1);
        pop[idx].seq.num_gaps = 0;
        for (size_t i = 0; i < cfg->continue_num_gaps && cfg->continue_gaps[i] <= max_N; i++) {
            pop[idx].seq.gaps[pop[idx].seq.num_gaps++] = cfg->continue_gaps[i];
        }
        pop[idx].evaluated = 0;
        idx++;

        /* Next 60% of population: mutations of the continued sequence */
        while (idx < pop_size * 3 / 5) {
            gap_sequence_copy(&pop[idx].seq, &pop[0].seq);
            /* Apply 1-3 mutations */
            int num_mutations = 1 + rng_uniform(&global_rng, 3);
            for (int m = 0; m < num_mutations; m++) {
                mutate(&pop[idx].seq, 1.0);
            }
            pop[idx].evaluated = 0;
            idx++;
        }

        /* Next 20%: heavier mutations */
        while (idx < pop_size * 4 / 5) {
            gap_sequence_copy(&pop[idx].seq, &pop[0].seq);
            for (int m = 0; m < 5; m++) mutate(&pop[idx].seq, 1.0);
            pop[idx].evaluated = 0;
            idx++;
        }

        /* Remaining 20%: baselines and random for diversity */
        gap_sequence_t baselines[NUM_BASELINES];
        gaps_all_baselines(baselines, max_N);
        for (int i = 0; i < NUM_BASELINES && idx < pop_size; i++) {
            gap_sequence_copy(&pop[idx].seq, &baselines[i]);
            pop[idx].evaluated = 0;
            idx++;
        }

        while (idx < pop_size) {
            double r = 2.0 + (rng_next(&global_rng) % 1000) / 1000.0;
            gaps_ratio(&pop[idx].seq, r, max_N, NULL);
            for (int m = 0; m < 2; m++) mutate(&pop[idx].seq, 1.0);
            pop[idx].evaluated = 0;
            idx++;
        }
        return;
    }

    /* Normal seeding (no continuation) */
    gap_sequence_t baselines[NUM_BASELINES];
    gaps_all_baselines(baselines, max_N);
    for (int i = 0; i < NUM_BASELINES && idx < pop_size; i++) {
        gap_sequence_copy(&pop[idx].seq, &baselines[i]);
        pop[idx].evaluated = 0;
        idx++;
    }

    for (double r = 2.1; r <= 2.5 && idx < pop_size; r += 0.05) {
        gaps_ratio(&pop[idx].seq, r, max_N, NULL);
        pop[idx].evaluated = 0;
        idx++;
    }

    /* Mutations of Ciura */
    while (idx < pop_size * 2 / 3) {
        gaps_ciura(&pop[idx].seq, max_N);
        for (int m = 0; m < 3; m++) mutate(&pop[idx].seq, 1.0);
        pop[idx].evaluated = 0;
        idx++;
    }

    /* Random ratio sequences */
    while (idx < pop_size) {
        double r = 2.0 + (rng_next(&global_rng) % 1000) / 1000.0;
        gaps_ratio(&pop[idx].seq, r, max_N, NULL);
        for (int m = 0; m < 2; m++) mutate(&pop[idx].seq, 1.0);
        pop[idx].evaluated = 0;
        idx++;
    }
}

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s --perms <dir> --out <dir> [options]\n", prog);
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  --perms <dir>       Permutation files\n");
    fprintf(stderr, "  --out <dir>         Output directory\n");
    fprintf(stderr, "  --status <file>     Status file for live monitoring (default: results/status.txt)\n");
    fprintf(stderr, "  --threads N         OpenMP threads\n");
    fprintf(stderr, "  --generations N     Number of generations\n");
    fprintf(stderr, "  --pop N             Population size\n");
    fprintf(stderr, "  --elite N           Elite count\n");
    fprintf(stderr, "  --mutation F        Mutation rate (0.0-1.0)\n");
    fprintf(stderr, "  --sizes <list>      Training sizes\n");
    fprintf(stderr, "  --seed <hex>        RNG seed\n");
    fprintf(stderr, "  --continue <gaps>   Continue from sequence (comma-separated gaps)\n");
    fprintf(stderr, "                      Seeds population with this sequence + mutations\n");
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
    cfg->generations = 100;
    cfg->pop_size = 60;
    cfg->elite_count = 8;
    cfg->mutation_rate = 0.3;
    cfg->rng_seed = 0xEE01EE42ULL;
    strcpy(cfg->status_file, "results/status.txt");

    cfg->sizes[0] = 1000;
    cfg->sizes[1] = 10000;
    cfg->sizes[2] = 100000;
    cfg->sizes[3] = 1000000;
    cfg->num_sizes = 4;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--perms") == 0 && i + 1 < argc) {
            strncpy(cfg->perms_dir, argv[++i], sizeof(cfg->perms_dir) - 1);
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            strncpy(cfg->out_dir, argv[++i], sizeof(cfg->out_dir) - 1);
        } else if (strcmp(argv[i], "--status") == 0 && i + 1 < argc) {
            strncpy(cfg->status_file, argv[++i], sizeof(cfg->status_file) - 1);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            cfg->threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--generations") == 0 && i + 1 < argc) {
            cfg->generations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--pop") == 0 && i + 1 < argc) {
            cfg->pop_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--elite") == 0 && i + 1 < argc) {
            cfg->elite_count = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--mutation") == 0 && i + 1 < argc) {
            cfg->mutation_rate = atof(argv[++i]);
        } else if (strcmp(argv[i], "--sizes") == 0 && i + 1 < argc) {
            parse_uint64_list(argv[++i], cfg->sizes, MAX_SIZES, &cfg->num_sizes);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            cfg->rng_seed = strtoull(argv[++i], NULL, 0);
        } else if (strcmp(argv[i], "--continue") == 0 && i + 1 < argc) {
            parse_uint64_list(argv[++i], cfg->continue_gaps, MAX_GAPS, &cfg->continue_num_gaps);
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        }
    }

    for (size_t i = 0; i < cfg->num_sizes; i++) {
        cfg->weights[i] = 1.0 / cfg->num_sizes;
        /* Set reduced search trials: 20 for 1M, 10 for 2M, scale for others */
        if (cfg->sizes[i] >= 2000000) {
            cfg->search_trials[i] = 10;
        } else if (cfg->sizes[i] >= 1000000) {
            cfg->search_trials[i] = 20;
        } else if (cfg->sizes[i] >= 100000) {
            cfg->search_trials[i] = 50;
        } else {
            cfg->search_trials[i] = 100;  /* Use more for smaller sizes (faster anyway) */
        }
    }

    if (cfg->perms_dir[0] == '\0' || cfg->out_dir[0] == '\0') {
        fprintf(stderr, "Error: --perms and --out required\n");
        return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    config_t cfg;
    if (parse_args(argc, argv, &cfg) < 0) {
        print_usage(argv[0]);
        return 1;
    }
    g_cfg = &cfg;
    g_start_time = time(NULL);

    int num_threads = cfg.threads;
#ifdef _OPENMP
    if (num_threads <= 0) num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
#else
    num_threads = 1;
#endif

    rng_seed(&global_rng, cfg.rng_seed);
    mkdir(cfg.out_dir, 0755);

    printf("Loading datasets...\n");
    perm_dataset_t *datasets = malloc(cfg.num_sizes * sizeof(perm_dataset_t));
    for (size_t i = 0; i < cfg.num_sizes; i++) {
        if (load_dataset(cfg.perms_dir, cfg.sizes[i], &datasets[i]) < 0) {
            fprintf(stderr, "Failed to load N=%lu\n", (unsigned long)cfg.sizes[i]);
            return 1;
        }
    }

    /* Get baseline (full evaluation) */
    uint64_t max_N = cfg.sizes[cfg.num_sizes - 1];
    gap_sequence_t ciura;
    gaps_ciura(&ciura, max_N);
    individual_t ciura_ind;
    gap_sequence_copy(&ciura_ind.seq, &ciura);
    ciura_ind.evaluated = 0;
    g_baseline_score = evaluate_individual_full(&ciura_ind, datasets, &cfg, num_threads);
    printf("Baseline (Ciura): %.2f\n", g_baseline_score);
    g_current_best_fitness = g_baseline_score;

    /* Track best scores per size for early termination */
    double best_scores_by_size[MAX_SIZES];
    for (size_t i = 0; i < cfg.num_sizes; i++) {
        best_scores_by_size[i] = ciura_ind.scores_by_size[i];
    }

    /* Initialize population */
    individual_t *population = malloc(cfg.pop_size * sizeof(individual_t));
    individual_t *next_gen = malloc(cfg.pop_size * sizeof(individual_t));

    printf("Seeding population...\n");
    seed_population(population, cfg.pop_size, max_N, &cfg);

    printf("Evaluating initial population (fast mode: %lu/%lu trials for 1M/2M)...\n",
           (unsigned long)cfg.search_trials[cfg.num_sizes > 1 ? cfg.num_sizes - 2 : 0],
           (unsigned long)cfg.search_trials[cfg.num_sizes - 1]);

    /* Parallel population evaluation */
    #pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    for (int i = 0; i < cfg.pop_size; i++) {
        evaluate_individual_fast(&population[i], datasets, &cfg, best_scores_by_size);
    }
    qsort(population, cfg.pop_size, sizeof(individual_t), compare_fitness);

    individual_t best_ever;
    memcpy(&best_ever, &population[0], sizeof(individual_t));
    g_current_best_fitness = best_ever.fitness;

    /* Update best scores per size */
    for (size_t i = 0; i < cfg.num_sizes; i++) {
        if (best_ever.scores_by_size[i] < best_scores_by_size[i]) {
            best_scores_by_size[i] = best_ever.scores_by_size[i];
        }
    }

    printf("Starting evolution... Monitor with: watch -n1 cat %s\n\n", cfg.status_file);

    int tournament_size = 4;
    double crossover_rate = 0.7;
    int plateau_count = 0;
    const int PLATEAU_LIMIT = 50;  /* Auto-stop after 50 gens without improvement */

    for (int gen = 1; gen <= cfg.generations; gen++) {
        /* Elitism */
        for (int i = 0; i < cfg.elite_count; i++) {
            memcpy(&next_gen[i], &population[i], sizeof(individual_t));
        }

        /* Generate offspring */
        for (int i = cfg.elite_count; i < cfg.pop_size; i++) {
            if ((double)rng_uniform(&global_rng, 100) / 100.0 < crossover_rate) {
                int p1 = tournament_select(population, cfg.pop_size, tournament_size);
                int p2 = tournament_select(population, cfg.pop_size, tournament_size);
                crossover(&population[p1].seq, &population[p2].seq, &next_gen[i].seq);
            } else {
                int p = tournament_select(population, cfg.pop_size, tournament_size);
                gap_sequence_copy(&next_gen[i].seq, &population[p].seq);
            }
            mutate(&next_gen[i].seq, cfg.mutation_rate);
            next_gen[i].evaluated = 0;
        }

        /* Parallel population evaluation with early termination */
        #pragma omp parallel for schedule(dynamic) num_threads(num_threads)
        for (int i = 0; i < cfg.pop_size; i++) {
            if (!next_gen[i].evaluated) {
                evaluate_individual_fast(&next_gen[i], datasets, &cfg, best_scores_by_size);
            }
        }

        /* Swap */
        individual_t *tmp = population;
        population = next_gen;
        next_gen = tmp;

        qsort(population, cfg.pop_size, sizeof(individual_t), compare_fitness);

        if (population[0].fitness < best_ever.fitness) {
            memcpy(&best_ever, &population[0], sizeof(individual_t));
            g_current_best_fitness = best_ever.fitness;
            plateau_count = 0;  /* Reset plateau counter on improvement */

            /* Update best scores per size for tighter early termination */
            for (size_t s = 0; s < cfg.num_sizes; s++) {
                if (best_ever.scores_by_size[s] < best_scores_by_size[s]) {
                    best_scores_by_size[s] = best_ever.scores_by_size[s];
                }
            }
        } else {
            plateau_count++;
        }

        /* Write status */
        write_status(gen, &best_ever, population, cfg.pop_size, plateau_count);

        /* Console output */
        double imp = (g_baseline_score - best_ever.fitness) / g_baseline_score * 100.0;
        printf("Gen %3d: best=%.2f (%+.4f%%)  pop_best=%.2f  plateau=%d\n",
               gen, best_ever.fitness, imp, population[0].fitness, plateau_count);

        /* Auto-stop on plateau */
        if (plateau_count >= PLATEAU_LIMIT) {
            printf("\n*** PLATEAU DETECTED: No improvement for %d generations. Stopping early. ***\n", PLATEAU_LIMIT);
            break;
        }
    }

    /* Final validation with FULL trials */
    printf("\n=== VALIDATING BEST WITH FULL TRIALS ===\n");
    individual_t validated_best;
    gap_sequence_copy(&validated_best.seq, &best_ever.seq);
    validated_best.evaluated = 0;
    double validated_fitness = evaluate_individual_full(&validated_best, datasets, &cfg, num_threads);

    printf("\n=== FINAL ===\n");
    printf("Search fitness:     %.2f (%+.4f%% vs Ciura)\n", best_ever.fitness,
           (g_baseline_score - best_ever.fitness) / g_baseline_score * 100.0);
    printf("Validated fitness:  %.2f (%+.4f%% vs Ciura)\n", validated_fitness,
           (g_baseline_score - validated_fitness) / g_baseline_score * 100.0);
    printf("Gaps: [");
    for (size_t i = 0; i < best_ever.seq.num_gaps; i++) {
        printf("%lu", (unsigned long)best_ever.seq.gaps[i]);
        if (i < best_ever.seq.num_gaps - 1) printf(", ");
    }
    printf("]\n");

    /* Save */
    time_t now = time(NULL);
    char ts[64];
    strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", localtime(&now));
    char path[1024];
    snprintf(path, sizeof(path), "%s/evolve_%s.txt", cfg.out_dir, ts);
    FILE *f = fopen(path, "w");
    if (f) {
        fprintf(f, "Evolutionary Search Results\n");
        fprintf(f, "===========================\n\n");
        fprintf(f, "Generations: %d, Population: %d\n", cfg.generations, cfg.pop_size);
        fprintf(f, "Search trials: ");
        for (size_t i = 0; i < cfg.num_sizes; i++) {
            fprintf(f, "%lu@N=%lu%s", (unsigned long)cfg.search_trials[i],
                    (unsigned long)cfg.sizes[i], i < cfg.num_sizes - 1 ? ", " : "\n");
        }
        fprintf(f, "Search fitness: %.2f\n", best_ever.fitness);
        fprintf(f, "Validated fitness: %.2f\n", validated_fitness);
        fprintf(f, "Ciura baseline: %.2f\n", g_baseline_score);
        fprintf(f, "Improvement (validated): %.4f%%\n",
                (g_baseline_score - validated_fitness) / g_baseline_score * 100.0);
        fprintf(f, "\nBest sequence gaps:\n[");
        for (size_t i = 0; i < best_ever.seq.num_gaps; i++) {
            fprintf(f, "%lu", (unsigned long)best_ever.seq.gaps[i]);
            if (i < best_ever.seq.num_gaps - 1) fprintf(f, ", ");
        }
        fprintf(f, "]\n");
        fclose(f);
        printf("Saved to %s\n", path);
    }

    for (size_t i = 0; i < cfg.num_sizes; i++) free_dataset(&datasets[i]);
    free(datasets);
    free(population);
    free(next_gen);
    return 0;
}
