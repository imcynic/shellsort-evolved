#define _GNU_SOURCE
/*
 * evolve.c - Evolutionary/genetic search for optimal Shellsort gap sequences
 *
 * Strategies:
 * 1. Seed population with known-good sequences (Ciura, Skean, Tokuda, etc.)
 * 2. Mutations: insert gap, delete gap, modify gap, swap adjacent gaps
 * 3. Crossover: combine gap subsequences from two parents
 * 4. Selection: tournament selection with elitism
 * 5. Multi-objective: primarily comparisons, secondarily sequence length
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
#define POP_SIZE 60
#define ELITE_COUNT 8
#define TOURNAMENT_SIZE 4
#define MAX_GENERATIONS 100
#define MUTATION_RATE 0.3
#define CROSSOVER_RATE 0.7

typedef struct {
    char perms_dir[512];
    char out_dir[512];
    int threads;
    int verbose;
    int generations;
    uint64_t sizes[MAX_SIZES];
    double weights[MAX_SIZES];
    size_t num_sizes;
    uint64_t rng_seed;
} config_t;

typedef struct {
    uint64_t N;
    uint64_t trials;
    int32_t *data;
} perm_dataset_t;

typedef struct {
    gap_sequence_t seq;
    double fitness;         /* Lower is better (mean comparisons) */
    double scores_by_size[MAX_SIZES];
    int evaluated;
} individual_t;

static rng_state_t global_rng;

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s --perms <dir> --out <dir> [options]\n", prog);
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  --perms <dir>       Permutation files directory\n");
    fprintf(stderr, "  --out <dir>         Output directory\n");
    fprintf(stderr, "  --threads N         OpenMP threads\n");
    fprintf(stderr, "  --generations N     Number of generations (default: 100)\n");
    fprintf(stderr, "  --sizes <list>      Training sizes\n");
    fprintf(stderr, "  --seed <hex>        RNG seed for evolution\n");
    fprintf(stderr, "  --verbose           Print details\n");
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
    cfg->generations = MAX_GENERATIONS;
    cfg->rng_seed = 0xDEADBEEF42ULL;

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
        } else if (strcmp(argv[i], "--generations") == 0 && i + 1 < argc) {
            cfg->generations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--sizes") == 0 && i + 1 < argc) {
            parse_uint64_list(argv[++i], cfg->sizes, MAX_SIZES, &cfg->num_sizes);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            cfg->rng_seed = strtoull(argv[++i], NULL, 0);
        } else if (strcmp(argv[i], "--verbose") == 0) {
            cfg->verbose = 1;
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        }
    }

    /* Equal weights */
    for (size_t i = 0; i < cfg->num_sizes; i++) {
        cfg->weights[i] = 1.0 / cfg->num_sizes;
    }

    if (cfg->perms_dir[0] == '\0' || cfg->out_dir[0] == '\0') {
        fprintf(stderr, "Error: --perms and --out required\n");
        return -1;
    }
    return 0;
}

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

static double evaluate_sequence(const perm_dataset_t *ds, const gap_sequence_t *seq, int threads) {
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

static double evaluate_individual(individual_t *ind, perm_dataset_t *datasets,
                                  const config_t *cfg, int threads) {
    if (ind->evaluated) return ind->fitness;

    double score = 0;
    for (size_t i = 0; i < cfg->num_sizes; i++) {
        gap_sequence_t seq;
        gap_sequence_copy(&seq, &ind->seq);

        /* Trim gaps >= N */
        while (seq.num_gaps > 0 && seq.gaps[seq.num_gaps - 1] >= datasets[i].N) {
            seq.num_gaps--;
        }

        char reason[256];
        if (!gap_sequence_valid(&seq, reason, sizeof(reason))) {
            ind->fitness = DBL_MAX;
            ind->evaluated = 1;
            return DBL_MAX;
        }

        double mean = evaluate_sequence(&datasets[i], &seq, threads);
        ind->scores_by_size[i] = mean;
        score += cfg->weights[i] * mean;
    }

    ind->fitness = score;
    ind->evaluated = 1;
    return score;
}

/* Ensure sequence is valid: starts with 1, strictly increasing */
static void repair_sequence(gap_sequence_t *seq) {
    if (seq->num_gaps == 0) {
        seq->gaps[0] = 1;
        seq->num_gaps = 1;
        return;
    }

    /* Sort gaps */
    for (size_t i = 0; i < seq->num_gaps; i++) {
        for (size_t j = i + 1; j < seq->num_gaps; j++) {
            if (seq->gaps[i] > seq->gaps[j]) {
                uint64_t tmp = seq->gaps[i];
                seq->gaps[i] = seq->gaps[j];
                seq->gaps[j] = tmp;
            }
        }
    }

    /* Ensure starts with 1 */
    if (seq->gaps[0] != 1) {
        /* Shift and insert 1 */
        if (seq->num_gaps < MAX_GAPS) {
            memmove(&seq->gaps[1], &seq->gaps[0], seq->num_gaps * sizeof(uint64_t));
            seq->gaps[0] = 1;
            seq->num_gaps++;
        } else {
            seq->gaps[0] = 1;
        }
    }

    /* Remove duplicates and ensure strictly increasing */
    size_t write = 1;
    for (size_t read = 1; read < seq->num_gaps; read++) {
        if (seq->gaps[read] > seq->gaps[write - 1]) {
            seq->gaps[write++] = seq->gaps[read];
        }
    }
    seq->num_gaps = write;
}

/* Mutation: insert a new gap */
static void mutate_insert(gap_sequence_t *seq) {
    if (seq->num_gaps >= MAX_GAPS - 1) return;

    /* Find a gap between existing gaps to insert */
    size_t pos = rng_uniform(&global_rng, seq->num_gaps - 1) + 1;
    uint64_t lo = seq->gaps[pos - 1];
    uint64_t hi = seq->gaps[pos];

    if (hi - lo <= 1) return;  /* No room */

    uint64_t new_gap = lo + 1 + rng_uniform(&global_rng, hi - lo - 1);

    /* Insert */
    memmove(&seq->gaps[pos + 1], &seq->gaps[pos], (seq->num_gaps - pos) * sizeof(uint64_t));
    seq->gaps[pos] = new_gap;
    seq->num_gaps++;
}

/* Mutation: delete a gap (not gap 1) */
static void mutate_delete(gap_sequence_t *seq) {
    if (seq->num_gaps <= 2) return;

    size_t pos = 1 + rng_uniform(&global_rng, seq->num_gaps - 1);
    memmove(&seq->gaps[pos], &seq->gaps[pos + 1], (seq->num_gaps - pos - 1) * sizeof(uint64_t));
    seq->num_gaps--;
}

/* Mutation: modify a gap value */
static void mutate_modify(gap_sequence_t *seq) {
    if (seq->num_gaps <= 1) return;

    size_t pos = 1 + rng_uniform(&global_rng, seq->num_gaps - 1);

    uint64_t lo = seq->gaps[pos - 1];
    uint64_t hi = (pos + 1 < seq->num_gaps) ? seq->gaps[pos + 1] : seq->gaps[pos] * 3;

    if (hi - lo <= 2) return;

    /* Random value in valid range */
    seq->gaps[pos] = lo + 1 + rng_uniform(&global_rng, hi - lo - 2);
}

/* Mutation: scale all gaps by a factor */
static void mutate_scale(gap_sequence_t *seq) {
    double factor = 0.9 + (rng_next(&global_rng) % 2001) / 10000.0;  /* 0.9 to 1.1 */

    for (size_t i = 1; i < seq->num_gaps; i++) {
        uint64_t new_val = (uint64_t)(seq->gaps[i] * factor);
        if (new_val <= seq->gaps[i - 1]) new_val = seq->gaps[i - 1] + 1;
        seq->gaps[i] = new_val;
    }
}

/* Mutation: adjust ratio between consecutive gaps */
static void mutate_ratio(gap_sequence_t *seq) {
    if (seq->num_gaps <= 2) return;

    size_t pos = 1 + rng_uniform(&global_rng, seq->num_gaps - 2);

    /* Compute current ratio and adjust */
    double ratio = (double)seq->gaps[pos + 1] / (double)seq->gaps[pos];
    double new_ratio = ratio * (0.95 + (rng_next(&global_rng) % 1001) / 10000.0);

    uint64_t new_val = (uint64_t)(seq->gaps[pos] * new_ratio);
    if (new_val <= seq->gaps[pos]) new_val = seq->gaps[pos] + 1;
    if (pos + 2 < seq->num_gaps && new_val >= seq->gaps[pos + 2]) {
        new_val = seq->gaps[pos + 2] - 1;
    }
    if (new_val > seq->gaps[pos]) {
        seq->gaps[pos + 1] = new_val;
    }
}

static void mutate(gap_sequence_t *seq) {
    int op = rng_uniform(&global_rng, 5);
    switch (op) {
        case 0: mutate_insert(seq); break;
        case 1: mutate_delete(seq); break;
        case 2: mutate_modify(seq); break;
        case 3: mutate_scale(seq); break;
        case 4: mutate_ratio(seq); break;
    }
    repair_sequence(seq);
}

/* Crossover: take gaps from both parents */
static void crossover(const gap_sequence_t *p1, const gap_sequence_t *p2, gap_sequence_t *child) {
    child->num_gaps = 0;

    /* Merge gaps from both parents, selecting randomly */
    size_t i = 0, j = 0;
    uint64_t last = 0;

    while ((i < p1->num_gaps || j < p2->num_gaps) && child->num_gaps < MAX_GAPS) {
        uint64_t g1 = (i < p1->num_gaps) ? p1->gaps[i] : UINT64_MAX;
        uint64_t g2 = (j < p2->num_gaps) ? p2->gaps[j] : UINT64_MAX;

        uint64_t next;
        if (g1 < g2) {
            next = g1;
            i++;
        } else if (g2 < g1) {
            next = g2;
            j++;
        } else {
            next = g1;
            i++;
            j++;
        }

        /* Randomly include or skip (but always include 1) */
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

/* Tournament selection */
static int tournament_select(individual_t *pop, int pop_size) {
    int best = rng_uniform(&global_rng, pop_size);
    for (int i = 1; i < TOURNAMENT_SIZE; i++) {
        int challenger = rng_uniform(&global_rng, pop_size);
        if (pop[challenger].fitness < pop[best].fitness) {
            best = challenger;
        }
    }
    return best;
}

/* Compare for sorting (lower fitness = better) */
static int compare_fitness(const void *a, const void *b) {
    const individual_t *ia = a;
    const individual_t *ib = b;
    if (ia->fitness < ib->fitness) return -1;
    if (ia->fitness > ib->fitness) return 1;
    return 0;
}

static void seed_population(individual_t *pop, uint64_t max_N) {
    int idx = 0;

    /* Add all baselines */
    gap_sequence_t baselines[NUM_BASELINES];
    gaps_all_baselines(baselines, max_N);
    for (int i = 0; i < NUM_BASELINES && idx < POP_SIZE; i++) {
        gap_sequence_copy(&pop[idx].seq, &baselines[i]);
        pop[idx].evaluated = 0;
        idx++;
    }

    /* Add ratio-based sequences */
    for (double r = 2.1; r <= 2.5 && idx < POP_SIZE; r += 0.05) {
        gaps_ratio(&pop[idx].seq, r, max_N, NULL);
        pop[idx].evaluated = 0;
        idx++;
    }

    /* Add mutations of Ciura */
    for (int i = 0; i < 20 && idx < POP_SIZE; i++) {
        gaps_ciura(&pop[idx].seq, max_N);
        for (int m = 0; m < 3; m++) mutate(&pop[idx].seq);
        pop[idx].evaluated = 0;
        idx++;
    }

    /* Add mutations of Skean */
    for (int i = 0; i < 15 && idx < POP_SIZE; i++) {
        gaps_skean(&pop[idx].seq, max_N);
        for (int m = 0; m < 3; m++) mutate(&pop[idx].seq);
        pop[idx].evaluated = 0;
        idx++;
    }

    /* Fill rest with random ratio sequences */
    while (idx < POP_SIZE) {
        double r = 2.0 + (rng_next(&global_rng) % 1000) / 1000.0;  /* 2.0 to 3.0 */
        gaps_ratio(&pop[idx].seq, r, max_N, NULL);
        for (int m = 0; m < 2; m++) mutate(&pop[idx].seq);
        pop[idx].evaluated = 0;
        idx++;
    }
}

int main(int argc, char **argv) {
    config_t cfg;
    if (parse_args(argc, argv, &cfg) < 0) {
        print_usage(argv[0]);
        return 1;
    }

    int num_threads = cfg.threads;
#ifdef _OPENMP
    if (num_threads <= 0) num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
#else
    num_threads = 1;
#endif

    rng_seed(&global_rng, cfg.rng_seed);
    mkdir(cfg.out_dir, 0755);

    printf("Evolutionary Gap Sequence Search\n");
    printf("=================================\n");
    printf("Population: %d, Generations: %d, Threads: %d\n", POP_SIZE, cfg.generations, num_threads);
    printf("Mutation rate: %.0f%%, Crossover rate: %.0f%%\n", MUTATION_RATE * 100, CROSSOVER_RATE * 100);
    printf("Training sizes: ");
    for (size_t i = 0; i < cfg.num_sizes; i++) {
        printf("%lu", (unsigned long)cfg.sizes[i]);
        if (i < cfg.num_sizes - 1) printf(", ");
    }
    printf("\n\n");

    /* Load datasets */
    perm_dataset_t *datasets = malloc(cfg.num_sizes * sizeof(perm_dataset_t));
    for (size_t i = 0; i < cfg.num_sizes; i++) {
        printf("Loading N=%lu...\n", (unsigned long)cfg.sizes[i]);
        if (load_dataset(cfg.perms_dir, cfg.sizes[i], &datasets[i]) < 0) {
            fprintf(stderr, "Failed to load dataset for N=%lu\n", (unsigned long)cfg.sizes[i]);
            return 1;
        }
    }
    printf("\n");

    /* Get baseline scores for reference */
    printf("=== Baseline Reference ===\n");
    uint64_t max_N = cfg.sizes[cfg.num_sizes - 1];
    gap_sequence_t ciura;
    gaps_ciura(&ciura, max_N);
    individual_t ciura_ind;
    gap_sequence_copy(&ciura_ind.seq, &ciura);
    ciura_ind.evaluated = 0;
    double ciura_score = evaluate_individual(&ciura_ind, datasets, &cfg, num_threads);
    printf("Ciura baseline: %.2f\n\n", ciura_score);

    /* Initialize population */
    individual_t *population = malloc(POP_SIZE * sizeof(individual_t));
    individual_t *next_gen = malloc(POP_SIZE * sizeof(individual_t));

    printf("Seeding population...\n");
    seed_population(population, max_N);

    /* Evaluate initial population */
    printf("Evaluating initial population...\n");
    for (int i = 0; i < POP_SIZE; i++) {
        evaluate_individual(&population[i], datasets, &cfg, num_threads);
    }
    qsort(population, POP_SIZE, sizeof(individual_t), compare_fitness);

    printf("Initial best: %.2f (%s)\n\n", population[0].fitness, population[0].seq.name);

    /* Evolution loop */
    double best_ever = population[0].fitness;
    gap_sequence_t best_seq;
    gap_sequence_copy(&best_seq, &population[0].seq);

    for (int gen = 0; gen < cfg.generations; gen++) {
        /* Elitism: keep best individuals */
        for (int i = 0; i < ELITE_COUNT; i++) {
            memcpy(&next_gen[i], &population[i], sizeof(individual_t));
        }

        /* Generate rest through selection, crossover, mutation */
        for (int i = ELITE_COUNT; i < POP_SIZE; i++) {
            if ((double)rng_uniform(&global_rng, 100) / 100.0 < CROSSOVER_RATE) {
                /* Crossover */
                int p1 = tournament_select(population, POP_SIZE);
                int p2 = tournament_select(population, POP_SIZE);
                crossover(&population[p1].seq, &population[p2].seq, &next_gen[i].seq);
            } else {
                /* Copy from tournament winner */
                int p = tournament_select(population, POP_SIZE);
                gap_sequence_copy(&next_gen[i].seq, &population[p].seq);
            }

            /* Mutation */
            if ((double)rng_uniform(&global_rng, 100) / 100.0 < MUTATION_RATE) {
                mutate(&next_gen[i].seq);
            }

            next_gen[i].evaluated = 0;
        }

        /* Evaluate new generation */
        for (int i = 0; i < POP_SIZE; i++) {
            evaluate_individual(&next_gen[i], datasets, &cfg, num_threads);
        }

        /* Swap populations */
        individual_t *tmp = population;
        population = next_gen;
        next_gen = tmp;

        /* Sort by fitness */
        qsort(population, POP_SIZE, sizeof(individual_t), compare_fitness);

        /* Track best */
        if (population[0].fitness < best_ever) {
            best_ever = population[0].fitness;
            gap_sequence_copy(&best_seq, &population[0].seq);
        }

        /* Progress report */
        double improvement = (ciura_score - population[0].fitness) / ciura_score * 100.0;
        printf("Gen %3d: best=%.2f (%.2f%% vs Ciura), avg=%.2f\n",
               gen + 1, population[0].fitness, improvement,
               (population[0].fitness + population[POP_SIZE/2].fitness) / 2);

        if (cfg.verbose) {
            printf("  Top sequence: ");
            for (size_t j = 0; j < population[0].seq.num_gaps && j < 15; j++) {
                printf("%lu ", (unsigned long)population[0].seq.gaps[j]);
            }
            if (population[0].seq.num_gaps > 15) printf("...");
            printf("\n");
        }
    }

    /* Final results */
    printf("\n=== FINAL RESULTS ===\n");
    printf("Best fitness: %.2f\n", best_ever);
    printf("Ciura baseline: %.2f\n", ciura_score);
    double final_improvement = (ciura_score - best_ever) / ciura_score * 100.0;
    printf("Improvement: %.4f%%\n\n", final_improvement);

    printf("Best sequence:\n");
    printf("  Name: %s\n", best_seq.name);
    printf("  Gaps (%zu): [", best_seq.num_gaps);
    for (size_t i = 0; i < best_seq.num_gaps; i++) {
        printf("%lu", (unsigned long)best_seq.gaps[i]);
        if (i < best_seq.num_gaps - 1) printf(", ");
    }
    printf("]\n\n");

    printf("Top 5 evolved sequences:\n");
    for (int i = 0; i < 5 && i < POP_SIZE; i++) {
        printf("%d. fitness=%.2f, gaps=[", i + 1, population[i].fitness);
        for (size_t j = 0; j < population[i].seq.num_gaps && j < 10; j++) {
            printf("%lu", (unsigned long)population[i].seq.gaps[j]);
            if (j < population[i].seq.num_gaps - 1 && j < 9) printf(", ");
        }
        if (population[i].seq.num_gaps > 10) printf(", ...");
        printf("] (%zu gaps)\n", population[i].seq.num_gaps);
    }

    /* Save results */
    time_t now = time(NULL);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", localtime(&now));

    char path[1024];
    snprintf(path, sizeof(path), "%s/evolve_%s.txt", cfg.out_dir, timestamp);
    FILE *f = fopen(path, "w");
    if (f) {
        fprintf(f, "Evolutionary Search Results\n");
        fprintf(f, "===========================\n\n");
        fprintf(f, "Generations: %d, Population: %d\n", cfg.generations, POP_SIZE);
        fprintf(f, "Best fitness: %.2f\n", best_ever);
        fprintf(f, "Ciura baseline: %.2f\n", ciura_score);
        fprintf(f, "Improvement: %.4f%%\n\n", final_improvement);
        fprintf(f, "Best sequence gaps:\n[");
        for (size_t i = 0; i < best_seq.num_gaps; i++) {
            fprintf(f, "%lu", (unsigned long)best_seq.gaps[i]);
            if (i < best_seq.num_gaps - 1) fprintf(f, ", ");
        }
        fprintf(f, "]\n");
        fclose(f);
        printf("\nResults saved to %s\n", path);
    }

    /* Cleanup */
    for (size_t i = 0; i < cfg.num_sizes; i++) {
        free_dataset(&datasets[i]);
    }
    free(datasets);
    free(population);
    free(next_gen);

    return 0;
}
