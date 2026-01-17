#define _GNU_SOURCE
/*
 * permgen.c - Generate reproducible permutation datasets for benchmarking
 *
 * Usage: ./permgen --out <dir> --seed <hex> --sizes <n1,n2,...> --trials <t1,t2,...>
 *
 * Output format per size:
 *   <dir>/perm_<N>.bin   - Binary file with TRIALS permutations
 *   <dir>/perm_<N>.meta  - Metadata (JSON-ish)
 *
 * Binary format:
 *   - uint64_t magic (0x5045524D47454E31 = "PERMGEN1")
 *   - uint64_t N
 *   - uint64_t TRIALS
 *   - uint64_t master_seed
 *   - int32_t data[TRIALS][N]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <errno.h>
#include <sys/stat.h>

#include "rng.h"

#define MAGIC 0x5045524D47454E31ULL  /* "PERMGEN1" */
#define MAX_SIZES 32

typedef struct {
    char out_dir[512];
    uint64_t master_seed;
    uint64_t sizes[MAX_SIZES];
    uint64_t trials[MAX_SIZES];
    size_t num_sizes;
} config_t;

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s --out <dir> --seed <hex> --sizes <n1,n2,...> --trials <t1,t2,...>\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --out <dir>       Output directory for permutation files\n");
    fprintf(stderr, "  --seed <hex>      Master seed in hex (e.g., 0xC0FFEE1234)\n");
    fprintf(stderr, "  --sizes <list>    Comma-separated list of N values\n");
    fprintf(stderr, "  --trials <list>   Comma-separated list of trial counts (one per size)\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Example:\n");
    fprintf(stderr, "  %s --out results/perms --seed 0xC0FFEE1234 \\\n", prog);
    fprintf(stderr, "    --sizes 1000,10000,100000,1000000 --trials 1000,1000,1000,100\n");
}

static int parse_uint64_list(const char *str, uint64_t *out, size_t max, size_t *count) {
    *count = 0;
    char *copy = strdup(str);
    if (!copy) return -1;

    char *tok = strtok(copy, ",");
    while (tok && *count < max) {
        char *end;
        out[*count] = strtoull(tok, &end, 0);
        if (*end != '\0' && *end != '\n') {
            free(copy);
            return -1;
        }
        (*count)++;
        tok = strtok(NULL, ",");
    }

    free(copy);
    return 0;
}

static int parse_args(int argc, char **argv, config_t *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->master_seed = 0xC0FFEE1234ULL;  /* Default */

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            strncpy(cfg->out_dir, argv[++i], sizeof(cfg->out_dir) - 1);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            cfg->master_seed = strtoull(argv[++i], NULL, 0);
        } else if (strcmp(argv[i], "--sizes") == 0 && i + 1 < argc) {
            if (parse_uint64_list(argv[++i], cfg->sizes, MAX_SIZES, &cfg->num_sizes) < 0) {
                fprintf(stderr, "Error: Invalid sizes list\n");
                return -1;
            }
        } else if (strcmp(argv[i], "--trials") == 0 && i + 1 < argc) {
            size_t count;
            if (parse_uint64_list(argv[++i], cfg->trials, MAX_SIZES, &count) < 0) {
                fprintf(stderr, "Error: Invalid trials list\n");
                return -1;
            }
            if (count != cfg->num_sizes) {
                fprintf(stderr, "Error: trials count (%zu) must match sizes count (%zu)\n",
                        count, cfg->num_sizes);
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

    if (cfg->out_dir[0] == '\0' || cfg->num_sizes == 0) {
        fprintf(stderr, "Error: --out and --sizes are required\n");
        return -1;
    }

    return 0;
}

static int generate_permutations(const config_t *cfg, size_t size_idx) {
    uint64_t N = cfg->sizes[size_idx];
    uint64_t trials = cfg->trials[size_idx];

    /* Allocate array for one permutation */
    int32_t *arr = malloc(N * sizeof(int32_t));
    if (!arr) {
        fprintf(stderr, "Error: Failed to allocate array for N=%lu\n", (unsigned long)N);
        return -1;
    }

    /* Open output files */
    char bin_path[1024], meta_path[1024];
    snprintf(bin_path, sizeof(bin_path), "%s/perm_%lu.bin", cfg->out_dir, (unsigned long)N);
    snprintf(meta_path, sizeof(meta_path), "%s/perm_%lu.meta", cfg->out_dir, (unsigned long)N);

    FILE *bin_file = fopen(bin_path, "wb");
    if (!bin_file) {
        fprintf(stderr, "Error: Cannot open %s: %s\n", bin_path, strerror(errno));
        free(arr);
        return -1;
    }

    /* Write header */
    uint64_t magic = MAGIC;
    fwrite(&magic, sizeof(magic), 1, bin_file);
    fwrite(&N, sizeof(N), 1, bin_file);
    fwrite(&trials, sizeof(trials), 1, bin_file);
    fwrite(&cfg->master_seed, sizeof(cfg->master_seed), 1, bin_file);

    printf("Generating N=%lu, trials=%lu...\n", (unsigned long)N, (unsigned long)trials);

    /* Generate and write each permutation */
    for (uint64_t t = 0; t < trials; t++) {
        /* Initialize array to identity */
        for (uint64_t i = 0; i < N; i++) {
            arr[i] = (int32_t)i;
        }

        /* Seed RNG for this trial */
        rng_state_t rng;
        uint64_t seed = derive_seed(cfg->master_seed, N, t);
        rng_seed(&rng, seed);

        /* Shuffle */
        rng_shuffle(&rng, arr, N);

        /* Write */
        if (fwrite(arr, sizeof(int32_t), N, bin_file) != N) {
            fprintf(stderr, "Error: Write failed for trial %lu\n", (unsigned long)t);
            fclose(bin_file);
            free(arr);
            return -1;
        }

        if ((t + 1) % 100 == 0 || t + 1 == trials) {
            printf("  %lu/%lu trials\r", (unsigned long)(t + 1), (unsigned long)trials);
            fflush(stdout);
        }
    }
    printf("\n");

    fclose(bin_file);
    free(arr);

    /* Write metadata file */
    FILE *meta_file = fopen(meta_path, "w");
    if (!meta_file) {
        fprintf(stderr, "Error: Cannot open %s: %s\n", meta_path, strerror(errno));
        return -1;
    }

    time_t now = time(NULL);
    char time_str[64];
    strftime(time_str, sizeof(time_str), "%Y-%m-%dT%H:%M:%SZ", gmtime(&now));

    fprintf(meta_file, "{\n");
    fprintf(meta_file, "  \"N\": %lu,\n", (unsigned long)N);
    fprintf(meta_file, "  \"trials\": %lu,\n", (unsigned long)trials);
    fprintf(meta_file, "  \"master_seed\": \"0x%lX\",\n", (unsigned long)cfg->master_seed);
    fprintf(meta_file, "  \"rng\": \"xoshiro256** seeded via splitmix64\",\n");
    fprintf(meta_file, "  \"seed_derivation\": \"derive_seed(master, N, trial)\",\n");
    fprintf(meta_file, "  \"generation_date\": \"%s\",\n", time_str);
    fprintf(meta_file, "  \"format\": \"binary int32, TRIALS permutations of N elements\"\n");
    fprintf(meta_file, "}\n");

    fclose(meta_file);

    printf("Wrote %s and %s\n", bin_path, meta_path);
    return 0;
}

int main(int argc, char **argv) {
    config_t cfg;

    if (parse_args(argc, argv, &cfg) < 0) {
        print_usage(argv[0]);
        return 1;
    }

    /* Create output directory if it doesn't exist */
    mkdir(cfg.out_dir, 0755);

    printf("Permutation Generator\n");
    printf("=====================\n");
    printf("Master seed: 0x%lX\n", (unsigned long)cfg.master_seed);
    printf("Output dir:  %s\n", cfg.out_dir);
    printf("Sizes:       ");
    for (size_t i = 0; i < cfg.num_sizes; i++) {
        printf("%lu", (unsigned long)cfg.sizes[i]);
        if (i < cfg.num_sizes - 1) printf(", ");
    }
    printf("\n");
    printf("Trials:      ");
    for (size_t i = 0; i < cfg.num_sizes; i++) {
        printf("%lu", (unsigned long)cfg.trials[i]);
        if (i < cfg.num_sizes - 1) printf(", ");
    }
    printf("\n\n");

    for (size_t i = 0; i < cfg.num_sizes; i++) {
        if (generate_permutations(&cfg, i) < 0) {
            return 1;
        }
    }

    printf("\nDone!\n");
    return 0;
}
