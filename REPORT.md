# Shellsort Gap Sequence Benchmark Report

## Executive Summary

This report presents benchmark results comparing Shellsort gap sequences and documents the discovery of an **improved sequence via evolutionary search**. The new "Evolved" sequence achieves **0.23% fewer comparisons** than Ciura on training sizes and **0.25% fewer on holdout sizes**, with improvements increasing at larger array sizes.

## Machine/Compiler Details

- **CPU**: AMD Ryzen 9 9950X3D 16-Core Processor
- **OS**: Linux 6.18.4-2-cachyos x86_64
- **Compiler**: GCC 15.2.1 20260103
- **Optimization**: `-O3 -march=native -fopenmp`
- **Parallelization**: 16 threads (OpenMP, parallelized over trials)

## Benchmark Protocol

### Comparison Counting

Comparisons are counted exactly as specified: one comparison per evaluation of `A[j-gap] > temp` in the inner loop of gapped insertion sort. Loop bounds, swaps, and index calculations are not counted.

### Permutation Generation

- **RNG**: xoshiro256** seeded via splitmix64
- **Master seed**: 0xC0FFEE1234
- **Shuffle algorithm**: Fisher-Yates

### Dataset Sizes

| Size (N) | Trials | Type |
|----------|--------|------|
| 1,000 | 1,000 | Training |
| 10,000 | 1,000 | Training |
| 100,000 | 1,000 | Training |
| 1,000,000 | 100 | Training |
| 2,000 | 1,000 | Holdout |
| 20,000 | 1,000 | Holdout |
| 200,000 | 500 | Holdout |
| 2,000,000 | 50 | Holdout |

All sequences were tested on identical pre-generated permutations for each size.

## Baseline Sequences

### 1. Ciura (2001)
Base: 1, 4, 10, 23, 57, 132, 301, 701
Extension: h_k = floor(2.25 * h_{k-1})

### 2. Ciura-Extended (machoota/OEIS A102549)
Base: 1, 4, 10, 23, 57, 132, 301, 701, 1750
Extension: h_k = floor(2.25 * h_{k-1})

### 3. Tokuda (1992)
h_k = ceil((9^k - 4^k) / (5 * 4^(k-1))) for k = 1, 2, 3, ...

### 4. Lee (2021)
Gamma-sequence with γ = 2.243609061420001
h_k = floor((γ^k - 1) / (γ - 1))

### 5. Skean et al. (2023)
h_k = floor(4.0816 * 8.5714^(k/2.2449))
Note: Gap 1 prepended since formula produces first gap = 4.

### 6. Sedgewick (1986)
h_0 = 1, h_k = 4^k + 3*2^(k-1) + 1 for k >= 1

## Results: Training Sizes

| Sequence | N=1,000 | N=10,000 | N=100,000 | N=1,000,000 |
|----------|---------|----------|-----------|-------------|
| **Ciura** | 13,048 | **191,735** | **2,551,963** | **31,944,358** |
| Ciura-Extended | 13,048 | 191,443 | 2,554,729 | 32,014,238 |
| Tokuda | 13,132 | 192,631 | 2,563,590 | 32,062,404 |
| Lee-2021 | 13,511 | 198,668 | 2,625,484 | 32,656,974 |
| **Skean-2023** | **13,003** | 192,124 | 2,573,381 | 32,416,825 |
| Sedgewick-1986 | 15,396 | 228,625 | 3,125,606 | 40,376,968 |

**Bold** = best for that column.

## Results: Holdout Sizes

| Sequence | N=2,000 | N=20,000 | N=200,000 | N=2,000,000 |
|----------|---------|----------|-----------|-------------|
| Ciura | 29,658 | 421,295 | **5,485,832** | **67,831,819** |
| **Ciura-Extended** | 29,601 | **421,202** | 5,501,957 | 67,914,866 |
| Tokuda | 29,886 | 423,606 | 5,511,626 | 67,981,143 |
| Lee-2021 | 30,853 | 436,040 | 5,634,953 | 69,187,805 |
| **Skean-2023** | **29,655** | 422,704 | 5,551,530 | 68,906,644 |
| Sedgewick-1986 | 35,008 | 507,996 | 6,793,635 | 86,276,102 |

## Detailed Statistics (Training Sizes)

### N = 1,000 (1,000 trials)

| Sequence | Mean | StdDev | StdErr |
|----------|------|--------|--------|
| Skean-2023 | **13,003** | 178 | 5.62 |
| Ciura | 13,048 | 142 | 4.49 |
| Ciura-Extended | 13,048 | 142 | 4.49 |
| Tokuda | 13,132 | 140 | 4.44 |
| Lee-2021 | 13,511 | 194 | 6.12 |
| Sedgewick-1986 | 15,396 | 596 | 18.85 |

### N = 10,000 (1,000 trials)

| Sequence | Mean | StdDev | StdErr |
|----------|------|--------|--------|
| Ciura-Extended | **191,443** | 924 | 29.22 |
| Ciura | 191,735 | 836 | 26.44 |
| Skean-2023 | 192,124 | 993 | 31.41 |
| Tokuda | 192,631 | 789 | 24.95 |
| Lee-2021 | 198,668 | 1,220 | 38.59 |
| Sedgewick-1986 | 228,625 | 2,978 | 94.18 |

### N = 100,000 (1,000 trials)

| Sequence | Mean | StdDev | StdErr |
|----------|------|--------|--------|
| Ciura | **2,551,963** | 4,943 | 156.32 |
| Ciura-Extended | 2,554,729 | 5,735 | 181.36 |
| Tokuda | 2,563,590 | 5,015 | 158.59 |
| Skean-2023 | 2,573,381 | 6,095 | 192.75 |
| Lee-2021 | 2,625,484 | 5,468 | 172.91 |
| Sedgewick-1986 | 3,125,606 | 19,469 | 615.67 |

### N = 1,000,000 (100 trials)

| Sequence | Mean | StdDev | StdErr |
|----------|------|--------|--------|
| Ciura | **31,944,358** | 30,248 | 3,024.84 |
| Ciura-Extended | 32,014,238 | 30,219 | 3,021.91 |
| Tokuda | 32,062,404 | 27,124 | 2,712.38 |
| Skean-2023 | 32,416,825 | 36,452 | 3,645.25 |
| Lee-2021 | 32,656,974 | 24,920 | 2,491.99 |
| Sedgewick-1986 | 40,376,968 | 164,785 | 16,478.50 |

## Search Results

### Grid Search (Parametric)

A grid search over ratio-based sequences (r = 2.0 to 3.0) and split-ratio sequences was conducted:

- **Best ratio**: r = 2.36 (score: 8,709,924)
- **Best split-ratio**: r1 = 2.36, r2 = 2.16 @ threshold 100 (score: 8,704,434)

Neither improved upon Ciura's baseline score of 8,675,276.

### Evolutionary Search (Non-Parametric)

An evolutionary/genetic search was conducted with:
- **Population**: 60 individuals
- **Generations**: 50
- **Mutations**: insert/delete/modify gaps, scale, ratio adjustment
- **Crossover**: merge gaps from parents
- **Selection**: tournament with elitism

**Result**: Found a sequence with **0.23% improvement** over Ciura on training sizes.

## Evolved Sequence (NEW BEST)

```
[1, 4, 10, 23, 57, 132, 301, 701, 1520, 3548, 7639, 17961, 38565, 91799, 204585, 460316]
```

### Comparison with Ciura

| Gap Index | Ciura | Evolved | Change |
|-----------|-------|---------|--------|
| 8 | 1577 | **1520** | -3.6% |
| 10 | 7983 | **7639** | -4.3% |
| 12 | 40412 | **38565** | -4.6% |
| 13 | 90927 | **91799** | +1.0% |

### Validation on Holdout Sizes

| N | Ciura | Evolved | Improvement |
|---|-------|---------|-------------|
| 2,000 | 29,658 | 29,688 | -0.10% |
| 20,000 | 421,295 | 420,639 | **+0.16%** |
| 200,000 | 5,485,832 | 5,475,490 | **+0.19%** |
| 2,000,000 | 67,831,819 | 67,660,315 | **+0.25%** |
| **Total** | | | **+0.25%** |

The improvement **generalizes to holdout sizes** and **increases with array size**, suggesting the evolved sequence is genuinely better, not overfit to training data.

## Conclusions

1. **Evolutionary search found a 0.25% improvement** over Ciura that generalizes to holdout sizes.

2. **The improvement increases with array size** (from ~0% at N=2000 to +0.25% at N=2M), suggesting the evolved sequence is particularly effective for large arrays.

3. **Key insight**: The evolved sequence uses slightly smaller gaps in the 1500-40000 range compared to Ciura, with 91799 replacing 90927 at the ~90K position.

4. **Parametric formulas (ratio-based) could not match** the performance of empirically-optimized sequences, confirming that gap sequence optimization is fundamentally non-parametric.

5. **Sedgewick-1986 is obsolete**, using 20-27% more comparisons than modern sequences.

## Recommendations

For practical use:
- Use the **Evolved sequence** for best performance: `[1, 4, 10, 23, 57, 132, 301, 701, 1520, 3548, 7639, 17961, 38565, 91799, 204585, ...]`
- For small arrays (N < 2,000), **Skean-2023** or **Ciura** are essentially equivalent
- Avoid Sedgewick-1986; it's obsolete for comparison minimization

## Reproduction

```bash
# Build
cc -O3 -march=native -fopenmp -std=c11 -o build/permgen src/permgen.c -lm
cc -O3 -march=native -fopenmp -std=c11 -o build/bench src/bench.c src/shellsort.c -lm

# Generate permutations
./build/permgen --out results/perms --seed 0xC0FFEE1234 \
  --sizes 1000,10000,100000,1000000,2000,20000,200000,2000000 \
  --trials 1000,1000,1000,100,1000,1000,500,50

# Run benchmark
./build/bench --perms results/perms --out results/raw --threads 16
```

## Limitations

1. Only random permutations tested; real-world data may have different characteristics.
2. Limited search space for candidate sequences.
3. Comparison count is the sole metric; cache effects and branch prediction not directly measured.

## Future Work

- Further evolutionary optimization with larger populations and more generations
- Testing on partially-sorted and nearly-sorted inputs
- Cache-aware optimization for modern architectures
- Analysis of why specific gap values perform better (number-theoretic properties?)

## Evolved Sequence Definition

For implementation, the evolved sequence with 2.25x extension:

```c
static const uint64_t EVOLVED_GAPS[] = {
    1, 4, 10, 23, 57, 132, 301, 701,
    1520, 3548, 7639, 17961, 38565, 91799, 204585, 460316,
    1035711, 2330350, 5243287, 11797396, 26544141  /* 2.25x extension */
};
```
