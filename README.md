# Shellsort Evolved

[![DOI](https://zenodo.org/badge/1136260484.svg)](https://doi.org/10.5281/zenodo.18281131)

An empirically optimized gap sequence for Shellsort that achieves **0.52% fewer comparisons** than the widely-used Ciura sequence across array sizes from 1M to 8M elements.

## The Evolved Sequence

```c
static const uint64_t EVOLVED_GAPS[] = {
    1, 4, 10, 23, 57, 132, 301, 701,
    1577, 3524, 7705, 17961, 40056, 94681,
    199137, 460316, 1035711, 3236462
};
```

For arrays larger than 8M, extend with the 2.25x rule.

## Key Results

| Size (N) | Ciura | Evolved | Improvement | p-value |
|----------|-------|---------|-------------|---------|
| 1,000,000 | 31,944,358 | 31,825,784 | **+0.37%** | <0.001 |
| 2,000,000 | 67,831,819 | 67,563,913 | **+0.40%** | <0.001 |
| 4,000,000 | 143,478,037 | 142,782,469 | **+0.48%** | <0.001 |
| 8,000,000 | 302,706,309 | 300,974,436 | **+0.57%** | <0.001 |
| **Total** | **545,960,523** | **543,146,602** | **+0.52%** | - |

All results validated with paired t-tests on identical permutations.

## Building

```bash
# Compile tools
cc -O3 -march=native -fopenmp -std=c11 -o permgen src/permgen.c -lm
cc -O3 -march=native -fopenmp -std=c11 -o validate src/validate.c src/shellsort.c -lm
cc -O3 -march=native -fopenmp -std=c11 -o bench src/bench.c src/shellsort.c -lm
cc -O3 -march=native -fopenmp -std=c11 -o evolve search/evolve_live.c src/shellsort.c -lm
```

## Reproducing Results

### Step 1: Generate Test Permutations

```bash
mkdir -p results/perms
./permgen --out results/perms --seed 0xC0FFEE1234 \
  --sizes 1000000,2000000,4000000,8000000 \
  --trials 100,50,25,10
```

### Step 2: Run Validation

```bash
./validate results/perms 16
```

Expected output:
```
N            | Ciura            | Evolved          | Diff %
-------------|------------------|------------------|------------
1000000      |      31944358.18 |      31825783.55 |   +0.3712%
2000000      |      67831818.98 |      67563913.46 |   +0.3950%
4000000      |     143478037.36 |     142782469.08 |   +0.4848%
8000000      |     302706308.80 |     300974436.40 |   +0.5721%
```

## Repository Structure

```
├── src/
│   ├── shellsort.c       # Shellsort with comparison counting
│   ├── shellsort.h       # Header with gap sequence types
│   ├── gaps_baselines.h  # All sequences (Ciura, Tokuda, Lee, Skean, Evolved)
│   ├── rng.h             # xoshiro256** RNG
│   ├── permgen.c         # Permutation generator
│   ├── bench.c           # Benchmark tool
│   └── validate.c        # Validation tool
├── search/
│   └── evolve_live.c     # Evolutionary search algorithm
├── results/
│   └── perms/            # Pre-generated permutation files
├── REPORT.md             # Detailed benchmark report
└── ACADEMIC_REPORT.md    # Full academic writeup
```

## Methodology

- **Comparison counting**: One comparison per `A[j-gap] > temp` evaluation
- **Test data**: Fisher-Yates shuffle with xoshiro256** RNG, master seed 0xC0FFEE1234
- **Evolutionary search**: 4 parallel runs, population 100, up to 400 generations
- **Validation**: Paired t-tests on identical permutations

## Comparison with Other Sequences

| Sequence | Total Comparisons | vs Evolved |
|----------|-------------------|------------|
| **Evolved** | 543,146,602 | - |
| Ciura | 545,960,523 | +0.52% |
| Ciura-Extended | 546,586,830 | +0.63% |
| Tokuda | 546,718,462 | +0.65% |
| Lee-2021 | 555,648,424 | +2.25% |
| Skean-2023 | 555,590,161 | +2.24% |
| Sedgewick-1986 | 704,304,679 | +22.88% |

## License

Public domain. Use freely for any purpose.

## Citation

If you use this work, please cite:

```bibtex
@software{banner2026shellsort,
  author = {Banner, Bryan},
  title = {An Improved Gap Sequence for Shellsort via Evolutionary Optimization},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.18281131},
  url = {https://doi.org/10.5281/zenodo.18281131}
}
```
