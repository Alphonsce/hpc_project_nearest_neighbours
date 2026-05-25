# hpc_project_nearest_neighbours

Speeding up cosine-similarity nearest-neighbour search on CPU (OpenMP) and GPU
(CUDA), benchmarked on GloVe word embeddings.

## What's implemented

- **Brute-force kNN** - CPU (OpenMP) and GPU (custom CUDA kernel) baselines.
- **SimHash LSH** ([cpp/lsh.cpp](cpp/lsh.cpp)) - L tables of K random
  hyperplanes; build phase is OpenMP-parallel over points, query phase has
  selectable parallel modes (`queries`, `tables`, `candidates`, `features`,
  `query_table`, `all`).
- **CUDA LSH** ([cpp/lsh_cuda.cu](cpp/lsh_cuda.cu)) - cuBLAS SGEMM for the
  projection step, custom `pack_signs` kernel, `thrust::sort_by_key` per table
  for bucket layout, custom kernels for binary-search lookup, candidate gather,
  and per-query rerank/top-k with id-dedup.
- **HNSW** ([cpp/hnsw.cpp](cpp/hnsw.cpp)) - graph-based approximate nearest
  neighbours with serial and OpenMP modes for query/build exploration.
- **CUDA HNSW query path** ([cpp/hnsw_cuda.cu](cpp/hnsw_cuda.cu)) - CPU graph
  construction with GPU brute/query kernels for comparing HNSW search against
  GPU brute force and LSH.
- **Datasets** - GloVe `6B` (Wikipedia, 400k vocab, dims 50/100/200/300) and
  `twitter.27B` (1.2M vocab, dims 25/50/100/200), prepared via
  [scripts/prepare_glove.py](scripts/prepare_glove.py).
- **Benchmarks** - OpenMP strong scaling, CPU-vs-GPU comparisons, HNSW `ef`
  recall-speed sweeps, and combined HNSW-vs-LSH plots.

## Setup

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install numpy scikit-learn matplotlib tqdm requests

# CPU build  (g++, OpenMP)
make -C cpp lsh hnsw

# GPU build  (nvcc, cuBLAS, thrust; needs CUDA 12+)
make -C cpp lsh_cuda hnsw_cuda

# all targets
make -C cpp
```

## Dataset prep

```bash
# small visualisable subset
python scripts/prepare_glove.py --source 6B --dim 50 --n 20000

# benchmark scale (downloads ~1.5 GB zip, cached in data/)
python scripts/prepare_glove.py --source twitter.27B --dim 100 --n 1000000

# raw binary for the C++/CUDA binaries
python scripts/export_for_cpp.py --dim 100 --n 1000000
```

Outputs in `data/`:

| file | type | notes |
|---|---|---|
| `glove<dim>_<N>.npy` | float32 (N, dim) | raw GloVe vectors |
| `glove<dim>_<N>_norm.npy` | float32 (N, dim) | L2-normalised |
| `glove<dim>_<N>.f32` | raw float32 | row-major, used by C++/CUDA |
| `glove<dim>_<N>.shape` | text | `N D` header |
| `glove<dim>_<N>_words.txt` | N lines | row-aligned tokens |

## Visualise neighbours

```bash
python scripts/visualize_neighbors.py --word king --k 10
# prints cosine kNN, writes data/nn_king.png (PCA-2D, query + neighbours highlighted)
```

## Running the algorithms

```bash
# CPU LSH
./cpp/lsh data/glove100_1000000 --L 32 --K 12 --queries 500 --topk 10 --mode queries

# CUDA LSH / brute
./cpp/lsh_cuda data/glove100_1000000 --L 32 --K 12 --queries 500 --topk 10 --mode cuda_lsh
./cpp/lsh_cuda data/glove100_1000000 --queries 500 --topk 10 --mode cuda_brute

# CPU HNSW
./cpp/hnsw data/glove100_1000000 --M 16 --ef_construction 200 --ef 50 --queries 500 --topk 10 --mode build

# CUDA HNSW / brute
./cpp/hnsw_cuda data/glove100_1000000 --M 16 --ef_construction 200 --ef 50 --queries 500 --topk 10 --mode cuda_hnsw
./cpp/hnsw_cuda data/glove100_1000000 --queries 500 --topk 10 --mode cuda_brute
```

## Benchmarks

### CPU OpenMP strong scaling

```bash
scripts/bench.sh                                   # default: dim=100, N=1M, L=32 K=12
MODES="queries tables candidates features" PLOT=1 scripts/bench.sh
```

Writes `results/scaling_modes_<...>.csv` (and PNG when `PLOT=1`). Below: same
workload, varying OpenMP threads and parallelism axis.

![CPU OpenMP modes](results/scaling_methods_no_dynamic_new_L32_K12.png)

Takeaways: parallel-over-queries is the only mode that scales near-ideal up to
the 8 physical cores; parallel-over-tables and -candidates plateau early due to
serial dedupe/merge tails; parallel-over-features is *slower* than serial — D=100
is too small for OpenMP to beat SIMD.

### LSH CPU vs CUDA

```bash
scripts/bench_cuda.sh   # produces results/cuda_compare_L32_K12.{csv,png}
```

![CPU vs CUDA](results/cuda_compare_L32_K12.png)

### HNSW CPU vs CUDA

```bash
# Small dataset, useful for fast iteration.
DIM=50 N=20000 SOURCE=6B QUERIES=500 PLOT=1 scripts/bench_hnsw_cuda.sh

# 1M dataset. Use CPU_MODE=build to use the parallel HNSW build path.
CPU_MODE=build CPU_THREADS="8" DIM=100 N=1000000 SOURCE=twitter.27B \
  QUERIES=500 PLOT=1 scripts/bench_hnsw_cuda.sh

# HNSW recall/speed sweep over ef on the small dataset.
DIM=50 N=20000 SOURCE=6B QUERIES=500 THREADS=8 MODES="queries" \
  PLOT=1 scripts/bench_hnsw_ef.sh
```

![HNSW CPU vs CUDA, 1M](results/hnsw_cuda_compare_1M_100d_M16_ef50.png)

Important note: HNSW index construction is much more expensive than LSH bucket
construction at 1M points. On the L40S run below, HNSW query time was
competitive, but the HNSW build took about 744 seconds.

### HNSW vs LSH

```bash
python scripts/plot_hnsw_lsh_summary.py
```

The combined summary script reads:

- `results/hnsw_lsh_small_N20000_summary.csv`
- `results/hnsw_lsh_1M_100d_summary.csv`

and writes:

- `results/hnsw_separate_visualization.png`
- `results/hnsw_vs_lsh_visualization.png`

![HNSW summary](results/hnsw_separate_visualization.png)

![HNSW vs LSH](results/hnsw_vs_lsh_visualization.png)

### Latest numbers on L40S vs 8-core CPU

LSH, `N=1M`, `D=100`, `Q=500`, `L=32`, `K=12`, top-10:

| method | wall time | recall | speedup vs CPU brute P=1 |
|---|---|---|---|
| CPU brute, P=1 | 26 062 ms | 1.00 | 1x |
| CPU LSH, P=1   | 2 310 ms  | 0.506 | 11x |
| CPU brute, P=8 | 5 104 ms  | 1.00 | 5x |
| CPU LSH, P=8   | 504 ms    | 0.506 | 52x |
| GPU brute      | 101 ms    | 1.00 | 259x |
| GPU LSH        | 22 ms     | 0.506 | 1207x |

HNSW, `N=1M`, `D=100`, `Q=500`, `M=16`, `ef=50`, top-10:

| method | wall time | recall | build time |
|---|---|---|---|
| CPU brute, P=8 | 5 082 ms | 1.00 | - |
| CPU HNSW, P=8  | 128 ms   | 0.517 | CPU build included before query |
| GPU brute      | 101 ms   | 1.00 | - |
| GPU HNSW       | 50 ms    | 0.505 | 743 661 ms |

Small dataset, `N=20k`, `D=50`, `Q=500`:

| method | wall time | recall | build time |
|---|---|---|---|
| CPU LSH, P=8  | 4.01 ms | 0.765 | - |
| CPU HNSW, P=8 | 3.61 ms | 0.972 | - |
| GPU LSH       | 1.15 ms | 0.765 | 52 ms |
| GPU HNSW      | 0.79 ms | 0.972 | 3 345 ms |

At these settings, HNSW gives much higher recall on `N=20k` and similar recall
to LSH on `N=1M`. GPU LSH remains the better throughput/build-time option at
1M because its index construction is orders of magnitude cheaper.

## Repo layout

```
cpp/
  lsh.cpp        # CPU LSH + brute (OpenMP, --mode flag)
  lsh_cuda.cu    # GPU LSH + brute (cuBLAS + thrust + custom kernels)
  hnsw.cpp       # CPU HNSW + brute (OpenMP modes)
  hnsw_cuda.cu   # GPU HNSW query/brute comparison
  Makefile       # builds CPU and CUDA binaries

scripts/
  prepare_glove.py    # download + slice GloVe
  export_for_cpp.py   # .npy -> .f32 + .shape
  visualize_neighbors.py
  bench.sh            # OpenMP strong-scaling sweep
  bench_cuda.sh       # LSH CPU vs GPU comparison
  bench_hnsw.sh       # HNSW OpenMP strong-scaling sweep
  bench_hnsw_cuda.sh  # HNSW CPU vs GPU comparison
  bench_hnsw_ef.sh    # HNSW recall/speed sweep over ef
  plot_scaling.py     # per-mode scaling plot
  plot_cuda.py        # LSH CPU-vs-GPU bar chart
  plot_hnsw_cuda.py
  plot_hnsw_ef.py
  plot_hnsw_lsh_summary.py

data/      # GloVe artifacts (gitignored)
results/   # CSVs + PNGs from benchmarks
```
