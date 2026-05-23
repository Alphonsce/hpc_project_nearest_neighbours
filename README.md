# hpc_project_nearest_neighbours

Speeding up cosine-similarity nearest-neighbour search on CPU (OpenMP) and GPU
(CUDA), benchmarked on GloVe word embeddings.

## What's implemented

- **Brute-force kNN** — CPU (OpenMP) and GPU (custom CUDA kernel) baselines.
- **SimHash LSH** ([cpp/lsh.cpp](cpp/lsh.cpp)) — L tables of K random hyperplanes;
  build phase is OpenMP-parallel over points, query phase has six selectable
  parallel modes (`--mode`): `queries`, `tables`, `candidates`, `features`,
  `query_table`, `all`.
- **CUDA LSH** ([cpp/lsh_cuda.cu](cpp/lsh_cuda.cu)) — cuBLAS SGEMM for the
  projection step, custom `pack_signs` kernel, `thrust::sort_by_key` per table
  for bucket layout, custom kernels for binary-search lookup, candidate gather
  (with `thrust::exclusive_scan` offsets), and per-query rerank/top-k with
  id-dedup.
- **Datasets** — GloVe `6B` (Wikipedia, 400k vocab, dims 50/100/200/300) and
  `twitter.27B` (1.2M vocab, dims 25/50/100/200), prepared via
  [scripts/prepare_glove.py](scripts/prepare_glove.py).
- **Benchmarks** — OpenMP strong scaling per mode ([scripts/bench.sh](scripts/bench.sh)),
  CPU-vs-GPU comparison ([scripts/bench_cuda.sh](scripts/bench_cuda.sh)).

## Setup

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install numpy scikit-learn matplotlib tqdm requests

# CPU build  (g++, OpenMP)
make -C cpp lsh

# GPU build  (nvcc, cuBLAS, thrust; needs CUDA 12+)
make -C cpp lsh_cuda
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

### CPU vs CUDA

```bash
scripts/bench_cuda.sh   # produces results/cuda_compare_L32_K12.{csv,png}
```

![CPU vs CUDA](results/cuda_compare_L32_K12.png)

### Numbers (N=1M, D=100, Q=500, L=32, K=12, top-10) on L40S vs 8-core CPU

| method | wall time | recall | speedup vs CPU brute P=1 |
|---|---|---|---|
| CPU brute, P=1 | 26 055 ms | 1.00 | 1× |
| CPU LSH, P=1   | 2 282 ms  | 0.51 | 11× |
| CPU brute, P=8 | 4 961 ms  | 1.00 | 5× |
| CPU LSH, P=8   | 483 ms    | 0.51 | 54× |
| GPU brute      | 100 ms    | 1.00 | **260×** |
| GPU LSH        | 22 ms     | 0.51 | **1205×** |

GPU LSH is ~22× faster than 8-thread CPU LSH at identical recall. GPU index
build is ~96 ms (one-time, amortised across query batches).

## Repo layout

```
cpp/
  lsh.cpp        # CPU LSH + brute (OpenMP, --mode flag)
  lsh_cuda.cu    # GPU LSH + brute (cuBLAS + thrust + custom kernels)
  Makefile       # builds both binaries

scripts/
  prepare_glove.py    # download + slice GloVe
  export_for_cpp.py   # .npy -> .f32 + .shape
  visualize_neighbors.py
  bench.sh            # OpenMP strong-scaling sweep
  bench_cuda.sh       # CPU vs GPU comparison
  plot_scaling.py     # per-mode scaling plot
  plot_cuda.py        # CPU-vs-GPU bar chart

data/      # GloVe artifacts (gitignored)
results/   # CSVs + PNGs from benchmarks
```
