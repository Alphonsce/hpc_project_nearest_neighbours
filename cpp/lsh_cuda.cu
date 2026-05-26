// CUDA LSH for cosine similarity. Same I/O contract as cpp/lsh.cpp so bench.sh
// can drive both. Output line starts with "mode=cuda_lsh" or "mode=cuda_brute".
//
// Pipeline (mode=cuda_lsh):
//   1. cuBLAS sgemm: scores[N, L*K] = points[N,D] @ planes[L*K,D]^T
//   2. pack_signs kernel: scores -> sigs[N, L] (uint64)
//   3. thrust::sort_by_key per table -> sorted_sigs[L,N], sorted_ids[L,N]
//   4. queries: sgemm + pack -> qsigs[Q, L]
//   5. bucket_lookup kernel: per (q,t) lower/upper bound -> lo[Q*L], hi[Q*L]
//   6. exclusive_scan(hi-lo) -> per-(q,t) offsets, total candidates
//   7. gather kernel: copy sorted_ids[t][lo..hi) into flat candidates buffer
//   8. rerank kernel: block per query, score candidates, top-k with dedup
//
// Build: make -C cpp lsh_cuda
// Run:   ./cpp/lsh_cuda data/glove100_1000000 --L 32 --K 12 --queries 500 --topk 10 --mode cuda_lsh

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

using std::size_t;
using std::uint64_t;
using std::uint32_t;

#define CUDA_OK(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) { \
    std::fprintf(stderr, "cuda: %s @ %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
    std::exit(1); } } while (0)
#define CUBLAS_OK(call) do { cublasStatus_t _s = (call); if (_s != CUBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "cublas error %d @ %s:%d\n", (int)_s, __FILE__, __LINE__); \
    std::exit(1); } } while (0)

enum class Mode { cuda_lsh, cuda_brute };

static const char* mode_name(Mode m) { return m == Mode::cuda_lsh ? "cuda_lsh" : "cuda_brute"; }
static Mode parse_mode(const std::string& s) {
    if (s == "cuda_lsh")   return Mode::cuda_lsh;
    if (s == "cuda_brute") return Mode::cuda_brute;
    std::fprintf(stderr, "unknown --mode: %s\n", s.c_str()); std::exit(1);
}

struct Dataset {
    int n = 0, d = 0;
    std::vector<float> x;
};

static Dataset load(const std::string& base) {
    Dataset ds;
    std::ifstream sh(base + ".shape");
    if (!sh) { std::fprintf(stderr, "cannot open %s.shape\n", base.c_str()); std::exit(1); }
    sh >> ds.n >> ds.d;
    std::ifstream f(base + ".f32", std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s.f32\n", base.c_str()); std::exit(1); }
    ds.x.resize(size_t(ds.n) * ds.d);
    f.read(reinterpret_cast<char*>(ds.x.data()), ds.x.size() * sizeof(float));
    std::fprintf(stderr, "loaded n=%d d=%d\n", ds.n, ds.d);
    return ds;
}

static double now_sec() {
    cudaDeviceSynchronize();
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ---- kernels ------------------------------------------------------------

// scores[i, t*K + k] >= 0 -> bit k of sigs[i, t]; K <= 64
__global__ void pack_signs_kernel(const float* scores, uint64_t* sigs,
                                  int N, int L, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;     // [0, N*L)
    if (idx >= N * L) return;
    int i = idx / L;
    int t = idx % L;
    const float* row = scores + size_t(i) * (L * K) + t * K;
    uint64_t h = 0;
    #pragma unroll
    for (int k = 0; k < 64; ++k) {
        if (k >= K) break;
        if (row[k] >= 0.f) h |= (uint64_t{1} << k);
    }
    sigs[size_t(i) * L + t] = h;
}

// For each (q, t): find lo/hi in sorted_sigs[t] matching q_sigs[q, t].
// lo[q*L+t], hi[q*L+t].  Branchless lower/upper bound (binary search).
__device__ inline int dev_lower_bound(const uint64_t* a, int n, uint64_t key) {
    int lo = 0, hi = n;
    while (lo < hi) { int m = (lo + hi) >> 1; if (a[m] < key) lo = m + 1; else hi = m; }
    return lo;
}
__device__ inline int dev_upper_bound(const uint64_t* a, int n, uint64_t key) {
    int lo = 0, hi = n;
    while (lo < hi) { int m = (lo + hi) >> 1; if (a[m] <= key) lo = m + 1; else hi = m; }
    return lo;
}

__global__ void bucket_lookup_kernel(const uint64_t* q_sigs, const uint64_t* sorted_sigs_all,
                                     int Q, int L, int N, int* lo, int* hi) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // [0, Q*L)
    if (idx >= Q * L) return;
    int q = idx / L, t = idx % L;
    uint64_t key = q_sigs[size_t(q) * L + t];
    const uint64_t* tbl = sorted_sigs_all + size_t(t) * N;
    lo[idx] = dev_lower_bound(tbl, N, key);
    hi[idx] = dev_upper_bound(tbl, N, key);
}

// counts[q*L+t] = hi[q*L+t] - lo[q*L+t]
__global__ void counts_kernel(const int* lo, const int* hi, int n_pairs, int* counts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pairs) return;
    counts[i] = hi[i] - lo[i];
}

// Gather: for each (q,t) range, copy sorted_ids[t][lo..hi) into out[offsets[q*L+t]..)
__global__ void gather_kernel(const int* sorted_ids_all, int N,
                              const int* lo, const int* hi, const int* offsets,
                              int Q, int L, int* out) {
    int qt = blockIdx.x;                                 // one block per (q,t)
    if (qt >= Q * L) return;
    int t = qt % L;
    int a = lo[qt], b = hi[qt], dst = offsets[qt];
    const int* src = sorted_ids_all + size_t(t) * N;
    for (int i = a + threadIdx.x; i < b; i += blockDim.x) {
        out[dst + (i - a)] = src[i];
    }
}

// Reranking: block per query, threads stream candidates, compute dot,
// per-block top-k with linear-insertion small heap in shared memory.
// Top-k may contain duplicate ids (from multiple table hits) — dedup in a final pass.
template <int TOPK_MAX>
__global__ void rerank_topk_kernel(const float* points, const float* queries,
                                   const int* q_offsets,  // [Q+1] per-query candidate offsets
                                   const int* cands,       // flat candidate ids
                                   int Q, int D, int topk,
                                   int* out_ids, float* out_scores) {
    int q = blockIdx.x;
    if (q >= Q) return;
    int a = q_offsets[q], b = q_offsets[q + 1];

    extern __shared__ float smem[];
    float* qvec = smem;                                  // [D]
    float* topv = smem + D;                              // [topk]
    int*   topi = (int*)(topv + TOPK_MAX);               // [topk]

    for (int j = threadIdx.x; j < D; j += blockDim.x) qvec[j] = queries[size_t(q) * D + j];
    if (threadIdx.x < topk) { topv[threadIdx.x] = -1e30f; topi[threadIdx.x] = -1; }
    __syncthreads();

    float my_v[TOPK_MAX]; int my_i[TOPK_MAX];
    #pragma unroll
    for (int t = 0; t < TOPK_MAX; ++t) { my_v[t] = -1e30f; my_i[t] = -1; }

    for (int c = a + threadIdx.x; c < b; c += blockDim.x) {
        int id = cands[c];
        // skip if this id is already in my local top-k (duplicates from multi-table)
        bool dup = false;
        for (int t = 0; t < topk; ++t) if (my_i[t] == id) { dup = true; break; }
        if (dup) continue;

        const float* p = points + size_t(id) * D;
        float dotv = 0.f;
        for (int j = 0; j < D; ++j) dotv += p[j] * qvec[j];

        int worst = 0; float worst_v = my_v[0];
        for (int t = 1; t < topk; ++t) if (my_v[t] < worst_v) { worst_v = my_v[t]; worst = t; }
        if (dotv > worst_v) { my_v[worst] = dotv; my_i[worst] = id; }
    }

    // Merge per-thread top-k into block top-k (one thread at a time, dedup ids)
    for (int round = 0; round < blockDim.x; ++round) {
        if (threadIdx.x == round) {
            for (int t = 0; t < topk; ++t) {
                int id = my_i[t];
                if (id < 0) continue;
                bool dup = false;
                for (int u = 0; u < topk; ++u) if (topi[u] == id) { dup = true; break; }
                if (dup) continue;
                float v = my_v[t];
                int worst = 0; float worst_v = topv[0];
                for (int u = 1; u < topk; ++u) if (topv[u] < worst_v) { worst_v = topv[u]; worst = u; }
                if (v > worst_v) { topv[worst] = v; topi[worst] = id; }
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        // sort topk desc by score (insertion sort, k tiny)
        for (int i = 1; i < topk; ++i) {
            float v = topv[i]; int id = topi[i]; int j = i - 1;
            while (j >= 0 && topv[j] < v) { topv[j+1] = topv[j]; topi[j+1] = topi[j]; --j; }
            topv[j+1] = v; topi[j+1] = id;
        }
        for (int i = 0; i < topk; ++i) {
            out_ids[size_t(q) * topk + i]    = topi[i];
            out_scores[size_t(q) * topk + i] = topv[i];
        }
    }
}

// Brute-force: block per query, threads stream all N points, top-k.
template <int TOPK_MAX>
__global__ void brute_topk_kernel(const float* points, const float* queries,
                                  int N, int D, int Q, int topk,
                                  int* out_ids, float* out_scores) {
    int q = blockIdx.x;
    if (q >= Q) return;

    extern __shared__ float smem[];
    float* qvec = smem;                                  // [D]
    float* topv = smem + D;                              // [topk]
    int*   topi = (int*)(topv + TOPK_MAX);               // [topk]

    for (int j = threadIdx.x; j < D; j += blockDim.x) qvec[j] = queries[size_t(q) * D + j];
    if (threadIdx.x < topk) { topv[threadIdx.x] = -1e30f; topi[threadIdx.x] = -1; }
    __syncthreads();

    float my_v[TOPK_MAX]; int my_i[TOPK_MAX];
    #pragma unroll
    for (int t = 0; t < TOPK_MAX; ++t) { my_v[t] = -1e30f; my_i[t] = -1; }

    for (int id = threadIdx.x; id < N; id += blockDim.x) {
        const float* p = points + size_t(id) * D;
        float dotv = 0.f;
        for (int j = 0; j < D; ++j) dotv += p[j] * qvec[j];
        int worst = 0; float worst_v = my_v[0];
        for (int t = 1; t < topk; ++t) if (my_v[t] < worst_v) { worst_v = my_v[t]; worst = t; }
        if (dotv > worst_v) { my_v[worst] = dotv; my_i[worst] = id; }
    }

    for (int round = 0; round < blockDim.x; ++round) {
        if (threadIdx.x == round) {
            for (int t = 0; t < topk; ++t) {
                float v = my_v[t]; int id = my_i[t];
                if (id < 0) continue;
                int worst = 0; float worst_v = topv[0];
                for (int u = 1; u < topk; ++u) if (topv[u] < worst_v) { worst_v = topv[u]; worst = u; }
                if (v > worst_v) { topv[worst] = v; topi[worst] = id; }
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        for (int i = 1; i < topk; ++i) {
            float v = topv[i]; int id = topi[i]; int j = i - 1;
            while (j >= 0 && topv[j] < v) { topv[j+1] = topv[j]; topi[j+1] = topi[j]; --j; }
            topv[j+1] = v; topi[j+1] = id;
        }
        for (int i = 0; i < topk; ++i) {
            out_ids[size_t(q) * topk + i]    = topi[i];
            out_scores[size_t(q) * topk + i] = topv[i];
        }
    }
}

// ---- main ---------------------------------------------------------------

struct Args {
    std::string base;
    int L = 32, K = 12, topk = 10, queries = 500;
    uint32_t seed = 1;
    Mode mode = Mode::cuda_lsh;
};

static Args parse(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: %s <data_base> [--L 32] [--K 12] [--topk 10] [--queries 500]\n"
            "                       [--mode cuda_lsh|cuda_brute] [--seed 1]\n", argv[0]);
        std::exit(1);
    }
    Args a; a.base = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]{ return std::string(argv[++i]); };
        if      (k == "--L")       a.L = std::stoi(next());
        else if (k == "--K")       a.K = std::stoi(next());
        else if (k == "--topk")    a.topk = std::stoi(next());
        else if (k == "--queries") a.queries = std::stoi(next());
        else if (k == "--seed")    a.seed = (uint32_t)std::stoul(next());
        else if (k == "--mode")    a.mode = parse_mode(next());
        else { std::fprintf(stderr, "unknown arg: %s\n", k.c_str()); std::exit(1); }
    }
    return a;
}

constexpr int TOPK_MAX = 32;

int main(int argc, char** argv) {
    Args a = parse(argc, argv);
    if (a.topk > TOPK_MAX) { std::fprintf(stderr, "topk > %d not supported\n", TOPK_MAX); return 1; }

    Dataset ds = load(a.base);
    const int N = ds.n, D = ds.d, Q = a.queries, L = a.L, K = a.K, TK = a.topk;

    cudaSetDevice(0);
    cublasHandle_t blas; CUBLAS_OK(cublasCreate(&blas));

    // upload points
    float* d_points = nullptr;
    CUDA_OK(cudaMalloc(&d_points, sizeof(float) * size_t(N) * D));
    CUDA_OK(cudaMemcpy(d_points, ds.x.data(), sizeof(float) * size_t(N) * D, cudaMemcpyHostToDevice));

    // pick queries
    std::mt19937 rng(a.seed ^ 0xC0FFEEu);
    std::uniform_int_distribution<int> u(0, N - 1);
    std::vector<int> q_ids(Q);
    for (auto& x : q_ids) x = u(rng);
    std::vector<float> q_host(size_t(Q) * D);
    for (int i = 0; i < Q; ++i) std::memcpy(q_host.data() + size_t(i)*D, ds.x.data() + size_t(q_ids[i])*D, sizeof(float)*D);
    float* d_queries = nullptr;
    CUDA_OK(cudaMalloc(&d_queries, sizeof(float) * size_t(Q) * D));
    CUDA_OK(cudaMemcpy(d_queries, q_host.data(), sizeof(float) * size_t(Q) * D, cudaMemcpyHostToDevice));

    // ground truth = GPU brute force (treat it as exact regardless of mode)
    int* d_gt_ids = nullptr; float* d_gt_scores = nullptr;
    CUDA_OK(cudaMalloc(&d_gt_ids, sizeof(int) * size_t(Q) * TK));
    CUDA_OK(cudaMalloc(&d_gt_scores, sizeof(float) * size_t(Q) * TK));
    double t0 = now_sec();
    {
        int block = 256;
        size_t smem = sizeof(float) * (D + TOPK_MAX) + sizeof(int) * TOPK_MAX;
        brute_topk_kernel<TOPK_MAX><<<Q, block, smem>>>(d_points, d_queries, N, D, Q, TK, d_gt_ids, d_gt_scores);
        CUDA_OK(cudaGetLastError());
    }
    double t_brute = now_sec() - t0;

    // ---------------- LSH ----------------
    double t_lsh = 0.0, t_build = 0.0;
    int* d_out_ids = nullptr; float* d_out_scores = nullptr;
    long long total_cand = 0;
    if (a.mode == Mode::cuda_lsh) {
        // planes: row-major [L*K, D]
        std::vector<float> planes(size_t(L) * K * D);
        std::mt19937 prng(a.seed);
        std::normal_distribution<float> g(0.f, 1.f);
        for (auto& v : planes) v = g(prng);

        float* d_planes = nullptr;
        CUDA_OK(cudaMalloc(&d_planes, sizeof(float) * size_t(L) * K * D));
        CUDA_OK(cudaMemcpy(d_planes, planes.data(), sizeof(float) * size_t(L) * K * D, cudaMemcpyHostToDevice));

        // ----- build -----
        double tb0 = now_sec();
        // scores[N, L*K] = points[N,D] @ planes[L*K,D]^T
        // Using cuBLAS column-major: compute planes @ points^T to get [L*K, N], then we interpret as scores[N, L*K] row-major (transpose-free trick by treating output as L*K rows).
        // Cleaner: set A=planes [L*K x D] col-major (i.e. stored as if D rows, L*K cols), use op_N on planes (rows=L*K) requires layout knowledge.
        // Simplest correct: use cublasSgemm with op_T on B (planes) and op_N on A (points) treating planes as col-major [D, L*K]
        // We provided planes as row-major [L*K, D]: equivalent to col-major [D, L*K]
        // Likewise points row-major [N, D] == col-major [D, N]
        // C = points @ planes^T  (row-major [N, L*K]) == planes_T @ points_T in col-major
        // In col-major land: A=planes col-major [D, L*K], B=points col-major [D, N]
        // C[L*K, N] col-major = A^T (L*K, D) * B (D, N) using gemm op(A)=T, op(B)=N
        // Row-major C[N, L*K] is the same memory as col-major C[L*K, N]
        float* d_scores = nullptr;
        CUDA_OK(cudaMalloc(&d_scores, sizeof(float) * size_t(N) * (L * K)));
        const float alpha = 1.f, beta = 0.f;
        CUBLAS_OK(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N,
                              L * K, N, D,
                              &alpha,
                              d_planes, D,
                              d_points, D,
                              &beta,
                              d_scores, L * K));

        // pack signs -> sigs[N, L] (row-major)
        uint64_t* d_sigs = nullptr;
        CUDA_OK(cudaMalloc(&d_sigs, sizeof(uint64_t) * size_t(N) * L));
        {
            int n_pairs = N * L;
            int block = 256;
            pack_signs_kernel<<<(n_pairs + block - 1) / block, block>>>(d_scores, d_sigs, N, L, K);
            CUDA_OK(cudaGetLastError());
        }
        CUDA_OK(cudaFree(d_scores));

        // sort per table: build sorted_sigs[L,N] (column-major-ish over tables) and sorted_ids[L,N]
        uint64_t* d_sorted_sigs = nullptr;
        int* d_sorted_ids = nullptr;
        CUDA_OK(cudaMalloc(&d_sorted_sigs, sizeof(uint64_t) * size_t(L) * N));
        CUDA_OK(cudaMalloc(&d_sorted_ids, sizeof(int) * size_t(L) * N));
        {
            // gather column t from d_sigs (stride L), then sort with ids
            thrust::device_vector<uint64_t> col(N);
            thrust::device_vector<int> ids(N);
            for (int t = 0; t < L; ++t) {
                // d_sigs[i * L + t] for i in [0, N) -> col[i]
                thrust::transform(thrust::device,
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(N),
                    col.begin(),
                    [d_sigs, L, t] __device__ (int i) { return d_sigs[size_t(i) * L + t]; });
                thrust::sequence(ids.begin(), ids.end());
                thrust::sort_by_key(col.begin(), col.end(), ids.begin());
                CUDA_OK(cudaMemcpy(d_sorted_sigs + size_t(t) * N,
                                   thrust::raw_pointer_cast(col.data()),
                                   sizeof(uint64_t) * N, cudaMemcpyDeviceToDevice));
                CUDA_OK(cudaMemcpy(d_sorted_ids + size_t(t) * N,
                                   thrust::raw_pointer_cast(ids.data()),
                                   sizeof(int) * N, cudaMemcpyDeviceToDevice));
            }
        }
        CUDA_OK(cudaFree(d_sigs));
        t_build = now_sec() - tb0;
        std::fprintf(stderr, "gpu build  L=%d K=%d  %.3fs\n", L, K, t_build);

        // ----- query -----
        double tq0 = now_sec();
        float* d_qscores = nullptr;
        CUDA_OK(cudaMalloc(&d_qscores, sizeof(float) * size_t(Q) * (L * K)));
        CUBLAS_OK(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N,
                              L * K, Q, D, &alpha,
                              d_planes, D, d_queries, D, &beta,
                              d_qscores, L * K));
        uint64_t* d_qsigs = nullptr;
        CUDA_OK(cudaMalloc(&d_qsigs, sizeof(uint64_t) * size_t(Q) * L));
        {
            int block = 256;
            pack_signs_kernel<<<(Q * L + block - 1) / block, block>>>(d_qscores, d_qsigs, Q, L, K);
            CUDA_OK(cudaGetLastError());
        }
        CUDA_OK(cudaFree(d_qscores));

        // bucket lookup -> lo, hi (Q*L)
        int* d_lo = nullptr; int* d_hi = nullptr; int* d_counts = nullptr;
        CUDA_OK(cudaMalloc(&d_lo, sizeof(int) * size_t(Q) * L));
        CUDA_OK(cudaMalloc(&d_hi, sizeof(int) * size_t(Q) * L));
        CUDA_OK(cudaMalloc(&d_counts, sizeof(int) * (size_t(Q) * L + 1)));
        {
            int block = 256;
            bucket_lookup_kernel<<<(Q * L + block - 1) / block, block>>>(
                d_qsigs, d_sorted_sigs, Q, L, N, d_lo, d_hi);
            counts_kernel<<<(Q * L + block - 1) / block, block>>>(d_lo, d_hi, Q * L, d_counts);
            CUDA_OK(cudaGetLastError());
        }
        CUDA_OK(cudaFree(d_qsigs));

        // exclusive scan of counts -> offsets [Q*L+1], total at offsets[Q*L]
        thrust::exclusive_scan(thrust::device, d_counts, d_counts + Q * L + 1, d_counts);
        int total = 0;
        CUDA_OK(cudaMemcpy(&total, d_counts + Q * L, sizeof(int), cudaMemcpyDeviceToHost));
        total_cand = total;

        int* d_cands = nullptr;
        CUDA_OK(cudaMalloc(&d_cands, sizeof(int) * size_t(total)));
        {
            int block = 128;
            gather_kernel<<<Q * L, block>>>(d_sorted_ids, N, d_lo, d_hi, d_counts, Q, L, d_cands);
            CUDA_OK(cudaGetLastError());
        }

        // build per-query offsets (sum over t of counts) so rerank knows ranges
        // counts/offsets above are per (q,t); to get per-q offset, we need offsets[q*L+0] for q's start
        // since for q the range is [counts[q*L+0], counts[(q+1)*L+0]) -> when we built exclusive_scan
        // over Q*L+1 entries, d_counts[q*L] is the start of query q. So q_offsets[q] = d_counts[q*L].
        int* d_q_offsets = nullptr;
        CUDA_OK(cudaMalloc(&d_q_offsets, sizeof(int) * (Q + 1)));
        thrust::transform(thrust::device,
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(Q + 1),
            d_q_offsets,
            [d_counts, L, Q, total] __device__ (int q) {
                return q < Q ? d_counts[q * L] : total;
            });

        CUDA_OK(cudaMalloc(&d_out_ids, sizeof(int) * size_t(Q) * TK));
        CUDA_OK(cudaMalloc(&d_out_scores, sizeof(float) * size_t(Q) * TK));
        {
            int block = 128;
            size_t smem = sizeof(float) * (D + TOPK_MAX) + sizeof(int) * TOPK_MAX;
            rerank_topk_kernel<TOPK_MAX><<<Q, block, smem>>>(
                d_points, d_queries, d_q_offsets, d_cands, Q, D, TK, d_out_ids, d_out_scores);
            CUDA_OK(cudaGetLastError());
        }
        t_lsh = now_sec() - tq0;

        CUDA_OK(cudaFree(d_cands));
        CUDA_OK(cudaFree(d_lo)); CUDA_OK(cudaFree(d_hi)); CUDA_OK(cudaFree(d_counts));
        CUDA_OK(cudaFree(d_q_offsets));
        CUDA_OK(cudaFree(d_sorted_sigs)); CUDA_OK(cudaFree(d_sorted_ids));
        CUDA_OK(cudaFree(d_planes));
    }

    // recall vs GPU brute ground truth (with id dedup in approx top-k)
    std::vector<int> gt_ids(size_t(Q) * TK), ap_ids(size_t(Q) * TK);
    CUDA_OK(cudaMemcpy(gt_ids.data(), d_gt_ids, sizeof(int) * size_t(Q) * TK, cudaMemcpyDeviceToHost));
    if (a.mode == Mode::cuda_lsh) {
        CUDA_OK(cudaMemcpy(ap_ids.data(), d_out_ids, sizeof(int) * size_t(Q) * TK, cudaMemcpyDeviceToHost));
    }

    double recall = 0.0;
    if (a.mode == Mode::cuda_lsh) {
        for (int q = 0; q < Q; ++q) {
            std::vector<int> g(gt_ids.begin() + size_t(q)*TK, gt_ids.begin() + size_t(q+1)*TK);
            std::sort(g.begin(), g.end());
            // dedup ap_ids within each top-k row
            std::vector<int> seen;
            for (int i = 0; i < TK; ++i) {
                int id = ap_ids[size_t(q)*TK + i];
                if (id >= 0 && std::find(seen.begin(), seen.end(), id) == seen.end()) seen.push_back(id);
            }
            int hit = 0;
            for (int x : seen) if (std::binary_search(g.begin(), g.end(), x)) ++hit;
            recall += double(hit) / TK;
        }
        recall /= Q;
    } else {
        recall = 1.0;  // brute mode: GT == approx
    }

    std::printf("mode=%s  N=%d D=%d  L=%d K=%d topk=%d queries=%d\n",
                mode_name(a.mode), N, D, L, K, TK, Q);
    std::printf("brute     : %.3f ms  (%.1f us/query)\n",
                t_brute * 1e3, t_brute * 1e6 / Q);
    if (a.mode == Mode::cuda_lsh) {
        std::printf("build     : %.3f ms\n", t_build * 1e3);
        std::printf("lsh       : %.3f ms  (%.1f us/query)  speedup x%.2f\n",
                    t_lsh * 1e3, t_lsh * 1e6 / Q, t_brute / t_lsh);
        std::printf("recall@%d  : %.4f   avg_candidates=%.1f\n",
                    TK, recall, double(total_cand) / Q);
    } else {
        std::printf("lsh       : -- (brute mode)  speedup x1.00\n");
        std::printf("recall@%d  : 1.0000   avg_candidates=%d\n", TK, N);
    }

    cublasDestroy(blas);
    return 0;
}
