// HNSW on GPU for cosine similarity.
// Same I/O contract as hnsw.cpp so bench_hnsw_cuda.sh can drive both.
//
// Strategy:
//   Build  : CPU (serial, OMP optional via hnsw.cpp logic embedded here).
//   Query  : two phases.
//     Phase 1 – upper-layer greedy descent (layers max_level..1, ef=1):
//               done on CPU; produces one layer-0 entry point per query.
//               Upper layers are tiny (O(log N) nodes per level) so CPU cost
//               is negligible compared with the main layer-0 search.
//     Phase 2 – layer-0 beam search (ef candidates):
//               done on GPU, one warp (32 threads) per query.
//               All 32 lanes cooperate on dot-product computation;
//               lane 0 drives control flow (heap pop/push, visited hash).
//
// Why upper layers stay on CPU:
//   The layer-0 graph has N nodes × M0 edges = dominant cost.
//   Upper layers together hold only ~N/M + N/M^2 + … ≈ N/(M-1) nodes total,
//   and traversal is O(max_level * M) ≈ 5*16 = 80 distance evals per query —
//   negligible. Storing sparse upper-layer graphs in GPU-friendly format
//   (CSR with binary-search lookup for ep) adds complexity without benefit.
//
// GPU kernel notes:
//   - BLOCK_DIM = 32 (one warp per query block).
//   - Shared memory per block ≈ 14 KB: fits 3+ blocks per SM on 48 KB devices.
//   - W (results) : flat sorted array (ascending distance), insertion-sorted by lane 0.
//   - C (candidates): flat unsorted array, linear-scan min-pop by lane 0.
//   - Visited       : open-addressed hash set in shared memory (HASH_CAP = 2048).
//   - Distances     : lanes 0..deg-1 each compute one full dot product (serial
//                     over D dims), write to smem, lane 0 reads and updates heaps.
//
// Build:  make -C cpp hnsw_cuda
// Run:    ./cpp/hnsw_cuda data/glove100_1000000 --M 16 --ef_construction 200
//                         --ef 50 --topk 10 --queries 500 --mode cuda_hnsw

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <queue>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

using std::size_t;
using std::uint32_t;

// ---------------------------------------------------------------------------
// Error helpers (mirrors lsh_cuda.cu)
// ---------------------------------------------------------------------------
#define CUDA_OK(call) do {                                                   \
    cudaError_t _e = (call);                                                 \
    if (_e != cudaSuccess) {                                                 \
        std::fprintf(stderr, "cuda: %s @ %s:%d\n",                          \
                     cudaGetErrorString(_e), __FILE__, __LINE__);            \
        std::exit(1);                                                        \
    }                                                                        \
} while (0)

// ---------------------------------------------------------------------------
// Mode
// ---------------------------------------------------------------------------
enum class Mode { cuda_hnsw, cuda_brute };

static const char* mode_name(Mode m) {
    return m == Mode::cuda_hnsw ? "cuda_hnsw" : "cuda_brute";
}
static Mode parse_mode(const std::string& s) {
    if (s == "cuda_hnsw")  return Mode::cuda_hnsw;
    if (s == "cuda_brute") return Mode::cuda_brute;
    std::fprintf(stderr, "unknown --mode: %s\n", s.c_str()); std::exit(1);
}

// ---------------------------------------------------------------------------
// Dataset  (identical to lsh_cuda.cu)
// ---------------------------------------------------------------------------
struct Dataset {
    int n = 0, d = 0;
    std::vector<float> x;
    const float* row(int i) const { return x.data() + (size_t)i * d; }
};

static Dataset load(const std::string& base) {
    Dataset ds;
    std::ifstream sh(base + ".shape");
    if (!sh) { std::fprintf(stderr, "cannot open %s.shape\n", base.c_str()); std::exit(1); }
    sh >> ds.n >> ds.d;
    std::ifstream f(base + ".f32", std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s.f32\n", base.c_str()); std::exit(1); }
    ds.x.resize((size_t)ds.n * ds.d);
    f.read(reinterpret_cast<char*>(ds.x.data()), ds.x.size() * sizeof(float));
    std::fprintf(stderr, "loaded n=%d d=%d\n", ds.n, ds.d);
    return ds;
}

// Synchronized wall clock (forces device sync before reading time).
static double now_sec() {
    cudaDeviceSynchronize();
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static inline float cpu_dot(const float* a, const float* b, int d) {
    float s = 0.f;
    for (int j = 0; j < d; ++j) s += a[j] * b[j];
    return s;
}

// ===========================================================================
// CPU HNSW  (serial build + upper-layer greedy descent)
// ===========================================================================
// Enough of hnsw.cpp to build the index and find layer-0 entry points.
// We keep it self-contained so hnsw_cuda.cu compiles independently.
// ===========================================================================

struct HNSWcpu {
    int M = 16, M0 = 32, ef_construction = 200, d = 0;
    float mL = 0.f;
    int entry_point = -1, max_level = -1;
    std::vector<std::vector<std::vector<int>>> nbrs;
    std::vector<int> node_level;
    const Dataset* ds_ = nullptr;

    // --- level generation ---
    int gen_level(std::mt19937& rng) const {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        double r = u(rng); if (r < 1e-15) r = 1e-15;
        return (int)std::floor(-std::log(r) * mL);
    }

    // --- distance (lower = better on L2-normalised vectors) ---
    float dist_qn(const float* q, int b) const {
        return -cpu_dot(q, ds_->row(b), d);
    }
    float dist_nn(int a, int b) const {
        return -cpu_dot(ds_->row(a), ds_->row(b), d);
    }

    // --- beam search at one layer (ef candidates, returns sorted ascending) ---
    using Pair = std::pair<float,int>;
    using MaxH = std::priority_queue<Pair>;
    using MinH = std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>>;

    std::vector<Pair> search_layer(const float* q, int ep, int ef, int layer,
                                   std::vector<uint32_t>& vis, uint32_t& stamp) const {
        if (++stamp == 0) { std::fill(vis.begin(), vis.end(), 0); stamp = 1; }
        vis[ep] = stamp;
        float d0 = dist_qn(q, ep);
        MaxH W; W.push({d0, ep});
        MinH C; C.push({d0, ep});
        while (!C.empty()) {
            auto [dc, c] = C.top(); C.pop();
            if (dc > W.top().first) break;
            for (int e : nbrs[c][layer]) {
                if (vis[e] == stamp) continue;
                vis[e] = stamp;
                float de = dist_qn(q, e);
                if (de < W.top().first || (int)W.size() < ef) {
                    C.push({de, e}); W.push({de, e});
                    if ((int)W.size() > ef) W.pop();
                }
            }
        }
        std::vector<Pair> res(W.size());
        for (int i = (int)res.size()-1; i >= 0; --i) { res[i] = W.top(); W.pop(); }
        return res;
    }

    // --- prune a neighbor list to at most M_max entries ---
    void prune(int nb, int layer, int M_max) {
        auto& nl = nbrs[nb][layer];
        if ((int)nl.size() <= M_max) return;
        std::vector<Pair> s; s.reserve(nl.size());
        for (int x : nl) s.push_back({dist_nn(nb, x), x});
        std::sort(s.begin(), s.end());
        nl.resize(M_max);
        for (int i = 0; i < M_max; ++i) nl[i] = s[i].second;
    }

    // --- insert one node (serial) ---
    void insert(int u, std::vector<uint32_t>& vis, uint32_t& stamp) {
        int lv = node_level[u];
        const float* q = ds_->row(u);
        if (entry_point == -1) { entry_point = u; max_level = lv; return; }

        int ep = entry_point, cur_max = max_level;
        for (int lc = cur_max; lc > lv; --lc) {
            auto W = search_layer(q, ep, 1, lc, vis, stamp);
            ep = W[0].second;
        }
        for (int lc = std::min(lv, cur_max); lc >= 0; --lc) {
            int M_max = (lc == 0) ? M0 : M;
            auto W = search_layer(q, ep, ef_construction, lc, vis, stamp);
            ep = W[0].second;
            int nn = std::min((int)W.size(), M_max);
            nbrs[u][lc].resize(nn);
            for (int i = 0; i < nn; ++i) nbrs[u][lc][i] = W[i].second;
            for (int i = 0; i < nn; ++i) {
                int nb = W[i].second;
                nbrs[nb][lc].push_back(u);
                if ((int)nbrs[nb][lc].size() > M_max) prune(nb, lc, M_max);
            }
        }
        if (lv > cur_max) { entry_point = u; max_level = lv; }
    }

    // --- build ---
    void build(const Dataset& dataset, int M_, int ef_construction_, uint32_t seed) {
        ds_ = &dataset;
        M = M_; M0 = 2*M; ef_construction = ef_construction_;
        d = dataset.d; mL = 1.f / std::log((float)M);
        entry_point = -1; max_level = -1;
        const int n = dataset.n;
        nbrs.resize(n); node_level.resize(n);
        std::mt19937 rng(seed);
        for (int i = 0; i < n; ++i) {
            node_level[i] = gen_level(rng);
            nbrs[i].resize(node_level[i] + 1);
        }
        std::vector<uint32_t> vis(n, 0); uint32_t stamp = 0;
        const double t0_b = [](){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
                                   return ts.tv_sec+ts.tv_nsec*1e-9; }();
        for (int i = 0; i < n; ++i) insert(i, vis, stamp);
        struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
        double elapsed = ts.tv_sec + ts.tv_nsec*1e-9 - t0_b;
        std::fprintf(stderr, "cpu build  M=%d ef=%d  %.3fs\n", M, ef_construction, elapsed);
    }

    // --- upper-layer greedy descent to find layer-0 entry point for one query ---
    int layer0_ep(const float* q) const {
        int ep = entry_point;
        for (int lc = max_level; lc > 0; --lc) {
            bool improved = true;
            while (improved) {
                improved = false;
                float ep_d = dist_qn(q, ep);
                for (int nb : nbrs[ep][lc]) {
                    float d = dist_qn(q, nb);
                    if (d < ep_d) { ep_d = d; ep = nb; improved = true; }
                }
            }
        }
        return ep;
    }
};

// ===========================================================================
// GPU kernel constants
// ===========================================================================
// Shared-memory budget per block (one warp = one query):
//   smq[D_MAX]        : 512 × 4  = 2 048 B   (query vector)
//   smNbDist[M_MAX]   :  64 × 4  =   256 B   (neighbor distances per step)
//   smNbIds[M_MAX]    :  64 × 4  =   256 B   (neighbor ids per step)
//   W_dist[EF_MAX]    : 128 × 4  =   512 B   (result set distances)
//   W_ids[EF_MAX]     : 128 × 4  =   512 B   (result set ids)
//   C_dist[C_MAX]     : 256 × 4  = 1 024 B   (candidates)
//   C_ids[C_MAX]      : 256 × 4  = 1 024 B
//   smCtrl[4]         :   4 × 4  =    16 B   (W_sz, C_sz, curr_c, stop)
//   vis_ht[HASH_CAP]  :2048 × 4  = 8 192 B   (visited hash set)
//   ─────────────────────────────────────────
//   Total             :              ≈ 13.8 KB
//   Blocks / SM       : ⌊48 KB / 13.8 KB⌋ = 3   (on 48 KB GPUs)
//                       ⌊128 KB / 13.8 KB⌋ = 9  (on L40S / A100)
// ===========================================================================
static constexpr int BLOCK_DIM = 32;   // one warp per query
static constexpr int D_MAX     = 512;
static constexpr int M_MAX     = 64;
static constexpr int EF_MAX    = 128;  // runtime ef must be ≤ this
static constexpr int C_MAX     = 256;  // candidate queue capacity
static constexpr int HASH_CAP  = 2048; // visited hash (power-of-2)
static constexpr int TOPK_MAX  = 32;

// ===========================================================================
// Device helper functions  (called by lane 0 only, no __syncwarp needed)
// ===========================================================================

// Negative dot product: smq in shared memory, v in global memory.
// Called by individual lanes (not warp-cooperative).
__device__ __forceinline__
float neg_dot_dev(const float* __restrict__ smq,
                  const float* __restrict__ v, int D) {
    float s = 0.f;
    for (int j = 0; j < D; ++j) s += smq[j] * v[j];
    return -s;
}

// Open-addressed hash set in shared memory.
// cap must be a power of 2; table must be initialised to -1.
// Returns true if id was already present; inserts and returns false otherwise.
// On hash-table full (should not happen with cap=2048, ef≤128), silently
// accepts re-visiting (only harms recall, not correctness / safety).
__device__ __forceinline__
bool hash_visit(int* __restrict__ ht, int id) {
    unsigned h = (unsigned)(id * 2654435761u) & (HASH_CAP - 1u);
    #pragma unroll 8
    for (int probe = 0; probe < HASH_CAP; ++probe) {
        int v = ht[h];
        if (v == id) return true;          // already visited
        if (v == -1) { ht[h] = id; return false; }  // new entry
        h = (h + 1u) & (HASH_CAP - 1u);   // linear probe
    }
    return false;  // table full: treat as not visited (accept revisit)
}

// Insert (d, id) into W sorted ascending by distance.
// W capacity = EF_MAX; if full and d >= W_dist[W_sz-1], do nothing.
__device__ __forceinline__
void push_W(float* __restrict__ W_dist, int* __restrict__ W_ids,
            int& W_sz, int ef, float d, int id) {
    if (W_sz >= ef && d >= W_dist[W_sz - 1]) return;
    // Insertion sort: shift right until correct position.
    int pos = (W_sz < ef) ? W_sz : W_sz - 1;
    while (pos > 0 && W_dist[pos - 1] > d) {
        W_dist[pos] = W_dist[pos - 1];
        W_ids[pos]  = W_ids[pos - 1];
        --pos;
    }
    W_dist[pos] = d;
    W_ids[pos]  = id;
    if (W_sz < ef) ++W_sz;
}

// Append (d, id) to the unsorted candidate list C.
__device__ __forceinline__
void push_C(float* __restrict__ C_dist, int* __restrict__ C_ids,
            int& C_sz, float d, int id) {
    if (C_sz < C_MAX) { C_dist[C_sz] = d; C_ids[C_sz] = id; ++C_sz; }
}

// Extract the minimum-distance element from C (linear scan + swap-with-last).
__device__ __forceinline__
int pop_C_min(float* __restrict__ C_dist, int* __restrict__ C_ids,
              int& C_sz, float& out_d) {
    int mi = 0;
    for (int i = 1; i < C_sz; ++i)
        if (C_dist[i] < C_dist[mi]) mi = i;
    int   id = C_ids[mi];  out_d = C_dist[mi];
    --C_sz;
    C_dist[mi] = C_dist[C_sz];
    C_ids[mi]  = C_ids[C_sz];
    return id;
}

// ===========================================================================
// HNSW layer-0 beam-search kernel
//   gridDim.x  = Q (one block per query)
//   blockDim.x = 32 (one warp per block)
// ===========================================================================
__global__ void hnsw_layer0_kernel(
    const int* __restrict__ d_nbrs0,     // [N * M0] padded neighbour ids
    const int* __restrict__ d_deg0,      // [N]      actual degree at layer 0
    int N, int D, int M0,
    const int* __restrict__ d_ep0,       // [Q]  layer-0 entry point per query
    const float* __restrict__ d_data,    // [N * D]
    const float* __restrict__ d_queries, // [Q * D]
    int Q, int ef, int topk,
    int* __restrict__ d_out              // [Q * topk]
) {
    const int qid  = blockIdx.x;
    const int lane = threadIdx.x;
    if (qid >= Q) return;

    // ---- shared memory layout ----
    __shared__ float smq      [D_MAX];    // query vector
    __shared__ float smNbDist [M_MAX];    // per-step: neighbour distances
    __shared__ int   smNbIds  [M_MAX];    // per-step: neighbour ids
    __shared__ int   smCtrl   [4];        // [0]=W_sz [1]=C_sz [2]=curr_c [3]=stop
    __shared__ float W_dist   [EF_MAX];   // result set (sorted ascending)
    __shared__ int   W_ids    [EF_MAX];
    __shared__ float C_dist   [C_MAX];    // candidate queue (unsorted)
    __shared__ int   C_ids    [C_MAX];
    __shared__ int   vis_ht   [HASH_CAP]; // visited hash set

    // ---- initialise shared memory (all lanes cooperate) ----
    for (int j = lane; j < D;        j += 32) smq[j]    = d_queries[qid * D + j];
    for (int j = lane; j < HASH_CAP; j += 32) vis_ht[j] = -1;
    __syncwarp();

    if (lane == 0) {
        smCtrl[0] = smCtrl[1] = smCtrl[3] = 0;  // W_sz=0, C_sz=0, stop=0
        int ep = d_ep0[qid];
        float dep = neg_dot_dev(smq, d_data + (size_t)ep * D, D);
        push_W(W_dist, W_ids, smCtrl[0], ef, dep, ep);
        push_C(C_dist, C_ids, smCtrl[1], dep, ep);
        hash_visit(vis_ht, ep);
    }
    __syncwarp();

    // ---- main beam-search loop ----
    while (!smCtrl[3]) {
        // Lane 0: pop best candidate, check stopping condition.
        if (lane == 0) {
            if (smCtrl[1] == 0) {
                smCtrl[3] = 1;  // C empty → stop
            } else {
                float dc;
                smCtrl[2] = pop_C_min(C_dist, C_ids, smCtrl[1], dc);
                float worst_W = (smCtrl[0] >= ef) ? W_dist[smCtrl[0] - 1] : 1e30f;
                if (dc > worst_W) smCtrl[3] = 1;  // pruning condition → stop
            }
        }
        __syncwarp();
        if (smCtrl[3]) break;

        int c   = smCtrl[2];
        int deg = d_deg0[c];

        // All 32 lanes compute distances to c's neighbours in batches of 32.
        // Lane k → neighbour index (base + k).
        for (int base = 0; base < deg; base += 32) {
            int k  = base + lane;
            int nb = (k < deg) ? d_nbrs0[(size_t)c * M0 + k] : -1;
            float di = (nb >= 0) ? neg_dot_dev(smq, d_data + (size_t)nb * D, D)
                                  : 1e30f;
            smNbDist[lane] = di;
            smNbIds [lane] = nb;
            __syncwarp();

            // Lane 0 only: update heaps from smNbDist / smNbIds.
            if (lane == 0) {
                int n_this   = min(32, deg - base);
                float worst_W = (smCtrl[0] >= ef) ? W_dist[smCtrl[0]-1] : 1e30f;
                for (int i = 0; i < n_this; ++i) {
                    int   nbi = smNbIds[i];
                    float ddi = smNbDist[i];
                    if (hash_visit(vis_ht, nbi)) continue;
                    if (ddi < worst_W || smCtrl[0] < ef) {
                        push_W(W_dist, W_ids, smCtrl[0], ef, ddi, nbi);
                        push_C(C_dist, C_ids, smCtrl[1], ddi, nbi);
                        worst_W = (smCtrl[0] >= ef) ? W_dist[smCtrl[0]-1] : 1e30f;
                    }
                }
            }
            __syncwarp();
        }
    }

    // ---- write topk results ----
    if (lane == 0) {
        int k = min(topk, smCtrl[0]);
        for (int i = 0; i < k; ++i)
            d_out[(size_t)qid * topk + i] = W_ids[i];
    }
}

// ===========================================================================
// GPU brute-force kernel (identical pattern to lsh_cuda.cu)
// ===========================================================================
template <int TK>
__global__ void brute_topk_kernel(const float* __restrict__ d_points,
                                   const float* __restrict__ d_queries,
                                   int N, int D, int Q, int topk,
                                   int* __restrict__ d_out,
                                   float* __restrict__ d_scores) {
    int q = blockIdx.x;
    if (q >= Q) return;

    extern __shared__ float smem[];
    float* qvec = smem;             // [D]
    float* topv = smem + D;         // [TK]
    int*   topi = (int*)(topv + TK);// [TK]

    for (int j = threadIdx.x; j < D; j += blockDim.x)
        qvec[j] = d_queries[(size_t)q * D + j];
    if (threadIdx.x < topk) { topv[threadIdx.x] = -1e30f; topi[threadIdx.x] = -1; }
    __syncthreads();

    float my_v[TK]; int my_i[TK];
    for (int t = 0; t < TK; ++t) { my_v[t] = -1e30f; my_i[t] = -1; }

    for (int id = threadIdx.x; id < N; id += blockDim.x) {
        const float* p = d_points + (size_t)id * D;
        float dotv = 0.f;
        for (int j = 0; j < D; ++j) dotv += p[j] * qvec[j];
        int w = 0; float wv = my_v[0];
        for (int t = 1; t < topk; ++t) if (my_v[t] < wv) { wv = my_v[t]; w = t; }
        if (dotv > wv) { my_v[w] = dotv; my_i[w] = id; }
    }

    for (int round = 0; round < (int)blockDim.x; ++round) {
        if ((int)threadIdx.x == round) {
            for (int t = 0; t < topk; ++t) {
                int id = my_i[t]; if (id < 0) continue;
                bool dup = false;
                for (int u = 0; u < topk; ++u) if (topi[u] == id) { dup = true; break; }
                if (dup) continue;
                float v = my_v[t];
                int w = 0; float wv = topv[0];
                for (int u = 1; u < topk; ++u) if (topv[u] < wv) { wv = topv[u]; w = u; }
                if (v > wv) { topv[w] = v; topi[w] = id; }
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        for (int i = 1; i < topk; ++i) {
            float v = topv[i]; int id = topi[i]; int j = i-1;
            while (j >= 0 && topv[j] < v) { topv[j+1]=topv[j]; topi[j+1]=topi[j]; --j; }
            topv[j+1] = v; topi[j+1] = id;
        }
        for (int i = 0; i < topk; ++i) {
            d_out[(size_t)q * topk + i]    = topi[i];
            d_scores[(size_t)q * topk + i] = topv[i];
        }
    }
}

// ===========================================================================
// Args + main
// ===========================================================================
struct Args {
    std::string base;
    int      M               = 16;
    int      ef_construction = 200;
    int      ef              = 50;
    int      topk            = 10;
    int      queries         = 500;
    uint32_t seed            = 1;
    Mode     mode            = Mode::cuda_hnsw;
};

static Args parse(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: %s <data_base> [--M 16] [--ef_construction 200] [--ef 50]\n"
            "                       [--topk 10] [--queries 500] [--seed 1]\n"
            "                       [--mode cuda_hnsw|cuda_brute]\n", argv[0]);
        std::exit(1);
    }
    Args a; a.base = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string k = argv[i];
        auto nxt = [&]{ return std::string(argv[++i]); };
        if      (k=="--M")               a.M               = std::stoi(nxt());
        else if (k=="--ef_construction") a.ef_construction = std::stoi(nxt());
        else if (k=="--ef")              a.ef              = std::stoi(nxt());
        else if (k=="--topk")            a.topk            = std::stoi(nxt());
        else if (k=="--queries")         a.queries         = std::stoi(nxt());
        else if (k=="--seed")            a.seed            = (uint32_t)std::stoul(nxt());
        else if (k=="--mode")            a.mode            = parse_mode(nxt());
        else { std::fprintf(stderr,"unknown arg: %s\n", k.c_str()); std::exit(1); }
    }
    return a;
}

int main(int argc, char** argv) {
    Args a = parse(argc, argv);
    if (a.topk > TOPK_MAX) {
        std::fprintf(stderr, "topk > %d not supported\n", TOPK_MAX); return 1;
    }
    if (a.ef > EF_MAX) {
        std::fprintf(stderr, "ef > %d; clamping to %d\n", EF_MAX, EF_MAX);
        a.ef = EF_MAX;
    }

    cudaSetDevice(0);
    Dataset ds = load(a.base);
    const int N = ds.n, D = ds.d, Q = a.queries, TK = a.topk;

    // ---- Upload dataset to GPU ----
    float* d_data = nullptr;
    CUDA_OK(cudaMalloc(&d_data, sizeof(float) * (size_t)N * D));
    CUDA_OK(cudaMemcpy(d_data, ds.x.data(), sizeof(float)*(size_t)N*D, cudaMemcpyHostToDevice));

    // ---- Select query vectors ----
    std::mt19937 rng(a.seed ^ 0xC0FFEEu);
    std::uniform_int_distribution<int> uid(0, N-1);
    std::vector<int> q_ids(Q);
    for (auto& x : q_ids) x = uid(rng);

    std::vector<float> q_host((size_t)Q * D);
    for (int i = 0; i < Q; ++i)
        std::memcpy(q_host.data() + (size_t)i*D, ds.x.data() + (size_t)q_ids[i]*D, sizeof(float)*D);

    float* d_queries = nullptr;
    CUDA_OK(cudaMalloc(&d_queries, sizeof(float)*(size_t)Q*D));
    CUDA_OK(cudaMemcpy(d_queries, q_host.data(), sizeof(float)*(size_t)Q*D, cudaMemcpyHostToDevice));

    // ---- GPU brute-force ground truth ----
    int*   d_gt_ids    = nullptr;
    float* d_gt_scores = nullptr;
    CUDA_OK(cudaMalloc(&d_gt_ids,    sizeof(int)  *(size_t)Q*TK));
    CUDA_OK(cudaMalloc(&d_gt_scores, sizeof(float)*(size_t)Q*TK));
    {
        int block = 256;
        size_t smem = sizeof(float)*(D + TOPK_MAX) + sizeof(int)*TOPK_MAX;
        brute_topk_kernel<TOPK_MAX><<<Q, block, smem>>>(
            d_data, d_queries, N, D, Q, TK, d_gt_ids, d_gt_scores);
        CUDA_OK(cudaGetLastError());
    }
    // Brute timing: re-run once after warm-up
    double t0 = now_sec();
    {
        int block = 256;
        size_t smem = sizeof(float)*(D + TOPK_MAX) + sizeof(int)*TOPK_MAX;
        brute_topk_kernel<TOPK_MAX><<<Q, block, smem>>>(
            d_data, d_queries, N, D, Q, TK, d_gt_ids, d_gt_scores);
        CUDA_OK(cudaGetLastError());
    }
    const double t_brute = now_sec() - t0;

    // ---- HNSW ----
    double t_hnsw = 0.0, t_build = 0.0;
    int* d_out_ids = nullptr;
    std::vector<int> out_ids;

    if (a.mode == Mode::cuda_hnsw) {
        // --- CPU build ---
        t0 = now_sec();
        HNSWcpu hnsw;
        hnsw.build(ds, a.M, a.ef_construction, a.seed);
        t_build = now_sec() - t0;

        // --- Upper-layer descent on CPU to find layer-0 entry points ---
        std::vector<int> ep0(Q);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < Q; ++i)
            ep0[i] = hnsw.layer0_ep(q_host.data() + (size_t)i * D);

        // --- Build GPU layer-0 graph ---
        const int M0 = hnsw.M0;
        std::vector<int> h_nbrs0((size_t)N * M0, -1);
        std::vector<int> h_deg0(N, 0);
        for (int i = 0; i < N; ++i) {
            int deg = (int)hnsw.nbrs[i][0].size();
            h_deg0[i] = deg;
            for (int k = 0; k < deg; ++k)
                h_nbrs0[(size_t)i * M0 + k] = hnsw.nbrs[i][0][k];
        }

        int* d_nbrs0 = nullptr; int* d_deg0 = nullptr; int* d_ep0 = nullptr;
        CUDA_OK(cudaMalloc(&d_nbrs0, sizeof(int)*(size_t)N*M0));
        CUDA_OK(cudaMalloc(&d_deg0,  sizeof(int)*(size_t)N));
        CUDA_OK(cudaMalloc(&d_ep0,   sizeof(int)*(size_t)Q));
        CUDA_OK(cudaMemcpy(d_nbrs0, h_nbrs0.data(), sizeof(int)*(size_t)N*M0, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_deg0,  h_deg0.data(),  sizeof(int)*(size_t)N,    cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_ep0,   ep0.data(),      sizeof(int)*(size_t)Q,    cudaMemcpyHostToDevice));

        CUDA_OK(cudaMalloc(&d_out_ids, sizeof(int)*(size_t)Q*TK));

        // --- GPU query ---
        t0 = now_sec();
        hnsw_layer0_kernel<<<Q, BLOCK_DIM>>>(
            d_nbrs0, d_deg0, N, D, M0, d_ep0,
            d_data, d_queries, Q, a.ef, TK, d_out_ids);
        CUDA_OK(cudaGetLastError());
        t_hnsw = now_sec() - t0;

        out_ids.resize((size_t)Q * TK);
        CUDA_OK(cudaMemcpy(out_ids.data(), d_out_ids,
                           sizeof(int)*(size_t)Q*TK, cudaMemcpyDeviceToHost));

        CUDA_OK(cudaFree(d_nbrs0)); CUDA_OK(cudaFree(d_deg0)); CUDA_OK(cudaFree(d_ep0));
        CUDA_OK(cudaFree(d_out_ids));
    }

    // ---- Ground truth → host ----
    std::vector<int> gt_ids((size_t)Q * TK);
    CUDA_OK(cudaMemcpy(gt_ids.data(), d_gt_ids,
                       sizeof(int)*(size_t)Q*TK, cudaMemcpyDeviceToHost));

    // ---- Recall ----
    double recall = 1.0;  // brute mode: exact by definition
    if (a.mode == Mode::cuda_hnsw) {
        recall = 0.0;
        for (int q = 0; q < Q; ++q) {
            std::vector<int> g(gt_ids.begin() + (size_t)q*TK,
                               gt_ids.begin() + (size_t)(q+1)*TK);
            std::sort(g.begin(), g.end());
            int hit = 0;
            for (int i = 0; i < TK; ++i) {
                int id = out_ids[(size_t)q*TK + i];
                if (id >= 0 && std::binary_search(g.begin(), g.end(), id)) ++hit;
            }
            recall += (double)hit / TK;
        }
        recall /= Q;
    }

    // ---- Output (same format as hnsw.cpp for bench_hnsw_cuda.sh) ----
    std::printf("mode=%s  N=%d D=%d  M=%d ef_construction=%d ef=%d topk=%d queries=%d\n",
                mode_name(a.mode), N, D, a.M, a.ef_construction, a.ef, TK, Q);
    std::printf("brute     : %.3f ms  (%.1f us/query)\n",
                t_brute*1e3, t_brute*1e6/Q);
    if (a.mode == Mode::cuda_hnsw) {
        std::printf("build     : %.3f ms\n", t_build*1e3);
        std::printf("hnsw      : %.3f ms  (%.1f us/query)  speedup x%.2f\n",
                    t_hnsw*1e3, t_hnsw*1e6/Q, t_brute/t_hnsw);
        std::printf("recall@%d  : %.4f\n", TK, recall);
    } else {
        std::printf("hnsw      : -- (brute mode)  speedup x1.00\n");
        std::printf("recall@%d  : 1.0000\n", TK);
    }

    CUDA_OK(cudaFree(d_data)); CUDA_OK(cudaFree(d_queries));
    CUDA_OK(cudaFree(d_gt_ids)); CUDA_OK(cudaFree(d_gt_scores));
    return 0;
}
