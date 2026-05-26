// CUDA KD-tree batch k-nearest-neighbour search.
// One GPU thread per query — same batch-level parallelism as the OpenMP serial mode,
// but running thousands of queries simultaneously on the GPU.
//
// Each thread carries:
//   - an explicit traversal stack of depth STACK_DEPTH in registers
//   - a fixed-size max-heap of size k in registers (k <= TOPK_MAX)
//
// Build: make kd_tree_cuda
// Run:   ./cpp/kd_tree_cuda data/glove25_400000 --k 10 --queries 200

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_OK(call) do { cudaError_t _e = (call);                          \
    if (_e != cudaSuccess) {                                                  \
        std::fprintf(stderr, "cuda: %s @ %s:%d\n",                           \
            cudaGetErrorString(_e), __FILE__, __LINE__); std::exit(1); } } while(0)

static constexpr int STACK_DEPTH = 64;   // > 2 * log2(400k) = 38
static constexpr int TOPK_MAX    = 32;

// ---------------------------------------------------------------------------
// Dataset
// ---------------------------------------------------------------------------

struct Dataset {
    int n = 0, d = 0;
    std::vector<float> x;
    const float* row(int i) const { return x.data() + (size_t)i * d; }
};

static Dataset load(const std::string& base)
{
    Dataset ds;
    std::ifstream sh(base + ".shape");
    if (!sh) { std::fprintf(stderr, "cannot open %s.shape\n", base.c_str()); std::exit(1); }
    sh >> ds.n >> ds.d;
    std::ifstream f(base + ".f32", std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s.f32\n", base.c_str()); std::exit(1); }
    ds.x.resize((size_t)ds.n * ds.d);
    f.read(reinterpret_cast<char*>(ds.x.data()), (std::streamsize)(ds.x.size() * sizeof(float)));
    std::fprintf(stderr, "loaded n=%d d=%d\n", ds.n, ds.d);
    return ds;
}

static double now_sec()
{
    cudaDeviceSynchronize();
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ---------------------------------------------------------------------------
// Flat tree (built on CPU, transferred to GPU)
// ---------------------------------------------------------------------------

struct FlatNode {
    int pointId;
    int axis;
    int left;    // index in flat array, -1 = null
    int right;
};

// Build helpers (reuse nth_element approach from kd_tree.cpp)

static int buildFlat(const Dataset& ds, std::vector<int>& idx,
    int lo, int hi, int depth, std::vector<FlatNode>& nodes)
{
    if (lo >= hi) return -1;
    int axis = depth % ds.d;
    int mid  = lo + (hi - lo) / 2;
    std::nth_element(idx.begin() + lo, idx.begin() + mid, idx.begin() + hi,
        [&](int a, int b){ return ds.row(a)[axis] < ds.row(b)[axis]; });
    int nodeIdx = (int)nodes.size();
    nodes.push_back({idx[mid], axis, -1, -1});
    nodes[nodeIdx].left  = buildFlat(ds, idx, lo,    mid,  depth + 1, nodes);
    nodes[nodeIdx].right = buildFlat(ds, idx, mid + 1, hi, depth + 1, nodes);
    return nodeIdx;
}

// ---------------------------------------------------------------------------
// GPU brute-force baseline (dot-product on normalised vectors)
// ---------------------------------------------------------------------------

__global__ void bruteKernel(const float* __restrict__ pts, const float* __restrict__ qs,
    int N, int D, int K, int* outIds, float* outDists)
{
    int qi = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= gridDim.x * blockDim.x) return;

    const float* q = qs + (size_t)qi * D;

    // fixed heap: max-heap on dist (small array, lives in registers for small K)
    float heapDist[TOPK_MAX];
    int   heapId  [TOPK_MAX];
    int   heapSz = 0;

    for (int j = 0; j < N; ++j) {
        const float* p = pts + (size_t)j * D;
        float d2 = 0.0f;
        for (int dim = 0; dim < D; ++dim) { float diff = q[dim] - p[dim]; d2 += diff * diff; }

        if (heapSz < K) {
            heapDist[heapSz] = d2; heapId[heapSz] = j; ++heapSz;
            // sift up
            int i = heapSz - 1;
            while (i > 0) {
                int parent = (i - 1) / 2;
                if (heapDist[parent] < heapDist[i]) {
                    float td = heapDist[parent]; heapDist[parent] = heapDist[i]; heapDist[i] = td;
                    int   ti = heapId  [parent]; heapId  [parent] = heapId  [i]; heapId  [i] = ti;
                    i = parent;
                } else break;
            }
        } else if (d2 < heapDist[0]) {
            heapDist[0] = d2; heapId[0] = j;
            // sift down
            int i = 0;
            while (true) {
                int l = 2*i+1, r = 2*i+2, largest = i;
                if (l < K && heapDist[l] > heapDist[largest]) largest = l;
                if (r < K && heapDist[r] > heapDist[largest]) largest = r;
                if (largest == i) break;
                float td = heapDist[i]; heapDist[i] = heapDist[largest]; heapDist[largest] = td;
                int   ti = heapId  [i]; heapId  [i] = heapId  [largest]; heapId  [largest] = ti;
                i = largest;
            }
        }
    }

    for (int i = 0; i < heapSz; ++i) {
        outDists[qi * K + i] = sqrtf(heapDist[i]);
        outIds  [qi * K + i] = heapId[i];
    }
}

// ---------------------------------------------------------------------------
// GPU KD-tree query kernel — one thread per query
// ---------------------------------------------------------------------------

__global__ void kdQueryKernel(const FlatNode* __restrict__ tree,
    const float* __restrict__ pts, const float* __restrict__ qs,
    int D, int K, int nQueries, int* outIds, float* outDists)
{
    int qi = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= nQueries) return;

    const float* q = qs + (size_t)qi * D;

    float heapDist[TOPK_MAX];
    int   heapId  [TOPK_MAX];
    int   heapSz = 0;

    // explicit stack: (nodeIdx, splitDiff²) pairs
    // splitDiff² stored so we can check pruning on the far child after visiting near
    struct Frame { int node; };
    Frame stack[STACK_DEPTH];
    int   stackTop = 0;

    stack[stackTop++] = {0};  // root is always index 0

    while (stackTop > 0) {
        int nodeIdx = stack[--stackTop].node;
        if (nodeIdx < 0) continue;

        const FlatNode& node = tree[nodeIdx];
        const float* p = pts + (size_t)node.pointId * D;

        float d2 = 0.0f;
        for (int dim = 0; dim < D; ++dim) { float diff = q[dim] - p[dim]; d2 += diff * diff; }

        if (heapSz < K) {
            heapDist[heapSz] = d2; heapId[heapSz] = node.pointId; ++heapSz;
            int i = heapSz - 1;
            while (i > 0) {
                int par = (i-1)/2;
                if (heapDist[par] < heapDist[i]) {
                    float td = heapDist[par]; heapDist[par] = heapDist[i]; heapDist[i] = td;
                    int   ti = heapId  [par]; heapId  [par] = heapId  [i]; heapId  [i] = ti;
                    i = par;
                } else break;
            }
        } else if (d2 < heapDist[0]) {
            heapDist[0] = d2; heapId[0] = node.pointId;
            int i = 0;
            while (true) {
                int l = 2*i+1, r = 2*i+2, lg = i;
                if (l < K && heapDist[l] > heapDist[lg]) lg = l;
                if (r < K && heapDist[r] > heapDist[lg]) lg = r;
                if (lg == i) break;
                float td = heapDist[i]; heapDist[i] = heapDist[lg]; heapDist[lg] = td;
                int   ti = heapId  [i]; heapId  [i] = heapId  [lg]; heapId  [lg] = ti;
                i = lg;
            }
        }

        float diff = q[node.axis] - p[node.axis];
        int near = diff <= 0 ? node.left  : node.right;
        int far  = diff <= 0 ? node.right : node.left;

        // push far first (visited last) — only if pruning bound allows
        if (far >= 0 && (heapSz < K || diff * diff < heapDist[0]))
            stack[stackTop++] = {far};

        if (near >= 0)
            stack[stackTop++] = {near};
    }

    for (int i = 0; i < heapSz; ++i) {
        outDists[qi * K + i] = sqrtf(heapDist[i]);
        outIds  [qi * K + i] = heapId[i];
    }
}

// ---------------------------------------------------------------------------
// Query generation
// ---------------------------------------------------------------------------

static std::vector<float> generateQueries(const Dataset& ds, int iters,
    float noise, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int>  idxDist(0, ds.n - 1);
    std::normal_distribution<float>     noiseDist(0.0f, noise);
    std::vector<float> queries((size_t)iters * ds.d);
    for (int i = 0; i < iters; ++i) {
        const float* src = ds.row(idxDist(rng));
        float* dst = queries.data() + (size_t)i * ds.d;
        for (int dim = 0; dim < ds.d; ++dim)
            dst[dim] = src[dim] + noiseDist(rng);
    }
    return queries;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::fprintf(stderr,
            "Usage: %s <data_base> [--k 10] [--queries 200] [--noise 0.01] [--seed 42]\n",
            argv[0]);
        return 1;
    }
    std::string base = argv[1];
    int      K       = 10;
    int      Q       = 200;
    float    noise   = 0.01f;
    uint64_t seed    = 42;

    for (int i = 2; i < argc; ++i) {
        std::string key = argv[i];
        auto next = [&]{ return std::string(argv[++i]); };
        if      (key == "--k")       K    = std::stoi(next());
        else if (key == "--queries") Q    = std::stoi(next());
        else if (key == "--noise")   noise = std::stof(next());
        else if (key == "--seed")    seed  = std::stoull(next());
        else { std::fprintf(stderr, "unknown arg: %s\n", key.c_str()); return 1; }
    }

    if (K > TOPK_MAX) {
        std::fprintf(stderr, "--k must be <= %d\n", TOPK_MAX); return 1;
    }

    Dataset ds = load(base);
    int N = ds.n, D = ds.d;

    // Build flat tree on CPU
    double t0 = now_sec();
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::vector<FlatNode> nodes;
    nodes.reserve(N);
    buildFlat(ds, idx, 0, N, 0, nodes);
    double buildMs = (now_sec() - t0) * 1e3;

    // Generate queries
    std::vector<float> queries = generateQueries(ds, Q, noise, seed);

    // Transfer to GPU
    float    *d_pts, *d_qs;
    FlatNode *d_tree;
    int      *d_out_ids,    *d_gt_ids;
    float    *d_out_dists,  *d_gt_dists;

    CUDA_OK(cudaMalloc(&d_pts,       sizeof(float)    * (size_t)N * D));
    CUDA_OK(cudaMalloc(&d_qs,        sizeof(float)    * (size_t)Q * D));
    CUDA_OK(cudaMalloc(&d_tree,      sizeof(FlatNode) * nodes.size()));
    CUDA_OK(cudaMalloc(&d_out_ids,   sizeof(int)      * (size_t)Q * K));
    CUDA_OK(cudaMalloc(&d_out_dists, sizeof(float)    * (size_t)Q * K));
    CUDA_OK(cudaMalloc(&d_gt_ids,    sizeof(int)      * (size_t)Q * K));
    CUDA_OK(cudaMalloc(&d_gt_dists,  sizeof(float)    * (size_t)Q * K));

    CUDA_OK(cudaMemcpy(d_pts,   ds.x.data(),      sizeof(float)    * (size_t)N * D, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_qs,    queries.data(),   sizeof(float)    * (size_t)Q * D, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_tree,  nodes.data(),     sizeof(FlatNode) * nodes.size(),  cudaMemcpyHostToDevice));

    int block = 128;

    // Brute-force baseline
    t0 = now_sec();
    bruteKernel<<<(Q + block - 1) / block, block>>>(d_pts, d_qs, N, D, K, d_gt_ids, d_gt_dists);
    CUDA_OK(cudaGetLastError());
    double bruteMs = (now_sec() - t0) * 1e3;

    // KD-tree query
    t0 = now_sec();
    kdQueryKernel<<<(Q + block - 1) / block, block>>>(d_tree, d_pts, d_qs, D, K, Q, d_out_ids, d_out_dists);
    CUDA_OK(cudaGetLastError());
    double queryMs = (now_sec() - t0) * 1e3;

    // Recall@K vs GPU brute-force
    std::vector<int>   gt_ids((size_t)Q * K),  out_ids((size_t)Q * K);
    std::vector<float> gt_dists((size_t)Q * K);
    CUDA_OK(cudaMemcpy(gt_ids.data(),   d_gt_ids,    sizeof(int)   * (size_t)Q * K, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(out_ids.data(),  d_out_ids,   sizeof(int)   * (size_t)Q * K, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(gt_dists.data(), d_gt_dists,  sizeof(float) * (size_t)Q * K, cudaMemcpyDeviceToHost));

    double recall = 0.0;
    for (int q = 0; q < Q; ++q) {
        std::vector<int> g(gt_ids.begin() + q*K, gt_ids.begin() + (q+1)*K);
        std::sort(g.begin(), g.end());
        int hit = 0;
        for (int i = 0; i < K; ++i)
            if (std::binary_search(g.begin(), g.end(), out_ids[q*K + i])) ++hit;
        recall += (double)hit / K;
    }
    recall /= Q;

    std::printf("N=%d D=%d  k=%d queries=%d\n", N, D, K, Q);
    std::printf("brute  : %.3f ms  (%.1f us/query)\n",
        bruteMs, bruteMs * 1e3 / Q);
    std::printf("build  : %.3f ms\n", buildMs);
    std::printf("query  : %.3f ms  (%.1f us/query)  speedup x%.2f\n",
        queryMs, queryMs * 1e3 / Q, bruteMs / queryMs);
    std::printf("recall@%d : %.4f\n", K, recall);

    cudaFree(d_pts); cudaFree(d_qs); cudaFree(d_tree);
    cudaFree(d_out_ids); cudaFree(d_out_dists);
    cudaFree(d_gt_ids);  cudaFree(d_gt_dists);
    return 0;
}
