// KD-tree k-nearest-neighbour search with multiple OpenMP parallelization modes.
//
// Build modes:
//   serial           single-threaded build (default)
//   parallel-tasks   omp task tree, depth cutoff = floor(log2(P))
//   parallel-flat    serial top phase + omp parallel for over subtrees
//
// Query modes:
//   serial           batch parallel for over queries, serial traversal per query (default)
//   local-heaps      parallel traversal per query, thread-local heaps, merge
//   atomic-global    parallel traversal per query, shared atomic worst bound
//
// Build:  make kd_tree
// Run:    ./cpp/kd_tree data/glove50_400000 [options]

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include <omp.h>

// ---------------------------------------------------------------------------
// Dataset
// ---------------------------------------------------------------------------

struct Dataset {
    int n = 0, d = 0;
    std::vector<float> x;
    const float* row(int i) const { return x.data() + (size_t)i * d; }
};

static Dataset load(const std::string &base)
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

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

using Point  = std::vector<float>;
using Points = std::vector<Point>;

struct Neighbour {
    float  dist{0.0f};
    size_t idx{0};
    bool operator<(const Neighbour &o) const { return dist < o.dist; }
};

using Neighbours = std::vector<Neighbour>;

struct QueryResult {
    Neighbours neighbours;
    size_t     nodesVisited{0};
};

using BatchQueryResult = std::vector<QueryResult>;

// ---------------------------------------------------------------------------
// Policy enums
// ---------------------------------------------------------------------------

enum class BuildMode { Serial, ParallelTasks, ParallelFlat };
enum class QueryMode { Serial, LocalHeaps, AtomicGlobal };

static BuildMode parseBuildMode(const std::string &s)
{
    if (s == "serial")         return BuildMode::Serial;
    if (s == "parallel-tasks") return BuildMode::ParallelTasks;
    if (s == "parallel-flat")  return BuildMode::ParallelFlat;
    std::fprintf(stderr, "unknown --build-mode: %s\n", s.c_str()); std::exit(1);
}

static QueryMode parseQueryMode(const std::string &s)
{
    if (s == "serial")        return QueryMode::Serial;
    if (s == "local-heaps")   return QueryMode::LocalHeaps;
    if (s == "atomic-global") return QueryMode::AtomicGlobal;
    std::fprintf(stderr, "unknown --query-mode: %s\n", s.c_str()); std::exit(1);
}

// ---------------------------------------------------------------------------
// KDTree
// ---------------------------------------------------------------------------

class KDTree {
public:
    explicit KDTree(BuildMode aBuild = BuildMode::Serial,
                    QueryMode aQuery = QueryMode::Serial)
        : buildMode(aBuild), queryMode(aQuery) {}

    KDTree(const KDTree &)            = delete;
    KDTree &operator=(const KDTree &) = delete;

    void build(const Points &aPoints)
    {
        points = &aPoints;
        std::vector<size_t> indices(aPoints.size());
        std::iota(indices.begin(), indices.end(), 0);
        int cutoff = Detail::taskCutoff();

        if (buildMode == BuildMode::ParallelFlat) {
            WorkList work;
            work.reserve(size_t(2) << cutoff);
            root = Detail::buildTop(aPoints, indices, 0, indices.size(), 0, cutoff, work);
            int n = (int)work.size();
#pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < n; ++i) {
                auto &[slot, lo, hi, depth] = work[i];
                *slot = Detail::buildSerial(aPoints, indices, lo, hi, depth);
            }
        } else if (buildMode == BuildMode::ParallelTasks) {
#pragma omp parallel
#pragma omp single
            root = Detail::buildTasks(aPoints, indices, 0, indices.size(), 0, cutoff);
        } else {
            root = Detail::buildSerial(aPoints, indices, 0, indices.size(), 0);
        }
    }

    BatchQueryResult query(const Points &aQueries, size_t aK) const
    {
        BatchQueryResult results(aQueries.size());
        if (queryMode == QueryMode::LocalHeaps) {
            for (size_t i = 0; i < aQueries.size(); ++i)
                results[i] = Detail::queryLocalHeaps(*points, root.get(), aQueries[i], aK);
        } else if (queryMode == QueryMode::AtomicGlobal) {
            for (size_t i = 0; i < aQueries.size(); ++i)
                results[i] = Detail::queryAtomicGlobal(*points, root.get(), aQueries[i], aK);
        } else {
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < aQueries.size(); ++i)
                results[i] = Detail::querySingle(*points, root.get(), aQueries[i], aK);
        }
        return results;
    }

private:
    struct Node {
        size_t pointId{0};
        int    axis{0};
        std::unique_ptr<Node> left, right;
    };

    using Heap     = std::priority_queue<Neighbour>;
    using WorkList = std::vector<std::tuple<std::unique_ptr<Node>*, size_t, size_t, int>>;

    struct Detail {

        static int taskCutoff()
        {
            int c = 0;
            for (int t = omp_get_max_threads(); (1 << c) < t; ++c);
            return c;
        }

        static float l2sq(const Point &a, const Point &b)
        {
            float s = 0.0f;
            for (size_t i = 0; i < a.size(); ++i) { float d = a[i]-b[i]; s += d*d; }
            return s;
        }

        static std::unique_ptr<Node> buildSerial(const Points &pts,
            std::vector<size_t> &idx, size_t lo, size_t hi, int depth)
        {
            if (lo >= hi) return nullptr;
            int    axis = depth % (int)pts[0].size();
            size_t mid  = lo + (hi - lo) / 2;
            std::nth_element(idx.begin()+lo, idx.begin()+mid, idx.begin()+hi,
                [&](size_t a, size_t b){ return pts[a][axis] < pts[b][axis]; });
            auto node     = std::make_unique<Node>();
            node->pointId = idx[mid];
            node->axis    = axis;
            node->left    = buildSerial(pts, idx, lo,    mid,  depth+1);
            node->right   = buildSerial(pts, idx, mid+1, hi,   depth+1);
            return node;
        }

        static std::unique_ptr<Node> buildTop(const Points &pts,
            std::vector<size_t> &idx, size_t lo, size_t hi, int depth,
            int cutoff, WorkList &work)
        {
            if (lo >= hi) return nullptr;
            int    axis = depth % (int)pts[0].size();
            size_t mid  = lo + (hi - lo) / 2;
            std::nth_element(idx.begin()+lo, idx.begin()+mid, idx.begin()+hi,
                [&](size_t a, size_t b){ return pts[a][axis] < pts[b][axis]; });
            auto node     = std::make_unique<Node>();
            node->pointId = idx[mid];
            node->axis    = axis;
            if (depth + 1 >= cutoff) {
                work.emplace_back(&node->left,  lo,    mid, depth+1);
                work.emplace_back(&node->right, mid+1, hi,  depth+1);
            } else {
                node->left  = buildTop(pts, idx, lo,    mid, depth+1, cutoff, work);
                node->right = buildTop(pts, idx, mid+1, hi,  depth+1, cutoff, work);
            }
            return node;
        }

        static std::unique_ptr<Node> buildTasks(const Points &pts,
            std::vector<size_t> &idx, size_t lo, size_t hi, int depth, int cutoff)
        {
            if (lo >= hi) return nullptr;
            int    axis = depth % (int)pts[0].size();
            size_t mid  = lo + (hi - lo) / 2;
            std::nth_element(idx.begin()+lo, idx.begin()+mid, idx.begin()+hi,
                [&](size_t a, size_t b){ return pts[a][axis] < pts[b][axis]; });
            auto node     = std::make_unique<Node>();
            node->pointId = idx[mid];
            node->axis    = axis;
            if (depth < cutoff) {
                std::unique_ptr<Node> l, r;
#pragma omp task shared(l)
                l = buildTasks(pts, idx, lo,    mid, depth+1, cutoff);
#pragma omp task shared(r)
                r = buildTasks(pts, idx, mid+1, hi,  depth+1, cutoff);
#pragma omp taskwait
                node->left  = std::move(l);
                node->right = std::move(r);
            } else {
                node->left  = buildSerial(pts, idx, lo,    mid, depth+1);
                node->right = buildSerial(pts, idx, mid+1, hi,  depth+1);
            }
            return node;
        }

        static void traverse(const Points &pts, const Node *node,
            const Point &q, size_t k, Heap &heap, size_t &visited)
        {
            if (!node) return;
            ++visited;
            float dist = l2sq(q, pts[node->pointId]);
            if (heap.size() < k)               heap.push({dist, node->pointId});
            else if (dist < heap.top().dist)   { heap.pop(); heap.push({dist, node->pointId}); }
            float diff = q[node->axis] - pts[node->pointId][node->axis];
            const Node *near = diff <= 0 ? node->left.get() : node->right.get();
            const Node *far  = diff <= 0 ? node->right.get() : node->left.get();
            traverse(pts, near, q, k, heap, visited);
            if (heap.size() < k || diff*diff < heap.top().dist)
                traverse(pts, far, q, k, heap, visited);
        }

        static void collectWork(const Points &pts, const Node *node,
            const Point &q, size_t k, int depth, int cutoff,
            Heap &upper, size_t &upperVisited, std::vector<const Node*> &work)
        {
            if (!node) return;
            ++upperVisited;
            float dist = l2sq(q, pts[node->pointId]);
            if (upper.size() < k)              upper.push({dist, node->pointId});
            else if (dist < upper.top().dist)  { upper.pop(); upper.push({dist, node->pointId}); }
            if (depth >= cutoff) { work.push_back(node); return; }
            collectWork(pts, node->left.get(),  q, k, depth+1, cutoff, upper, upperVisited, work);
            collectWork(pts, node->right.get(), q, k, depth+1, cutoff, upper, upperVisited, work);
        }

        static void mergeInto(Heap &dst, Heap &src, size_t k)
        {
            while (!src.empty()) {
                const Neighbour &n = src.top();
                if (dst.size() < k)              dst.push(n);
                else if (n.dist < dst.top().dist){ dst.pop(); dst.push(n); }
                src.pop();
            }
        }

        static QueryResult extractResult(Heap &heap, size_t visited)
        {
            Neighbours nbrs;
            nbrs.reserve(heap.size());
            while (!heap.empty()) {
                auto n = heap.top(); heap.pop();
                n.dist = std::sqrt(n.dist);
                nbrs.push_back(n);
            }
            std::sort(nbrs.begin(), nbrs.end());
            return {std::move(nbrs), visited};
        }

        static void updateGlobalWorst(std::atomic<float> &g, float val)
        {
            float cur = g.load(std::memory_order_relaxed);
            while (val < cur && !g.compare_exchange_weak(cur, val, std::memory_order_relaxed));
        }

        static void traverseAtomic(const Points &pts, const Node *node,
            const Point &q, size_t k, Heap &heap, size_t &visited,
            std::atomic<float> &gw)
        {
            if (!node) return;
            ++visited;
            float dist = l2sq(q, pts[node->pointId]);
            if (heap.size() < k) {
                heap.push({dist, node->pointId});
                if (heap.size() == k) updateGlobalWorst(gw, heap.top().dist);
            } else if (dist < heap.top().dist) {
                heap.pop(); heap.push({dist, node->pointId});
                updateGlobalWorst(gw, heap.top().dist);
            }
            float diff = q[node->axis] - pts[node->pointId][node->axis];
            const Node *near = diff <= 0 ? node->left.get() : node->right.get();
            const Node *far  = diff <= 0 ? node->right.get() : node->left.get();
            traverseAtomic(pts, near, q, k, heap, visited, gw);
            float bound = std::min(
                heap.size() < k ? std::numeric_limits<float>::infinity() : heap.top().dist,
                gw.load(std::memory_order_relaxed));
            if (heap.size() < k || diff*diff < bound)
                traverseAtomic(pts, far, q, k, heap, visited, gw);
        }

        static QueryResult querySingle(const Points &pts, const Node *root,
            const Point &q, size_t k)
        {
            Heap heap; size_t visited = 0;
            traverse(pts, root, q, k, heap, visited);
            return extractResult(heap, visited);
        }

        static QueryResult queryLocalHeaps(const Points &pts, const Node *root,
            const Point &q, size_t k)
        {
            int cutoff = taskCutoff();
            Heap upper; size_t upperVisited = 0;
            std::vector<const Node*> work;
            collectWork(pts, root, q, k, 0, cutoff, upper, upperVisited, work);
            int n = (int)work.size();
            std::vector<Heap>   localHeaps(n);
            std::vector<size_t> localVisited(n, 0);
#pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < n; ++i)
                traverse(pts, work[i], q, k, localHeaps[i], localVisited[i]);
            size_t total = upperVisited;
            for (int i = 0; i < n; ++i) {
                mergeInto(upper, localHeaps[i], k);
                total += localVisited[i];
            }
            return extractResult(upper, total);
        }

        static QueryResult queryAtomicGlobal(const Points &pts, const Node *root,
            const Point &q, size_t k)
        {
            int cutoff = taskCutoff();
            Heap upper; size_t upperVisited = 0;
            std::vector<const Node*> work;
            collectWork(pts, root, q, k, 0, cutoff, upper, upperVisited, work);
            float initWorst = upper.size() == k
                ? upper.top().dist : std::numeric_limits<float>::infinity();
            std::atomic<float> gw{initWorst};
            int n = (int)work.size();
            std::vector<Heap>   localHeaps(n);
            std::vector<size_t> localVisited(n, 0);
#pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < n; ++i)
                traverseAtomic(pts, work[i], q, k, localHeaps[i], localVisited[i], gw);
            size_t total = upperVisited;
            for (int i = 0; i < n; ++i) {
                mergeInto(upper, localHeaps[i], k);
                total += localVisited[i];
            }
            return extractResult(upper, total);
        }
    };

    BuildMode             buildMode;
    QueryMode             queryMode;
    const Points         *points{nullptr};
    std::unique_ptr<Node> root;
};

// ---------------------------------------------------------------------------
// Brute-force baseline — maximise dot product (equiv. to min L2 on normalised vecs)
// ---------------------------------------------------------------------------

static BatchQueryResult bruteForce(const Dataset &ds, const Points &queries, size_t k)
{
    BatchQueryResult results(queries.size());
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < queries.size(); ++i) {
        const Point &q = queries[i];
        std::vector<std::pair<float, size_t>> s(ds.n);
        for (int j = 0; j < ds.n; ++j) {
            const float *p = ds.row(j);
            float d = 0.0f;
            for (int dim = 0; dim < ds.d; ++dim) d += q[dim] * p[dim];
            s[j] = {d, (size_t)j};
        }
        std::partial_sort(s.begin(), s.begin() + k, s.end(),
            [](const auto &a, const auto &b){ return a.first > b.first; });
        Neighbours nbrs(k);
        for (size_t ki = 0; ki < k; ++ki) {
            float diff2 = 0.0f;
            const float *p = ds.row((int)s[ki].second);
            for (int dim = 0; dim < ds.d; ++dim) {
                float d = q[dim] - p[dim]; diff2 += d * d;
            }
            nbrs[ki] = {std::sqrt(diff2), s[ki].second};
        }
        results[i] = {std::move(nbrs), (size_t)ds.n};
    }
    return results;
}

// ---------------------------------------------------------------------------
// Query generation
// ---------------------------------------------------------------------------

static Points generateQueries(const Points &pts, int iters, float noise, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> idxDist(0, pts.size()-1);
    std::normal_distribution<float>       noiseDist(0.0f, noise);
    Points queries;
    queries.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        Point q = pts[idxDist(rng)];
        for (auto &v : q) v += noiseDist(rng);
        queries.push_back(std::move(q));
    }
    return queries;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

static const char* buildModeName(BuildMode m)
{
    switch (m) {
        case BuildMode::Serial:        return "serial";
        case BuildMode::ParallelTasks: return "parallel-tasks";
        case BuildMode::ParallelFlat:  return "parallel-flat";
    }
    return "?";
}

static const char* queryModeName(QueryMode m)
{
    switch (m) {
        case QueryMode::Serial:       return "serial";
        case QueryMode::LocalHeaps:   return "local-heaps";
        case QueryMode::AtomicGlobal: return "atomic-global";
    }
    return "?";
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::fprintf(stderr,
            "Usage: %s <data_base> [--build-mode serial|parallel-tasks|parallel-flat]\n"
            "                      [--query-mode serial|local-heaps|atomic-global]\n"
            "                      [--k 10] [--iters 200] [--noise 0.01] [--seed 42]\n"
            "                      [--threads 1]\n", argv[0]);
        return 1;
    }

    std::string base     = argv[1];
    BuildMode   buildMode = BuildMode::Serial;
    QueryMode   queryMode = QueryMode::Serial;
    int         k        = 10;
    int         iters    = 200;
    float       noise    = 0.01f;
    uint64_t    seed     = 42;
    int         threads  = 1;

    for (int i = 2; i < argc; ++i) {
        std::string key = argv[i];
        auto next = [&]{ return std::string(argv[++i]); };
        if      (key == "--build-mode") buildMode = parseBuildMode(next());
        else if (key == "--query-mode") queryMode = parseQueryMode(next());
        else if (key == "--k")          k       = std::stoi(next());
        else if (key == "--iters")      iters   = std::stoi(next());
        else if (key == "--noise")      noise   = std::stof(next());
        else if (key == "--seed")       seed    = std::stoull(next());
        else if (key == "--threads")    threads = std::stoi(next());
        else { std::fprintf(stderr, "unknown arg: %s\n", key.c_str()); return 1; }
    }
    omp_set_num_threads(threads);

    Dataset ds = load(base);
    Points pts(ds.n);
    for (int i = 0; i < ds.n; ++i)
        pts[i] = Point(ds.row(i), ds.row(i) + ds.d);

    Points queries = generateQueries(pts, iters, noise, seed);

    double t0 = omp_get_wtime();
    bruteForce(ds, queries, k);
    double bruteMs = (omp_get_wtime() - t0) * 1e3;

    KDTree tree(buildMode, queryMode);

    t0 = omp_get_wtime();
    tree.build(pts);
    double buildMs = (omp_get_wtime() - t0) * 1e3;

    t0 = omp_get_wtime();
    BatchQueryResult results = tree.query(queries, k);
    double queryMs = (omp_get_wtime() - t0) * 1e3;

    double nodesMean = 0.0;
    for (const auto &r : results) nodesMean += (double)r.nodesVisited;
    nodesMean /= results.size();

    std::printf("build_mode=%s  query_mode=%s  threads=%d  N=%d D=%d  k=%d iters=%d\n",
        buildModeName(buildMode), queryModeName(queryMode),
        omp_get_max_threads(), ds.n, ds.d, k, iters);
    std::printf("brute  : %.3f ms  (%.1f us/query)\n",
        bruteMs, bruteMs * 1e3 / iters);
    std::printf("build  : %.3f ms\n", buildMs);
    std::printf("query  : %.3f ms  (%.1f us/query)  nodes_mean=%.0f  speedup x%.2f\n",
        queryMs, queryMs * 1e3 / iters, nodesMean, bruteMs / queryMs);

    return 0;
}
