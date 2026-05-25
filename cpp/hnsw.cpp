// HNSW (Hierarchical Navigable Small World) for cosine similarity.
// Mirrors the interface and output format of lsh.cpp.
//
// Six parallelization modes (--mode):
//
//   serial        build serial, queries serial                        (baseline)
//   queries       build serial, outer parallel-for queries (static)
//   queries_dyn   build serial, outer parallel-for queries (dynamic,1)
//   neighbors     build serial, serial outer queries,
//                   inner OMP over each candidate's neighbor list
//                   (parallelize the M distance evals per expansion step)
//   features      build serial, serial outer queries,
//                   inner OMP parallel-reduction over D features
//                   inside every single dot-product call
//   build         parallel build (striped per-node omp locks), serial queries
//
// Build:  make
// Run:    ./hnsw data/glove100_1000000 --M 16 --ef_construction 200 --ef 50
//                --mode queries --queries 500 --topk 10

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <queue>
#include <random>
#include <string>
#include <vector>

#include <omp.h>

using std::size_t;

// ---------------------------------------------------------------------------
// Mode
// ---------------------------------------------------------------------------
enum class Mode { serial, queries, queries_dyn, neighbors, features, build };

static const char* mode_name(Mode m) {
    switch (m) {
        case Mode::serial:      return "serial";
        case Mode::queries:     return "queries";
        case Mode::queries_dyn: return "queries_dyn";
        case Mode::neighbors:   return "neighbors";
        case Mode::features:    return "features";
        case Mode::build:       return "build";
    }
    return "?";
}

static Mode parse_mode(const std::string& s) {
    if (s == "serial")      return Mode::serial;
    if (s == "queries")     return Mode::queries;
    if (s == "queries_dyn") return Mode::queries_dyn;
    if (s == "neighbors")   return Mode::neighbors;
    if (s == "features")    return Mode::features;
    if (s == "build")       return Mode::build;
    std::fprintf(stderr, "unknown --mode: %s\n", s.c_str()); std::exit(1);
}

// Returns true when the outer query loop should be parallelized.
static bool outer_parallel(Mode m) {
    return m == Mode::queries || m == Mode::queries_dyn;
}

// ---------------------------------------------------------------------------
// Dataset  (identical to lsh.cpp)
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

static double now_sec() { return omp_get_wtime(); }

// Serial dot product (inner loop auto-vectorised by the compiler with -march=native).
static inline float dot(const float* a, const float* b, int d) {
    float s = 0.f;
    for (int j = 0; j < d; ++j) s += a[j] * b[j];
    return s;
}

// Parallel dot product: OMP reduction over d dimensions.
// Useful only when d is large (>= 300); for d=100 OMP overhead dominates.
static inline float dot_par(const float* a, const float* b, int d) {
    float s = 0.f;
    #pragma omp parallel for reduction(+:s) schedule(static)
    for (int j = 0; j < d; ++j) s += a[j] * b[j];
    return s;
}

// ---------------------------------------------------------------------------
// Brute-force baseline  (identical to lsh.cpp)
// ---------------------------------------------------------------------------
static std::vector<int> brute_topk(const Dataset& ds, const float* q, int k) {
    std::vector<std::pair<float,int>> s(ds.n);
    for (int i = 0; i < ds.n; ++i) s[i] = {dot(ds.row(i), q, ds.d), i};
    std::partial_sort(s.begin(), s.begin() + k, s.end(),
                      [](auto& a, auto& b){ return a.first > b.first; });
    std::vector<int> out(k);
    for (int i = 0; i < k; ++i) out[i] = s[i].second;
    return out;
}

// ---------------------------------------------------------------------------
// HNSW index
// ---------------------------------------------------------------------------
struct HNSW {
    // ---- parameters ----
    int   M               = 16;   // max neighbors per layer (layer > 0)
    int   M0              = 32;   // max neighbors at layer 0  (= 2*M)
    int   ef_construction = 200;  // beam width during build
    int   d               = 0;    // feature dimension

    float mL = 0.f;               // level normalisation: 1 / ln(M)
    int   entry_point = -1;
    int   max_level   = -1;

    // ---- graph ----
    // nbrs[node][layer] = list of neighbor ids at that layer
    std::vector<std::vector<std::vector<int>>> nbrs;
    std::vector<int> node_level;  // highest layer each node lives in

    // ---- striped omp locks for parallel build ----
    // Node u uses lock slot (u & (NLOCK-1)) — avoids 1-lock-per-node overhead
    // while keeping contention low (16 384 stripes for 1 M nodes).
    static constexpr int NLOCK = 1 << 14;
    std::vector<omp_lock_t> locks_;
    bool locks_ready_ = false;

    void init_locks() {
        if (locks_ready_) return;
        locks_.resize(NLOCK);
        for (auto& lk : locks_) omp_init_lock(&lk);
        locks_ready_ = true;
    }
    ~HNSW() {
        if (locks_ready_)
            for (auto& lk : locks_) omp_destroy_lock(&lk);
    }
    void lock_node  (int u) { omp_set_lock  (&locks_[u & (NLOCK-1)]); }
    void unlock_node(int u) { omp_unset_lock(&locks_[u & (NLOCK-1)]); }

    // ---- per-thread search context ----
    // Stamp counter: "reset" the visited array in O(1) by incrementing stamp.
    // No memset between queries — just ++stamp (wrap-around handled explicitly).
    struct Ctx {
        std::vector<uint32_t> vis;
        uint32_t stamp = 0;

        void init(int n) { vis.assign(n, 0); stamp = 0; }
        void new_search() {
            if (++stamp == 0) { std::fill(vis.begin(), vis.end(), 0); stamp = 1; }
        }
        bool visited(int i) const { return vis[i] == stamp; }
        void mark    (int i)       { vis[i] = stamp; }
    };
    mutable std::vector<Ctx> ctxs_;   // one per OMP thread

    void init_ctxs(int n, int T) {
        ctxs_.resize(T);
        for (auto& c : ctxs_) c.init(n);
    }
    Ctx& my_ctx() const { return ctxs_[omp_get_thread_num()]; }

    // ---- distance helpers (lower = closer on L2-normalised vectors) ----
    const Dataset* ds_ = nullptr;
    float dist_nn    (int a, int b)       const { return -dot    (ds_->row(a), ds_->row(b), d); }
    float dist_qn    (const float* q, int b) const { return -dot    (q, ds_->row(b), d); }
    float dist_qn_par(const float* q, int b) const { return -dot_par(q, ds_->row(b), d); }

    // ---- random level generation ----
    // floor(-ln(uniform(0,1)) * mL) gives exponentially distributed levels.
    int gen_level(std::mt19937& rng) const {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        double r = u(rng);
        if (r < 1e-15) r = 1e-15;
        return static_cast<int>(std::floor(-std::log(r) * mL));
    }

    // ---- priority queue types ----
    using Pair = std::pair<float,int>;
    using MaxH = std::priority_queue<Pair>;
    using MinH = std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>>;

    // Drain a max-heap into a vector sorted ascending (closest first).
    static std::vector<Pair> drain(MaxH& W) {
        std::vector<Pair> res(W.size());
        for (int i = (int)res.size() - 1; i >= 0; --i) { res[i] = W.top(); W.pop(); }
        return res;
    }

    // =========================================================================
    // search_layer variants — Algorithm 2 from the HNSW paper.
    // All return candidates sorted ascending by distance (closest = res[0]).
    // =========================================================================

    // --- (1) SERIAL: standard single-threaded beam search ---
    std::vector<Pair>
    search_layer_serial(const float* q, int ep, int ef, int layer,
                        Ctx& ctx, bool locked = false) const {
        ctx.new_search();
        const float d0 = dist_qn(q, ep);
        ctx.mark(ep);

        MaxH W;  W.push({d0, ep});
        MinH C;  C.push({d0, ep});

        while (!C.empty()) {
            auto [dc, c] = C.top();  C.pop();
            if (dc > W.top().first) break;

            // Read neighbor list (copy under lock during parallel build).
            std::vector<int> nb_buf;
            const std::vector<int>* nb_ptr;
            if (locked) {
                omp_set_lock(const_cast<omp_lock_t*>(&locks_[c & (NLOCK-1)]));
                nb_buf = nbrs[c][layer];
                omp_unset_lock(const_cast<omp_lock_t*>(&locks_[c & (NLOCK-1)]));
                nb_ptr = &nb_buf;
            } else {
                nb_ptr = &nbrs[c][layer];
            }

            for (int e : *nb_ptr) {
                if (ctx.visited(e)) continue;
                ctx.mark(e);
                const float de = dist_qn(q, e);
                if (de < W.top().first || (int)W.size() < ef) {
                    C.push({de, e}); W.push({de, e});
                    if ((int)W.size() > ef) W.pop();
                }
            }
        }
        return drain(W);
    }

    // --- (2) NEIGHBORS mode: parallel scoring over each candidate's neighbor list ---
    //
    // For every candidate c popped from C, collect its unvisited neighbors,
    // then evaluate their distances in an OMP parallel-for before updating the
    // heaps serially.  The parallelism grain = M (typically 16-32 nodes × D floats).
    // Beneficial when M*D is large enough to amortise the fork overhead.
    std::vector<Pair>
    search_layer_par_nbrs(const float* q, int ep, int ef, int layer, Ctx& ctx) const {
        ctx.new_search();
        const float d0 = dist_qn(q, ep);
        ctx.mark(ep);

        MaxH W;  W.push({d0, ep});
        MinH C;  C.push({d0, ep});

        std::vector<int>   to_eval;
        std::vector<float> dists;

        while (!C.empty()) {
            auto [dc, c] = C.top();  C.pop();
            if (dc > W.top().first) break;

            // Collect unvisited neighbors (serial — visited state is per-thread).
            to_eval.clear();
            for (int e : nbrs[c][layer]) {
                if (!ctx.visited(e)) { ctx.mark(e); to_eval.push_back(e); }
            }

            // Parallel distance evaluations over the neighbor list.
            const int ne = (int)to_eval.size();
            dists.resize(ne);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < ne; ++i)
                dists[i] = dist_qn(q, to_eval[i]);

            // Serial heap update (heap structures are not thread-safe).
            for (int i = 0; i < ne; ++i) {
                if (dists[i] < W.top().first || (int)W.size() < ef) {
                    C.push({dists[i], to_eval[i]});
                    W.push({dists[i], to_eval[i]});
                    if ((int)W.size() > ef) W.pop();
                }
            }
        }
        return drain(W);
    }

    // --- (3) FEATURES mode: OMP parallel reduction inside each dot product ---
    //
    // Identical search control-flow to the serial variant, but every
    // dist_qn() call uses dot_par() which spawns an OMP parallel reduction
    // over the D feature dimensions.
    // Useful only for very large D (>= 300).  For D=100 the fork overhead
    // dominates and this mode will be *slower* than serial — same result as
    // LSH's "features" mode.
    std::vector<Pair>
    search_layer_par_feat(const float* q, int ep, int ef, int layer, Ctx& ctx) const {
        ctx.new_search();
        const float d0 = dist_qn_par(q, ep);
        ctx.mark(ep);

        MaxH W;  W.push({d0, ep});
        MinH C;  C.push({d0, ep});

        while (!C.empty()) {
            auto [dc, c] = C.top();  C.pop();
            if (dc > W.top().first) break;

            for (int e : nbrs[c][layer]) {
                if (ctx.visited(e)) continue;
                ctx.mark(e);
                const float de = dist_qn_par(q, e);
                if (de < W.top().first || (int)W.size() < ef) {
                    C.push({de, e}); W.push({de, e});
                    if ((int)W.size() > ef) W.pop();
                }
            }
        }
        return drain(W);
    }

    // =========================================================================
    // Internal helpers for build
    // =========================================================================

    // Prune the neighbor list of node `nb` at `layer` to at most M_max entries,
    // keeping the M_max closest.  Must be called with nb's lock held (or during
    // serial build where no concurrent modification can occur).
    void prune_nbrs(int nb, int layer, int M_max) {
        auto& nlist = nbrs[nb][layer];
        if ((int)nlist.size() <= M_max) return;
        std::vector<Pair> s;
        s.reserve(nlist.size());
        for (int x : nlist) s.push_back({dist_nn(nb, x), x});
        std::sort(s.begin(), s.end());
        nlist.resize(M_max);
        for (int i = 0; i < M_max; ++i) nlist[i] = s[i].second;
    }

    // Insert node u into the graph.
    // parallel_mode = false: lock-free serial insertion.
    // parallel_mode = true:  concurrent insertion with striped per-node locks.
    void insert(int u, bool parallel_mode) {
        const int    lv = node_level[u];
        const float*  q = ds_->row(u);
        Ctx&        ctx = my_ctx();

        // Handle first-ever node.
        if (!parallel_mode) {
            if (entry_point == -1) { entry_point = u; max_level = lv; return; }
        } else {
            bool is_first = false;
            #pragma omp critical(hnsw_ep)
            { if (entry_point == -1) { entry_point = u; max_level = lv; is_first = true; } }
            if (is_first) return;
        }

        // Snapshot (may be slightly stale in parallel — acceptable for approx. ANN).
        int ep      = entry_point;
        int cur_max = max_level;

        // Phase 1: greedy descent from top layer to lv+1 (ef=1 per layer).
        for (int lc = cur_max; lc > lv; --lc) {
            auto W = search_layer_serial(q, ep, 1, lc, ctx, parallel_mode);
            ep = W[0].second;
        }

        // Phase 2: insert at layers min(lv, cur_max) down to 0.
        for (int lc = std::min(lv, cur_max); lc >= 0; --lc) {
            const int M_max = (lc == 0) ? M0 : M;

            auto W = search_layer_serial(q, ep, ef_construction, lc, ctx, parallel_mode);
            ep = W[0].second;  // best entry for the next lower layer

            const int n_nbrs = std::min((int)W.size(), M_max);

            // Write u's neighbor list.  Node u is not yet reachable in the graph
            // (no back-links exist yet), so no lock is required on u itself.
            nbrs[u][lc].resize(n_nbrs);
            for (int i = 0; i < n_nbrs; ++i) nbrs[u][lc][i] = W[i].second;

            // Add back-links: append u to each chosen neighbor's list, then prune.
            for (int i = 0; i < n_nbrs; ++i) {
                int nb = W[i].second;
                if (parallel_mode) lock_node(nb);
                nbrs[nb][lc].push_back(u);
                if ((int)nbrs[nb][lc].size() > M_max)
                    prune_nbrs(nb, lc, M_max);
                if (parallel_mode) unlock_node(nb);
            }
        }

        // Update global entry point if u occupies a higher level.
        if (lv > cur_max) {
            if (!parallel_mode) {
                entry_point = u; max_level = lv;
            } else {
                #pragma omp critical(hnsw_ep)
                { if (lv > max_level) { entry_point = u; max_level = lv; } }
            }
        }
    }

    // =========================================================================
    // Public API
    // =========================================================================

    void build(const Dataset& dataset, int M_, int ef_construction_,
               uint32_t seed, bool parallel) {
        ds_ = &dataset;
        M = M_;  M0 = 2 * M;  ef_construction = ef_construction_;
        d = dataset.d;
        mL = 1.f / std::log((float)M);
        entry_point = -1;  max_level = -1;

        const int n = dataset.n;
        const int T = parallel ? omp_get_max_threads() : 1;

        nbrs.resize(n);
        node_level.resize(n);

        // Pre-generate random levels serially — result is independent of thread count.
        {
            std::mt19937 rng(seed);
            for (int i = 0; i < n; ++i) {
                node_level[i] = gen_level(rng);
                nbrs[i].resize(node_level[i] + 1);
            }
        }

        init_ctxs(n, T);
        if (parallel) init_locks();

        const double t0 = now_sec();
        if (!parallel) {
            for (int i = 0; i < n; ++i) insert(i, false);
        } else {
            // Dynamic scheduling: nodes at high levels trigger more search_layer calls.
            #pragma omp parallel for schedule(dynamic, 64)
            for (int i = 0; i < n; ++i) insert(i, true);
        }

        std::fprintf(stderr, "build  M=%d ef_construction=%d %s  %.3fs\n",
                     M, ef_construction,
                     parallel ? "(parallel)" : "(serial)",
                     now_sec() - t0);
    }

    // Query: find topk nearest neighbors of q using the given mode.
    // Greedy descent on upper layers always uses the serial variant (ef=1 means
    // only a single candidate — no inner-parallel benefit).
    // Inner-parallel modes kick in for the dense layer-0 search.
    std::vector<int> query(const float* q, int topk, int ef, Mode m) const {
        if (entry_point == -1) return {};
        Ctx& ctx = my_ctx();

        int ep = entry_point;

        // Greedy descent: layers max_level … 1  (serial, ef=1)
        for (int lc = max_level; lc > 0; --lc) {
            auto W = search_layer_serial(q, ep, 1, lc, ctx);
            ep = W[0].second;
        }

        // Full beam search at layer 0 — dispatch on mode.
        const int ef_actual = std::max(ef, topk);
        std::vector<Pair> W;
        switch (m) {
            case Mode::neighbors:
                W = search_layer_par_nbrs(q, ep, ef_actual, 0, ctx); break;
            case Mode::features:
                W = search_layer_par_feat(q, ep, ef_actual, 0, ctx); break;
            default:
                W = search_layer_serial  (q, ep, ef_actual, 0, ctx); break;
        }

        const int k = std::min(topk, (int)W.size());
        std::vector<int> out(k);
        for (int i = 0; i < k; ++i) out[i] = W[i].second;
        return out;
    }
};

// ---------------------------------------------------------------------------
// Args + main  (mirrors lsh.cpp structure and output format)
// ---------------------------------------------------------------------------
struct Args {
    std::string base;
    int      M               = 16;
    int      ef_construction = 200;
    int      ef              = 50;
    int      topk            = 10;
    int      queries         = 1000;
    uint32_t seed            = 1;
    Mode     mode            = Mode::queries;
};

static Args parse(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: %s <data_base> [--M 16] [--ef_construction 200] [--ef 50]\n"
            "                       [--topk 10] [--queries 1000] [--seed 1]\n"
            "                       [--mode serial|queries|queries_dyn|neighbors|features|build]\n",
            argv[0]);
        std::exit(1);
    }
    Args a;  a.base = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]{ return std::string(argv[++i]); };
        if      (k == "--M")               a.M               = std::stoi(next());
        else if (k == "--ef_construction") a.ef_construction = std::stoi(next());
        else if (k == "--ef")              a.ef              = std::stoi(next());
        else if (k == "--topk")            a.topk            = std::stoi(next());
        else if (k == "--queries")         a.queries         = std::stoi(next());
        else if (k == "--seed")            a.seed            = (uint32_t)std::stoul(next());
        else if (k == "--mode")            a.mode            = parse_mode(next());
        else { std::fprintf(stderr, "unknown arg: %s\n", k.c_str()); std::exit(1); }
    }
    return a;
}

int main(int argc, char** argv) {
    Args a = parse(argc, argv);
    // Allow nested OMP regions for neighbors/features inside an outer parallel for.
    omp_set_max_active_levels(3);

    Dataset ds = load(a.base);

    // Build: "build" mode = parallel insertions; all other modes = serial build.
    HNSW hnsw;
    const bool parallel_build = (a.mode == Mode::build);
    hnsw.build(ds, a.M, a.ef_construction, a.seed, parallel_build);

    // Query IDs (same RNG convention as lsh.cpp).
    std::mt19937 rng(a.seed ^ 0xC0FFEEu);
    std::uniform_int_distribution<int> uid(0, ds.n - 1);
    std::vector<int> q_ids(a.queries);
    for (auto& x : q_ids) x = uid(rng);

    // Ground truth via brute-force (always parallelised over queries).
    std::vector<std::vector<int>> gt(a.queries);
    {
        double t0 = now_sec();
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < a.queries; ++i)
            gt[i] = brute_topk(ds, ds.row(q_ids[i]), a.topk);
        std::fprintf(stderr, "brute-force gt: %.3f s\n", now_sec() - t0);
    }
    // (We measure the brute-force *query* time below, inside the timed section.)

    // Re-run brute for timing (separate pass so gt is already computed).
    std::vector<std::vector<int>> bt(a.queries);
    double t_brute;
    {
        double t0 = now_sec();
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < a.queries; ++i)
            bt[i] = brute_topk(ds, ds.row(q_ids[i]), a.topk);
        t_brute = now_sec() - t0;
    }

    // Initialise per-thread contexts for the query phase.
    const int T_query = outer_parallel(a.mode) ? omp_get_max_threads() : 1;
    hnsw.init_ctxs(ds.n, T_query);

    // HNSW queries.
    std::vector<std::vector<int>> ap_(a.queries);
    double t0 = now_sec();

    if (a.mode == Mode::queries) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < a.queries; ++i)
            ap_[i] = hnsw.query(ds.row(q_ids[i]), a.topk, a.ef, a.mode);

    } else if (a.mode == Mode::queries_dyn) {
        #pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < a.queries; ++i)
            ap_[i] = hnsw.query(ds.row(q_ids[i]), a.topk, a.ef, a.mode);

    } else {
        // serial / neighbors / features / build: outer loop is serial,
        // inner OMP parallelism (if any) is activated inside query().
        for (int i = 0; i < a.queries; ++i)
            ap_[i] = hnsw.query(ds.row(q_ids[i]), a.topk, a.ef, a.mode);
    }

    const double t_hnsw = now_sec() - t0;

    // Recall@k.
    double recall = 0.0;
    for (int i = 0; i < a.queries; ++i) {
        std::vector<int> g = gt[i];
        std::sort(g.begin(), g.end());
        int hit = 0;
        for (int x : ap_[i])
            if (std::binary_search(g.begin(), g.end(), x)) ++hit;
        recall += (double)hit / a.topk;
    }
    recall /= a.queries;

    // --- Output (same field layout as lsh.cpp so bench scripts can parse it) ---
    std::printf("mode=%s  threads=%d  N=%d D=%d  M=%d ef_construction=%d ef=%d topk=%d queries=%d\n",
                mode_name(a.mode), omp_get_max_threads(),
                ds.n, ds.d, a.M, a.ef_construction, a.ef, a.topk, a.queries);
    std::printf("brute     : %.3f ms  (%.1f us/query)\n",
                t_brute * 1e3, t_brute * 1e6 / a.queries);
    std::printf("hnsw      : %.3f ms  (%.1f us/query)  speedup x%.2f\n",
                t_hnsw * 1e3, t_hnsw * 1e6 / a.queries, t_brute / t_hnsw);
    std::printf("recall@%d  : %.4f\n", a.topk, recall);
    return 0;
}
