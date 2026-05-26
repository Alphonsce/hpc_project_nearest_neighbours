// SimHash LSH for cosine similarity with multiple OpenMP parallelization modes.
//   queries       outer parallel for over queries, serial body  (default)
//   queries_dyn   outer dynamic parallel for over queries, serial body
//   tables        outer serial, inner parallel over L tables
//   candidates    outer serial, inner parallel over candidate rerank
//   features      outer serial, inner parallel reduction over D inside each dot
//   query_table   nested: outer parallel queries, inner parallel tables
//   all           nested: outer queries, inner tables, inner-inner candidates
//
// Build:  make
// Run:    ./lsh data/glove100_1000000 --L 32 --K 12 --mode queries --queries 500 --topk 10

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <omp.h>

using std::size_t;
using std::uint64_t;

enum class Mode { queries, queries_dyn, tables, candidates, features, query_table, all };

static const char* mode_name(Mode m) {
    switch (m) {
        case Mode::queries:     return "queries";
        case Mode::queries_dyn: return "queries_dyn";
        case Mode::tables:      return "tables";
        case Mode::candidates:  return "candidates";
        case Mode::features:    return "features";
        case Mode::query_table: return "query_table";
        case Mode::all:         return "all";
    }
    return "?";
}

static Mode parse_mode(const std::string& s) {
    if (s == "queries")     return Mode::queries;
    if (s == "queries_dyn") return Mode::queries_dyn;
    if (s == "tables")      return Mode::tables;
    if (s == "candidates")  return Mode::candidates;
    if (s == "features")    return Mode::features;
    if (s == "query_table") return Mode::query_table;
    if (s == "all")         return Mode::all;
    std::fprintf(stderr, "unknown --mode: %s\n", s.c_str()); std::exit(1);
}

static bool outer_parallel(Mode m) {
    return m == Mode::queries || m == Mode::queries_dyn || m == Mode::query_table || m == Mode::all;
}

struct Dataset {
    int n = 0, d = 0;
    std::vector<float> x;
    const float* row(int i) const { return x.data() + size_t(i) * d; }
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

static double now_sec() { return omp_get_wtime(); }

static inline float dot(const float* a, const float* b, int d) {
    float s = 0.f;
    for (int j = 0; j < d; ++j) s += a[j] * b[j];
    return s;
}

// ---- brute-force baseline ----------------------------------------------

static std::vector<int> brute_topk(const Dataset& ds, const float* q, int k) {
    std::vector<std::pair<float,int>> s(ds.n);
    for (int i = 0; i < ds.n; ++i) s[i] = {dot(ds.row(i), q, ds.d), i};
    std::partial_sort(s.begin(), s.begin() + k, s.end(),
                      [](auto& a, auto& b){ return a.first > b.first; });
    std::vector<int> out(k);
    for (int i = 0; i < k; ++i) out[i] = s[i].second;
    return out;
}

// ---- LSH index ----------------------------------------------------------

struct LSH {
    int L = 0, K = 0, d = 0;
    std::vector<float> planes;
    std::vector<std::unordered_map<uint64_t, std::vector<int>>> tables;

    void build(const Dataset& ds, int L_, int K_, uint32_t seed) {
        L = L_; K = K_; d = ds.d;
        if (K > 64) { std::fprintf(stderr, "K must be <= 64\n"); std::exit(1); }
        planes.assign(size_t(L) * K * d, 0.f);
        std::mt19937 rng(seed);
        std::normal_distribution<float> g(0.f, 1.f);
        for (auto& v : planes) v = g(rng);

        tables.assign(L, {});
        for (int t = 0; t < L; ++t) tables[t].reserve(ds.n * 2);

        double t0 = now_sec();
        #pragma omp parallel
        {
            std::vector<std::vector<std::pair<uint64_t,int>>> local(L);
            #pragma omp for schedule(static)
            for (int i = 0; i < ds.n; ++i)
                for (int t = 0; t < L; ++t)
                    local[t].emplace_back(sig(ds.row(i), t, false), i);
            #pragma omp critical
            for (int t = 0; t < L; ++t)
                for (auto& [h, i] : local[t]) tables[t][h].push_back(i);
        }
        std::fprintf(stderr, "build  L=%d K=%d  %.3fs\n", L, K, now_sec() - t0);
    }

    uint64_t sig(const float* x, int table, bool par_features) const {
        if (par_features) return sig_features(x, table);

        const float* P = planes.data() + size_t(table) * K * d;
        uint64_t h = 0;
        for (int k = 0; k < K; ++k) {
            const float* p = P + size_t(k) * d;
            float v = dot(p, x, d);
            if (v >= 0.f) h |= (uint64_t{1} << k);
        }
        return h;
    }

    uint64_t sig_features(const float* x, int table) const {
        const float* P = planes.data() + size_t(table) * K * d;
        int T = omp_get_max_threads();
        std::vector<std::vector<float>> partial(T, std::vector<float>(K, 0.f));

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto& sums = partial[tid];
            #pragma omp for schedule(static)
            for (int j = 0; j < d; ++j) {
                float xj = x[j];
                for (int k = 0; k < K; ++k) sums[k] += P[size_t(k) * d + j] * xj;
            }
        }

        uint64_t h = 0;
        for (int k = 0; k < K; ++k) {
            float v = 0.f;
            for (int tid = 0; tid < T; ++tid) v += partial[tid][k];
            if (v >= 0.f) h |= (uint64_t{1} << k);
        }
        return h;
    }

    // Serial gather: union bucket members across L tables.
    void gather_serial(std::vector<char>& seen, std::vector<int>& cand,
                       const float* q, bool par_features) const {
        for (int t = 0; t < L; ++t) {
            auto it = tables[t].find(sig(q, t, par_features));
            if (it == tables[t].end()) continue;
            for (int i : it->second)
                if (!seen[i]) { seen[i] = 1; cand.push_back(i); }
        }
    }

    // Parallel-over-tables gather: each thread takes a slice of tables and
    // builds a local id list; serial dedupe pass after the parallel region.
    void gather_tables(std::vector<char>& seen, std::vector<int>& cand,
                       const float* q) const {
        int T = omp_get_max_threads();
        std::vector<std::vector<int>> local(T);
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto& mine = local[tid];
            #pragma omp for schedule(static) nowait
            for (int t = 0; t < L; ++t) {
                auto it = tables[t].find(sig(q, t, false));
                if (it == tables[t].end()) continue;
                mine.insert(mine.end(), it->second.begin(), it->second.end());
            }
        }
        for (auto& v : local)
            for (int i : v) if (!seen[i]) { seen[i] = 1; cand.push_back(i); }
    }

    std::vector<int> rerank_serial(const Dataset& ds, const float* q,
                                   const std::vector<int>& cand, int topk,
                                   bool par_features) const {
        if ((int)cand.size() <= topk) return cand;
        std::vector<std::pair<float,int>> s(cand.size());
        for (size_t i = 0; i < cand.size(); ++i)
            s[i] = {dot(ds.row(cand[i]), q, ds.d), cand[i]};
        std::partial_sort(s.begin(), s.begin() + topk, s.end(),
                          [](auto& a, auto& b){ return a.first > b.first; });
        std::vector<int> out(topk);
        for (int i = 0; i < topk; ++i) out[i] = s[i].second;
        return out;
    }

    std::vector<int> rerank_features_parallel(const Dataset& ds, const float* q,
                                              const std::vector<int>& cand, int topk) const {
        if ((int)cand.size() <= topk) return cand;
        int n = (int)cand.size();
        int T = omp_get_max_threads();
        std::vector<std::vector<float>> partial(T, std::vector<float>(n, 0.f));

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto& mine = partial[tid];
            #pragma omp for schedule(static)
            for (int j = 0; j < ds.d; ++j) {
                float qj = q[j];
                for (int i = 0; i < n; ++i) mine[i] += ds.row(cand[i])[j] * qj;
            }
        }

        std::vector<std::pair<float,int>> s(n);
        for (int i = 0; i < n; ++i) {
            float score = 0.f;
            for (int tid = 0; tid < T; ++tid) score += partial[tid][i];
            s[i] = {score, cand[i]};
        }
        std::partial_sort(s.begin(), s.begin() + topk, s.end(),
                          [](auto& a, auto& b){ return a.first > b.first; });
        std::vector<int> out(topk);
        for (int i = 0; i < topk; ++i) out[i] = s[i].second;
        return out;
    }

    std::vector<int> rerank_parallel(const Dataset& ds, const float* q,
                                     const std::vector<int>& cand, int topk) const {
        if ((int)cand.size() <= topk) return cand;
        int n = (int)cand.size();
        std::vector<std::pair<float,int>> s(n);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i)
            s[i] = {dot(ds.row(cand[i]), q, ds.d), cand[i]};
        std::partial_sort(s.begin(), s.begin() + topk, s.end(),
                          [](auto& a, auto& b){ return a.first > b.first; });
        std::vector<int> out(topk);
        for (int i = 0; i < topk; ++i) out[i] = s[i].second;
        return out;
    }

    std::vector<int> query(const Dataset& ds, const float* q, int topk, Mode m,
                           int* n_candidates) const {
        std::vector<char> seen(ds.n, 0);
        std::vector<int> cand;
        cand.reserve(256);

        const bool inner_tables = (m == Mode::tables || m == Mode::query_table || m == Mode::all);
        const bool inner_cands  = (m == Mode::candidates || m == Mode::all);
        const bool par_features = (m == Mode::features && omp_get_max_threads() > 1);

        if (inner_tables) gather_tables(seen, cand, q);
        else              gather_serial(seen, cand, q, par_features);

        if (n_candidates) *n_candidates = (int)cand.size();
        if (par_features) return rerank_features_parallel(ds, q, cand, topk);
        return inner_cands ? rerank_parallel(ds, q, cand, topk)
                           : rerank_serial(ds, q, cand, topk, par_features);
    }
};

// ---- driver -------------------------------------------------------------

struct Args {
    std::string base;
    int L = 8, K = 16, topk = 10, queries = 1000;
    uint32_t seed = 1;
    Mode mode = Mode::queries;
};

static Args parse(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: %s <data_base> [--L 8] [--K 16] [--topk 10] [--queries 1000]\n"
            "                       [--mode queries|queries_dyn|tables|candidates|features|query_table|all]\n"
            "                       [--seed 1]\n", argv[0]);
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

int main(int argc, char** argv) {
    Args a = parse(argc, argv);
    omp_set_max_active_levels(3);  // allow nested for query_table / all

    Dataset ds = load(a.base);
    LSH lsh;
    lsh.build(ds, a.L, a.K, a.seed);

    std::mt19937 rng(a.seed ^ 0xC0FFEEu);
    std::vector<int> q_ids(a.queries);
    {
        std::uniform_int_distribution<int> u(0, ds.n - 1);
        for (auto& x : q_ids) x = u(rng);
    }

    // ground truth (always: outer parallel over queries)
    std::vector<std::vector<int>> gt(a.queries);
    double t0 = now_sec();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < a.queries; ++i)
        gt[i] = brute_topk(ds, ds.row(q_ids[i]), a.topk);
    double t_brute = now_sec() - t0;

    // LSH queries
    std::vector<std::vector<int>> ap_(a.queries);
    std::vector<int> ncand(a.queries, 0);
    t0 = now_sec();
    if (a.mode == Mode::queries_dyn) {
        #pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < a.queries; ++i)
            ap_[i] = lsh.query(ds, ds.row(q_ids[i]), a.topk, a.mode, &ncand[i]);
    } else {
        const bool outer = outer_parallel(a.mode);
        #pragma omp parallel for schedule(static) if(outer)
        for (int i = 0; i < a.queries; ++i)
            ap_[i] = lsh.query(ds, ds.row(q_ids[i]), a.topk, a.mode, &ncand[i]);
    }
    double t_lsh = now_sec() - t0;

    // recall@k
    double recall = 0.0; long long cand_sum = 0;
    for (int i = 0; i < a.queries; ++i) {
        std::vector<int> g = gt[i]; std::sort(g.begin(), g.end());
        int hit = 0;
        for (int x : ap_[i]) if (std::binary_search(g.begin(), g.end(), x)) ++hit;
        recall += double(hit) / a.topk;
        cand_sum += ncand[i];
    }
    recall /= a.queries;

    std::printf("mode=%s  threads=%d  N=%d D=%d  L=%d K=%d topk=%d queries=%d\n",
                mode_name(a.mode), omp_get_max_threads(), ds.n, ds.d,
                a.L, a.K, a.topk, a.queries);
    std::printf("brute     : %.3f ms  (%.1f us/query)\n",
                t_brute * 1e3, t_brute * 1e6 / a.queries);
    std::printf("lsh       : %.3f ms  (%.1f us/query)  speedup x%.2f\n",
                t_lsh * 1e3, t_lsh * 1e6 / a.queries, t_brute / t_lsh);
    std::printf("recall@%d  : %.4f   avg_candidates=%.1f\n",
                a.topk, recall, double(cand_sum) / a.queries);
    return 0;
}
