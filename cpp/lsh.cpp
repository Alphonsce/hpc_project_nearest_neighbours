// SimHash LSH for cosine similarity, brute-force baseline, OpenMP parallel queries.
// Build:  make
// Run:    ./lsh data/glove50_20000 --L 8 --K 16 --queries 1000 --topk 10

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

struct Dataset {
    int n = 0, d = 0;
    std::vector<float> x;  // row-major, L2-normalised
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

static double now_sec() {
    return omp_get_wtime();
}

// ---- brute-force cosine kNN (data already L2-normalised) ----------------

static std::vector<int> brute_topk(const Dataset& ds, const float* q, int k) {
    std::vector<std::pair<float,int>> s(ds.n);
    for (int i = 0; i < ds.n; ++i) {
        float dot = 0.f;
        const float* r = ds.row(i);
        for (int j = 0; j < ds.d; ++j) dot += r[j] * q[j];
        s[i] = {dot, i};
    }
    std::partial_sort(s.begin(), s.begin() + k, s.end(),
                      [](auto& a, auto& b){ return a.first > b.first; });
    std::vector<int> out(k);
    for (int i = 0; i < k; ++i) out[i] = s[i].second;
    return out;
}

// ---- SimHash LSH index --------------------------------------------------

struct LSH {
    int L;        // number of hash tables
    int K;        // bits per signature (<= 64)
    int d;
    std::vector<float> planes;                                  // L*K*d row-major
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
            for (int i = 0; i < ds.n; ++i) {
                for (int t = 0; t < L; ++t) {
                    local[t].emplace_back(sig(ds.row(i), t), i);
                }
            }
            #pragma omp critical
            for (int t = 0; t < L; ++t)
                for (auto& [h, i] : local[t]) tables[t][h].push_back(i);
        }
        std::fprintf(stderr, "build  L=%d K=%d  %.3fs\n", L, K, now_sec() - t0);
    }

    uint64_t sig(const float* x, int table) const {
        const float* P = planes.data() + size_t(table) * K * d;
        uint64_t h = 0;
        for (int k = 0; k < K; ++k) {
            float dot = 0.f;
            const float* p = P + size_t(k) * d;
            for (int j = 0; j < d; ++j) dot += p[j] * x[j];
            if (dot >= 0.f) h |= (uint64_t{1} << k);
        }
        return h;
    }

    // returns top-k by cosine over the union of bucket members
    std::vector<int> query(const Dataset& ds, const float* q, int topk, int* n_candidates = nullptr) const {
        std::vector<char> seen(ds.n, 0);
        std::vector<int> cand;
        cand.reserve(256);
        for (int t = 0; t < L; ++t) {
            auto it = tables[t].find(sig(q, t));
            if (it == tables[t].end()) continue;
            for (int i : it->second) {
                if (!seen[i]) { seen[i] = 1; cand.push_back(i); }
            }
        }
        if (n_candidates) *n_candidates = (int)cand.size();
        if ((int)cand.size() <= topk) return cand;

        std::vector<std::pair<float,int>> s(cand.size());
        for (size_t i = 0; i < cand.size(); ++i) {
            float dot = 0.f;
            const float* r = ds.row(cand[i]);
            for (int j = 0; j < ds.d; ++j) dot += r[j] * q[j];
            s[i] = {dot, cand[i]};
        }
        std::partial_sort(s.begin(), s.begin() + topk, s.end(),
                          [](auto& a, auto& b){ return a.first > b.first; });
        std::vector<int> out(topk);
        for (int i = 0; i < topk; ++i) out[i] = s[i].second;
        return out;
    }
};

// ---- benchmark driver ---------------------------------------------------

struct Args {
    std::string base;
    int L = 8, K = 16, topk = 10, queries = 1000;
    uint32_t seed = 1;
};

static Args parse(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <data_base> [--L 8] [--K 16] [--topk 10] [--queries 1000] [--seed 1]\n", argv[0]);
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
        else { std::fprintf(stderr, "unknown arg: %s\n", k.c_str()); std::exit(1); }
    }
    return a;
}

int main(int argc, char** argv) {
    Args a = parse(argc, argv);
    Dataset ds = load(a.base);

    LSH lsh;
    lsh.build(ds, a.L, a.K, a.seed);

    std::mt19937 rng(a.seed ^ 0xC0FFEEu);
    std::vector<int> q_ids(a.queries);
    {
        std::uniform_int_distribution<int> u(0, ds.n - 1);
        for (auto& x : q_ids) x = u(rng);
    }

    // ground truth (brute force, parallel)
    std::vector<std::vector<int>> gt(a.queries);
    double t0 = now_sec();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < a.queries; ++i) {
        gt[i] = brute_topk(ds, ds.row(q_ids[i]), a.topk);
    }
    double t_brute = now_sec() - t0;

    // LSH queries
    std::vector<std::vector<int>> ap_(a.queries);
    std::vector<int> ncand(a.queries, 0);
    t0 = now_sec();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < a.queries; ++i) {
        ap_[i] = lsh.query(ds, ds.row(q_ids[i]), a.topk, &ncand[i]);
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

    std::printf("threads=%d  N=%d D=%d  L=%d K=%d topk=%d queries=%d\n",
                omp_get_max_threads(), ds.n, ds.d, a.L, a.K, a.topk, a.queries);
    std::printf("brute     : %.3f ms  (%.1f us/query)\n",
                t_brute * 1e3, t_brute * 1e6 / a.queries);
    std::printf("lsh       : %.3f ms  (%.1f us/query)  speedup x%.2f\n",
                t_lsh * 1e3, t_lsh * 1e6 / a.queries, t_brute / t_lsh);
    std::printf("recall@%d  : %.4f   avg_candidates=%.1f\n",
                a.topk, recall, double(cand_sum) / a.queries);
    return 0;
}
