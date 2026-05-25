#ifndef SOURCE_BENCHMARK_BENCHMARK_HH_
#define SOURCE_BENCHMARK_BENCHMARK_HH_

#include "generic/index.hh"
#include "generic/types.hh"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <omp.h>
#include <random>
#include <string>
#include <vector>

class Benchmark {
public:
	// aBuildMode / aQueryMode are written verbatim into the CSV.
	// Pass "-" for algorithms that don't have these knobs (e.g. exhaustive).
	Benchmark(std::unique_ptr<NNSIndex> aIndex, size_t aK, size_t aIters,
		std::string aBuildMode, std::string aQueryMode,
		float aNoise = 0.01f, uint64_t aSeed = 42) :
		index(std::move(aIndex)),
		k(aK),
		iters(aIters),
		buildMode(std::move(aBuildMode)),
		queryMode(std::move(aQueryMode)),
		noise(aNoise),
		seed(aSeed)
	{
	}

	void run(PointsPtr aPoints)
	{
		points	 = aPoints;
		nThreads = omp_get_max_threads();

		auto t0 = Clock::now();
		index->build(aPoints);
		buildMs = toMs(Clock::now() - t0);

		auto queries = generateQueries();

		auto batchT0		 = Clock::now();
		BatchQueryResult res = index->query(queries, k);
		batchMs				 = toMs(Clock::now() - batchT0);

		std::vector<double> nodesVec(res.size());
		for (size_t i = 0; i < res.size(); ++i)
			nodesVec[i] = static_cast<double>(res[i].nodesVisited);

		nodesMean	= mean(nodesVec);
		nodesStd	= std(nodesVec, nodesMean);
		queryMeanUs = batchMs * 1000.0 / static_cast<double>(iters);

		firstQuery	= queries[0];
		firstResult = std::move(res[0]);
	}

	// Append one build row to <algo>_build.csv.
	// Schema: build_mode, n, d, seed, threads, build_ms
	void writeBuildStats(const std::filesystem::path &aPath) const
	{
		bool isNew = !std::filesystem::exists(aPath)
			|| std::filesystem::file_size(aPath) == 0;

		std::ofstream f(aPath, std::ios::app);
		if (isNew)
			f << "build_mode,n,d,seed,threads,build_ms\n";

		f << buildMode << ','
		  << points->size() << ',' << points->at(0).size() << ','
		  << seed << ',' << nThreads << ',' << buildMs << '\n';
	}

	// Append one query row to <algo>_query.csv.
	// Schema: build_mode, query_mode, n, d, seed, k, iters, threads,
	//         batch_ms, query_mean_us, nodes_mean, nodes_std
	void writeQueryStats(const std::filesystem::path &aPath) const
	{
		bool isNew = !std::filesystem::exists(aPath)
			|| std::filesystem::file_size(aPath) == 0;

		std::ofstream f(aPath, std::ios::app);
		if (isNew)
			f << "build_mode,query_mode,n,d,seed,k,iters,threads,"
				 "batch_ms,query_mean_us,nodes_mean,nodes_std\n";

		f << buildMode << ',' << queryMode << ','
		  << points->size() << ',' << points->at(0).size() << ','
		  << seed << ',' << k << ',' << iters << ',' << nThreads << ','
		  << batchMs << ',' << queryMeanUs << ','
		  << nodesMean << ',' << nodesStd << '\n';
	}

	// Write neighbours of the first query point (for visualisation).
	// Format: idx,dist,x0,x1,...  — first row is the query point (idx=-1, dist=0).
	void writeNeighbours(const std::filesystem::path &aPath) const
	{
		size_t		  d = points->at(0).size();
		std::ofstream f(aPath);

		f << "idx,dist";
		for (size_t i = 0; i < d; ++i) f << ",x" << i;
		f << '\n';

		f << "-1,0";
		for (float v : firstQuery) f << ',' << v;
		f << '\n';

		for (const auto &n : firstResult.neighbours) {
			f << n.idx << ',' << n.dist;
			for (auto v : points->at(n.idx)) f << ',' << v;
			f << '\n';
		}
	}

private:
	using Clock = std::chrono::high_resolution_clock;

	template<typename D>
	static double toMs(D aD)
	{
		return std::chrono::duration<double, std::milli>(aD).count();
	}

	static double mean(const std::vector<double> &aV)
	{
		return std::accumulate(aV.begin(), aV.end(), 0.0) / aV.size();
	}

	static double std(const std::vector<double> &aV, double aMean)
	{
		double acc = 0.0;
		for (double x : aV) acc += (x - aMean) * (x - aMean);
		return std::sqrt(acc / aV.size());
	}

	Points generateQueries() const
	{
		std::mt19937_64					  rng(seed);
		std::uniform_int_distribution<size_t> idxDist(0, points->size() - 1);
		std::normal_distribution<float>		  noiseDist(0.0f, noise);

		Points queries;
		queries.reserve(iters);
		for (size_t i = 0; i < iters; ++i) {
			Point q = points->at(idxDist(rng));
			for (auto &v : q) v += noiseDist(rng);
			queries.push_back(std::move(q));
		}
		return queries;
	}

	std::unique_ptr<NNSIndex> index;
	size_t					  k;
	size_t					  iters;
	std::string				  buildMode;
	std::string				  queryMode;
	float					  noise;
	uint64_t				  seed;
	PointsPtr				  points;

	int	   nThreads{1};
	double buildMs{0};
	double batchMs{0};
	double queryMeanUs{0};
	double nodesMean{0};
	double nodesStd{0};

	Point		firstQuery;
	QueryResult firstResult;
};

#endif /* SOURCE_BENCHMARK_BENCHMARK_HH_ */
