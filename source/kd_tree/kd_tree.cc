#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include <omp.h>

#include "benchmark/arg_helper.hh"
#include "benchmark/benchmark.hh"
#include "benchmark/dataloader.hh"
#include "exhaustive_knn/exhaustive_knn.hh"
#include "kd_tree/kd_tree.hh"

static inline std::filesystem::path dataDir{DATA_DIR};
static inline std::filesystem::path statsDir{STATS_DIR};

static void printUsage()
{
	std::cerr
		<< "Usage: kd-tree --data <filename> [options]\n"
		   "\n"
		   "  --data <filename>      .npy file (relative to DATA_DIR)\n"
		   "  --algo <algo>          kdtree (default) | exhaustive\n"
		   "  --build-mode <mode>    serial (default) | parallel      [kdtree]\n"
		   "  --query-mode <mode>    serial (default) | local-heaps   [kdtree]\n"
		   "                                         | atomic-global  [kdtree]\n"
		   "  --k <int>              neighbours (default: 10)\n"
		   "  --iters <int>          query iterations (default: 500)\n"
		   "  --threads <int>        OpenMP threads (default: 1)\n"
		   "  --noise <float>        query noise stddev (default: 0.01)\n"
		   "  --seed <int>           rng seed (default: 42)\n"
		   "  --save-neighbours      write neighbours CSV\n"
		   "\n"
		   "Output: out/stats/<algo>_build.csv, out/stats/<algo>_query.csv\n";
}

int main(int argc, char **argv)
{
	std::span<char *> args(argv, argc);

	if (hasFlag(args, "--help") || hasFlag(args, "-h")) {
		printUsage();
		return 0;
	}

	auto dataFile = getArg(args, "--data");
	if (dataFile.empty()) {
		printUsage();
		return 1;
	}

	auto	 algo	   = getArg(args, "--algo", "kdtree");
	auto	 buildMode = getArg(args, "--build-mode", "serial");
	auto	 queryMode = getArg(args, "--query-mode", "serial");
	size_t	 k		   = std::stoul(std::string(getArg(args, "--k", "10")));
	size_t	 iters	   = std::stoul(std::string(getArg(args, "--iters", "500")));
	int		 threads   = std::stoi(std::string(getArg(args, "--threads", "1")));
	float	 noise	   = std::stof(std::string(getArg(args, "--noise", "0.01")));
	uint64_t seed	   = std::stoull(std::string(getArg(args, "--seed", "42")));
	bool	 saveNeigh = hasFlag(args, "--save-neighbours");

	if (threads < 1) {
		std::cerr << "--threads must be >= 1\n";
		return 1;
	}
	omp_set_num_threads(threads);

	std::unique_ptr<NNSIndex> index;
	std::string				  buildMeta, queryMeta;
	try {
		if (algo == "kdtree") {
			buildMeta = std::string(buildMode);
			queryMeta = std::string(queryMode);
			index	  = std::make_unique<KDTree>(
				parseBuildMode(buildMode), parseQueryMode(queryMode));
		} else if (algo == "exhaustive") {
			index	  = std::make_unique<ExhaustiveKNN>();
			buildMeta = "-";
			queryMeta = "-";
		} else {
			std::cerr << "Unknown --algo: " << algo << '\n';
			return 1;
		}
	} catch (const std::invalid_argument &e) {
		std::cerr << e.what() << '\n';
		return 1;
	}

	std::filesystem::create_directories(statsDir);
	auto buildPath = statsDir / (std::string(algo) + "_build.csv");
	auto queryPath = statsDir / (std::string(algo) + "_query.csv");
	auto neighPath = statsDir / (std::string(algo) + "_"
		+ std::filesystem::path(dataFile).stem().string() + "_neighbours.csv");

	DataLoader loader{dataDir / dataFile};
	Benchmark  bench(std::move(index), k, iters, buildMeta, queryMeta, noise, seed);
	bench.run(loader.get());

	if (algo != "exhaustive") {
		bench.writeBuildStats(buildPath);
		std::cout << "build  -> " << buildPath << '\n';
	}
	bench.writeQueryStats(queryPath);
	std::cout << "query  -> " << queryPath << '\n';

	if (saveNeigh) {
		bench.writeNeighbours(neighPath);
		std::cout << "neighb -> " << neighPath << '\n';
	}

	return 0;
}
