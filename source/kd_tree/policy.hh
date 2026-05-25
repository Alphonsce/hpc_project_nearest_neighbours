#ifndef SOURCE_KD_TREE_POLICY_HH_
#define SOURCE_KD_TREE_POLICY_HH_

#include <stdexcept>
#include <string>
#include <string_view>

enum class BuildMode {
	Serial,        // single-threaded nth_element recursion
	ParallelTasks, // omp task tree, depth cutoff = floor(log2(nThreads))
	ParallelFlat,  // serial top phase + omp parallel for over subtrees
};

enum class QueryMode {
	Serial,       // Stage 1: serial traversal per query, batch parallelised
	LocalHeaps,   // Stage 3A: parallel traversal, thread-local heaps, merge
	AtomicGlobal, // Stage 3B: parallel traversal, shared atomic worst bound
};

inline BuildMode parseBuildMode(std::string_view s)
{
	if (s == "serial")          return BuildMode::Serial;
	if (s == "parallel-tasks")  return BuildMode::ParallelTasks;
	if (s == "parallel-flat")   return BuildMode::ParallelFlat;
	throw std::invalid_argument("unknown --build-mode '" + std::string(s)
		+ "' (serial|parallel-tasks|parallel-flat)");
}

inline QueryMode parseQueryMode(std::string_view s)
{
	if (s == "serial")         return QueryMode::Serial;
	if (s == "local-heaps")    return QueryMode::LocalHeaps;
	if (s == "atomic-global")  return QueryMode::AtomicGlobal;
	throw std::invalid_argument("unknown --query-mode '" + std::string(s)
		+ "' (serial|local-heaps|atomic-global)");
}

#endif /* SOURCE_KD_TREE_POLICY_HH_ */
