#ifndef SOURCE_EXHAUSTIVE_KNN_EXHAUSTIVE_KNN_HH_
#define SOURCE_EXHAUSTIVE_KNN_EXHAUSTIVE_KNN_HH_

#include "generic/functions.hh"
#include "generic/index.hh"

#include <algorithm>
#include <cmath>
#include <omp.h>
#include <queue>

// Flat linear-scan KNN.
//
// Build: O(1)      — stores the data pointer.
// Query: O(n log k) per query — scans all n points, maintains a max-heap of
//                   size k.  Parallelised across the batch via OpenMP.
class ExhaustiveKNN : public NNSIndex {
public:
	ExhaustiveKNN()									= default;
	ExhaustiveKNN(const ExhaustiveKNN &)			= delete;
	ExhaustiveKNN &operator=(const ExhaustiveKNN &) = delete;
	ExhaustiveKNN(ExhaustiveKNN &&)					= delete;
	ExhaustiveKNN &operator=(ExhaustiveKNN &&)		= delete;

	void build(PointsPtr aPoints) final
	{
		if (!aPoints || aPoints->empty())
			throw std::invalid_argument("ExhaustiveKNN::build: empty dataset");
		points = std::move(aPoints);
	}

	BatchQueryResult query(const Points &aQueries, size_t aK) const final
	{
		if (!points)
			throw std::runtime_error(
				"ExhaustiveKNN::query: call build() before query()");
		if (aK == 0)
			throw std::invalid_argument(
				"ExhaustiveKNN::query: k must be positive");

		BatchQueryResult results(aQueries.size());

#pragma omp parallel for schedule(static)
		for (size_t i = 0; i < aQueries.size(); ++i) {
			const Point &q = aQueries[i];

			// Distances in the heap are *squared* L2; sqrt only on final k.
			std::priority_queue<Neighbour> heap;

			for (size_t j = 0; j < points->size(); ++j) {
				float norm = L2norm(q, (*points)[j]);
				if (heap.size() < aK) {
					heap.push({norm, j});
				} else if (norm < heap.top().dist) {
					heap.pop();
					heap.push({norm, j});
				}
			}

			Neighbours nbrs;
			nbrs.reserve(heap.size());
			while (!heap.empty()) {
				auto n = heap.top();
				heap.pop();
				n.dist = std::sqrt(n.dist);
				nbrs.push_back(n);
			}
			std::sort(nbrs.begin(), nbrs.end());
			results[i] = {std::move(nbrs), points->size()};
		}

		return results;
	}
};

#endif /* SOURCE_EXHAUSTIVE_KNN_EXHAUSTIVE_KNN_HH_ */
