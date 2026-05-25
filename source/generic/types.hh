#ifndef SOURCE_GENERIC_TYPES_HH_
#define SOURCE_GENERIC_TYPES_HH_

#include <cstddef>
#include <memory>
#include <vector>

struct Neighbour {
	float  dist{0.0};
	size_t idx{0};

	bool operator<(const Neighbour &aOther) const { return dist < aOther.dist; }
};

using Point		 = std::vector<float>;
using Points	 = std::vector<Point>;
using PointsPtr	 = std::shared_ptr<Points>;
using Neighbours = std::vector<Neighbour>;

struct QueryResult {
	Neighbours neighbours;
	size_t	   nodesVisited{0};
};

// One QueryResult per input query point.
using BatchQueryResult = std::vector<QueryResult>;

#endif /* SOURCE_GENERIC_TYPES_HH_ */
