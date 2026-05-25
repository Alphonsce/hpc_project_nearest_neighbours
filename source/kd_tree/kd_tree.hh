#ifndef SOURCE_KD_TREE_KD_TREE_HH_
#define SOURCE_KD_TREE_KD_TREE_HH_

#include "generic/functions.hh"
#include "generic/index.hh"
#include "kd_tree/policy.hh"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <numeric>
#include <omp.h>
#include <queue>
#include <stdexcept>

class KDTree : public NNSIndex {
public:
	explicit KDTree(BuildMode aBuild = BuildMode::Serial,
		QueryMode aQuery			 = QueryMode::Serial) :
		buildMode(aBuild), queryMode(aQuery)
	{
	}

	KDTree(const KDTree &)			  = delete;
	KDTree &operator=(const KDTree &) = delete;
	KDTree(KDTree &&)				  = delete;
	KDTree &operator=(KDTree &&)	  = delete;

	void build(PointsPtr aPoints) final
	{
		if (!aPoints || aPoints->empty())
			throw std::invalid_argument("KDTree::build: empty dataset");
		points = std::move(aPoints);

		std::vector<size_t> indices(points->size());
		std::iota(indices.begin(), indices.end(), 0);

		if (buildMode == BuildMode::ParallelFlat) {
			int cutoff = Detail::taskCutoff();
			std::vector<std::tuple<std::unique_ptr<Node>*, size_t, size_t, int>> work;
			work.reserve(size_t(2) << cutoff);

			root = Detail::buildTop(*points, indices, 0, indices.size(), 0, cutoff, work);

			int n = static_cast<int>(work.size());
#pragma omp parallel for schedule(dynamic)
			for (int i = 0; i < n; ++i) {
				auto &[slot, lo, hi, depth] = work[i];
				*slot = Detail::build(*points, indices, lo, hi, depth);
			}
		} else if (buildMode == BuildMode::ParallelTasks) {
			int cutoff = Detail::taskCutoff();
#pragma omp parallel
#pragma omp single
			root = Detail::buildTasks(
				*points, indices, 0, indices.size(), 0, cutoff);
		} else {
			root = Detail::build(*points, indices, 0, indices.size(), 0);
		}
	}

	BatchQueryResult query(const Points &aQueries, size_t aK) const final
	{
		if (!points)
			throw std::runtime_error(
				"KDTree::query: call build() before query()");
		if (aK == 0)
			throw std::invalid_argument("KDTree::query: k must be positive");

		BatchQueryResult results(aQueries.size());

		if (queryMode == QueryMode::LocalHeaps) {
			for (size_t i = 0; i < aQueries.size(); ++i)
				results[i] = Detail::queryLocalHeaps(
					*points, root.get(), aQueries[i], aK);

		} else if (queryMode == QueryMode::AtomicGlobal) {
			for (size_t i = 0; i < aQueries.size(); ++i)
				results[i] = Detail::queryAtomicGlobal(
					*points, root.get(), aQueries[i], aK);

		} else {
#pragma omp parallel for schedule(static)
			for (size_t i = 0; i < aQueries.size(); ++i)
				results[i]
					= Detail::querySingle(*points, root.get(), aQueries[i], aK);
		}

		return results;
	}

private:
	struct Node {
		size_t pointId{0};
		int axis{0};
		std::unique_ptr<Node> left;
		std::unique_ptr<Node> right;
	};

	struct Detail {

		static std::unique_ptr<Node> build(const Points &aPoints,
			std::vector<size_t> &aIndices, size_t aLo, size_t aHi, int aDepth)
		{
			if (aLo >= aHi)
				return nullptr;

			int axis   = aDepth % static_cast<int>(aPoints[0].size());
			size_t mid = aLo + (aHi - aLo) / 2;

			std::nth_element(aIndices.begin() + aLo, aIndices.begin() + mid,
				aIndices.begin() + aHi, [&](size_t a, size_t b) {
					return aPoints[a][axis] < aPoints[b][axis];
				});

			auto node	  = std::make_unique<Node>();
			node->pointId = aIndices[mid];
			node->axis	  = axis;
			node->left	  = build(aPoints, aIndices, aLo, mid, aDepth + 1);
			node->right	  = build(aPoints, aIndices, mid + 1, aHi, aDepth + 1);
			return node;
		}

		using WorkList = std::vector<std::tuple<std::unique_ptr<Node>*, size_t, size_t, int>>;

		static std::unique_ptr<Node> buildTop(const Points &aPoints,
			std::vector<size_t> &aIndices, size_t aLo, size_t aHi, int aDepth,
			int aCutoff, WorkList &aWork)
		{
			if (aLo >= aHi)
				return nullptr;

			int axis   = aDepth % static_cast<int>(aPoints[0].size());
			size_t mid = aLo + (aHi - aLo) / 2;

			std::nth_element(aIndices.begin() + aLo, aIndices.begin() + mid,
				aIndices.begin() + aHi, [&](size_t a, size_t b) {
					return aPoints[a][axis] < aPoints[b][axis];
				});

			auto node	  = std::make_unique<Node>();
			node->pointId = aIndices[mid];
			node->axis	  = axis;

			if (aDepth + 1 >= aCutoff) {
				aWork.emplace_back(&node->left,  aLo,     mid, aDepth + 1);
				aWork.emplace_back(&node->right, mid + 1, aHi, aDepth + 1);
			} else {
				node->left  = buildTop(aPoints, aIndices, aLo,     mid, aDepth + 1, aCutoff, aWork);
				node->right = buildTop(aPoints, aIndices, mid + 1, aHi, aDepth + 1, aCutoff, aWork);
			}

			return node;
		}

		static std::unique_ptr<Node> buildTasks(const Points &aPoints,
			std::vector<size_t> &aIndices, size_t aLo, size_t aHi, int aDepth,
			int aCutoff)
		{
			if (aLo >= aHi)
				return nullptr;

			int axis   = aDepth % static_cast<int>(aPoints[0].size());
			size_t mid = aLo + (aHi - aLo) / 2;

			std::nth_element(aIndices.begin() + aLo, aIndices.begin() + mid,
				aIndices.begin() + aHi, [&](size_t a, size_t b) {
					return aPoints[a][axis] < aPoints[b][axis];
				});

			auto node	  = std::make_unique<Node>();
			node->pointId = aIndices[mid];
			node->axis	  = axis;

			if (aDepth < aCutoff) {
				std::unique_ptr<Node> leftNode, rightNode;

#pragma omp task shared(leftNode)
				leftNode = buildTasks(
					aPoints, aIndices, aLo, mid, aDepth + 1, aCutoff);
#pragma omp task shared(rightNode)
				rightNode = buildTasks(
					aPoints, aIndices, mid + 1, aHi, aDepth + 1, aCutoff);
#pragma omp taskwait

				node->left	= std::move(leftNode);
				node->right = std::move(rightNode);
			} else {
				node->left  = build(aPoints, aIndices, aLo, mid, aDepth + 1);
				node->right = build(aPoints, aIndices, mid + 1, aHi, aDepth + 1);
			}

			return node;
		}

		static void traverse(const Points &aPoints, const Node *aNode,
			const Point &aQ, size_t aK, std::priority_queue<Neighbour> &aHeap,
			size_t &aNodesVisited)
		{
			if (!aNode)
				return;

			++aNodesVisited;

			float dist = L2norm(aQ, aPoints[aNode->pointId]);
			if (aHeap.size() < aK) {
				aHeap.push({dist, aNode->pointId});
			} else if (dist < aHeap.top().dist) {
				aHeap.pop();
				aHeap.push({dist, aNode->pointId});
			}

			float diff = aQ[aNode->axis] - aPoints[aNode->pointId][aNode->axis];

			const auto *near
				= diff <= 0 ? aNode->left.get() : aNode->right.get();
			const auto *far
				= diff <= 0 ? aNode->right.get() : aNode->left.get();

			traverse(aPoints, near, aQ, aK, aHeap, aNodesVisited);

			if (aHeap.size() < aK || diff * diff < aHeap.top().dist)
				traverse(aPoints, far, aQ, aK, aHeap, aNodesVisited);
		}

		static int taskCutoff()
		{
			int c = 0;
			for (int t = omp_get_max_threads(); (1 << c) < t; ++c);
			return c;
		}

		static void collectWork(const Points &aPoints, const Node *aNode,
			const Point &aQ, size_t aK, int aDepth, int aCutoff,
			std::priority_queue<Neighbour> &aUpperHeap, size_t &aUpperVisited,
			std::vector<const Node *> &aWork)
		{
			if (!aNode)
				return;

			++aUpperVisited;

			float dist = L2norm(aQ, aPoints[aNode->pointId]);
			if (aUpperHeap.size() < aK) {
				aUpperHeap.push({dist, aNode->pointId});
			} else if (dist < aUpperHeap.top().dist) {
				aUpperHeap.pop();
				aUpperHeap.push({dist, aNode->pointId});
			}

			if (aDepth >= aCutoff) {
				aWork.push_back(aNode);
				return;
			}

			collectWork(aPoints, aNode->left.get(), aQ, aK, aDepth + 1, aCutoff,
				aUpperHeap, aUpperVisited, aWork);
			collectWork(aPoints, aNode->right.get(), aQ, aK, aDepth + 1,
				aCutoff, aUpperHeap, aUpperVisited, aWork);
		}

		static void mergeInto(std::priority_queue<Neighbour> &aDst,
			std::priority_queue<Neighbour> &aSrc, size_t aK)
		{
			while (!aSrc.empty()) {
				const Neighbour &n = aSrc.top();
				if (aDst.size() < aK) {
					aDst.push(n);
				} else if (n.dist < aDst.top().dist) {
					aDst.pop();
					aDst.push(n);
				}
				aSrc.pop();
			}
		}

		static QueryResult extractResult(
			std::priority_queue<Neighbour> &aHeap, size_t aNodesVisited)
		{
			Neighbours nbrs;
			nbrs.reserve(aHeap.size());
			while (!aHeap.empty()) {
				auto n = aHeap.top();
				aHeap.pop();
				n.dist = std::sqrt(n.dist);
				nbrs.push_back(n);
			}
			std::sort(nbrs.begin(), nbrs.end());
			return {std::move(nbrs), aNodesVisited};
		}

		static QueryResult querySingle(const Points &aPoints, const Node *aRoot,
			const Point &aQ, size_t aK)
		{
			std::priority_queue<Neighbour> heap;
			size_t nodesVisited = 0;
			traverse(aPoints, aRoot, aQ, aK, heap, nodesVisited);
			return extractResult(heap, nodesVisited);
		}

		static QueryResult queryLocalHeaps(const Points &aPoints,
			const Node *aRoot, const Point &aQ, size_t aK)
		{
			int cutoff = taskCutoff();

			std::priority_queue<Neighbour> upperHeap;
			size_t upperVisited = 0;
			std::vector<const Node *> work;

			collectWork(aPoints, aRoot, aQ, aK, 0, cutoff, upperHeap,
				upperVisited, work);

			int n = static_cast<int>(work.size());
			std::vector<std::priority_queue<Neighbour>> localHeaps(n);
			std::vector<size_t> localVisited(n, 0);

#pragma omp parallel for schedule(dynamic)
			for (int i = 0; i < n; ++i)
				traverse(
					aPoints, work[i], aQ, aK, localHeaps[i], localVisited[i]);

			size_t totalVisited = upperVisited;
			for (int i = 0; i < n; ++i) {
				mergeInto(upperHeap, localHeaps[i], aK);
				totalVisited += localVisited[i];
			}

			return extractResult(upperHeap, totalVisited);
		}

		static void updateGlobalWorst(std::atomic<float> &aGlobal, float aVal)
		{
			float cur = aGlobal.load(std::memory_order_relaxed);
			while (aVal < cur
				&& !aGlobal.compare_exchange_weak(
					cur, aVal, std::memory_order_relaxed));
		}

		static void traverseAtomic(const Points &aPoints, const Node *aNode,
			const Point &aQ, size_t aK, std::priority_queue<Neighbour> &aHeap,
			size_t &aNodesVisited, std::atomic<float> &aGlobalWorst)
		{
			if (!aNode)
				return;

			++aNodesVisited;

			float dist = L2norm(aQ, aPoints[aNode->pointId]);
			if (aHeap.size() < aK) {
				aHeap.push({dist, aNode->pointId});
				if (aHeap.size() == aK)
					updateGlobalWorst(aGlobalWorst, aHeap.top().dist);
			} else if (dist < aHeap.top().dist) {
				aHeap.pop();
				aHeap.push({dist, aNode->pointId});
				updateGlobalWorst(aGlobalWorst, aHeap.top().dist);
			}

			float diff = aQ[aNode->axis] - aPoints[aNode->pointId][aNode->axis];

			const auto *near
				= diff <= 0 ? aNode->left.get() : aNode->right.get();
			const auto *far
				= diff <= 0 ? aNode->right.get() : aNode->left.get();

			traverseAtomic(
				aPoints, near, aQ, aK, aHeap, aNodesVisited, aGlobalWorst);

			float bound = std::min(aHeap.size() < aK ?
					std::numeric_limits<float>::infinity() :
					aHeap.top().dist,
				aGlobalWorst.load(std::memory_order_relaxed));

			if (aHeap.size() < aK || diff * diff < bound)
				traverseAtomic(
					aPoints, far, aQ, aK, aHeap, aNodesVisited, aGlobalWorst);
		}

		static QueryResult queryAtomicGlobal(const Points &aPoints,
			const Node *aRoot, const Point &aQ, size_t aK)
		{
			int cutoff = taskCutoff();

			std::priority_queue<Neighbour> upperHeap;
			size_t upperVisited = 0;
			std::vector<const Node *> work;

			collectWork(aPoints, aRoot, aQ, aK, 0, cutoff, upperHeap,
				upperVisited, work);

			float initWorst = upperHeap.size() == aK ?
				upperHeap.top().dist :
				std::numeric_limits<float>::infinity();
			std::atomic<float> globalWorst{initWorst};

			int n = static_cast<int>(work.size());
			std::vector<std::priority_queue<Neighbour>> localHeaps(n);
			std::vector<size_t> localVisited(n, 0);

#pragma omp parallel for schedule(dynamic)
			for (int i = 0; i < n; ++i)
				traverseAtomic(aPoints, work[i], aQ, aK, localHeaps[i],
					localVisited[i], globalWorst);

			size_t totalVisited = upperVisited;
			for (int i = 0; i < n; ++i) {
				mergeInto(upperHeap, localHeaps[i], aK);
				totalVisited += localVisited[i];
			}

			return extractResult(upperHeap, totalVisited);
		}
	};

	BuildMode buildMode;
	QueryMode queryMode;
	std::unique_ptr<Node> root;
};

#endif /* SOURCE_KD_TREE_KD_TREE_HH_ */
