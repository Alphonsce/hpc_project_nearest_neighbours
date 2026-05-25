#ifndef SOURCE_GENERIC_FUNCTIONS_HH_
#define SOURCE_GENERIC_FUNCTIONS_HH_

#include <cstddef>
#include <stdexcept>

#include "types.hh"

// Squared L2 distance — used internally by all NNS implementations.
// Avoids sqrt on every comparison; callers take sqrt only on final results.
inline float L2norm(const Point &aA, const Point &aB)
{
	if (aA.size() != aB.size())
		throw std::invalid_argument("L2norm: inconsistent vector sizes");

	float sum = 0.0f;
	for (size_t i = 0; i < aA.size(); ++i) {
		const float d = aA[i] - aB[i];
		sum += d * d;
	}
	return sum;
}

#endif /* SOURCE_GENERIC_FUNCTIONS_HH_ */
