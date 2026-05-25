#ifndef SOURCE_GENERIC_INDEX_HH_
#define SOURCE_GENERIC_INDEX_HH_

#include "types.hh"

class NNSIndex {
public:
	NNSIndex()					= default;
	virtual ~NNSIndex()			= default;

	virtual void			 build(PointsPtr aPoints)								  = 0;
	virtual BatchQueryResult query(const Points &aQueries, size_t aK) const = 0;

protected:
	PointsPtr points;
};

#endif /* SOURCE_GENERIC_INDEX_HH_ */
