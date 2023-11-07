#pragma once

#include <vector>
#include "AlignedArray.h"

struct Run {
	int x, y, length;
};


class RLEVector {
public:

	void add(const Run& run);
	void add(int x, int y);
	bool contains(int x, int y);
	void clear();
	int size();

	void add(const RLEVector& vector);
	void subtract(const RLEVector& vector);
	std::vector<Run> getPart(int start, int end);

	const std::vector<Run>& getRuns() { return runs; }

	std::shared_ptr<AlignedArray> scanArea(AlignedArrayPool& arrayPool);

private:
	std::vector<Run> runs;
};
