#pragma once

#include <utility>
#include <vector>
#include "AlignedArray.h"

struct Run {
	int x, y, length;
};


class RLEVector {
public:
	RLEVector() = default;
	explicit RLEVector(const std::vector<Run>& runs): runs(runs) {}

	void add(const Run& run);
	void remove(const Run& run);
	void add(int x, int y);
	bool contains(int x, int y);
	void clear();
	int size();

	void add(const RLEVector& vector);
	void remove(const RLEVector& vector);
	RLEVector getPart(int start, int end);

	const std::vector<Run>& getRuns() { return runs; }

	std::shared_ptr<CLArray> scanArea(AlignedArrayPool& arrayPool);

private:
	std::vector<Run> runs;
};
