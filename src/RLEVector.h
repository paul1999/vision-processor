#pragma once

#include <vector>

struct Run {
	int x, y, length;
};


class RLEVector {
public:

	void add(int x, int y);
	void clear();
	std::vector<Run>& getRuns() { return runs; }

private:
	std::vector<Run> runs;
};
