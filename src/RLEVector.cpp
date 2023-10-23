#include "RLEVector.h"

void RLEVector::add(int x, int y) {
	for(auto it = runs.begin(); it != runs.end(); it++) {
		Run& run = *it;
		if(run.y < y)
			continue;

		if(run.y > y) {
			runs.insert(it, {x, y, 1});
			return;
		}

		if(run.x-1 > x) {
			runs.insert(it, {x, y, 1});
			return;
		}

		if(run.x + run.length <= x) {
			run.x = std::min(run.x, x);
			run.length = std::max(run.length, 1 + x - run.x);
			return;
		}
	}

	runs.push_back({x, y, 1});
}

void RLEVector::clear() {
	runs.clear();
}
