#include "RLEVector.h"

void RLEVector::add(int x, int y) {
	//TODO remove runs on run merging
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

bool RLEVector::contains(int x, int y) {
	auto it = runs.cbegin();
	auto end = runs.cend();
	while(it != end && it->y < y)
		it++;

	while(it != end && it->y == y) {
		if(it->x <= x && it->x + it->length > x)
			return true;

		it++;
	}

	return false;
}

void RLEVector::clear() {
	runs.clear();
}

int RLEVector::size() {
	int size = 0;
	for(const Run& run : runs) {
		size += run.length;
	}
	return size;
}

std::vector<Run> RLEVector::getPart(int start, int end) {
	std::vector<Run> result;

	int pos = 0;
	auto it = runs.cbegin();
	while(pos+it->length < start) {
		pos += it->length;
		it++;
	}

	while(pos+it->length < end) {
		result.push_back(*it++);
	}

	if(pos < end) {
		result.push_back({it->x, it->y, end - pos});
	}

	return result;
}

void RLEVector::subtract(const RLEVector &vector) {
	//TODO
}

void RLEVector::add(const RLEVector &vector) {
	for(const auto& run : vector.runs) {
		for(int x = run.x; x < run.x+run.length; x++)
			add(x, run.y);
	}
}
