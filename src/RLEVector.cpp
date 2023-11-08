#include <algorithm>
#include "RLEVector.h"

void RLEVector::add(int x, int y) {
	add({x, y, 1});
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

static inline bool lowerOrEqual(const Run& value, const Run& element) {
	//return value.y < element.y || (value.y == element.y && value.x <= element.x); //Das letzte Element, für das diese Bedingung gilt oder .end()
	//return element.y < value.y || (element.y == value.y && element.x < value.x); //first value where this is false (lower_bound)
	//return value.y < element.y || (value.y == element.y && value.x < element.x); //first value where this is true
	return element.y < value.y || (element.y == value.y && element.x <= value.x);
	//return element.y < value.y || (element.y == value.y && element.x < value.x);
}

void RLEVector::add(const Run& run) {
	auto rpos = std::upper_bound(runs.rbegin(), runs.rend(), run, lowerOrEqual);
	auto pos = rpos.base();

	if(rpos == runs.rend()) {
		pos = runs.insert(pos, run);
	} else if(rpos->y < run.y || rpos->x + rpos->length < run.x) {
		pos = runs.insert(pos, run);
	} else {
		if (rpos->x + rpos->length >= run.x + run.length)
			return; // Already inside

		rpos->length = run.x + run.length - rpos->x;
		pos--;
	}

	auto next = pos+1;
	while(next != runs.end() && pos->x + pos->length >= next->x && pos->y == next->y) {
		pos->length = next->x + next->length - pos->x;
		next = runs.erase(next);
	}
}

void RLEVector::add(const RLEVector &vector) {
	for(const auto& run : vector.runs) {
		add(run);
	}
}

std::shared_ptr<AlignedArray> RLEVector::scanArea(AlignedArrayPool& arrayPool) {
	auto alignedArray = arrayPool.acquire<int>(2*size());
	int* array = alignedArray->mapWrite<int>();
	int i = 0;
	for(const Run& run : runs) {
		for(int x = run.x; x < run.x+run.length; x++) {
			array[i++] = x;
			array[i++] = run.y;
		}
	}
	alignedArray->unmap();
	return alignedArray;
}
