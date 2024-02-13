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

static inline bool lowerOrEqual(const Run& value, const Run& element) {
	return element.y < value.y || (element.y == value.y && element.x <= value.x);
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
		pos->length = std::max(next->x + next->length, pos->x + pos->length) - pos->x;
		next = runs.erase(next);
	}
}

void RLEVector::remove(const Run &run) {
	auto rpos = std::upper_bound(runs.rbegin(), runs.rend(), run, lowerOrEqual); // rpos prior to first conflicting run
	auto pos = rpos.base(); //base points to first element after rpos

	if(rpos == runs.rend() || rpos->y < run.y || rpos->x + rpos->length < run.x) {
		// at start or rpos completely prior to first run
	} else if (rpos->x < run.x) {
		// rpos starts prior to run
		int length = rpos->length;
		rpos->length = run.x - rpos->x;
		if(rpos->x + length > run.x + run.length) {
			// and stops prior to run
			add(Run{run.x+run.length, rpos->y, (rpos->x+length) - (run.x+run.length)});
			return;
		}
	} else {
		pos--;
	}

	// from here on run start guaranteed prior to pos
	while(pos != runs.end() && run.y == pos->y && run.x + run.length >= pos->x) {
		if (pos->x + pos->length > run.x + run.length) {
			pos->length = (pos->x + pos->length) - (run.x + run.length);
			pos->x = run.x + run.length;
			return;
		}

		pos = runs.erase(pos);
	}
}

void RLEVector::add(const RLEVector &vector) {
	for(const auto& run : vector.runs) {
		add(run);
	}
}

void RLEVector::remove(const RLEVector &vector) {
	for(const auto& run : vector.runs) {
		remove(run);
	}
}

std::shared_ptr<CLArray> RLEVector::scanArea(AlignedArrayPool& arrayPool) {
	auto alignedArray = arrayPool.acquire<int>(2*size());
	CLMap<int> array = alignedArray->write<int>();
	int i = 0;
	for(const Run& run : runs) {
		for(int x = run.x; x < run.x+run.length; x++) {
			array[i++] = x;
			array[i++] = run.y;
		}
	}
	return alignedArray;
}