#pragma once

#include "opencl.h"

#include <map>

//Acquiring PAGE_SIZE (adapted from https://stackoverflow.com/a/2460809 CC BY-SA 2.5 by Giuseppe Guerrini)
#ifdef _WIN32
#include <w32api/ddk/winddk.h>
#else
#include <sys/user.h>
#include <any>

#endif

//TODO threadsafety?
//Pool design adapted from Jonathan Mee https://stackoverflow.com/a/27828584 CC BY-SA 3.0
class AlignedArrayPool {
public:
	template<typename T>
	std::shared_ptr<CLArray> acquire(int size) {
		size *= sizeof(T);
		if(size % PAGE_SIZE != 0) {
			size -= size % PAGE_SIZE;
			size += PAGE_SIZE;
		}

		auto& sizedPool = pool[size];

		auto iterator = std::find_if(sizedPool.begin(), sizedPool.end(), [](const std::shared_ptr<CLArray>& i){return i.use_count() == 1;});
		if(iterator != sizedPool.end())
			return *iterator;

		auto array = std::make_shared<CLArray>(size);
		sizedPool.push_back(array);
		return std::move(array);
	}

private:
	std::map<int, std::vector<std::shared_ptr<CLArray>>> pool;
};
