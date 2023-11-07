#pragma once

#include <CL/opencl.hpp>
#include <sys/user.h>
#include <iostream>
#include <map>

class AlignedArray {
public:
	explicit AlignedArray(int size): size(size) {
		data = std::aligned_alloc(PAGE_SIZE, size);
		buffer = cl::Buffer((cl_mem_flags) CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, (cl::size_type) size, data, nullptr);
	}
	~AlignedArray() {
		std::free(data);
	}

	template<typename T>
	const T* mapRead() {
		map = cl::enqueueMapBuffer(buffer, true, CL_MAP_READ, 0, size);
		return (T*)map;
	}
	template<typename T>
	T* mapWrite() {
		map = cl::enqueueMapBuffer(buffer, true, CL_MAP_WRITE_INVALIDATE_REGION, 0, size);
		return (T*)map;
	}
	template<typename T>
	T* mapPartialWrite() {
		map = cl::enqueueMapBuffer(buffer, true, CL_MAP_WRITE, 0, size);
		return (T*)map;
	}
	void unmap() {
		cl::enqueueUnmapMemObject(buffer, map);
	}

	[[nodiscard]] cl::Buffer getBuffer() { return buffer; }

	template<typename T>
	[[nodiscard]] int getSize() const {
		return size/sizeof(T);
	}

private:
	const int size;

	void* data;
	void* map;
	cl::Buffer buffer;
};

//TODO threadsafety?
//Pool design adapted from Jonathan Mee https://stackoverflow.com/a/27828584 CC BY-SA 3.0
class AlignedArrayPool {
public:
	template<typename T>
	std::shared_ptr<AlignedArray> acquire(int size) {
		size *= sizeof(T);
		if(size % PAGE_SIZE != 0) {
			size -= size % PAGE_SIZE;
			size += PAGE_SIZE;
		}

		auto& sizedPool = pool[size];

		auto iterator = std::find_if(sizedPool.begin(), sizedPool.end(), [](const std::shared_ptr<AlignedArray>& i){return i.use_count() == 1;});
		if(iterator != sizedPool.end())
			return *iterator;

		auto array = std::make_shared<AlignedArray>(size);
		sizedPool.push_back(array);
		return std::move(array);
	}

private:
	std::map<int, std::vector<std::shared_ptr<AlignedArray>>> pool;
};
