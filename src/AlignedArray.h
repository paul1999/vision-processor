#pragma once

#include <CL/opencl.hpp>
#include <sys/user.h>
#include <iostream>

template<typename T>
class AlignedArray {
public:
	explicit AlignedArray(int size): size(size) {
		data = std::aligned_alloc(PAGE_SIZE, size * sizeof(T));
		buffer = cl::Buffer((cl_mem_flags) CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, (cl::size_type) size * sizeof(T), data, nullptr);
	}

	~AlignedArray() {
		std::free(data);
	}

	const T* mapRead() {
		if(cl::enqueueMapBuffer(buffer, true, CL_MAP_READ, 0, size*sizeof(T)) != data) {
			std::cerr << "[AlignedArray] mapped memory region differs, read will fail." << std::endl;
		}
		return data;
	}
	T* mapWrite() {
		if(cl::enqueueMapBuffer(buffer, true, CL_MAP_WRITE_INVALIDATE_REGION, 0, size*sizeof(T)) != data) {
			std::cerr << "[AlignedArray] mapped memory region differs, read will fail." << std::endl;
		}
		return data;
	}
	void unmap() {
		cl::enqueueUnmapMemObject(buffer, data);
	}

	[[nodiscard]] int getSize() const {
		return size;
	}

private:
	const int size;

	T* data;
	cl::Buffer buffer;
};
