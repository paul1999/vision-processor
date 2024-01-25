#pragma once

#include <iostream>
#include <CL/opencl.hpp>
#include <utility>


template<typename T>
class CLMap {
public:
	explicit CLMap(const cl::Buffer& buffer, int size, int clRWType): buffer(buffer) {
		map = (T*) cl::enqueueMapBuffer(buffer, true, clRWType, 0, size);
	}
	~CLMap() {
		if(unmoved)
			cl::enqueueUnmapMemObject(buffer, map);
	}

	CLMap (CLMap&& other) noexcept: buffer(std::move(other.buffer)), map(std::move(other.map)) {
		other.unmoved = false;
	}
	CLMap ( const CLMap & ) = delete;
	CLMap& operator= ( const CLMap & ) = delete;
	T*& operator*() { return map; }
	T* operator-> () { return map; }
	T& operator [] (int i) { return map[i]; }
	const T* const& operator*() const { return map; }
	const T* operator-> () const { return map; }
	const T& operator [] (int i) const { return map[i]; }

private:
	const cl::Buffer buffer;
	T* map;
	bool unmoved = true;
};

class CLArray {
public:
	explicit CLArray(int size): size(size), buffer((cl_mem_flags) CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, (cl::size_type) size, nullptr, nullptr) {}
	//data = std::aligned_alloc(PAGE_SIZE, size);
	CLArray(void* data, const int size) : size(size), buffer((cl_mem_flags) CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, (cl::size_type) size, data, nullptr) {}

	template<typename T> CLMap<T> read() const { return std::move(CLMap<T>(buffer, size, CL_MAP_READ)); }
	template<typename T> CLMap<T> write() { return std::move(CLMap<T>(buffer, size, CL_MAP_WRITE_INVALIDATE_REGION)); }
	template<typename T> CLMap<T> readWrite() { return std::move(CLMap<T>(buffer, size, CL_MAP_WRITE)); }

	const cl::Buffer buffer; //TODO releaseMemObject

private:
	const int size;
};