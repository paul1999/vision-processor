#pragma once

#include <CL/opencl.hpp>

#include <vector>
#include <iostream>


class OpenCL {
public:
	OpenCL();

	cl::Kernel compile(const std::string& code, const std::string& options = "");

	template<typename... Ts>
	cl::Event run(cl::Kernel kernel, const cl::EnqueueArgs& args, Ts... ts) {
		cl::KernelFunctor<Ts...> functor(std::move(kernel));
		cl_int error;
		cl::Event event = functor(args, std::forward<Ts>(ts)..., error);
		if(error != CL_SUCCESS) {
			std::cerr << "[OpenCL] Enqueue kernel error: " << error << std::endl;
			exit(1);
		}
		return event;
	}

	static void wait(const cl::Event& event);

private:
	bool searchDevice(const std::vector<cl::Platform>& platforms, cl_device_type type);

	cl::Device device;
	cl::Context context;
	cl::CommandQueue queue;
};

template<typename T>
class CLMap {
public:
	explicit CLMap(const cl::Buffer& buffer, int size, int clRWType): buffer(buffer) {
		int error;
		map = (T*) cl::enqueueMapBuffer(buffer, true, clRWType, 0, size, nullptr, nullptr, &error);
		if(error != CL_SUCCESS) {
			std::cerr << "[OpenCL] Enqueue map buffer error: " << error << std::endl;
			exit(1);
		}
	}
	~CLMap() {
		if(unmoved) {
			cl::Event event;
			int error = cl::enqueueUnmapMemObject(buffer, map, nullptr, &event);
			if(error != CL_SUCCESS) {
				std::cerr << "[OpenCL] Enqueue unmap buffer error: " << error << std::endl;
				exit(1);
			}
			event.wait(); //TODO utilize EventList to increase asynchronicity?
		}
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
	explicit CLArray(int size);
	CLArray(void* data, int size);

	template<typename T> CLMap<T> read() const { return std::move(CLMap<T>(buffer, size, CL_MAP_READ)); }
	template<typename T> CLMap<T> write() { return std::move(CLMap<T>(buffer, size, CL_MAP_WRITE_INVALIDATE_REGION)); }
	template<typename T> CLMap<T> readWrite() { return std::move(CLMap<T>(buffer, size, CL_MAP_WRITE)); }

	const cl::Buffer buffer;
	const int size;
};
