#pragma once

#include <CL/opencl.hpp>

#include <vector>
#include <iostream>
#include <opencv2/core/mat.hpp>


class OpenCL {
public:
	OpenCL();

	cl::Kernel compileFile(const std::string& path, const std::string& options = "");
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
			OpenCL::wait(event);
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

template<typename T>
class CLImageMap {
public:
	explicit CLImageMap(const cl::Image2D& image, int width, int height, int clRWType): image(image) {
		int error;
		size_t origin[]{0, 0, 0};
		size_t region[]{(size_t)width, (size_t)height, 0};
		map = (T*) clEnqueueMapImage(cl::CommandQueue::getDefault()(), image(), true, clRWType, origin, region, &bytePitch, nullptr, 0, nullptr, nullptr, &error);
		if(error != CL_SUCCESS) {
			std::cerr << "[OpenCL] Enqueue map image error: " << error << std::endl;
			exit(1);
		}
		rowPitch = bytePitch/sizeof(T);
		cv = ::cv::Mat(height, width, CV_8SC4, *map);
		cv.step[0] = bytePitch;
	}
	~CLImageMap() {
		if(unmoved) {
			cl::Event event;
			int error = cl::enqueueUnmapMemObject(image, map, nullptr, &event);
			if(error != CL_SUCCESS) {
				std::cerr << "[OpenCL] Enqueue unmap image error: " << error << std::endl;
				exit(1);
			}
			OpenCL::wait(event);
		}
	}

	CLImageMap (CLImageMap&& other) noexcept: image(std::move(other.image)), map(std::move(other.map)), bytePitch(other.bytePitch), rowPitch(other.rowPitch) {
		other.unmoved = false;
	}
	CLImageMap ( const CLImageMap & ) = delete;
	CLImageMap& operator= ( const CLImageMap & ) = delete;
	T*& operator*() { return map; }
	T* operator-> () { return map; }
	T& operator [] (int i) { return map[i]; }
	T& operator()(int x, int y) { return map[x + y * rowPitch]; }
	const T* const& operator*() const { return map; }
	const T* operator-> () const { return map; }
	const T& operator [] (int i) const { return map[i]; }

	size_t bytePitch;
	size_t rowPitch;
	cv::Mat cv;

private:
	const cl::Image2D image;
	T* map;
	bool unmoved = true;
};

class CLImage {
public:
	CLImage(int width, int height, bool u);

	template<typename T> CLImageMap<T> read() const { return std::move(CLImageMap<T>(image, width, height, CL_MAP_READ)); }
	template<typename T> CLImageMap<T> write() { return std::move(CLImageMap<T>(image, width, height, CL_MAP_WRITE_INVALIDATE_REGION)); }
	template<typename T> CLImageMap<T> readWrite() { return std::move(CLImageMap<T>(image, width, height, CL_MAP_WRITE)); }

	const cl::Image2D image;
	const int width;
	const int height;
};
