#pragma once

#include <CL/opencl.hpp>

#include <vector>
#include <iostream>
#include <opencv2/core/mat.hpp>


class PixelFormat {
public:
	static const PixelFormat RGBA8;
	static const PixelFormat RGGB8;
	static const PixelFormat BGR888;
	static const PixelFormat U8;
	static const PixelFormat I8;
	static const PixelFormat F32;
	static const PixelFormat NV12;

	[[nodiscard]] int pixelSize() const { return stride*rowStride; }

	const int stride;
	const int rowStride;
	const bool color;
	const int cvType;

	const std::string clKernelToNV12;
private:
	PixelFormat(int stride, int rowStride, bool color, int cvType, std::string clKernelToNV12): stride(stride), rowStride(rowStride), color(color), cvType(cvType), clKernelToNV12(std::move(clKernelToNV12)) {}
};


typedef struct __attribute__ ((packed)) RGBA {
	cl_uchar r;
	cl_uchar g;
	cl_uchar b;
	cl_uchar a;
} RGBA;


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

	template<typename... Ts>
	void await(cl::Kernel kernel, const cl::EnqueueArgs& args, Ts... ts) {
		wait(run(kernel, args, std::forward<Ts>(ts)...));
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
class CLImageMap;


class CLImage {
public:
	explicit CLImage(const PixelFormat* format);
	CLImage(const PixelFormat* format, int width, int height, std::string name);

	template<typename T> CLImageMap<T> read() const { return std::move(CLImageMap<T>(*this, CL_MAP_READ)); }
	template<typename T> CLImageMap<T> write() { return std::move(CLImageMap<T>(*this, CL_MAP_WRITE_INVALIDATE_REGION)); }
	template<typename T> CLImageMap<T> readWrite() { return std::move(CLImageMap<T>(*this, CL_MAP_WRITE)); }

	void save(const std::string& suffix, float factor = 1.0f, float offset = 0.0f) const;

	cl::Image2D image;

	const PixelFormat* format;
	int width;
	int height;
	std::string name;
};


template<typename T>
class CLImageMap {
public:
	explicit CLImageMap(const CLImage& image, int clRWType): image(image.image) {
		int error;
		size_t origin[]{0, 0, 0};
		size_t region[]{(size_t)image.width, (size_t)image.height, 1};
		map = (T*) clEnqueueMapImage(cl::CommandQueue::getDefault()(), image.image(), true, clRWType, origin, region, &bytePitch, nullptr, 0, nullptr, nullptr, &error);
		if(error != CL_SUCCESS) {
			std::cerr << "[OpenCL] Enqueue map image error: " << error << std::endl;
			exit(1);
		}
		rowPitch = bytePitch/sizeof(T);
		cv = ::cv::Mat(image.height, image.width, image.format->cvType, map, bytePitch);
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

	CLImageMap (CLImageMap&& other) noexcept: image(std::move(other.image)), map(std::move(other.map)), bytePitch(other.bytePitch), rowPitch(other.rowPitch), cv(other.cv) {
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