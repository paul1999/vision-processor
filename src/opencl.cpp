#include "opencl.h"

#include <iostream>
#include <fstream>


OpenCL::OpenCL() {
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	if (platforms.empty()) {
		std::cerr << "No platforms found. Check OpenCL installation!" << std::endl;
		exit(1);
	}

	if(!searchDevice(platforms, CL_DEVICE_TYPE_GPU) && !searchDevice(platforms, CL_DEVICE_TYPE_ALL)) {
		std::cerr << "No GPU devices found. Check OpenCL installation!" << std::endl;
		exit(1);
	}

	context = cl::Context(device);
	cl::Context::setDefault(context);
	queue = cl::CommandQueue(context, device);
	cl::CommandQueue::setDefault(queue);
}

bool OpenCL::searchDevice(const std::vector<cl::Platform>& platforms, cl_device_type type) {
	for(const cl::Platform& platform : platforms) {
		std::vector<cl::Device> devices;
		platform.getDevices(type, &devices);

		for(cl::Device& d : devices) {
			device = d;
			std::cout << "[OpenCL] Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl
					  << "[OpenCL] Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
			return true;
		}
	}

	return false;
}

cl::Kernel OpenCL::compile(const std::string& code, const std::string& options) {
	cl::Program::Sources sources;
	sources.emplace_back(code.c_str(), code.length());

	cl::Program program(context, sources);
	if (program.build({device}, options.c_str()) != CL_SUCCESS) {
		std::cerr << "[OpenCL] Error during kernel compilation: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
		exit(1);
	}

	std::vector<cl::Kernel> kernels;
	int error = program.createKernels(&kernels);
	if(error != CL_SUCCESS) {
		std::cerr << "[OpenCL] Error during kernel creation: " << error << std::endl;
		exit(1);
	}
	if(kernels.empty()) {
		std::cerr << "[OpenCL] Kernel missing: " << code << std::endl;
		exit(1);
	}
	return kernels[0];
}

void OpenCL::wait(const cl::Event& event) {
	int error = event.wait();
	if(error != CL_SUCCESS) {
		std::cerr << "[OpenCL] Error during kernel execution: " << error << std::endl;
		exit(1);
	}
}

cl::Kernel OpenCL::compileFile(const std::string& path, const std::string& options) {
	std::ifstream sourceFile(path);
	std::string sourceCode( std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
	return compile(sourceCode, options);
}

static inline cl::Buffer clAlloc(cl_mem_flags type, cl::size_type size, void* data) {
	int error;
	cl::Buffer buffer(type | CL_MEM_READ_WRITE, size, data, &error);
	if(error != CL_SUCCESS) {
		std::cerr << "[OpenCL] Error during image allocation: " << error << std::endl;
		exit(1);
	}
	return std::move(buffer);
}

CLArray::CLArray(int size): size(size), buffer(clAlloc((cl_mem_flags) CL_MEM_ALLOC_HOST_PTR, (cl::size_type) size, nullptr)) {}
CLArray::CLArray(void* data, const int size) : size(size), buffer(clAlloc((cl_mem_flags) CL_MEM_COPY_HOST_PTR, (cl::size_type) size, data)) {}

static inline cl::Image2D allocImage(int width, int height, bool f) {
	int error;
	cl::Image2D image = cl::Image2D(cl::Context::getDefault(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, f ? cl::ImageFormat(CL_INTENSITY, CL_FLOAT) : cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8), width, height, 0, nullptr, &error);
	if(error != CL_SUCCESS) {
		std::cerr << "bgr Image creation error: " << error << std::endl;
		exit(1);
	}
	return std::move(image);
}

CLImage::CLImage(int width, int height, bool f): image(allocImage(width, height, f)), width(width), height(height), f(f) {}
