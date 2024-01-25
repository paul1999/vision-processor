#include "opencl.h"

#include <iostream>
#include <utility>


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

	//CL_DEVICE_GLOBAL_MEM_CACHE_TYPE
	//CL_DEVICE_EXTENSIONS

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
	program.createKernels(&kernels);
	return kernels[0];
}