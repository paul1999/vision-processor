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

	bool deviceFound = false;
	for(const cl::Platform& platform : platforms) {
		std::vector<cl::Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

		for(cl::Device& d : devices) {
			cl_device_type type = d.getInfo<CL_DEVICE_TYPE>();
			if(type & CL_DEVICE_TYPE_GPU) {
				deviceFound = true;
				device = d;
				std::cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
				std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << "\n";
				break;
			}
		}

		if(deviceFound)
			break;
	}

	if(!deviceFound){
		std::cerr << "No GPU devices found. Check OpenCL installation!" << std::endl;
		exit(1);
	}

	context = cl::Context(device);
	cl::Context::setDefault(context);
	queue = cl::CommandQueue(context, device);
	cl::CommandQueue::setDefault(queue);
}

cl::Kernel OpenCL::compile(const std::string& name, const std::string& code) {
	cl::Program::Sources sources;
	sources.emplace_back(code.c_str(), code.length());

	cl::Program program(context, sources);
	if (program.build({device}) != CL_SUCCESS) {
		std::cout << "Error during OpenCL kernel compilation: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
		exit(1);
	}

	return {program, name.c_str()};
}

template<typename... Ts>
void OpenCL::run(cl::Kernel kernel, const cl::EnqueueArgs& args, Ts... ts) {
	cl::KernelFunctor functor(std::move(kernel));
	//queue, cl::NullRange, cl::NDRange(10), cl::NullRange
	//TODO defaultqueue?
	//cl::EnqueueArgs args(queue, cl::NDRange(10));
	functor(args, std::forward<Ts>(ts)...).wait();
}
