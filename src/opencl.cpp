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

	//device.getInfo<CL_UNIFIED_SHARED_MEMORY_ACCESS_INTEL>();

	context = cl::Context(device);
	cl::Context::setDefault(context);
	queue = cl::CommandQueue(context, device);
	cl::CommandQueue::setDefault(queue);
}

cl::Kernel OpenCL::compile(const std::string& code, const std::string& options) {
	cl::Program::Sources sources;
	sources.emplace_back(code.c_str(), code.length());

	cl::Program program(context, sources);
	if (program.build({device}, options.c_str()) != CL_SUCCESS) {
		std::cerr << "Error during OpenCL kernel compilation: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
		exit(1);
	}

	std::vector<cl::Kernel> kernels;
	program.createKernels(&kernels);
	return kernels[0];
}

cl::Buffer OpenCL::toBuffer(bool read, std::shared_ptr<Image>& image) {
	return {(cl_mem_flags) CL_MEM_USE_HOST_PTR | (read ? CL_MEM_READ_ONLY : CL_MEM_WRITE_ONLY), (cl::size_type) image->getWidth()*image->getHeight()*image->pixelSize(), image->getData(), nullptr};
}
