#pragma once

#include <CL/opencl.hpp>

#include <vector>


class OpenCL {
public:
	OpenCL();

	cl::Kernel compile(const std::string& name, const std::string& code);

	template<typename... Ts>
	void run(cl::Kernel kernel, const cl::EnqueueArgs& args, Ts... ts);

private:
	cl::Device device;
	cl::Context context;
	cl::CommandQueue queue;
};
