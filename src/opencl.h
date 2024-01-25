#pragma once

#include <CL/opencl.hpp>

#include <vector>
#include "image.h"


class OpenCL {
public:
	OpenCL();

	cl::Kernel compile(const std::string& code, const std::string& options = "");

	template<typename... Ts>
	cl::Event run(cl::Kernel kernel, const cl::EnqueueArgs& args, Ts... ts) {
		cl::KernelFunctor<Ts...> functor(std::move(kernel));
		return functor(args, std::forward<Ts>(ts)...);
	}

private:
	bool searchDevice(const std::vector<cl::Platform>& platforms, cl_device_type type);

	cl::Device device;
	cl::Context context;
	cl::CommandQueue queue;
};
