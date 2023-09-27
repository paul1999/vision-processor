#pragma once

#include <CL/opencl.hpp>

#include <vector>


class OpenCL {
public:
	OpenCL();

	cl::Kernel compile(const std::string& name, const std::string& code);

	template<typename... Ts>
	cl::Event run(cl::Kernel kernel, const cl::EnqueueArgs& args, Ts... ts) {
		cl::KernelFunctor<Ts...> functor(std::move(kernel));
		return functor(args, std::forward<Ts>(ts)...);
	}

private:
	cl::Device device;
	cl::Context context;
	cl::CommandQueue queue;
};
