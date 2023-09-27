#pragma once

#include <CL/opencl.hpp>

#include <vector>


class OpenCL {
public:
	OpenCL();

	cl::Kernel compile(const std::string& name, const std::string& code);

	template<typename... Ts>
	void run(cl::Kernel kernel, const cl::EnqueueArgs& args, Ts... ts) {
		cl::KernelFunctor functor(std::move(kernel));
		//queue, cl::NullRange, cl::NDRange(10), cl::NullRange
		//TODO defaultqueue?
		//cl::EnqueueArgs args(queue, cl::NDRange(10));
		functor(args, std::forward<Ts>(ts)...).wait();
	}

private:
	cl::Device device;
	cl::Context context;
	cl::CommandQueue queue;
};
