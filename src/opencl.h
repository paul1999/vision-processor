#pragma once

#include <CL/opencl.hpp>

#include <vector>


class OpenCL {
public:
	OpenCL();

private:
	cl::Device device;
	cl::Context context;
	cl::CommandQueue queue;
};
