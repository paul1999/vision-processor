#include "opencl.h"

#include <iostream>
#include <fstream>
#include <utility>
#include <opencv2/imgcodecs.hpp>


const PixelFormat PixelFormat::RGBA8 = PixelFormat(4, 1, true, CV_8UC4, "NOT IMPLEMENTED");
const PixelFormat PixelFormat::RGGB8 = PixelFormat(2, 2, true, CV_8UC1, "void kernel c(global const uchar* in, global uchar* out) {"
																		"	const int i0 = 2*get_global_id(0) + 2*get_global_id(1)*2*get_global_size(0);"
																		"	const int i1 = i0 + 2*get_global_size(0);"
																		"	const int uvout = UV_OFFSET + get_global_id(0)/2*2 + get_global_id(1)/2*get_global_size(0);"
																		"	const short r = in[i0]; const short g0 = in[i0+1]; const short g1 = in[i1]; const short b = in[i1+1];"
																		"	out[get_global_id(0) + get_global_id(1)*get_global_size(0)] = (uchar)((66*r + 64*g0 + 65*g1 + 25*b) / 256 + 16);"
																		"	out[uvout] = (uchar)((-38*r + -37*g0 + -37*g1 + 112*b) / 256 + 128);"
																		"  out[uvout+1] = (uchar)((112*r + -47*g0 + -47*g1 + -18*b) / 256 + 128);"
																		"}");
const PixelFormat PixelFormat::BGR888 = PixelFormat(3, 1, true, CV_8UC3, "void kernel c(global const uchar* in, global uchar* out) {"
																		 "	const int i = 3*get_global_id(0) + get_global_id(1)*3*get_global_size(0);"
																		 "	const int uvout = UV_OFFSET + get_global_id(0)/2*2 + get_global_id(1)/2*get_global_size(0);"
																		 "	const short b = in[i]; const short g = in[i+1]; const short r = in[i+2];"
																		 "	out[get_global_id(0) + get_global_id(1)*get_global_size(0)] = (uchar)((66*r + 129*g + 25*b) / 256 + 16);"
																		 "	out[uvout] = (uchar)((-38*r + -74*g + 112*b) / 256 + 128);"
																		 "  out[uvout+1] = (uchar)((112*r + -94*g + -18*b) / 256 + 128);"
																		 "}");
const PixelFormat PixelFormat::U8 = PixelFormat(1, 1, false, CV_8UC1, "void kernel c(global const uchar* in, global uchar* out) { int i = get_global_id(0) + get_global_id(1)*get_global_size(0); out[i] = in[i]; }");
const PixelFormat PixelFormat::I8 = PixelFormat(1, 1, false, CV_8UC1, "void kernel c(global const char* in, global uchar* out) { int i = get_global_id(0) + get_global_id(1)*get_global_size(0); out[i] = (uchar)in[i] + 127; }");
const PixelFormat PixelFormat::F32 = PixelFormat(4, 1, false, CV_32FC1, "void kernel c(global const float* in, global uchar* out) { int i = get_global_id(0) + get_global_id(1)*get_global_size(0); out[i] = (uchar)in[i]; }");  //(uchar)fabs(in[i]) + 127
//TODO oversized du to planar architecture (1.5)
const PixelFormat PixelFormat::NV12 = PixelFormat(1, 2, true, CV_8UC1, "void kernel c(global const uchar* in, global uchar* out) {"
																	   "	const int yi = get_global_id(0) + get_global_id(1)*get_global_size(0);"
																	   "	const int uvi = UV_OFFSET + get_global_id(0)/2 + get_global_id(1)/2*get_global_size(0);"
																	   "	out[yi] = in[yi];"
																	   "	out[uvi] = in[uvi];"
																	   "	out[uvi+1] = in[uvi+1];"
																	   "}");


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

	/*std::map<int, std::string> table;
	table[CL_R] = "CL_R";
	table[CL_A] = "CL_A";
	table[CL_RG] = "CL_RG";
	table[CL_RGBA] = "CL_RGBA";
	table[CL_BGRA] = "CL_BGRA";
	table[CL_INTENSITY] = "CL_INTENSITY";
	table[CL_LUMINANCE] = "CL_LUMINANCE";
	table[CL_DEPTH] = "CL_DEPTH";

	table[CL_UNORM_INT8] = "CL_UNORM_INT8";
	table[CL_FLOAT] = "CL_FLOAT";
	cl_image_format formats[64];
	unsigned int numFormats;
	clGetSupportedImageFormats(cl::Context::getDefault()(), 0, CL_MEM_OBJECT_IMAGE2D, 64, formats, &numFormats);
	for(int i = 0; i < numFormats; i++) {
		std::cout << std::hex << table[formats[i].image_channel_order] << " " << table[formats[i].image_channel_data_type] << std::endl;
	}*/
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

static inline cl::Image2D allocImage(int width, int height, const PixelFormat* format) {
	//TODO embed into PixelFormat as value
	cl::ImageFormat clFormat;
	if(format == &PixelFormat::RGBA8)
		clFormat = cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8);
	else if(format == &PixelFormat::F32)
		clFormat = cl::ImageFormat(CL_LUMINANCE, CL_FLOAT);
	else {
		std::cerr << "[OpenCL] cannot alloc CLImage with given PixelFormat " << std::endl;
		exit(1);
	}

	int error;
	cl::Image2D image = cl::Image2D(cl::Context::getDefault(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, clFormat, width, height, 0, nullptr, &error);
	if(error != CL_SUCCESS) {
		std::cerr << "bgr Image creation error: " << error << std::endl;
		exit(1);
	}
	return std::move(image);
}

CLImage::CLImage(const PixelFormat* format): format(format), width(0), height(0) {}
CLImage::CLImage(const PixelFormat* format, int width, int height, std::string name): image(allocImage(width, height, format)), format(format), width(width), height(height), name(std::move(name)) {}

void CLImage::save(const std::string &suffix, float factor, float offset) const {
	if(format == &PixelFormat::F32) {
		cv::Mat grayscale(height, width, CV_8UC1);
		CLImageMap<float> map = read<float>();
		for(int y = 0; y < height; y++) {
			for(int x = 0; x < width; x++) {
				grayscale.data[x + width * y] = cv::saturate_cast<uint8_t>(factor * map[x + width * y] + offset);
			}
		}

		cv::imwrite("img/" + name + suffix, grayscale);
	} else {
		cv::imwrite("img/" + name + suffix, read<RGBA>().cv);
	}
}