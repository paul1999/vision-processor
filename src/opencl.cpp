/*
     Copyright 2024 Felix Weinmann

     Licensed under the Apache License, Version 2.0 (the "License");
     you may not use this file except in compliance with the License.
     You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

     Unless required by applicable law or agreed to in writing, software
     distributed under the License is distributed on an "AS IS" BASIS,
     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     See the License for the specific language governing permissions and
     limitations under the License.
 */
#include "opencl.h"

#include <iostream>
#include <utility>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>


const PixelFormat PixelFormat::RGBA8 = PixelFormat(4, 1, true, CV_8UC4);
const PixelFormat PixelFormat::RGGB8 = PixelFormat(2, 2, true, CV_8UC1);
const PixelFormat PixelFormat::BGR888 = PixelFormat(3, 1, true, CV_8UC3);
const PixelFormat PixelFormat::U8 = PixelFormat(1, 1, false, CV_8UC1);
const PixelFormat PixelFormat::I8 = PixelFormat(1, 1, false, CV_8SC1);
const PixelFormat PixelFormat::F32 = PixelFormat(4, 1, false, CV_32FC1);


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

cl::Kernel OpenCL::compile(const char *code, const char* codeEnd, const std::string &options) {
	cl::Program::Sources sources;
	sources.emplace_back(code, codeEnd - code);

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

//Pool design adapted from Jonathan Mee https://stackoverflow.com/a/27828584 CC BY-SA 3.0
std::shared_ptr<CLImage> OpenCL::acquire(const PixelFormat* format, int width, int height, const std::string& name) {
	std::vector<std::shared_ptr<CLImage>>& formatPool = pool[format];

	auto iterator = std::find_if(formatPool.begin(), formatPool.end(), [&](const std::shared_ptr<CLImage>& i){
		return i.use_count() == 1 && i->width == width && i->height == height;
	});
	if(iterator != formatPool.end()) {
		(*iterator)->name = name;
		return *iterator;
	}

	auto array = std::make_shared<CLImage>(format, width, height, name);
	formatPool.push_back(array);
	return array;
}

static inline cl::Buffer clAlloc(cl_mem_flags type, cl::size_type size, void* data) {
	int error;
	cl::Buffer buffer(type | CL_MEM_READ_WRITE, size, data, &error);
	if(error != CL_SUCCESS) {
		std::cerr << "[OpenCL] Error during image allocation: " << error << std::endl;
		exit(1);
	}
	return buffer;
}

CLArray::CLArray(int size): buffer(clAlloc((cl_mem_flags) CL_MEM_ALLOC_HOST_PTR, (cl::size_type) size, nullptr)), size(size) {}
CLArray::CLArray(void* data, const int size): buffer(clAlloc((cl_mem_flags) CL_MEM_COPY_HOST_PTR, (cl::size_type) size, data)), size(size) {}

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
		std::cerr << "bgr Image creation error: " << error << " " << width << "," << height << " " << (format == &PixelFormat::RGBA8) << std::endl;
		exit(1);
	}
	return image;
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
	} else if(format == &PixelFormat::RGBA8) {
		cv::Mat bgr;
		cv::cvtColor(read<RGBA>().cv, bgr, cv::COLOR_RGBA2BGR);
		cv::imwrite("img/" + name + suffix, bgr);
	} else {
		cv::imwrite("img/" + name + suffix, read<RGBA>().cv);
	}
}