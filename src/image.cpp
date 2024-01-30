#include <cmath>
#include <opencv2/imgproc.hpp>
#include "image.h"

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

CVMap Image::cvRead() const {
	return std::move(CVMap(*this, CL_MAP_READ));
}

CVMap Image::cvWrite() {
	return std::move(CVMap(*this, CL_MAP_WRITE)); //_INVALIDATE_REGION
}

CVMap Image::cvReadWrite() {
	return std::move(CVMap(*this, CL_MAP_WRITE));
}

Image Image::toGrayscale() const {
	if (format == &PixelFormat::U8) {
		return *this;
	} else if(format == &PixelFormat::BGR888) {
		Image image(&PixelFormat::U8, width, height);
		cv::cvtColor(*cvRead(), *image.cvWrite(), cv::COLOR_BGR2GRAY);
		return image;
	} else if(format == &PixelFormat::RGGB8) {
		Image image(&PixelFormat::U8, 2*width, 2*height);
		cv::cvtColor(*cvRead(), *image.cvWrite(), cv::COLOR_BayerBG2GRAY);
		return image;
	} else {
		std::cerr << "[Image] Unimplemented conversion to grayscale" << std::endl;
		exit(1);
	}
}

Image Image::toBGR() const {
	if(format == &PixelFormat::BGR888) {
		return *this;
	} else if(format == &PixelFormat::RGGB8) {
		Image image(&PixelFormat::BGR888, 2*width, 2*height);
		cv::cvtColor(*cvRead(), *image.cvWrite(), cv::COLOR_BayerBG2BGR);
		return image;
	} else {
		std::cerr << "[Image] Unimplemented conversion to BGR" << std::endl;
		exit(1);
	}
}

Image Image::toRGGB() const {
	if(format == &PixelFormat::RGGB8) {
		return *this;
	} else if(format == &PixelFormat::BGR888) {
		Image image(&PixelFormat::RGGB8, width/2, height/2);
		CLMap<uint8_t> read = ::Image::read<uint8_t>();
		CLMap<uint8_t> write = image.write<uint8_t>();
		for(int y = 0; y < height; y++) {
			for(int x = 0; x < width; x++) {
				write[x + width * y] = (y % 2 ?
						(x%2 ? read[3 * (x + width * y) + 0] : read[3 * (x + width * y) + 1]) :
						(x%2 ? read[3 * (x + width * y) + 1] : read[3 * (x + width * y) + 2])
				);
			}
		}
		return image;
	} else {
		std::cerr << "[Image] Unimplemented conversion to RGGB" << std::endl;
		exit(1);
	}
}

CVMap::CVMap(const Image& image, int clRWType): buffer(image.buffer) {
	int size = image.height*image.width*image.format->pixelSize();
	int error;
	map = cl::enqueueMapBuffer(buffer, true, clRWType, 0, size, nullptr, nullptr, &error);
	if(error != 0)
		std::cerr << "[CLMap] enqueue map buffer returned " << error << std::endl;

	if(image.format->cvType == CV_8UC1)
		mat = cv::Mat(image.height*image.format->rowStride, image.width*image.format->stride, image.format->cvType, map);
	else
		mat = cv::Mat(image.height, image.width, image.format->cvType, map);
}

CVMap::~CVMap() {
	if(unmoved)
		cl::enqueueUnmapMemObject(buffer, map);
}

CVMap::CVMap(CVMap&& other) noexcept: buffer(other.buffer), map(other.map), mat(std::move(other.mat)) {
	other.unmoved = false;
}
