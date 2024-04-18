#pragma once


#include <memory>
#include <string>
#include <opencv2/core/mat.hpp>
#include <utility>
#include "opencl.h"

class PixelFormat {
public:
	static const PixelFormat RGGB8;
	static const PixelFormat BGR888;
	static const PixelFormat U8;
	static const PixelFormat I8;
	static const PixelFormat F32;
	static const PixelFormat NV12;

	[[nodiscard]] int pixelSize() const { return stride*rowStride; }

	const int stride;
	const int rowStride;
	const bool color;
	const int cvType;

	const std::string clKernelToNV12;
private:
	PixelFormat(int stride, int rowStride, bool color, int cvType, std::string clKernelToNV12): stride(stride), rowStride(rowStride), color(color), cvType(cvType), clKernelToNV12(std::move(clKernelToNV12)) {}
};


class CVMap;


class Image : public CLArray {
public:
	Image(const Image& other) = default;
	Image(const PixelFormat* format, int width, int height): CLArray(width*height*format->pixelSize()), format(format), width(width), height(height), timestamp(0), name() {}
	Image(const PixelFormat* format, int width, int height, std::string name): CLArray(width*height*format->pixelSize()), format(format), width(width), height(height), timestamp(0), name(std::move(name)) {}
	Image(const PixelFormat* format, int width, int height, double timestamp): CLArray(width*height*format->pixelSize()), format(format), width(width), height(height), timestamp(timestamp), name() {}

	//Only use these constructors if not possible otherwise due to necessary copy (because of potential alignment mismatch for zero-copy support)
	Image(const PixelFormat* format, int width, int height, unsigned char* data): CLArray(data, width*height*format->pixelSize()), format(format), width(width), height(height), timestamp(0) {}
	Image(const PixelFormat* format, int width, int height, double timestamp, unsigned char* data): CLArray(data, width*height*format->pixelSize()), format(format), width(width), height(height), timestamp(timestamp) {}

	virtual ~Image() = default;

	[[nodiscard]] CVMap cvRead() const;
	[[nodiscard]] CVMap cvWrite();
	[[nodiscard]] CVMap cvReadWrite();

	[[nodiscard]] Image toGrayscale() const;
	[[nodiscard]] Image toBGR() const;
	[[nodiscard]] Image toRGGB() const;
	[[nodiscard]] Image toUpscaleRGGB() const;

	void save(const std::string& suffix, float factor = 1.0f) const;

	const PixelFormat* format;
	const int width;
	const int height;
	// timestamp of 0 indicates unavailability
	const double timestamp;
	const std::string name;
};

/*class CLImage {
	explicit CLImage(int width, int height, int planes);

	template<typename T> CLMap<T> read() const { return std::move(CLMap<T>(buffer, size, CL_MAP_READ)); }
	template<typename T> CLMap<T> write() { return std::move(CLMap<T>(buffer, size, CL_MAP_WRITE_INVALIDATE_REGION)); }
	template<typename T> CLMap<T> readWrite() { return std::move(CLMap<T>(buffer, size, CL_MAP_WRITE)); }

	const cl::Image2DArray image;
	const int width;
	const int height;
	const int planes;
};*/


class CVMap {
public:
	explicit CVMap(const Image& image, int clRWType);
	~CVMap() = default;

	CVMap(CVMap&& other) noexcept = default;
	CVMap ( const CVMap & ) = delete;
	CVMap& operator= ( const CVMap & ) = delete;
	cv::Mat& operator*() { return mat; }
	cv::Mat* operator-> () { return &mat; }
	const cv::Mat& operator*() const { return mat; }
	const cv::Mat* operator-> () const { return &mat; }

private:
	CLMap<uint8_t> map;
	cv::Mat mat;
};