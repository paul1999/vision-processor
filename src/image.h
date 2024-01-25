#pragma once


#include <memory>
#include <string>
#include <opencv2/core/mat.hpp>
#include "CLArray.h"

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
	Image(const PixelFormat* format, int width, int height): CLArray(width*height*format->pixelSize()), format(format), width(width), height(height), timestamp(0) {}
	Image(const PixelFormat* format, int width, int height, double timestamp): CLArray(width*height*format->pixelSize()), format(format), width(width), height(height), timestamp(timestamp) {}

	//Only use these constructors if not possible otherwise
	Image(const PixelFormat* format, int width, int height, unsigned char* data): CLArray(data, width*height*format->pixelSize()), format(format), width(width), height(height), timestamp(0) {}
	Image(const PixelFormat* format, int width, int height, double timestamp, unsigned char* data): CLArray(data, width*height*format->pixelSize()), format(format), width(width), height(height), timestamp(timestamp) {}

	virtual ~Image() = default;

	[[nodiscard]] CVMap cvRead() const;
	[[nodiscard]] CVMap cvWrite();
	[[nodiscard]] CVMap cvReadWrite();

	[[nodiscard]] Image toGrayscale() const;
	[[nodiscard]] Image toBGR() const;
	[[nodiscard]] Image toRGGB() const;

	const PixelFormat* format;
	const int width;
	const int height;
	// timestamp of 0 indicates unavailability
	const double timestamp;
};


class CVMap {
public:
	explicit CVMap(const Image& image, int clRWType);
	~CVMap();

	CVMap(CVMap&& other) noexcept;
	CVMap ( const CVMap & ) = delete;
	CVMap& operator= ( const CVMap & ) = delete;
	cv::Mat& operator*() { return mat; }
	cv::Mat* operator-> () { return &mat; }
	const cv::Mat& operator*() const { return mat; }
	const cv::Mat* operator-> () const { return &mat; }

private:
	const cl::Buffer buffer;
	void* map;
	cv::Mat mat;
	bool unmoved = true;
};