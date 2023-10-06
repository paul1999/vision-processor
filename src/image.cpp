#include <iostream>
#include <cmath>
#include "image.h"

BufferImage::BufferImage(PixelFormat format, int width, int height, unsigned char* data) : Image(format, width, height, data) {}

/* TODO
https://stackoverflow.com/a/3351994 CC BY-SA 2.5
#include <unistd.h>
long sz = sysconf (_SC_PAGESIZE);

https://stackoverflow.com/a/50958005 CC BY-SA 4.0
SYSTEM_INFO sysInfo;
GetSystemInfo(&sysInfo);
printf("%s %d\n\n", "PageSize[Bytes] :", sysInfo.dwPageSize);
*/
static const int PAGE_SIZE = 4096; // Required for OpenCL

std::shared_ptr<Image> BufferImage::create(PixelFormat format, int width, int height) {
	int pixelWidthSize = 1;
	int pixelHeightSize = 1;
	switch(format) {
		case RGGB8:
			pixelWidthSize = 2;
			pixelHeightSize = 2;
			break;
		case F32:
			pixelWidthSize = 4;
			break;
		case BGR888:
			pixelWidthSize = 3;
			break;
		case NV12:
			pixelHeightSize = 2; // TODO 1,5 Image planes
	}

	auto* buffer = (unsigned char*)std::aligned_alloc(PAGE_SIZE, width * height * pixelHeightSize * pixelWidthSize);
	return std::make_shared<BufferImage>(format, width, height, buffer);
}

BufferImage::~BufferImage() {
	std::free(getData());
}

int Image::pixelSize() {
	return pixelWidth()*pixelHeight();
}

int Image::pixelWidth() {
	switch(format) {
		case F32:
			return 4;
		case BGR888:
			return 3;
		case RGGB8:
			return 2; // 2 Image planes, TODO 1.5
		case NV12:
		case U8:
		case I8:
			return 1;
	}
}

int Image::pixelHeight() {
	switch(format) {
		case RGGB8:
		case NV12:
			return 2;
		case BGR888:
		case F32:
		case U8:
		case I8:
			return 1;
	}
}
