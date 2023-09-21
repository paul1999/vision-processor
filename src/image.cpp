#include <iostream>
#include "image.h"

BufferImage::BufferImage(PixelFormat format, int width, int height, std::vector<unsigned char*> buffer) : Image(format, width, height, (unsigned char*)buffer.data()), buffer(std::move(buffer)) {}

std::shared_ptr<Image> BufferImage::create(PixelFormat format, int width, int height) {
	int pixelSize;
	switch(format) {
		case RGGB8:
			pixelSize = 4;
			break;
		case BGR888:
			pixelSize = 3;
			break;
		case U8:
		case I8:
			pixelSize = 1;
			break;
	}

	std::vector<unsigned char*> buffer;
	buffer.resize(width*height*pixelSize);
	return std::make_shared<BufferImage>(format, width, height, std::move(buffer));
}
