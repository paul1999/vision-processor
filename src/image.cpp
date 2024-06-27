#include <cmath>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "image.h"

CVMap Image::cvRead() const {
	return std::move(CVMap(*this, CL_MAP_READ));
}

CVMap Image::cvWrite() {
	return std::move(CVMap(*this, CL_MAP_WRITE_INVALIDATE_REGION));
}

CVMap Image::cvReadWrite() {
	return std::move(CVMap(*this, CL_MAP_WRITE));
}

Image Image::toGrayscale() const {
	if (format == &PixelFormat::U8) {
		return *this;
	} else if(format == &PixelFormat::BGR888) {
		Image image(&PixelFormat::U8, width, height, name);
		cv::cvtColor(*cvRead(), *image.cvWrite(), cv::COLOR_BGR2GRAY);
		return image;
	} else if(format == &PixelFormat::RGGB8) {
		Image image(&PixelFormat::U8, 2*width, 2*height, name);
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
		Image image(&PixelFormat::BGR888, 2*width, 2*height, name);
		cv::cvtColor(*cvRead(), *image.cvWrite(), cv::COLOR_BayerBG2BGR);
		/*CLMap<uint8_t> read = ::Image::read<uint8_t>();
		CLMap<uint8_t> write = image.write<uint8_t>();
		for(int y = 0; y < height; y++) {
			for(int x = 0; x < width; x++) {
				write[3*(x + width * y) + 0] = read[2*x+1 + 2*width * (2*y+1)];
				write[3*(x + width * y) + 1] = ((uint16_t)read[2*x+1 + 2*width * 2*y] + read[2*x + 2*width * (2*y+1)]) / 2;
				write[3*(x + width * y) + 2] = read[2*x + 2*width * 2*y];
			}
		}*/
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
		Image image(&PixelFormat::RGGB8, width/2, height/2, name);
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

Image Image::toUpscaleRGGB() const {
	if(format == &PixelFormat::RGGB8) {
		return *this;
	} else if(format == &PixelFormat::BGR888) {
		Image image(&PixelFormat::RGGB8, width, height, name);
		CLMap<uint8_t> read = ::Image::read<uint8_t>();
		CLMap<uint8_t> write = image.write<uint8_t>();
		for(int y = 0; y < height; y++) {
			for(int x = 0; x < width; x++) {
				int readpos = 3 * (x + width * y);
				int writepos = 2*x + 2*y * 2*width;
				write[writepos] = read[readpos+2];
				write[writepos+1] = read[readpos+1];
				write[writepos+2*width] = read[readpos+1];
				write[writepos+2*width+1] = read[readpos];
			}
		}
		return image;
	} else {
		std::cerr << "[Image] Unimplemented conversion to upscaled RGGB" << std::endl;
		exit(1);
	}
}

void Image::save(const std::string &suffix, float factor) const {
	if(format == &PixelFormat::F32) {
		Image grayscale(&PixelFormat::U8, width, height, name);
		{
			CLMap<uint8_t> write = grayscale.write<uint8_t>();
			if(factor == 1.0f) {
				CLMap<int> read = ::Image::read<int>();
				for(int y = 0; y < height; y++) {
					for(int x = 0; x < width; x++) {
						write[x + width * y] = cv::saturate_cast<uint8_t>(read[x + width * y]);
					}
				}
			} else {
				CLMap<float> read = ::Image::read<float>();
				for(int y = 0; y < height; y++) {
					for(int x = 0; x < width; x++) {
						write[x + width * y] = cv::saturate_cast<uint8_t>(factor * read[x + width * y]);
					}
				}
			}
		}
		cv::imwrite("img/" + name + suffix, *grayscale.cvRead());
	} else if(format == &PixelFormat::I8) {
		cv::Mat u8;
		cv::add(*cvRead(), 128, u8, cv::noArray(), CV_8UC1);
		cv::imwrite("img/" + name + suffix, u8);
	} else {
		cv::imwrite("img/" + name + suffix, *cvRead());
	}
}

CVMap::CVMap(const Image& image, int clRWType): map(std::move(CLMap<uint8_t>(image.buffer, image.height*image.width*image.format->pixelSize(), clRWType))) {
	if(image.format->cvType == CV_8UC1)
		mat = cv::Mat(image.height*image.format->rowStride, image.width*image.format->stride, image.format->cvType, *map);
	else
		mat = cv::Mat(image.height, image.width, image.format->cvType, *map);
}
