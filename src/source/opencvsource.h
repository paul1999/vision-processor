#pragma once

#include <opencv2/videoio.hpp>

#include "videosource.h"

class OpenCVSource : public VideoSource {
public:
	explicit OpenCVSource(const std::string& path): capture(path) {}

	std::shared_ptr<Image> readImage() override {
		cv::Mat mat;
		capture.read(mat);
		if(mat.empty())
			return nullptr;

		uint8_t* read = mat.data;
		std::shared_ptr<Image> image = std::make_shared<Image>(&PixelFormat::RGGB8, mat.cols/2, mat.rows/2);
		CLMap<uint8_t> write = image->write<uint8_t>();
		for(int y = 0; y < mat.rows; y++) {
			for(int x = 0; x < mat.cols; x++) {
				write[x + mat.cols * y] = (y % 2 ?
										(x%2 ? read[3 * (x + mat.cols * y) + 0] : read[3 * (x + mat.cols * y) + 1]) :
										(x%2 ? read[3 * (x + mat.cols * y) + 1] : read[3 * (x + mat.cols * y) + 2])
				);
			}
		}
		/*std::shared_ptr<Image> image = std::make_shared<Image>(&PixelFormat::BGR888, mat.cols, mat.rows);
		CLMap<uint8_t> data = image->write<uint8_t>();
		for(int i = 0; i < mat.cols*mat.rows*3; i++)
			data[i] = mat.data[i];*/

		return image;
	}

private:
	cv::VideoCapture capture;
};
