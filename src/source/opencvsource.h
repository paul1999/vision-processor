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

		std::shared_ptr<Image> image = std::make_shared<Image>(&PixelFormat::BGR888, mat.cols, mat.rows);
		CLMap<uint8_t> data = image->write<uint8_t>();
		for(int i = 0; i < mat.cols*mat.rows*3; i++)
			data[i] = mat.data[i];

		return image;
	}

private:
	cv::VideoCapture capture;
};
