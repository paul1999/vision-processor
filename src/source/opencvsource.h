#pragma once

#include <opencv2/videoio.hpp>

#include "videosource.h"

class OpenCVSource : public VideoSource {
public:
	explicit OpenCVSource(const std::string& path): capture(path) {}

	std::shared_ptr<Image> readImage() override {
		cv::Mat mat;
		capture.read(mat);

		std::shared_ptr<Image> image = BufferImage::create(BGR888, mat.cols, mat.rows);
		for(int i = 0; i < mat.cols*mat.rows*3; i++)
			image->getData()[i] = mat.data[i];

		return image;
	}

private:
	cv::VideoCapture capture;
};
