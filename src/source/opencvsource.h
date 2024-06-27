#pragma once

#include <opencv2/videoio.hpp>

#include "videosource.h"

class OpenCVSource : public VideoSource {
public:
	explicit OpenCVSource(const std::string& path): capture(path, cv::CAP_ANY, {cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_ANY}), name(path) {
		std::replace(name.begin(), name.end(), '/', '_');
	}

	std::shared_ptr<Image> readImage() override {
		//TODO better image pooling
		if(image == nullptr || !image.unique())
			image = std::make_shared<Image>(&PixelFormat::BGR888, capture.get(cv::CAP_PROP_FRAME_WIDTH), capture.get(cv::CAP_PROP_FRAME_HEIGHT), name);

		CLMap<uint8_t> map = image->write<uint8_t>();
		cv::Mat mat(cv::Size(image->width, image->height), CV_8UC3, (void*)*map);
		if(!capture.read(mat))
			return nullptr;

		return image;
	}

private:
	cv::VideoCapture capture;
	std::shared_ptr<Image> image = nullptr;
	std::string name;
};
