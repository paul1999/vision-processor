#pragma once

#include <cstdlib>
#include <vector>
#include <thread>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "image.h"
#include "videosource.h"

class ImageSource : public VideoSource {
public:
	explicit ImageSource(const std::vector<std::string>& paths) {
		for(auto& path : paths) {
			cv::Mat mat = cv::imread(path);
			int i = path.find("/")+1;
			std::shared_ptr<Image> image = std::make_shared<Image>(&PixelFormat::BGR888, mat.cols, mat.rows, path.substr(i, path.find("/", i)-i));
			mat.copyTo(*image->cvWrite());
			images.push_back(image);
		}
	}

	std::shared_ptr<Image> readImage() override {
		std::this_thread::sleep_for(std::chrono::microseconds(33333));
		return images[std::rand() % images.size()];
	}

private:
	std::vector<std::shared_ptr<Image>> images;
};
