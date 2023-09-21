#pragma once

#include <cstdlib>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>

#include "image.h"
#include "videosource.h"


class CVImage : public Image {
public:
	CVImage(cv::Mat img, PixelFormat format): Image(format, img.cols, img.rows, img.data), img(std::move(img)) {}
	explicit CVImage(cv::Mat img): Image(BGR888, img.cols, img.rows, img.data), img(std::move(img)) {}
private:
	const cv::Mat img;
};

class ImageSource : public VideoSource {
public:
	explicit ImageSource(const std::vector<std::string>& paths) {
		for(auto& path : paths) {
			images.push_back(std::make_shared<CVImage>(cv::imread(path)));
		}
	}

	std::shared_ptr<Image> readImage() override {
		return images[std::rand() % images.size()];
	}

private:
	std::vector<std::shared_ptr<Image>> images;
};
