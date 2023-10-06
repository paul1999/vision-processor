#pragma once

#include <cstdlib>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>

#include "image.h"
#include "videosource.h"

class ImageSource : public VideoSource {
public:
	explicit ImageSource(const std::vector<std::string>& paths) {
		for(auto& path : paths) {
			cv::Mat mat = cv::imread(path);
			std::shared_ptr<Image> image = BufferImage::create(BGR888, mat.cols, mat.rows);
			for(int i = 0; i < mat.cols*mat.rows*3; i++)
				image->getData()[i] = mat.data[i];
			images.push_back(image);
		}
	}

	std::shared_ptr<Image> readImage() override {
		return images[std::rand() % images.size()];
	}

private:
	std::vector<std::shared_ptr<Image>> images;
};
