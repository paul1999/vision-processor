#pragma once

#include <cstdlib>
#include <vector>

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
			std::shared_ptr<Image> image = BufferImage::create(RGGB8, mat.cols/2, mat.rows/2);
			for(int y = 0; y < mat.rows; y++) {
				for(int x = 0; x < mat.cols; x++) {
					image->getData()[x + mat.cols*y] = (y%2 ?
							(x%2 ? mat.data[3*(x + mat.cols*y)+0] : mat.data[3*(x + mat.cols*y)+1]) :
							(x%2 ? mat.data[3*(x + mat.cols*y)+1] : mat.data[3*(x + mat.cols*y)+2])
						);
				}
			}
			images.push_back(image);
		}
	}

	std::shared_ptr<Image> readImage() override {
		return images[std::rand() % images.size()];
	}

private:
	std::vector<std::shared_ptr<Image>> images;
};
