/*
     Copyright 2024 Felix Weinmann

     Licensed under the Apache License, Version 2.0 (the "License");
     you may not use this file except in compliance with the License.
     You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

     Unless required by applicable law or agreed to in writing, software
     distributed under the License is distributed on an "AS IS" BASIS,
     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     See the License for the specific language governing permissions and
     limitations under the License.
 */
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
