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

#include <opencv2/videoio.hpp>

#include "cameradriver.h"

class OpenCVDriver : public CameraDriver {
public:
	explicit OpenCVDriver(const std::string& path): capture(path, cv::CAP_ANY, {cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_ANY}), name(path) {
		std::replace(name.begin(), name.end(), '/', '_');
	}

	std::shared_ptr<Image> readImage() override {
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
