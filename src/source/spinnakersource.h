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
#ifdef SPINNAKER

#include "videosource.h"
#include "Spinnaker.h"

class SpinnakerSource : public VideoSource {
public:
	explicit SpinnakerSource(int id);
	~SpinnakerSource() override;

	std::shared_ptr<Image> readImage() override;

	std::shared_ptr<Image> borrow(const Spinnaker::ImagePtr& pImage);
	void restore(const Image& image);

private:
	Spinnaker::SystemPtr pSystem;
	Spinnaker::CameraPtr pCam;

	std::map<std::shared_ptr<Image>, std::unique_ptr<CLMap<uint8_t>>> buffers; // Use own image buffers for page size alignment (OpenCL pinned memory and zero copy)
};

#endif