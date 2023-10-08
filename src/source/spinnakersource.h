#pragma once

#include "videosource.h"

#ifdef SPINNAKER
#include "Spinnaker.h"

class SpinnakerSource : public VideoSource {
public:
	explicit SpinnakerSource(int id);
	~SpinnakerSource() override;

	std::shared_ptr<Image> readImage() override;

private:
	Spinnaker::SystemPtr pSystem;
	Spinnaker::CameraPtr pCam;

	std::vector<std::shared_ptr<Image>> buffers;  // Use own image buffers for page size alignment (Intel OpenCL zero copy)
};

#endif