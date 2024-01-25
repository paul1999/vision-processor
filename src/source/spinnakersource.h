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