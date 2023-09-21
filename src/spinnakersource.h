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
};

class SpinnakerImage : public Image {
public:
	//TODO image format
	// Image size halfed (RGB resolution)
	explicit SpinnakerImage(const Spinnaker::ImagePtr& pImage): Image(RGGB8, (int)pImage->GetWidth() / 2, (int)pImage->GetHeight() / 2, (unsigned char*)pImage->GetData()), pImage(pImage) {}
	~SpinnakerImage() override { pImage->Release(); }

private:
	const Spinnaker::ImagePtr pImage;
};

#endif