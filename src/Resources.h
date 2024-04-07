#pragma once


#include <string>
#include <memory>
#include <yaml-cpp/node/node.h>
#include "source/spinnakersource.h"
#include "source/videosource.h"
#include "source/opencvsource.h"
#include "source/imagesource.h"
#include "rtpstreamer.h"
#include "Mask.h"
#include "udpsocket.h"
#include "Perspective.h"
#include "opencl.h"


typedef struct __attribute__ ((packed)) RGB {
	cl_uchar r;
	cl_uchar g;
	cl_uchar b;

	auto operator<=>(const RGB&) const = default;
} RGB;


class Resources {
public:
	explicit Resources(YAML::Node config);

	std::unique_ptr<VideoSource> camera = nullptr;

	int camId;
	int cameraAmount;
	double botRadius;
	double maxBotAcceleration;
	double sideBlobDistance;
	double centerBlobRadius;
	double sideBlobRadius;
	double maxBallVelocity;
	double ballRadius;
	double minTrackingRadius;

	std::string groundTruth;

	double contrast = 1.0;
	RGB orange = {255, 64, 0};
	float orangeMedian = 1000.0f;
	//RGB yellow = {255, 192, 128};
	RGB yellow = {192, 128, 64};
	float yellowMedian = 1000.0f;
	RGB blue = {0, 0, 255};
	float blueMedian = 1000.0f;
	RGB green = {64, 255, 64};
	float greenMedian = 1000.0f;
	RGB pink = {255, 0, 255};
	float pinkMedian = 1000.0f;

	uint8_t orangeHue = 21; //30°
	//uint8_t yellowHue = 43; //60°
	uint8_t yellowHue = 35;
	uint8_t blueHue = 149; //210°
	uint8_t greenHue = 85; //120°
	//uint8_t pinkHue = 213; //300°
	uint8_t pinkHue = 230;

	std::shared_ptr<GCSocket> gcSocket;
	std::shared_ptr<VisionSocket> socket;
	std::shared_ptr<Perspective> perspective;
	std::shared_ptr<OpenCL> openCl;
	std::shared_ptr<AlignedArrayPool> arrayPool;
	std::shared_ptr<Mask> mask;
	std::shared_ptr<RTPStreamer> rtpStreamer;

	cl::Kernel blurkernel;
	cl::Kernel gradientkernel;
	cl::Kernel yuvkernel;
	cl::Kernel bgkernel;
	cl::Kernel diffkernel;
	cl::Kernel ringkernel;
	cl::Kernel botkernel;
	cl::Kernel sidekernel;
	cl::Kernel ballkernel;
};
