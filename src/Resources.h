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


typedef struct __attribute__ ((packed)) {
	cl_uchar r;
	cl_uchar g;
	cl_uchar b;
} RGB;


class Resources {
public:
	explicit Resources(YAML::Node config);

	std::unique_ptr<VideoSource> camera = nullptr;

	int camId;
	int cameraAmount;
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
	float orangeMedian = 0.0f;
	RGB yellow = {255, 255, 64};
	float yellowMedian = 0.0f;
	RGB blue = {0, 255, 255};
	float blueMedian = 0.0f;
	RGB green = {64, 255, 64};
	float greenMedian = 0.0f;
	RGB pink = {255, 0, 255};
	float pinkMedian = 0.0f;

	std::shared_ptr<GCSocket> gcSocket;
	std::shared_ptr<VisionSocket> socket;
	std::shared_ptr<Perspective> perspective;
	std::shared_ptr<OpenCL> openCl;
	std::shared_ptr<AlignedArrayPool> arrayPool;
	std::shared_ptr<Mask> mask;
	std::shared_ptr<RTPStreamer> rtpStreamer;

	cl::Kernel yuvkernel;
	cl::Kernel bgkernel;
	cl::Kernel diffkernel;
	cl::Kernel ringkernel;
	cl::Kernel botkernel;
	cl::Kernel sidekernel;
	cl::Kernel ballkernel;
};
