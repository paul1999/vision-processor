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

	std::shared_ptr<GCSocket> gcSocket;
	std::shared_ptr<VisionSocket> socket;
	std::shared_ptr<Perspective> perspective;
	std::shared_ptr<OpenCL> openCl;
	std::shared_ptr<AlignedArrayPool> arrayPool;
	std::shared_ptr<Mask> mask;
	std::shared_ptr<RTPStreamer> rtpStreamer;

	cl::Kernel diffkernel;
	cl::Kernel ringkernel;
	cl::Kernel botkernel;
	cl::Kernel sidekernel;
	cl::Kernel ballkernel;
};
