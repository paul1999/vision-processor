#pragma once


#include <string>
#include <memory>
#include <yaml-cpp/node/node.h>
#include "source/imagesource.h"
#include "rtpstreamer.h"
#include "udpsocket.h"
#include "Perspective.h"
#include "opencl.h"

double getTime();

void ensureSize(CLImage& image, int width, int height, std::string name);


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

	double sideBlobDistance;
	double centerBlobRadius;
	double sideBlobRadius;
	double ballRadius;

	double minTrackingRadius;
	double maxBallVelocity;
	double maxBotAcceleration;

	double minCircularity;
	uint8_t minSaturation;
	uint8_t minBrightness;

	double contrast = 1.0; //deprecated
	RGB orange = {255, 64, 0};
	RGB yellow = {255, 255, 64};
	//RGB yellow = {255, 192, 128};
	//RGB yellow = {192, 128, 64};
	RGB blue = {0, 0, 255};
	RGB green = {64, 255, 64};
	RGB pink = {255, 0, 255};

	uint8_t orangeHue;
	uint8_t yellowHue;
	uint8_t blueHue;
	uint8_t greenHue;
	uint8_t pinkHue;

	int cameraAmount;
	double cameraHeight; // Just for calibration, do not use elsewhere (0.0 as special value for automatic calibration)
	uint8_t fieldLineThreshold;
	double minLineSegmentLength;
	double minMajorLineLength;
	double maxIntersectionDistance;
	double maxLineSegmentOffset;
	double maxLineSegmentAngle;

	std::string groundTruth;
	bool waitForGeometry;
	bool debugImages;

	std::shared_ptr<GCSocket> gcSocket;
	std::shared_ptr<VisionSocket> socket;
	std::shared_ptr<Perspective> perspective;
	std::shared_ptr<OpenCL> openCl;
	std::shared_ptr<AlignedArrayPool> arrayPool;
	std::shared_ptr<RTPStreamer> rtpStreamer;

	cl::Kernel blurkernel;
	cl::Kernel gradientkernel;
	cl::Kernel diffkernel;
	cl::Kernel ringkernel;
};
