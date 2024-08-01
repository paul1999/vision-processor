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


#include <string>
#include <memory>
#include <yaml-cpp/node/node.h>
#include "driver/cameradriver.h"
#include "rtpstreamer.h"
#include "udpsocket.h"
#include "Perspective.h"
#include "opencl.h"

double getTime();


typedef struct __attribute__ ((packed)) RGB {
	cl_uchar r;
	cl_uchar g;
	cl_uchar b;

	auto operator<=>(const RGB&) const = default;
} RGB;


class Resources {
public:
	explicit Resources(const YAML::Node& config);

	std::unique_ptr<CameraDriver> camera = nullptr;

	int camId;

	double sideBlobDistance;
	double centerBlobRadius;
	double sideBlobRadius;
	double minBlobRadius;
	double maxBlobRadius;

	double minTrackingRadius;
	double maxBallVelocity;
	double maxBotAcceleration;

	double minCircularity;
	double minScore;
	int maxBlobs;

	Eigen::Vector3i falseOrangeReference = {128, 128, 128};
	Eigen::Vector3i orangeReference = {255, 128, 0};
	Eigen::Vector3i yellowReference = {255, 128, 0};
	Eigen::Vector3i blueReference = {0, 128, 255};
	Eigen::Vector3i greenReference = {0, 255, 128};
	Eigen::Vector3i pinkReference = {255, 0, 128};
	Eigen::Vector3i falseOrange = falseOrangeReference;
	Eigen::Vector3i orange = orangeReference;
	Eigen::Vector3i yellow = yellowReference;
	Eigen::Vector3i blue = blueReference;
	Eigen::Vector3i green = greenReference;
	Eigen::Vector3i pink = pinkReference;

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
	std::shared_ptr<RTPStreamer> rtpStreamer;
};
