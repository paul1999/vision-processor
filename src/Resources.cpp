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
#include <yaml-cpp/yaml.h>

#include "cl_kernels.h"
#include "Resources.h"
#include "driver/spinnakerdriver.h"
#include "driver/mvimpactdriver.h"
#include "driver/cameradriver.h"
#include "driver/opencvdriver.h"

template<>
struct YAML::convert<Eigen::Vector2f> {
	static YAML::Node encode(const Eigen::Vector2f& rhs) {
		YAML::Node node;
		node.push_back(rhs.x());
		node.push_back(rhs.y());
		return node;
	}

	static bool decode(const YAML::Node& node, Eigen::Vector2f& rhs) {
		if(!node.IsSequence() || node.size() != 2) {
			return false;
		}

		rhs.x() = node[0].as<float>();
		rhs.y() = node[1].as<float>();
		return true;
	}
};

template<>
struct YAML::convert<Eigen::Vector3i> {
	static YAML::Node encode(const Eigen::Vector3i& rhs) {
		YAML::Node node;
		node.push_back(rhs.x());
		node.push_back(rhs.y());
		node.push_back(rhs.z());
		return node;
	}

	static bool decode(const YAML::Node& node, Eigen::Vector3i& rhs) {
		if(!node.IsSequence() || node.size() != 3) {
			return false;
		}

		rhs.x() = node[0].as<int>();
		rhs.y() = node[1].as<int>();
		rhs.z() = node[2].as<int>();
		return true;
	}
};

static YAML::Node getOptional(const YAML::Node& node) {
	return node.IsDefined() ? node : YAML::Node();
}

Resources::Resources(const YAML::Node& config) {
	openCl = std::make_shared<OpenCL>();

	YAML::Node cam = getOptional(config["camera"]);

	auto driver = cam["driver"].as<std::string>("SPINNAKER");
	int driver_id = cam["id"].as<int>(0);
	auto exposure = cam["exposure"].as<double>(0.0);
	auto gain = cam["gain"].as<double>(0.0);
	auto gamma = cam["gamma"].as<double>(1.0);

	YAML::Node wbNode = cam["white_balance"];
	WhiteBalanceType wbType = WhiteBalanceType_Manual;
	std::vector<double> wbValues;
	if(wbNode.IsSequence()) {
		wbValues = wbNode.as<std::vector<double>>();
	} else {
		auto wbTypeString = wbNode.as<std::string>("OUTDOOR");
		wbType = wbTypeString == "OUTDOOR" ? WhiteBalanceType_AutoOutdoor : WhiteBalanceType_AutoIndoor;
	}

#ifdef SPINNAKER
	if(driver == "SPINNAKER")
		camera = std::make_unique<SpinnakerDriver>(driver_id, exposure, gain, gamma, wbType, wbValues);
#endif

#ifdef MVIMPACT
	if(driver == "MVIMPACT")
		camera = std::make_unique<MVImpactDriver>(driver_id, exposure, gain, wbType, wbValues);
#endif

	if(driver == "OPENCV")
		camera = std::make_unique<OpenCVDriver>(cam["path"].as<std::string>("/dev/video" + std::to_string(driver_id)), exposure, gain, gamma, wbType, wbValues);

	if(camera == nullptr) {
		std::cerr << "[Resources] Unknown camera/image driver defined: " << driver << std::endl;
		exit(1);
	}

	camId = config["cam_id"].as<int>(0);
	if (camId < 0 || camId > 7) {
		std::cerr << "[Resources] Invalid camera ID, must be >= 0 and <= 7: " << camId << std::endl;
		exit(1);
	}

	YAML::Node thresholds = getOptional(config["thresholds"]);
	minCircularity = thresholds["circularity"].as<double>(25.0);
	minScore = thresholds["score"].as<double>(0.0);
	maxBlobs = thresholds["blobs"].as<int>(2000);
	minBotConfidence = thresholds["min_bot_confidence"].as<float>(0.1f);

	YAML::Node tracking = getOptional(config["tracking"]);
	minTrackingRadius = tracking["min_tracking_radius"].as<double>(20.0);
	maxBotAcceleration = 1000 * tracking["max_bot_acceleration"].as<double>(6.5);

	YAML::Node geometry = getOptional(config["geometry"]);
	cameraAmount = geometry["camera_amount"].as<int>(1);
	cameraHeight = geometry["camera_height"].as<double>(0.0);
	lineCorners = geometry["line_corners"].as<std::vector<Eigen::Vector2f>>(std::vector<Eigen::Vector2f>());
	fieldLineThreshold = geometry["field_line_threshold"].as<int>(5);
	minLineSegmentLength = geometry["min_line_segment_length"].as<double>(10.0);
	maxLineSegmentOffset = geometry["max_line_segment_offset"].as<double>(10.0);
	maxLineSegmentAngle = geometry["max_line_segment_angle"].as<double>(3.0) * M_PI/180.0;

	YAML::Node color = getOptional(config["color"]);
	referenceForce = color["reference_force"].as<float>(0.1f);
	historyForce = color["history_force"].as<float>(0.7f);
	orangeReference = color["orange"].as<Eigen::Vector3i>(Eigen::Vector3i{255, 128, 0});
	fieldReference = color["field"].as<Eigen::Vector3i>(Eigen::Vector3i{128, 128, 128});
	yellowReference = color["yellow"].as<Eigen::Vector3i>(Eigen::Vector3i{255, 128, 0});
	blueReference = color["blue"].as<Eigen::Vector3i>(Eigen::Vector3i{0, 128, 255});
	greenReference = color["green"].as<Eigen::Vector3i>(Eigen::Vector3i{0, 255, 128});
	pinkReference = color["pink"].as<Eigen::Vector3i>(Eigen::Vector3i{255, 0, 128});
	orange = orangeReference;
	field = fieldReference;
	yellow = yellowReference;
	blue = blueReference;
	green = greenReference;
	pink = pinkReference;

	YAML::Node debug = getOptional(config["debug"]);
	groundTruth = debug["ground_truth"].as<std::string>("gt.yml");
	bool waitForGeometry = debug["wait_for_geometry"].as<bool>(false);
	debugImages = debug["debug_images"].as<bool>(false);

	YAML::Node network = getOptional(config["network"]);
	gcSocket = std::make_shared<GCSocket>(network["gc_ip"].as<std::string>("224.5.23.1"), network["gc_port"].as<int>(10003), YAML::LoadFile(config["bot_heights_file"].as<std::string>("robot-heights.yml")).as<std::map<std::string, double>>());
	socket = std::make_shared<VisionSocket>(network["vision_ip"].as<std::string>("224.5.23.2"), network["vision_port"].as<int>(10006), gcSocket->defaultBotHeight);
	perspective = std::make_shared<Perspective>(socket, camId);

	YAML::Node stream = getOptional(config["stream"]);
	rtpStreamer = std::make_shared<RTPStreamer>(stream["active"].as<bool>(true), openCl, "rtp://" + stream["ip_base_prefix"].as<std::string>("224.5.23.") + std::to_string(stream["ip_base_end"].as<int>(100) + camId) + ":" + std::to_string(stream["port"].as<int>(10100)));
	rawFeed = stream["raw_feed"].as<bool>(false);

	raw2rgbaKernel = openCl->compile(camera->format().toRgbaKernel, camera->format().toRgbaKernelEnd);
	resampling = openCl->compile(kernel_resampling_cl, kernel_resampling_cl_end);
	gradientDot = openCl->compile(kernel_gradientDot_cl, kernel_gradientDot_cl_end);
	satHorizontal = openCl->compile(kernel_satHorizontal_cl, kernel_satHorizontal_cl_end);
	satVertical = openCl->compile(kernel_satVertical_cl, kernel_satVertical_cl_end);
	satBlobCenter = openCl->compile(kernel_satBlobCenter_cl, kernel_satBlobCenter_cl_end);

	while(waitForGeometry && !socket->getGeometryVersion()) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
		socket->geometryCheck();
	}
}

std::shared_ptr<CLImage> Resources::raw2rgba(const RawImage& img) {
	std::shared_ptr<CLImage> clImg = openCl->acquire(&PixelFormat::RGBA8, img.width, img.height, img.name);
	OpenCL::await(raw2rgbaKernel, cl::EnqueueArgs(cl::NDRange(img.width, img.height)), img.buffer, clImg->image);
	return clImg;
}

void Resources::rgba2blobCenter(const CLImage& rgba, std::shared_ptr<CLImage>& flat, std::shared_ptr<CLImage>& gradDot, std::shared_ptr<CLImage>& blobCenter) {
	cl::NDRange visibleFieldRange(perspective->reprojectedFieldSize[0], perspective->reprojectedFieldSize[1]);
	flat = openCl->acquire(&PixelFormat::RGBA8, perspective->reprojectedFieldSize[0], perspective->reprojectedFieldSize[1], rgba.name);
	gradDot = openCl->acquire(&PixelFormat::F32, perspective->reprojectedFieldSize[0], perspective->reprojectedFieldSize[1], rgba.name);
	std::shared_ptr<CLImage> gradDotHor = openCl->acquire(&PixelFormat::F32, perspective->reprojectedFieldSize[0], perspective->reprojectedFieldSize[1], rgba.name);
	std::shared_ptr<CLImage> gradDotSat = openCl->acquire(&PixelFormat::F32, perspective->reprojectedFieldSize[0], perspective->reprojectedFieldSize[1], rgba.name);
	blobCenter = openCl->acquire(&PixelFormat::F32, perspective->reprojectedFieldSize[0], perspective->reprojectedFieldSize[1], rgba.name);

	cl::Event e1 = OpenCL::run(resampling, cl::EnqueueArgs(visibleFieldRange), rgba.image, flat->image, perspective->getCLCameraModel(), (float)gcSocket->maxBotHeight, perspective->fieldScale, perspective->visibleFieldExtent[0], perspective->visibleFieldExtent[2]);
	cl::Event e2 = OpenCL::run(gradientDot, cl::EnqueueArgs(e1, visibleFieldRange), flat->image, gradDot->image, (int)ceilf(perspective->maxBlobRadius / perspective->fieldScale) / 3);
	cl::Event e3 = OpenCL::run(satHorizontal, cl::EnqueueArgs(e2, cl::NDRange(perspective->reprojectedFieldSize[1])), gradDot->image, gradDotHor->image);
	cl::Event e4 = OpenCL::run(satVertical, cl::EnqueueArgs(e3, cl::NDRange(perspective->reprojectedFieldSize[0])), gradDotHor->image, gradDotSat->image);
	OpenCL::await(satBlobCenter, cl::EnqueueArgs(e4, visibleFieldRange), gradDotSat->image, blobCenter->image, (int)ceilf(perspective->maxBlobRadius / perspective->fieldScale));
}
