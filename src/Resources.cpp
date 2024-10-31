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
	[[maybe_unused]] auto exposure = cam["exposure"].as<double>(0.0);
	[[maybe_unused]] auto gain = cam["gain"].as<double>(0.0);

	YAML::Node wbNode = cam["white_balance"];
	[[maybe_unused]] WhiteBalanceType wbType = WhiteBalanceType_Manual;
	[[maybe_unused]] std::vector<double> wbValues;
	if(wbNode.IsSequence()) {
		wbValues = wbNode.as<std::vector<double>>();
	} else {
		auto wbTypeString = wbNode.as<std::string>("OUTDOOR");
		wbType = wbTypeString == "OUTDOOR" ? WhiteBalanceType_AutoOutdoor : WhiteBalanceType_AutoIndoor;
	}

#ifdef SPINNAKER
	if(driver == "SPINNAKER")
		camera = std::make_unique<SpinnakerDriver>(driver_id, exposure, gain, wbType, wbValues);
#endif

#ifdef MVIMPACT
	if(driver == "MVIMPACT")
		camera = std::make_unique<MVImpactDriver>(driver_id);
#endif

	if(driver == "OPENCV")
		camera = std::make_unique<OpenCVDriver>(cam["path"].as<std::string>("/dev/video" + std::to_string(driver_id)));

	if(camera == nullptr) {
		std::cerr << "[Resources] No camera/image source defined." << std::endl;
		exit(1);
	}

	camId = config["cam_id"].as<int>(0);

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
	minMajorLineLength = geometry["min_major_line_length"].as<double>(0.5);
	maxIntersectionDistance = geometry["max_intersection_distance"].as<double>(0.2);
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
	waitForGeometry = debug["wait_for_geometry"].as<bool>(false);
	debugImages = debug["debug_images"].as<bool>(false);
	rawFeed = debug["raw_feed"].as<bool>(false);

	YAML::Node sizes = getOptional(config["sizes"]);
	YAML::Node network = getOptional(config["network"]);
	gcSocket = std::make_shared<GCSocket>(network["gc_ip"].as<std::string>("224.5.23.1"), network["gc_port"].as<int>(10003), YAML::LoadFile(sizes["bot_heights_file"].as<std::string>("robot-heights.yml")).as<std::map<std::string, double>>());
	socket = std::make_shared<VisionSocket>(network["vision_ip"].as<std::string>("224.5.23.2"), network["vision_port"].as<int>(10006), gcSocket->defaultBotHeight, 21.5); //TODO
	perspective = std::make_shared<Perspective>(socket, camId);
	rtpStreamer = std::make_shared<RTPStreamer>(openCl, "rtp://" + network["stream_ip_base_prefix"].as<std::string>("224.5.23.") + std::to_string(network["stream_ip_base_end"].as<int>(100) + camId) + ":" + std::to_string(network["stream_port"].as<int>(10100)));
}
