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
	minCircularity = thresholds["circularity"].as<double>(25.0); // 10.0
	minScore = thresholds["score"].as<double>(0.0); // 8.0
	maxBlobs = thresholds["blobs"].as<int>(2000);

	YAML::Node sizes = getOptional(config["sizes"]);
	sideBlobDistance = sizes["side_blob_distance"].as<double>(65.0);
	centerBlobRadius = sizes["center_blob_radius"].as<double>(25.0);
	sideBlobRadius = sizes["side_blob_radius"].as<double>(20.0);
	minBlobRadius = std::min({centerBlobRadius, sideBlobRadius, 21.5}); //TODO
	maxBlobRadius = std::max({centerBlobRadius, sideBlobRadius, 21.5}); //TODO

	YAML::Node tracking = getOptional(config["tracking"]);
	minTrackingRadius = tracking["min_tracking_radius"].as<double>(20.0);
	maxBallVelocity = 1000*tracking["max_ball_velocity"].as<double>(8.0);
	maxBotAcceleration = 1000*tracking["max_bot_acceleration"].as<double>(6.5);

	YAML::Node geometry = getOptional(config["geometry"]);
	cameraAmount = geometry["camera_amount"].as<int>(1);
	cameraHeight = geometry["camera_height"].as<double>(0.0);
	fieldLineThreshold = geometry["field_line_threshold"].as<int>(5);
	minLineSegmentLength = geometry["min_line_segment_length"].as<double>(10.0);
	minMajorLineLength = geometry["min_major_line_length"].as<double>(0.5);
	maxIntersectionDistance = geometry["max_intersection_distance"].as<double>(0.2);
	maxLineSegmentOffset = geometry["max_line_segment_offset"].as<double>(10.0);
	maxLineSegmentAngle = geometry["max_line_segment_angle"].as<double>(3.0) * M_PI/180.0;

	YAML::Node benchmark = getOptional(config["debug"]);
	groundTruth = benchmark["ground_truth"].as<std::string>("gt.yml");
	waitForGeometry = benchmark["wait_for_geometry"].as<bool>(false);
	debugImages = benchmark["debug_images"].as<bool>(false);

	YAML::Node network = getOptional(config["network"]);
	gcSocket = std::make_shared<GCSocket>(network["gc_ip"].as<std::string>("224.5.23.1"), network["gc_port"].as<int>(10003), YAML::LoadFile(sizes["bot_heights_file"].as<std::string>("robot-heights.yml")).as<std::map<std::string, double>>());
	socket = std::make_shared<VisionSocket>(network["vision_ip"].as<std::string>("224.5.23.2"), network["vision_port"].as<int>(10006), gcSocket->defaultBotHeight, 21.5); //TODO
	perspective = std::make_shared<Perspective>(socket, camId);
	rtpStreamer = std::make_shared<RTPStreamer>(openCl, "rtp://" + network["stream_ip_base_prefix"].as<std::string>("224.5.23.") + std::to_string(network["stream_ip_base_end"].as<int>(100) + camId) + ":" + std::to_string(network["stream_port"].as<int>(10100)));
}
