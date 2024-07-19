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

#include <utility>
#include "Resources.h"

#include "source/spinnakersource.h"
#include "source/mvimpactsource.h"
#include "source/videosource.h"
#include "source/opencvsource.h"

static uint8_t readHue(const YAML::Node& node, double fallback) {
	return node.as<double>(fallback) * 256.0 / 360.0;
}

double getTime() {
	return (double)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count() / 1e6;
}

Resources::Resources(const YAML::Node& config) {
	openCl = std::make_shared<OpenCL>();

	auto source = config["source"].as<std::string>("SPINNAKER");
	int source_id = config["source_id"].as<int>(0);

#ifdef SPINNAKER
	if(source == "SPINNAKER")
		camera = std::make_unique<SpinnakerSource>(source_id);
#endif

#ifdef MVIMPACT
	if(source == "MVIMPACT")
		camera = std::make_unique<MVImpactSource>(source_id);
#endif

	if(source == "OPENCV")
		camera = std::make_unique<OpenCVSource>(config["opencv_path"].as<std::string>("/dev/video" + std::to_string(source_id)));

	if(source == "IMAGES") {
		auto paths = config["images"].as<std::vector<std::string>>();

		if(paths.empty()) {
			std::cerr << "[Resources] Source IMAGES needs at least one image." << std::endl;
			exit(1);
		}

		camera = std::make_unique<ImageSource>(paths);
	}

	if(camera == nullptr) {
		std::cerr << "[Resources] No camera/image source defined." << std::endl;
		exit(1);
	}

	camId = config["cam_id"].as<int>(0);

	YAML::Node thresholds = config["thresholds"].IsDefined() ? config["thresholds"] : YAML::Node();
	minCircularity = thresholds["circularity"].as<double>(10.0);
	minScore = thresholds["score"].as<double>(64.0); // 32.0
	maxBlobs = thresholds["blobs"].as<int>(2000);

	YAML::Node sizes = config["sizes"].IsDefined() ? config["sizes"] : YAML::Node();
	sideBlobDistance = sizes["side_blob_distance"].as<double>(65.0);
	centerBlobRadius = sizes["center_blob_radius"].as<double>(25.0);
	sideBlobRadius = sizes["side_blob_radius"].as<double>(20.0);
	ballRadius = sizes["ball_radius"].as<double>(21.5);
	minBlobRadius = std::min({centerBlobRadius, sideBlobRadius, ballRadius});
	maxBlobRadius = std::max({centerBlobRadius, sideBlobRadius, ballRadius});

	YAML::Node tracking = config["tracking"].IsDefined() ? config["tracking"] : YAML::Node();
	minTrackingRadius = tracking["min_tracking_radius"].as<double>(30.0);
	maxBallVelocity = 1000*tracking["max_ball_velocity"].as<double>(8.0);
	maxBotAcceleration = 1000*tracking["max_bot_acceleration"].as<double>(6.5);

	YAML::Node geometry = config["geometry"].IsDefined() ? config["geometry"] : YAML::Node();
	cameraAmount = geometry["camera_amount"].as<int>(1);
	cameraHeight = geometry["camera_height"].as<double>(0.0);
	fieldLineThreshold = geometry["field_line_threshold"].as<int>(5);
	minLineSegmentLength = geometry["min_line_segment_length"].as<double>(10.0);
	minMajorLineLength = geometry["min_major_line_length"].as<double>(0.5);
	maxIntersectionDistance = geometry["max_intersection_distance"].as<double>(0.2);
	maxLineSegmentOffset = geometry["max_line_segment_offset"].as<double>(10.0);
	maxLineSegmentAngle = geometry["max_line_segment_angle"].as<double>(3.0) * M_PI/180.0;

	YAML::Node benchmark = config["benchmark"].IsDefined() ? config["benchmark"] : YAML::Node();
	groundTruth = benchmark["ground_truth"].as<std::string>("gt.yml");
	waitForGeometry = benchmark["wait_for_geometry"].as<bool>(false);
	debugImages = benchmark["debug_images"].as<bool>(false);

	YAML::Node network = config["network"].IsDefined() ? config["network"] : YAML::Node();
	gcSocket = std::make_shared<GCSocket>(network["gc_ip"].as<std::string>("224.5.23.1"), network["gc_port"].as<int>(10003), YAML::LoadFile(sizes["bot_heights_file"].as<std::string>("robot-heights.yml")).as<std::map<std::string, double>>());
	socket = std::make_shared<VisionSocket>(network["vision_ip"].as<std::string>("224.5.23.2"), network["vision_port"].as<int>(10006), gcSocket->defaultBotHeight, ballRadius);
	perspective = std::make_shared<Perspective>(socket, camId);
	rtpStreamer = std::make_shared<RTPStreamer>(openCl, "rtp://" + network["stream_ip_base_prefix"].as<std::string>("224.5.23.") + std::to_string(network["stream_ip_base_end"].as<int>(100) + camId) + ":" + std::to_string(network["stream_port"].as<int>(10100)));
}
