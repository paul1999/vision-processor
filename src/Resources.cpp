#include <yaml-cpp/yaml.h>
#include "Resources.h"

static uint8_t readHue(YAML::Node node, double fallback) {
	return node.as<double>(fallback) * 256.0 / 360.0;
}

Resources::Resources(YAML::Node config) {
	openCl = std::make_shared<OpenCL>();
	arrayPool = std::make_shared<AlignedArrayPool>();

	auto source = config["source"].as<std::string>("SPINNAKER");

#ifdef SPINNAKER
	if(source == "SPINNAKER")
		camera = std::make_unique<SpinnakerSource>(config["spinnaker_id"].as<int>(0));
#endif

	if(source == "OPENCV")
		camera = std::make_unique<OpenCVSource>(config["opencv_path"].as<std::string>("/dev/video0"));

	if(source == "IMAGES") {
		auto paths = config["images"].as<std::vector<std::string>>();

		if(paths.empty()) {
			std::cerr << "Source IMAGES needs at least one image." << std::endl;
			return;
		}

		camera = std::make_unique<ImageSource>(paths);
	}

	if(camera == nullptr) {
		std::cerr << "No camera/image source defined." << std::endl;
		return;
	}

	camId = config["cam_id"].as<int>(0);
	cameraAmount = config["camera_amount"].as<int>(1);
	sideBlobDistance = config["side_blob_distance"].as<double>(65.0);
	centerBlobRadius = config["center_blob_radius"].as<double>(25.0);
	sideBlobRadius = config["side_blob_radius"].as<double>(20.0);
	ballRadius = config["ball_radius"].as<double>(21.5);
	groundTruth = config["ground_truth"].as<std::string>("");

	YAML::Node thresholds = config["thresholds"].IsDefined() ? config["thresholds"] : YAML::Node();
	minCircularity = thresholds["circularity"].as<double>(0.625);
	minSaturation = thresholds["saturation"].as<int>(24);
	minBrightness = thresholds["brightness"].as<int>(32);

	YAML::Node hues = config["hues"].IsDefined() ? config["hues"] : YAML::Node();
	orangeHue = readHue(hues["orange"], 30.0);
	yellowHue = readHue(hues["yellow"], 60.0);
	blueHue = readHue(hues["blue"], 210.0);
	greenHue = readHue(hues["green"], 120.0);
	pinkHue = readHue(hues["pink"], 300.0);

	YAML::Node tracking = config["tracking"].IsDefined() ? config["tracking"] : YAML::Node();
	minTrackingRadius = tracking["min_tracking_radius"].as<double>(30.0);
	maxBallVelocity = 1000*tracking["max_ball_velocity"].as<double>(8.0);
	maxBotAcceleration = 1000*tracking["max_bot_acceleration"].as<double>(6.5);


	YAML::Node network = config["network"].IsDefined() ? config["network"] : YAML::Node();
	gcSocket = std::make_shared<GCSocket>(network["gc_ip"].as<std::string>("224.5.23.1"), network["gc_port"].as<int>(10003), YAML::LoadFile(config["bot_heights_file"].as<std::string>("robot-heights.yml")).as<std::map<std::string, double>>());
	socket = std::make_shared<VisionSocket>(network["vision_ip"].as<std::string>("224.5.23.2"), network["vision_port"].as<int>(10006), gcSocket->defaultBotHeight, ballRadius);
	perspective = std::make_shared<Perspective>(socket, camId);
	mask = std::make_shared<Mask>(perspective, gcSocket->maxBotHeight, ballRadius);
	rtpStreamer = std::make_shared<RTPStreamer>(openCl, "rtp://" + network["stream_ip_base_prefix"].as<std::string>("224.5.23.") + std::to_string(network["stream_ip_base_end"].as<int>(100) + camId) + ":" + std::to_string(network["stream_port"].as<int>(10100)));

	blurkernel = openCl->compileFile("kernel/blur.cl");
	gradientkernel = openCl->compileFile("kernel/gradient.cl", "-D RGGB");
	diffkernel = openCl->compileFile("kernel/delta.cl");
	ringkernel = openCl->compileFile("kernel/midpointssd.cl", "-D RGGB");
}
