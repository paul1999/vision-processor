#include "GroundTruth.h"

#include <yaml-cpp/yaml.h>

#include "messages_robocup_ssl_wrapper.pb.h"


namespace YAML {
	template<> struct convert<SSL_DetectionBall> {
		static bool decode(const Node& node, SSL_DetectionBall& ball) {
			ball.set_confidence(node["confidence"].as<float>());
			if(node["area"].IsDefined())
				ball.set_area(node["area"].as<int>());

			ball.set_x(node["x"].as<float>());
			ball.set_y(node["y"].as<float>());
			if(node["z"].IsDefined())
				ball.set_z(node["z"].as<float>());

			ball.set_pixel_x(node["pixel_x"].as<float>());
			ball.set_pixel_y(node["pixel_y"].as<float>());
			return true;
		}
	};

	template<> struct convert<SSL_DetectionRobot> {
		static bool decode(const Node& node, SSL_DetectionRobot& robot) {
			robot.set_confidence(node["confidence"].as<float>());
			if(node["robot_id"].IsDefined())
				robot.set_robot_id(node["robot_id"].as<int>());

			robot.set_x(node["x"].as<float>());
			robot.set_y(node["y"].as<float>());
			if(node["orientation"].IsDefined())
				robot.set_orientation(node["orientation"].as<float>());

			robot.set_pixel_x(node["pixel_x"].as<float>());
			robot.set_pixel_y(node["pixel_y"].as<float>());
			if(node["height"].IsDefined())
				robot.set_height(node["height"].as<float>());

			return true;
		}
	};

	template<> struct convert<SSL_DetectionFrame> {
		static bool decode(const Node& node, SSL_DetectionFrame& detection) {
			detection.set_camera_id(node["camera_id"].as<int>());
			detection.set_frame_number(node["frame_number"].as<int>());
			detection.set_t_capture(node["t_capture"].as<double>());
			detection.set_t_sent(node["t_sent"].as<double>());
			if(node["t_capture_camera"].IsDefined())
				detection.set_t_capture_camera(node["t_capture_camera"].as<double>());

			for (const auto &item : node["balls"])
				detection.mutable_balls()->Add(item.as<SSL_DetectionBall>());
			for (const auto &item : node["robots_blue"])
				detection.mutable_robots_blue()->Add(item.as<SSL_DetectionRobot>());
			for (const auto &item : node["robots_yellow"])
				detection.mutable_robots_yellow()->Add(item.as<SSL_DetectionRobot>());

			return true;
		}
	};
}


GroundTruth::GroundTruth(const std::string &source, int cameraId, double timestamp) {
	SSL_WrapperPacket wrapper;
	SSL_DetectionFrame* detection = wrapper.mutable_detection();
	detection->CopyFrom(YAML::LoadFile(source).as<SSL_DetectionFrame>());
	detection->set_camera_id(cameraId);
	detection->set_t_capture(timestamp);
	detection->set_t_sent(timestamp);
	message = std::make_unique<SSL_WrapperPacket>(wrapper);
}
