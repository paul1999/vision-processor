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
#include <yaml-cpp/node/parse.h>
#include "Resources.h"
#include "GroundTruth.h"
#include "proto/ssl_vision_wrapper.pb.h"
#include "calib/LineDetection.h"
#include "calib/GeomModel.h"

static void moveRobot(const Resources& r, SSL_DetectionRobot& robot) {
	Eigen::Vector3f pos = r.perspective->model.image2field({robot.pixel_x(), robot.pixel_y()}, robot.height());
	robot.set_x(pos.x());
	robot.set_y(pos.y());
}

static void geometryBenchmark(Resources& r, const uint32_t frameId) {
	std::vector<SSL_DetectionFrame> groundTruth = parseGroundTruth(r.groundTruth);

	SSL_WrapperPacket wrapper;
	SSL_DetectionFrame* detection = wrapper.mutable_detection();
	detection->CopyFrom(getCorrespondingFrame(groundTruth, frameId));

	for(SSL_DetectionRobot& robot : *detection->mutable_robots_yellow())
		moveRobot(r, robot);
	for(SSL_DetectionRobot& robot : *detection->mutable_robots_blue())
		moveRobot(r, robot);

	for(SSL_DetectionBall& ball : *detection->mutable_balls()) {
		Eigen::Vector3f pos = r.perspective->model.image2field({ball.pixel_x(), ball.pixel_y()}, r.perspective->field.ball_radius());
		ball.set_x(pos.x());
		ball.set_y(pos.y());
	}

	detection->set_camera_id(r.camId);
	detection->set_t_capture(getRealTime());
	detection->set_t_sent(getRealTime());
	r.socket->send(wrapper);
}

int main(int argc, char* argv[]) {
	Resources r(YAML::LoadFile(argc > 1 ? argv[1] : "config.yml"));

	while(r.waitForGeometry && !r.socket->getGeometryVersion()) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
		r.socket->geometryCheck();
	}

	std::shared_ptr<Image> img = r.camera->readImage();
	r.perspective->geometryCheck(r.cameraAmount, img->width, img->height, r.gcSocket->maxBotHeight);

	Image gray = img->toGrayscale();
	Image thresholded = thresholdImage(r, gray, halfLineWidthEstimation(r, gray));
	r.perspective->model.ensureSize({thresholded.width, thresholded.height});
	const std::vector<Eigen::Vector2f> linePixels = getLinePixels(thresholded);
	const int error = modelError(r, r.perspective->model, linePixels);

	std::cout << "[Model score] " << (error/(float)linePixels.size()) << std::endl;
	geometryBenchmark(r, 1);
}