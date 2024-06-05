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
		Eigen::Vector3f pos = r.perspective->model.image2field({ball.pixel_x(), ball.pixel_y()}, r.ballRadius);
		ball.set_x(pos.x());
		ball.set_y(pos.y());
	}

	detection->set_camera_id(r.camId);
	detection->set_t_capture(getTime());
	detection->set_t_sent(getTime());
	r.socket->send(wrapper);
}

int main(int argc, char* argv[]) {
	Resources r(YAML::LoadFile(argc > 1 ? argv[1] : "config.yml"));

	while(r.waitForGeometry && !r.socket->getGeometryVersion()) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
		r.socket->geometryCheck();
	}

	std::shared_ptr<Image> img = r.camera->readImage();
	r.perspective->geometryCheck(img->width, img->height, r.gcSocket->maxBotHeight);

	Image gray = img->toGrayscale();
	Image thresholded = thresholdImage(r, gray, halfLineWidthEstimation(r, gray));
	r.perspective->model.ensureSize({thresholded.width, thresholded.height});
	const std::vector<Eigen::Vector2f> linePixels = getLinePixels(thresholded);
	const int error = modelError(r, r.perspective->model, linePixels);

	std::cout << "[Model score] " << (error/(float)linePixels.size()) << std::endl;
	geometryBenchmark(r, 1);
}