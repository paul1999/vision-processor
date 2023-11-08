#include <iostream>

#include "source/imagesource.h"
#include "rtpstreamer.h"
#include "source/spinnakersource.h"
#include "opencl.h"
#include "udpsocket.h"
#include "Perspective.h"
#include "Mask.h"
#include "GroundTruth.h"
#include "messages_robocup_ssl_wrapper.pb.h"
#include "AlignedArray.h"

#include <yaml-cpp/yaml.h>

double getTime() {
	return (double)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count() / 1e6;
}

typedef struct __attribute__ ((packed)) {
	cl_uchar r;
	cl_uchar g;
	cl_uchar b;
} RGB;

const int patterns[16] = {
		0b0100, // 0
		0b1100, // 1
		0b1101, // 2
		0b0101, // 3
		0b0010, // 4
		0b1010, // 5
		0b1011, // 6
		0b0011, // 7
		0b1111, // 8
		0b0000, // 9
		0b0110, //10
		0b1001, //11
		0b1110, //12
		0b1000, //13
		0b0111, //14
		0b0001  //15
};

void addBot(const std::shared_ptr<Perspective>& perspective, const GroundTruth& groundTruth, const double maxBotHeight, const I2& blob, SSL_DetectionRobot* bot) {
	V2 botPos = perspective->image2field({blob.x / 2.0, blob.y / 2.0}, maxBotHeight);

	std::vector<std::pair<double, bool>> nearestBlobs;
	for(const I2& green : groundTruth.getGreen()) {
		V2 blobPos = perspective->image2field({green.x / 2.0, green.y / 2.0}, maxBotHeight);
		V2 diff = {
				blobPos.x - botPos.x,
				blobPos.y - botPos.y
		};

		if(sqrt(diff.x*diff.x + diff.y*diff.y) < 90.0)
			nearestBlobs.emplace_back(atan2(diff.y, diff.x), true);
	}
	for(const I2& pink : groundTruth.getPink()) {
		V2 blobPos = perspective->image2field({pink.x / 2.0, pink.y / 2.0}, maxBotHeight);
		V2 diff = {
				blobPos.x - botPos.x,
				blobPos.y - botPos.y
		};

		if(sqrt(diff.x*diff.x + diff.y*diff.y) < 90.0)
			nearestBlobs.emplace_back(atan2(diff.y, diff.x), false);
	}
	if(nearestBlobs.size() != 4) {
		std::cerr << "Incomplete pattern: " << nearestBlobs.size() << " " << blob.x << "," << blob.y << std::endl;
		return;
	}

	std::sort(nearestBlobs.begin(), nearestBlobs.end());

	double largestAngle = 2*M_PI + nearestBlobs[0].first - nearestBlobs[3].first;
	double orientation = nearestBlobs[3].first + largestAngle/2; //potentially outside of -pi / +pi
	int rotation = 0;
	for(int i = 0; i < 3; i++) {
		double angle = nearestBlobs[i+1].first - nearestBlobs[i].first;
		if(angle > largestAngle) {
			largestAngle = angle;
			orientation = nearestBlobs[i].first + angle/2;
			rotation = i+1;
		}
	}

	std::rotate(nearestBlobs.begin(), nearestBlobs.begin()+rotation, nearestBlobs.end());
	int robotCode = (nearestBlobs[0].second << 3) + (nearestBlobs[1].second << 2) + (nearestBlobs[2].second << 1) + nearestBlobs[3].second;

	bot->set_confidence(1.0f);
	for(int i = 0; i < 16; i++) {
		if(patterns[i] == robotCode) {
			bot->set_robot_id(i);
			break;
		}
	}
	bot->set_x(botPos.x);
	bot->set_y(botPos.y);
	bot->set_orientation(orientation);
	bot->set_pixel_x(blob.x);
	bot->set_pixel_y(blob.y);
	bot->set_height(maxBotHeight);
}

int main() {
	YAML::Node config = YAML::LoadFile("config.yml");

	auto source = config["source"].as<std::string>("SPINNAKER");
	std::unique_ptr<VideoSource> camera = nullptr;

#ifdef SPINNAKER
	if(source == "SPINNAKER")
		camera = std::make_unique<SpinnakerSource>(config["spinnaker_id"].as<int>(0));
#endif

	if(source == "IMAGES") {
		auto paths = config["images"].as<std::vector<std::string>>();

		if(paths.empty()) {
			std::cerr << "Source IMAGES needs at least one image." << std::endl;
			return 1;
		}

		camera = std::make_unique<ImageSource>(paths);
	}

	if(camera == nullptr) {
		std::cerr << "No camera/image source defined." << std::endl;
		return 1;
	}

	auto camId = config["cam_id"].as<int>(0);
	auto defaultBotHeight = config["default_bot_height"].as<double>(150.0);
	auto maxBotAcceleration = 1000*config["max_bot_acceleration"].as<double>(6.5);
	auto maxBallVelocity = 1000*config["max_ball_velocity"].as<double>(8.0);
	auto minTrackingRadius = config["min_tracking_radius"].as<double>(30.0);

	std::shared_ptr<UDPSocket> socket = std::make_shared<UDPSocket>(config["vision_ip"].as<std::string>("224.5.23.2"), config["vision_port"].as<int>(10006), defaultBotHeight, 21.5f);
	std::shared_ptr<Perspective> perspective = std::make_shared<Perspective>(socket, camId);
	std::shared_ptr<OpenCL> openCl = std::make_shared<OpenCL>();
	std::shared_ptr<AlignedArrayPool> arrayPool = std::make_shared<AlignedArrayPool>();
	Mask mask(perspective, defaultBotHeight);
	GroundTruth groundTruth("test-data/rc2022/bots-balls-many-1/gt.yml");
	RTPStreamer rtpStreamer(openCl, "rtp://" + config["vision_ip"].as<std::string>("224.5.23.2") + ":" + std::to_string(config["stream_base_port"].as<int>(10100) + camId));

	cl::Kernel kernel = openCl->compile((
#include "image2field.cl"
#include "ssd.cl"
	), "-D RGGB");
	cl::Kernel ballkernel = openCl->compile((
#include "image2field.cl"
#include "ballssd.cl"
	), "-D RGGB");

	//cv::Ptr<cv::LineSegmentDetector> detector = cv::createLineSegmentDetector();

	uint32_t frameNumber = 0;

	while(true) {
		std::shared_ptr<Image> img = camera->readImage();

		perspective->geometryCheck();
		mask.geometryCheck();

		double startTime = getTime();
		double timestamp = img->getTimestamp() == 0 ? startTime : img->getTimestamp();

		if(perspective->getGeometryVersion()) {
			SSL_WrapperPacket wrapper;
			SSL_DetectionFrame* detection = wrapper.mutable_detection();
			detection->set_frame_number(frameNumber++);
			detection->set_t_capture(startTime);
			if(img->getTimestamp() != 0)
				detection->set_t_capture_camera(img->getTimestamp());
			detection->set_camera_id(camId);

			cl::Buffer clBuffer = openCl->toBuffer(false, img); //TODO aligned array with OpenCL buffer already provided

			if(socket->getTrackedObjects().count(camId)) {
				//TODO do for all camIds, filter already detected from other cameras
				for(const TrackingState& object : socket->getTrackedObjects()[camId]) {
					double timeDelta = timestamp - object.timestamp;
					//double timeDelta = 0.033333;
					double height = object.z + object.vz * timeDelta;
					V2 position = perspective->field2image({
						object.x + object.vx * timeDelta,
						object.y + object.vy * timeDelta,
						height,
					});

					if(position.x < 0 || position.y < 0 || position.x >= perspective->getWidth() || position.y >= perspective->getHeight())
						continue;

					//TODO fix correct search radius (accel/decel)
					RLEVector searchArea = perspective->getRing(
						position,
						height,
						0.0,
						std::max(minTrackingRadius, object.id != -1 ? maxBotAcceleration*timeDelta*timeDelta/2.0 : maxBallVelocity*timeDelta)
					);

					auto posArray = searchArea.scanArea(*arrayPool);
					int searchAreaSize = searchArea.size();
					auto resultArray = arrayPool->acquire<float>(searchAreaSize);

					if(object.id == -1) {
						openCl->run(ballkernel, cl::EnqueueArgs(cl::NDRange(searchAreaSize)), clBuffer, posArray->getBuffer(), resultArray->getBuffer(), perspective->getClPerspective(), height, 21.5f, (RGB) {255, 128, 0}).wait();
						//openCl->run(ballkernel, cl::EnqueueArgs(cl::NDRange(searchAreaSize)), clBuffer, posArray->getBuffer(), resultArray->getBuffer(), perspective->getClPerspective(), height, 21.5f, (RGB) {150, 130, 90}).wait();
					} else if(object.id < 16) {
						openCl->run(kernel, cl::EnqueueArgs(cl::NDRange(searchAreaSize)), clBuffer, posArray->getBuffer(), resultArray->getBuffer(), perspective->getClPerspective(), (float)defaultBotHeight, 25.0f, (RGB) {255, 255, 0}, 45.0f, (RGB) {0, 0, 0}).wait();
					} else {
						openCl->run(kernel, cl::EnqueueArgs(cl::NDRange(searchAreaSize)), clBuffer, posArray->getBuffer(), resultArray->getBuffer(), perspective->getClPerspective(), (float)defaultBotHeight, 25.0f, (RGB) {0, 128, 255}, 45.0f, (RGB) {0, 0, 0}).wait();
					}

					auto* result = resultArray->mapRead<float>();
					int best = std::distance(result, std::min_element(result, result + searchAreaSize));
					resultArray->unmap();

					auto* pos = posArray->mapRead<int>();
					int rawX = pos[2*best];
					int rawY = pos[2*best + 1];
					posArray->unmap();

					V2 bestPos = perspective->image2field({
							(double)rawX,
							(double)rawY
					}, height);

					//TODO thresholding (detection)
					//TODO bot orientation

					if(object.id == -1) {
						SSL_DetectionBall* ball = detection->add_balls();
						ball->set_confidence(1.0f);
						//ball->set_area(0);
						ball->set_x(bestPos.x);
						ball->set_y(bestPos.y);
						//ball->set_z(0.0f);
						ball->set_pixel_x(rawX * 2);
						ball->set_pixel_y(rawY * 2);
					} else {
						SSL_DetectionRobot* bot = object.id < 16 ? detection->add_robots_yellow() : detection->add_robots_blue();
						bot->set_confidence(1.0f);
						bot->set_robot_id(object.id % 16);
						bot->set_x(bestPos.x);
						bot->set_y(bestPos.y);
						bot->set_orientation(object.w);
						bot->set_pixel_x(rawX * 2);
						bot->set_pixel_y(rawY * 2);
						bot->set_height(height);
					}
				}

				/*auto posArray = mask.scanArea(*arrayPool);
				int maskSize = mask.getRuns().size();
				auto resultArray = arrayPool->acquire<float>(maskSize);
				//yellow peak SNR 1.8 PSR 1.15
				//openCl->run(kernel, cl::EnqueueArgs(cl::NDRange(result.size())), clBuffer, clPos, clResult, perspective->getClPerspective(), (float)defaultBotHeight, 25.0f, (RGB) {255, 255, 0}, 45.0f, (RGB) {0, 0, 0}).wait();
				//yellow optimized color peak SNR 5.8 openCl->run(kernel, cl::EnqueueArgs(cl::NDRange(result.size())), clBuffer, clPos, clResult, perspective->getClPerspective(), (float)defaultBotHeight, 25.0f, (RGB) {109, 150, 120}, 45.0f, (RGB) {42, 57, 73}).wait();
				//TODO issues with goal and floor outside field. Black background consideration necessary
				//blue openCl->run(kernel, cl::EnqueueArgs(cl::NDRange(result.size())), clBuffer, clPos, clResult, perspective->getClPerspective(), (float)defaultBotHeight, 25.0f, (RGB) {0, 128, 255}).wait();
				//ball
				openCl->run(ballkernel, cl::EnqueueArgs(cl::NDRange(maskSize)), clBuffer, posArray->getBuffer(), resultArray->getBuffer(), perspective->getClPerspective(), 21.5f, 21.5f, (RGB) {255, 128, 0}).wait();
				///openCl->run(kernel, cl::EnqueueArgs(cl::NDRange(maskSize)), clBuffer, posArray->getBuffer(), resultArray->getBuffer(), perspective->getClPerspective(), 21.5f, 21.5f, (RGB) {255, 128, 0}, 21.5f, (RGB){0, 0, 0}).wait();

				auto* result = resultArray->mapRead<float>();
				float min = *std::min_element(result, result + maskSize);
				float factor = 255 / (*std::max_element(result, result + maskSize) - min);
				std::cout << factor << " " << min << std::endl;

				auto* pos = posArray->mapRead<int>();
				std::shared_ptr<Image> r = BufferImage::create(PixelFormat::F32, img->getWidth(), img->getHeight());
				for(int i = 0; i < maskSize; i++) {
					((float*)r->getData())[img->getWidth()*pos[2*i+1] + pos[2*i]] = (result[i] - min) * factor;
				}
				resultArray->unmap();
				posArray->unmap();
				rtpStreamer.sendFrame(r);*/

				/*for(Run& run : mask.getRuns()) {
					//TODO only RGGB
					uint8_t* row0 = img->getData() + 4*img->getWidth()*run.y + 2*run.x;
					uint8_t* row1 = row0 + 2*img->getWidth();
					for(int i = 0; i < run.length; i++) {
						row0[2*i + 0] = 127;
						row0[2*i + 1] = 127;
						row1[2*i + 0] = 127;
						row1[2*i + 1] = 127;
					}
				}*/

				/*RLEVector yellowGroundTruth;
				for(const I2& point : groundTruth.getYellow()) {
					yellowGroundTruth.add(perspective->getRing({point.x / 2.0, point.y / 2.0}, defaultBotHeight, 0.0, 25.0));
				}

				/*float min = *std::min_element(result.begin(), result.end());
				float sidelobeMin = INFINITY;
				for(int i = 0; i < result.size(); i++) {
					if(result[i] < sidelobeMin && !yellowGroundTruth.contains(pos[2*i], pos[2*i+1])) {
						sidelobeMin = result[i];
					}
				}*/
				/*float max = *std::max_element(result.begin(), result.end());
				float sidelobeMax = -INFINITY;
				for(int i = 0; i < result.size(); i++) {
					if(result[i] > sidelobeMax && !yellowGroundTruth.contains(pos[2 * i], pos[2 * i + 1])) {
						sidelobeMax = result[i];
					}
				}

				/*for(RLEVector& vector : yellowGroundTruth) {
					for(const Run& run : vector.getRuns()) {
						uint8_t* row0 = img->getData() + 4*img->getWidth()*run.y + 2*run.x;
						uint8_t* row1 = row0 + 2*img->getWidth();
						for(int i = 0; i < run.length; i++) {
							row0[2*i + 0] = 255;
							row0[2*i + 1] = 255;
							row1[2*i + 0] = 255;
							row1[2*i + 1] = 255;
						}
					}
				}*/

				//float threshold = min*1.15;
				/*float threshold = max*0.9;
				for(int i = 0; i < result.size(); i++) {
					if(result[i] > threshold) {
						uint8_t* row0 = img->getData() + 4*img->getWidth()*pos[2*i+1] + 2*pos[2*i];
						uint8_t* row1 = row0 + 2*img->getWidth();
						row0[0] = 255;
						row0[1] = 255;
						row1[0] = 255;
						row1[1] = 0;
					}
				}
				//std::sort(result.begin(), result.end());
				//std::cout << "minmax " << result[0] << " " << result[result.size()/2] << " " << result[result.size()-1] << std::endl;
				//std::cout << "min sidelobe " << min << " " << sidelobeMin << std::endl;
				std::cout << "max sidelobe " << max << " " << sidelobeMax << std::endl;*/
			} else {
				for(const I2& orange : groundTruth.getOrange()) {
					V2 ballPos = perspective->image2field({orange.x/2.0, orange.y/2.0}, 21.5);
					SSL_DetectionBall* ball = detection->add_balls();
					ball->set_confidence(1.0f);
					//ball->set_area(0);
					ball->set_x(ballPos.x);
					ball->set_y(ballPos.y);
					//ball->set_z(0.0f);
					ball->set_pixel_x(orange.x);
					ball->set_pixel_y(orange.y);
				}

				for(const I2& blob : groundTruth.getYellow())
					addBot(perspective, groundTruth, defaultBotHeight, blob, detection->add_robots_yellow());

				for(const I2& blob : groundTruth.getBlue())
					addBot(perspective, groundTruth, defaultBotHeight, blob, detection->add_robots_blue());
			}

			detection->set_t_sent(getTime());
			socket->send(wrapper);

			/*if(frameNumber == 2) {
				std::cout << "main " << (getTime() - startTime) * 1000.0 << " ms" << std::endl;
				socket->close();
				return 0;
			}*/
		}

		/*cv::Mat cvImg(img->getHeight(), img->getWidth(), CV_8UC3, (uint8_t*)img->getData());

		//BGR -> Grayscale
		cv::Mat gray;
		cv::cvtColor(cvImg, gray, cv::COLOR_BGR2GRAY);
		//Threshold (128)
		cv::threshold(gray, gray, 93, 255, cv::THRESH_BINARY);
		//Line thinning (Morphological)
		//Hough transform
		//peak picking
		cv::Mat4f lines;
		//Pixel resolution, angular resolution, accumulator threshold, min line length, max line gap
		//cv::HoughLinesP(gray, lines, 10.0, 0.31415, 100, 100, 5);
		cv::HoughLinesP(gray, lines, 20.0, 0.31415, 100, 100, 5);
		//detector->detect(gray, lines);
		cv::Mat cvOut(cvImg.rows, cvImg.cols, CV_8UC3, cv::Scalar(0));
		detector->drawSegments(cvOut, lines);
		std::shared_ptr<Image> out = std::make_shared<CVImage>(cvOut, BGR888);
		rtpStreamer.sendFrame(out);*/

		rtpStreamer.sendFrame(img);
		std::cout << "main " << (getTime() - startTime) * 1000.0 << " ms" << std::endl;
		std::this_thread::sleep_for(std::chrono::microseconds(33333 - (int64_t)((getTime() - startTime) * 1e6)));
	}
}
