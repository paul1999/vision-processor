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
#include "source/opencvsource.h"
#include "distortion.h"

#include <yaml-cpp/yaml.h>

double getTime() {
	return (double)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count() / 1e6;
}

typedef struct __attribute__ ((packed)) {
	cl_uchar r;
	cl_uchar g;
	cl_uchar b;
} RGB;

//1 indicates green, 0 pink, increasing 2d angle starting from bot orientation most significant bit to least significant bit
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

const double patternAngles[4] = {
		1.0021839078803572,
		2.5729802346752537,
		-2.5729802346752537, //3.7102050725043325
		-1.0021839078803572 //5.281001399299229
};

const int patternLUT[16] = { 9, 15, 4, 7, 0, 3, 10, 14, 13, 11, 5, 6, 1, 2, 12, 8 };

std::pair<double, int> orientationAndPatternId(std::vector<std::pair<double, bool>>& blobs) {
	std::sort(blobs.begin(), blobs.end());

	double largestAngle = 2*M_PI + blobs[0].first - blobs[3].first;
	double orientation = blobs[3].first + largestAngle/2; //potentially outside of -pi / +pi
	int rotation = 0;
	for(int i = 0; i < 3; i++) {
		double angle = blobs[i+1].first - blobs[i].first;
		if(angle > largestAngle) {
			largestAngle = angle;
			orientation = blobs[i].first + angle/2;
			rotation = i+1;
		}
	}

	std::rotate(blobs.begin(), blobs.begin()+rotation, blobs.end());
	return {orientation, patternLUT[(blobs[0].second << 3) + (blobs[1].second << 2) + (blobs[2].second << 1) + blobs[3].second]};
}

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

	std::pair<double, int> OAPID = orientationAndPatternId(nearestBlobs);

	bot->set_confidence(1.0f);
	bot->set_robot_id(OAPID.second);
	bot->set_x(botPos.x);
	bot->set_y(botPos.y);
	bot->set_orientation(OAPID.first);
	bot->set_pixel_x(blob.x);
	bot->set_pixel_y(blob.y);
	bot->set_height(maxBotHeight);
}

class Resources {
public:
	explicit Resources(YAML::Node config) {
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
		maxBotAcceleration = 1000*config["max_bot_acceleration"].as<double>(6.5);
		sideBlobDistance = config["side_blob_distance"].as<double>(65.0);
		centerBlobRadius = config["center_blob_radius"].as<double>(25.0);
		sideBlobRadius = config["side_blob_radius"].as<double>(20.0);
		maxBallVelocity = 1000*config["max_ball_velocity"].as<double>(8.0);
		ballRadius = config["ball_radius"].as<double>(21.5);
		minTrackingRadius = config["min_tracking_radius"].as<double>(30.0);

		YAML::Node network = config["network"];
		gcSocket = std::make_shared<GCSocket>(network["gc_ip"].as<std::string>("224.5.23.1"), network["gc_port"].as<int>(10003), YAML::LoadFile(config["bot_heights_file"].as<std::string>("robot-heights.yml")).as<std::map<std::string, double>>());
		socket = std::make_shared<VisionSocket>(network["vision_ip"].as<std::string>("224.5.23.2"), network["vision_port"].as<int>(10006), gcSocket->getDefaultBotHeight(), ballRadius);
		perspective = std::make_shared<Perspective>(socket, camId);
		openCl = std::make_shared<OpenCL>();
		arrayPool = std::make_shared<AlignedArrayPool>();
		mask = std::make_shared<Mask>(perspective, gcSocket->getDefaultBotHeight());
		rtpStreamer = std::make_shared<RTPStreamer>(openCl, "rtp://" + network["stream_ip_base_prefix"].as<std::string>("224.5.23.") + std::to_string(network["stream_ip_base_end"].as<int>(100) + camId) + ":" + std::to_string(network["stream_port"].as<int>(10100)));

		botkernel = openCl->compile((
#include "image2field.cl"
#include "botssd.cl"
		), "-D RGGB");
		sidekernel = openCl->compile((
#include "image2field.cl"
#include "ssd.cl"
		), "-D RGGB");
		ballkernel = openCl->compile((
#include "image2field.cl"
#include "ballssd.cl"
		), "-D RGGB");
	}

	std::unique_ptr<VideoSource> camera = nullptr;

	int camId;
	int cameraAmount;
	double maxBotAcceleration;
	double sideBlobDistance;
	double centerBlobRadius;
	double sideBlobRadius;
	double maxBallVelocity;
	double ballRadius;
	double minTrackingRadius;

	std::shared_ptr<GCSocket> gcSocket;
	std::shared_ptr<VisionSocket> socket;
	std::shared_ptr<Perspective> perspective;
	std::shared_ptr<OpenCL> openCl;
	std::shared_ptr<AlignedArrayPool> arrayPool;
	std::shared_ptr<Mask> mask;
	std::shared_ptr<RTPStreamer> rtpStreamer;

	cl::Kernel botkernel;
	cl::Kernel sidekernel;
	cl::Kernel ballkernel;
};

void trackObjects(Resources& r, const double timestamp, const cl::Buffer& clBuffer, const std::vector<TrackingState>& objects, SSL_DetectionFrame* detection, std::vector<int>& filtered, uint8_t* data) {
	std::vector<int> tracked;
	for(const TrackingState& object : objects) {
		if(std::any_of(filtered.begin(), filtered.end(), [&] (int i) { return i == object.id; }))
			continue;

		double timeDelta = timestamp - object.timestamp;
		//double timeDelta = 0.033333;
		double height = object.z + object.vz * timeDelta;
		V2 position = r.perspective->field2image({
			object.x + object.vx * timeDelta,
			object.y + object.vy * timeDelta,
			height,
		});

		if(position.x < 0 || position.y < 0 || position.x >= r.perspective->getWidth() || position.y >= r.perspective->getHeight()) {
			std::cout << "Lost " << object.id << " " << timeDelta << std::endl;
			continue;
		}

		//TODO fix correct search radius (accel/decel)
		RLEVector searchArea = r.perspective->getRing(
				position,
				height,
				0.0,
				std::max(r.minTrackingRadius, object.id != -1 ? r.maxBotAcceleration*timeDelta*timeDelta/2.0 : r.maxBallVelocity*timeDelta)
		);

		int rawX, rawY;
		{
			auto posArray = searchArea.scanArea(*r.arrayPool);
			int searchAreaSize = searchArea.size();
			auto resultArray = r.arrayPool->acquire<float>(searchAreaSize);

			if(object.id == -1) {
				//r.openCl->run(r.ballkernel, cl::EnqueueArgs(cl::NDRange(searchAreaSize)), clBuffer, posArray->getBuffer(), resultArray->getBuffer(), r.perspective->getClPerspective(), height, (float)r.ballRadius, (RGB) {255, 128, 0}).wait();
				r.openCl->run(r.sidekernel, cl::EnqueueArgs(cl::NDRange(searchAreaSize)), clBuffer, posArray->getBuffer(), resultArray->getBuffer(), r.perspective->getClPerspective(), height, (float)r.ballRadius, (RGB) {255, 128, 0}).wait();
			} else if(object.id < 16) {
				//r.openCl->run(r.botkernel, cl::EnqueueArgs(cl::NDRange(searchAreaSize)), clBuffer, posArray->getBuffer(), resultArray->getBuffer(), r.perspective->getClPerspective(), (float)r.defaultBotHeight, (float)r.centerBlobRadius, (RGB) {255, 255, 0}, (float)(r.sideBlobDistance - r.sideBlobRadius), (RGB) {0, 0, 0}).wait();
				r.openCl->run(r.botkernel, cl::EnqueueArgs(cl::NDRange(searchAreaSize)), clBuffer, posArray->getBuffer(), resultArray->getBuffer(), r.perspective->getClPerspective(), (float)r.gcSocket->getYellowBotHeight(), (float)r.centerBlobRadius, (RGB) {255, 255, 128}, (float)(r.sideBlobDistance - r.sideBlobRadius), (RGB) {32, 32, 32}).wait();
			} else {
				//r.openCl->run(r.botkernel, cl::EnqueueArgs(cl::NDRange(searchAreaSize)), clBuffer, posArray->getBuffer(), resultArray->getBuffer(), r.perspective->getClPerspective(), (float)r.defaultBotHeight, (float)r.centerBlobRadius, (RGB) {0, 128, 255}, (float)(r.sideBlobDistance - r.sideBlobRadius), (RGB) {0, 0, 0}).wait();
				r.openCl->run(r.botkernel, cl::EnqueueArgs(cl::NDRange(searchAreaSize)), clBuffer, posArray->getBuffer(), resultArray->getBuffer(), r.perspective->getClPerspective(), (float)r.gcSocket->getBlueBotHeight(), (float)r.centerBlobRadius, (RGB) {0, 128, 255}, (float)(r.sideBlobDistance - r.sideBlobRadius), (RGB) {0, 0, 0}).wait();
			}

			auto* result = resultArray->mapRead<float>();
			int best = std::distance(result, std::min_element(result, result + searchAreaSize));
			resultArray->unmap();

			auto* pos = posArray->mapRead<int>();
			rawX = pos[2*best];
			rawY = pos[2*best + 1];
			posArray->unmap();
		}

		V2 bestPos = r.perspective->image2field({ (double)rawX, (double)rawY }, height);

		//TODO subsampling/supersampling (constant computational time/constant resolution)
		//ssd starting from 16 pixels +2 steps...? (only half of computational cost increase)

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
			//TODO better pattern matching/orientation determination
			RLEVector sideSearchArea = r.perspective->getRing(position, height, std::max(0.0, r.sideBlobDistance - r.minTrackingRadius/2), r.sideBlobDistance + r.minTrackingRadius/2);
			int pattern = patterns[object.id % 16];
			int searchAreaSize = sideSearchArea.size();
			auto posArray = sideSearchArea.scanArea(*r.arrayPool);
			auto greenArray = r.arrayPool->acquire<float>(searchAreaSize);
			auto pinkArray = r.arrayPool->acquire<float>(searchAreaSize);
			//r.openCl->run(r.sidekernel, cl::EnqueueArgs(cl::NDRange(searchAreaSize)), clBuffer, posArray->getBuffer(), greenArray->getBuffer(), r.perspective->getClPerspective(), height, (float)r.sideBlobRadius, (RGB) {0, 255, 128}).wait();
			r.openCl->run(r.sidekernel, cl::EnqueueArgs(cl::NDRange(searchAreaSize)), clBuffer, posArray->getBuffer(), greenArray->getBuffer(), r.perspective->getClPerspective(), height, (float)r.sideBlobRadius, (RGB) {64, 128, 48}).wait();
			//r.openCl->run(r.sidekernel, cl::EnqueueArgs(cl::NDRange(searchAreaSize)), clBuffer, posArray->getBuffer(), pinkArray->getBuffer(), r.perspective->getClPerspective(), height, (float)r.sideBlobRadius, (RGB) {255, 0, 255}).wait();
			r.openCl->run(r.sidekernel, cl::EnqueueArgs(cl::NDRange(searchAreaSize)), clBuffer, posArray->getBuffer(), pinkArray->getBuffer(), r.perspective->getClPerspective(), height, (float)r.sideBlobRadius, (RGB) {255, 128, 128}).wait();

			auto* green = greenArray->mapRead<float>();
			auto* pink = pinkArray->mapRead<float>();
			auto* pos = posArray->mapRead<int>();
			auto bestGreen = std::min_element(green, green + searchAreaSize);
			auto bestPink = std::min_element(pink, pink + searchAreaSize);
			int anchor = ((*bestGreen < *bestPink && pattern != 0b0000) || pattern == 0b1111) ? 1 : 0;
			int anchorIndex = std::distance(anchor ? green : pink, anchor ? bestGreen : bestPink);
			V2 anchorPos = r.perspective->image2field({(double)pos[2*anchorIndex], (double)pos[2*anchorIndex+1]}, height);
			double anchorAngle = atan2(anchorPos.y - bestPos.y, anchorPos.x - bestPos.x);
			double bestScore = INFINITY;
			int bestPermutation = 0;
			for(int i = 0; i < 4; i++) {
				if((bool)(pattern & (1 << (3-i))) != (bool)anchor)
					continue;

				double score = anchor ? *bestGreen : *bestPink;
				double patternAngle = anchorAngle - patternAngles[i];
				for(int j = 1; j < 4; j++) {
					auto* color = (pattern & (1 << (3-(i+j)%4))) ? green : pink;
					double angle = patternAngle + patternAngles[(i+j)%4];
					V2 targetPos = { bestPos.x + cos(angle)*r.sideBlobDistance, bestPos.y + sin(angle)*r.sideBlobDistance };

					double nextDistSq = INFINITY;
					double nextScore = 0;
					for(int k = 0; k < searchAreaSize; k++) {
						V2 testPos = r.perspective->image2field({(double)pos[2*k], (double)pos[2*k+1]}, height);
						V2 diff = {testPos.x - targetPos.x, testPos.y - targetPos.y};
						double distSq = diff.x * diff.x + diff.y * diff.y;
						if(distSq < nextDistSq) {
							nextScore = color[k];
							nextDistSq = distSq;
						}
					}
					score += nextScore;
				}

				if(score < bestScore) {
					bestScore = score;
					bestPermutation = i;
				}
			}
			greenArray->unmap();
			pinkArray->unmap();
			posArray->unmap();

			SSL_DetectionRobot* bot = object.id < 16 ? detection->add_robots_yellow() : detection->add_robots_blue();
			bot->set_confidence(1.0f);
			bot->set_robot_id(object.id % 16);
			bot->set_x(bestPos.x);
			bot->set_y(bestPos.y);
			bot->set_orientation(anchorAngle - patternAngles[bestPermutation]);
			bot->set_pixel_x(rawX * 2);
			bot->set_pixel_y(rawY * 2);
			bot->set_height(height);
		}

		RLEVector debugArea = r.perspective->getRing({ (double)rawX, (double)rawY }, height, 15.0, 25.0);
		for(const Run& run : debugArea.getRuns()) {
			//TODO only RGGB
			uint8_t* row0 = data + 4*r.perspective->getWidth()*run.y + 2*run.x;
			uint8_t* row1 = row0 + 2*r.perspective->getWidth();
			for(int i = 0; i < run.length; i++) {
				row0[2*i + 0] = 255;
				row0[2*i + 1] = 255;
				row1[2*i + 0] = 255;
				row1[2*i + 1] = 255;
			}
		}

		if(std::none_of(tracked.begin(), tracked.end(), [&] (int i) { return i == object.id; }))
			tracked.push_back(object.id);
	}

	filtered.insert(filtered.end(), tracked.begin(), tracked.end());
}

float l2norm(float dx, float dy) {
	return sqrtf(dx*dx + dy*dy);
}

//Bases 0|0: rootEnd -> segmentStart
std::pair<std::list<cv::Vec4f>::const_iterator, float> getBest(
		const float maxRange, const float maxAngle, const float maxPerpendicularDistance,
		const std::list<cv::Vec4f>& lines, const int rootBase, const int segBase) {
	const cv::Vec4f& root = lines.front();
	float rootAngle = atan2f(root[(rootBase+3)%4] - root[rootBase+1], root[(rootBase+2)%4] - root[rootBase]);

	auto best = lines.cend();
	float bestScore = MAXFLOAT;

	auto it = lines.cbegin();
	while(++it != lines.cend()) {
		const cv::Vec4f& segment = *it;
		float range = l2norm(segment[segBase] - root[(rootBase+2)%4], segment[segBase+1] - root[(rootBase+3)%4]);
		if(range > maxRange)
			continue;

		float segmentAngle = atan2f(segment[(segBase+3)%4] - segment[segBase+1], segment[(segBase+2)%4] - segment[segBase]);
		float angleDiff = atan2f(sinf(segmentAngle-rootAngle), cosf(segmentAngle-rootAngle));
		if(abs(angleDiff) > maxAngle)
			continue;

		//TODO max perpendicular distance increasing with range?
		float perpendicualarDistance = cosf(rootAngle) * (root[(rootBase+3)%4] - segment[segBase+1]) - sinf(rootAngle) * (root[(rootBase+2)%4] - segment[segBase]);
		if(abs(perpendicualarDistance) > maxPerpendicularDistance)
			continue;

		if(perpendicualarDistance > bestScore)
			continue;

		//std::cout << rootAngle << " " << segmentAngle << std::endl;
		bestScore = perpendicualarDistance;
		best = it;
	}

	return {best, bestScore};
}

Eigen::Vector2f undistort(const Eigen::Vector2f k, const int width, const Eigen::Vector2f p) {
	Eigen::Vector2f n = p/width;
	n(0) -= 0.5f;
	n(1) -= 0.5f;

	float r2 = n(0)*n(0) + n(1)*n(1);
	float factor = 1 + k(0)*r2 + k(1)*r2*r2;

	n *= factor;
	n(0) += 0.5f;
	n(1) += 0.5f;
	return n*width;
}

int main() {
	Resources r(YAML::LoadFile("config.yml"));

	uint32_t frameNumber = 0;
	bool gtLoaded = false;

	while(true) {
		std::shared_ptr<Image> img = r.camera->readImage();

		r.perspective->geometryCheck();
		r.mask->geometryCheck();

		double startTime = getTime();
		//double timestamp = img->getTimestamp() == 0 ? startTime : img->getTimestamp();
		double timestamp = startTime;

		if(r.perspective->getGeometryVersion()) {
			SSL_WrapperPacket wrapper;
			SSL_DetectionFrame* detection = wrapper.mutable_detection();
			detection->set_frame_number(frameNumber++);
			detection->set_t_capture(startTime);
			if(img->getTimestamp() != 0)
				detection->set_t_capture_camera(img->getTimestamp());
			detection->set_camera_id(r.camId);

			cl::Buffer clBuffer = r.openCl->toBuffer(false, img); //TODO aligned array with OpenCL buffer already provided

			if(!gtLoaded) {
				GroundTruth groundTruth("test-data/rc2022/bots-balls-many-1/gt.yml");
				for(const I2& orange : groundTruth.getOrange()) {
					V2 ballPos = r.perspective->image2field({orange.x/2.0, orange.y/2.0}, r.ballRadius);
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
					addBot(r.perspective, groundTruth, r.gcSocket->getYellowBotHeight(), blob, detection->add_robots_yellow());

				for(const I2& blob : groundTruth.getBlue())
					addBot(r.perspective, groundTruth, r.gcSocket->getBlueBotHeight(), blob, detection->add_robots_blue());

				gtLoaded = true;
			} else {
				std::vector<int> filtered;
				trackObjects(r, timestamp, clBuffer, r.socket->getTrackedObjects()[r.camId], detection, filtered, img->getData());
				for(auto& tracked : r.socket->getTrackedObjects()) {
					trackObjects(r, startTime, clBuffer, tracked.second, detection, filtered, img->getData());
				}

				//TODO thresholding (detection/search)

				/*auto posArray = mask.scanArea(*arrayPool);
				int maskSize = mask.getRuns().size();
				auto resultArray = arrayPool->acquire<float>(maskSize);
				//yellow peak SNR 1.8 PSR 1.15
				//openCl->run(botkernel, cl::EnqueueArgs(cl::NDRange(result.size())), clBuffer, clPos, clResult, perspective->getClPerspective(), (float)defaultBotHeight, 25.0f, (RGB) {255, 255, 0}, 45.0f, (RGB) {0, 0, 0}).wait();
				//yellow optimized color peak SNR 5.8 openCl->run(botkernel, cl::EnqueueArgs(cl::NDRange(result.size())), clBuffer, clPos, clResult, perspective->getClPerspective(), (float)defaultBotHeight, 25.0f, (RGB) {109, 150, 120}, 45.0f, (RGB) {42, 57, 73}).wait();
				//blue openCl->run(botkernel, cl::EnqueueArgs(cl::NDRange(result.size())), clBuffer, clPos, clResult, perspective->getClPerspective(), (float)defaultBotHeight, 25.0f, (RGB) {0, 128, 255}).wait();
				//ball
				openCl->run(ballkernel, cl::EnqueueArgs(cl::NDRange(maskSize)), clBuffer, posArray->getBuffer(), resultArray->getBuffer(), perspective->getClPerspective(), 21.5f, 21.5f, (RGB) {255, 128, 0}).wait();
				///openCl->run(botkernel, cl::EnqueueArgs(cl::NDRange(maskSize)), clBuffer, posArray->getBuffer(), resultArray->getBuffer(), perspective->getClPerspective(), 21.5f, 21.5f, (RGB) {255, 128, 0}, 21.5f, (RGB){0, 0, 0}).wait();

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

				/*for(const Run& run : r.mask->getRuns().getRuns()) {
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
			}

			detection->set_t_sent(getTime());
			r.socket->send(wrapper);
		} else if(r.socket->getGeometryVersion()) {
			const SSL_GeometryFieldSize& field = r.socket->getGeometry().field();
			int xSize = 1;
			int ySize = 1;
			for(int i = r.cameraAmount; i > 1; i /= 2) {
				if(field.field_length()/xSize >= field.field_width()/ySize)
					xSize *= 2;
				else
					ySize *= 2;
			}

			int xPos = 0;
			int yPos = 0;
			for(int i = r.camId % r.cameraAmount; i > 0; i--) {
				yPos++;
				if(yPos == ySize) {
					yPos = 0;
					xPos++;
				}
			}

			int extentLarge = field.field_length()/xSize + (xPos == 0 ? field.boundary_width() : 0) + (xPos == xSize-1 ? field.boundary_width() : 0);
			int extentSmall = field.field_width()/ySize + (yPos == 0 ? field.boundary_width() : 0) + (yPos == ySize-1 ? field.boundary_width() : 0);
			if(extentLarge < extentSmall)
				std::swap(extentSmall, extentLarge);

			int cameraLarge = img->getWidth();
			int cameraSmall = img->getHeight();
			if(cameraLarge < cameraSmall)
				std::swap(cameraSmall, cameraLarge);

			float largeRatio = cameraLarge/(float)extentLarge;
			float smallRatio = cameraSmall/(float)extentSmall;
			if(largeRatio < smallRatio)
				std::swap(smallRatio, largeRatio);

			int halfLineWidth = std::ceil(largeRatio*field.line_thickness()/2.f);
			std::cout << "Line width: " << halfLineWidth << std::endl;
		} else {
			//TODO only RGGB
			cv::Mat cvImg(img->getHeight()*img->pixelHeight(), img->getWidth()*img->pixelWidth(), CV_8UC1, img->getData());
			std::shared_ptr<Image> gray = BufferImage::create(PixelFormat::U8, img->getWidth()*2, img->getHeight()*2);
			cv::Mat cvBgr;
			cv::cvtColor(cvImg, cvBgr, cv::COLOR_BayerBG2BGR);
			cv::Mat cvGray(gray->getHeight(), gray->getWidth(), CV_8UC1, gray->getData());
			cv::cvtColor(cvImg, cvGray, cv::COLOR_BayerBG2GRAY);

			int halfLineWidth = 4; //3, 7;
			const int width = gray->getWidth();
			std::shared_ptr<Image> thresholded = BufferImage::create(PixelFormat::U8, width, gray->getHeight());
			cv::Mat cvThresholded(gray->getHeight(), width, CV_8UC1, thresholded->getData());

			/*cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {halfLineWidth,halfLineWidth});
			//cv::morphologyEx(cvGray, cvThresholded, cv::MORPH_TOPHAT, kernel);*/

			const uint8_t* data = gray->getData();
			int diff = 5; //TODO config
			for(int y = halfLineWidth; y < gray->getHeight() - halfLineWidth; y++) {
				for(int x = halfLineWidth; x < width - halfLineWidth; x++) {
					int value = data[x+y*width];
					thresholded->getData()[x+y*width] = (
																(value - data[x - halfLineWidth + y * width] > diff && value - data[x + halfLineWidth + y * width] > diff) ||
																(value - data[x+ (y - halfLineWidth) * width] > diff && value - data[x + (y + halfLineWidth) * width] > diff)
					) ? 255 : 0;
				}
			}

			cv::Ptr<cv::LineSegmentDetector> detector = cv::createLineSegmentDetector();
			cv::Mat4f linesMat;
			detector->detect(cvThresholded, linesMat);
			//std::cout << linesMat << std::endl;
			/*for(int i = 0; i < width*gray->getHeight(); i++) {
				img->getData()[i] = 0;
			}*/
			/*//Line thinning (Morphological) - Hough transform - peak picking
			//Pixel resolution, angular resolution, accumulator threshold, min line length, max line gap
			//cv::HoughLinesP(gray, linesMat, 10.0, 0.31415, 100, 100, 5);
			cv::HoughLinesP(gray, linesMat, 20.0, 0.31415, 100, 100, 5);*/

			std::list<cv::Vec4f> lines;
			for(int i = 0; i < linesMat.rows; i++)
				lines.push_back(linesMat(0, i));

			std::cout << "Line segments: " << lines.size() << std::endl;

			std::vector<std::vector<cv::Vec4f>> compoundLines;
			auto it = lines.begin();
			while(it != lines.end()) {
				std::vector<cv::Vec4f> compound;
				compound.push_back(*it);

				cv::Vec4f root;
				do {
					root = compound.front();

					//TODO certify functionality
					//TODO compare values
					auto bestEnd = getBest(20.0, 0.1, 4.0, lines, 2, 2); // segmentEnd -> rootStart
					auto bestStart = getBest(20.0, 0.1, 4.0, lines, 2, 0); // segmentStart -> rootStart

					if(bestEnd.second < bestStart.second) {
						compound.insert(compound.cbegin(), *bestEnd.first);
						lines.erase(bestEnd.first);
					} else if(bestStart.second < MAXFLOAT) {
						const cv::Vec4f& bestResult = *bestStart.first;
						compound.insert(compound.cbegin(), cv::Vec4f(bestResult[2], bestResult[3], bestResult[0], bestResult[1]));
						lines.erase(bestStart.first);
					}
				} while(root != compound.front());

				do {
					root = compound.back();

					auto bestStart = getBest(20.0, 0.1, 4.0, lines, 0, 0); //rootEnd -> segmentStart
					auto bestEnd = getBest(20.0, 0.1, 4.0, lines, 0, 2); //rootEnd -> segmentEnd

					if(bestStart.second < bestEnd.second) {
						compound.push_back(*bestStart.first);
						lines.erase(bestStart.first);
					} else if(bestEnd.second < MAXFLOAT) {
						const cv::Vec4f& bestResult = *bestEnd.first;
						compound.emplace_back(bestResult[2], bestResult[3], bestResult[0], bestResult[1]);
						lines.erase(bestEnd.first);
					}
				} while(root != compound.back());

				it = lines.erase(it);
				compoundLines.push_back(compound);
			}

			std::cout << "Compound lines: " << compoundLines.size() << std::endl;
			std::cout << "linesearch: " << (getTime() - startTime) * 1000.0 << " ms" << std::endl;

			cv::imwrite("thresholded.png", cvThresholded);

			detector->drawSegments(cvBgr, linesMat);
			cv::imwrite("lineSegments.png", cvBgr);

			std::vector<std::vector<Eigen::Vector2f>> l;
			for(const auto& compound : compoundLines) {
				for(int i = 1; i < compound.size(); i++) {
					cv::arrowedLine(cvBgr, {(int)compound[i-1][2], (int)compound[i-1][3]}, {(int)compound[i][0], (int)compound[i][1]}, CV_RGB(0, 0, 255));
				}

				std::vector<Eigen::Vector2f> points;
				for(const auto& segment : compound) {
					//std::cout << " " << segment[0] << "," << segment[1] << "->" << segment[2] << "," << segment[3];
					//TODO horrendously wrong scaling
					//points.emplace_back(segment[0]*2/img->getWidth() - 0.5f, segment[1]*2/img->getHeight() - 0.5f);
					//points.emplace_back(segment[2]*2/img->getWidth() - 0.5f, segment[3]*2/img->getHeight() - 0.5f);
					points.emplace_back(segment[0]/img->getWidth() - 0.5f, segment[1]/img->getWidth() - 0.5f);
					points.emplace_back(segment[2]/img->getWidth() - 0.5f, segment[3]/img->getWidth() - 0.5f);
				}
				//std::cout << std::endl;
				l.push_back(points);
			}

			cv::imwrite("lines.png", cvBgr);
			Eigen::Vector2f k = distortion(l);

			for(const auto& compound : compoundLines) {
				if(compound.size() == 1)
					continue;

				cv::arrowedLine(cvBgr, {(int)compound.front()[0], (int)compound.front()[1]}, {(int)compound.back()[2], (int)compound.back()[3]}, CV_RGB(0, 255, 0));
			}
			cv::imwrite("lines2.png", cvBgr);

			for(const auto& compound : compoundLines) {
				if(compound.size() == 1)
					continue;

				Eigen::Vector2f start = undistort(k, img->getWidth(), {compound.front()[0], compound.front()[1]});
				Eigen::Vector2f end = undistort(k, img->getWidth(), {compound.back()[2], compound.back()[3]});
				cv::arrowedLine(cvBgr, {(int)start(0), (int)start(1)}, {(int)end(0), (int)end(1)}, CV_RGB(0, 255, 255));
			}
			cv::imwrite("lines3.png", cvBgr);

			img = thresholded;
		}

		r.rtpStreamer->sendFrame(img);
		std::cout << "main " << (getTime() - startTime) * 1000.0 << " ms" << std::endl;
		std::this_thread::sleep_for(std::chrono::microseconds(33333 - (int64_t)((getTime() - startTime) * 1e6)));
		break;
	}

	r.socket->close();
	r.rtpStreamer = nullptr;
	return 0;
}
