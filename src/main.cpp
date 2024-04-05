#include <iostream>
#include <opencv2/bgsegm.hpp>
#include <yaml-cpp/yaml.h>

#include "messages_robocup_ssl_wrapper.pb.h"
#include "Resources.h"
#include "GroundTruth.h"
#include <opencv2/video/background_segm.hpp>

#define DRAW_DEBUG_IMAGES false
#define DEBUG_PRINT false

static double getTime() {
	return (double)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count() / 1e6;
}

//1 indicates green, 0 pink, increasing 2d angle starting from bot orientation most significant bit to least significant bit
/*const int patterns[16] = {
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
};*/
const int patternLUT[16] = { 9, 15, 4, 7, 0, 3, 10, 14, 13, 11, 5, 6, 1, 2, 12, 8 };

const double patternAngles[4] = {
		1.0021839078803572,
		2.5729802346752537,
		-2.5729802346752537, //3.7102050725043325
		-1.0021839078803572 //5.281001399299229
};

struct Match {
	int x, y;
	float score;
	RGB color;
	float radius;
	float height;

	auto operator<=>(const Match&) const = default;
};

static void filterMatches(const Resources& r, std::list<Match>& matches, std::list<Match>& matches2, const float radius) {
	std::erase_if(matches, [&](const Match& match) {
		V2 pos = r.perspective->image2field({(double)match.x, (double)match.y}, match.height);

		return std::ranges::any_of(matches2, [&](const Match& match2) {
			if(match2.score >= match.score)
				return false;

			V2 pos2 = r.perspective->image2field({(double)match2.x, (double)match2.y}, match2.height);
			return dist({(float)pos.x, (float)pos.y}, {(float)pos2.x, (float)pos2.y}) < 2*radius;
		});
	});
}

static std::list<Match> getMatchesFromResult(const Resources& r, Image& img, RGB color, const float height, const float radius, const float threshold, CLArray& pos, CLArray& result, const int areaSize) {
	auto map = result.read<float>();
	auto posMap = pos.read<int>();

	std::list<Match> matches;
	for(int i = 0; i < areaSize; i++) {
		if(map[i] < threshold) {
			matches.push_back({posMap[2*i], posMap[2*i+1], map[i], color, radius, height});
		}
	}

	// filter matches
	filterMatches(r, matches, matches, radius);

	if(DRAW_DEBUG_IMAGES) {
		CVMap map = img.cvReadWrite();
		for(const Match& match : matches) {
			cv::drawMarker(*map, cv::Point(2*match.x, 2*match.y), CV_RGB(match.color.r, match.color.g, match.color.b), cv::MARKER_CROSS, 10);
			std::stringstream score;
			score << std::fixed << std::setprecision(2) << match.score;
			cv::putText(*map, score.str(), cv::Point(2*match.x, 2*match.y), cv::FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(match.color.r, match.color.g, match.color.b));
		}
	}

	return matches;
}

static void getResult(const Resources& r, Image& gxy, Image& gx, Image& gy, RGB color, const float height, const float radius, CLArray& pos, CLArray& result, const int areaSize) {
	color.r = color.r * r.contrast;
	color.g = color.g * r.contrast;
	color.b = color.b * r.contrast;

	OpenCL::wait(r.openCl->run(r.ringkernel, cl::EnqueueArgs(cl::NDRange(areaSize)), gxy.buffer, gx.buffer, gy.buffer, pos.buffer, result.buffer, r.perspective->getClPerspective(), height, radius, color));
}

static void getResultUpdateThreshold(Resources& r, Image& gxy, Image& gx, Image& gy, CLArray& pos, CLArray& result, const int fieldSize, const int areaSize, const RGB& color, float height, float radius, float& threshold, const std::string& name) {
	getResult(r, gxy, gx, gy, color, height, radius, pos, result, areaSize);

	//TODO slow update/convergence: color
	const float updateFraction = (float)areaSize / (float)fieldSize;
	auto map = result.read<float>();

	if(DRAW_DEBUG_IMAGES) {
		//TODO outdated threshold
		auto posMap = pos.read<int>();
		Image debug(&PixelFormat::U8, gxy.width, gxy.height);
		{
			auto debugMap = debug.write<uint8_t>();
			for (int i = 0; i < areaSize; i++) {
				debugMap[posMap[2 * i + 1] * gxy.width + posMap[2 * i]] = std::min(map[i] / threshold * 128, 255.0f);
			}
		}
		cv::imwrite(name, *debug.cvRead());
	}

	//TODO hardcoded factor
	int percentile = areaSize/10; //Top 10%
	std::nth_element(*map, *map+percentile, *map+areaSize);
	float newmedian = map[percentile];
	//TODO hardcoded factor
	//TODO search for largest break/valley in histogram below 10% percentile instead
	//TODO determine PSR
	threshold = (1.0f - updateFraction) * threshold + updateFraction * 0.5f * newmedian;
}

static std::list<Match> getMatches(Resources& r, Image& img, Image& gxy, Image& gx, Image& gy, RLEVector& area, RGB color, const float height, const float radius, const float threshold) {
	int areaSize = area.size();
	auto pos = area.scanArea(*r.arrayPool);
	auto result = r.arrayPool->acquire<float>(areaSize);
	getResult(r, gxy, gx, gy, color, height, radius, *pos, *result, areaSize);
	return getMatchesFromResult(r, img, color, height, radius, threshold, *pos, *result, areaSize);
}

static std::list<Match> getMatchesUpdateThreshold(Resources& r, Image& img, Image& gxy, Image& gx, Image& gy, const int fieldSize, RLEVector& area, const RGB& color, float height, float radius, float& threshold, const std::string& name) {
	int areaSize = area.size();
	auto pos = area.scanArea(*r.arrayPool);
	auto result = r.arrayPool->acquire<float>(areaSize);
	getResultUpdateThreshold(r, gxy, gx, gy, *pos, *result, fieldSize, areaSize, color, height, radius, threshold, name);
	return getMatchesFromResult(r, img, color, height, radius, threshold, *pos, *result, areaSize);
}

static void findBots(Resources& r, Image& img, Image& gxy, Image& gx, Image& gy, std::list<Match>& centerBlobs, SSL_DetectionFrame* detection, bool yellow) {
	auto it = centerBlobs.cbegin();
	while(it != centerBlobs.cend()) {
		const Match& match = *it;
		V2 imgPos {(double)match.x, (double)match.y};
		RLEVector sideSearchArea = r.perspective->getRing(imgPos, match.height, std::max(0.0, r.sideBlobDistance - r.minTrackingRadius/2), r.sideBlobDistance + r.minTrackingRadius/2);
		std::list<Match> green = getMatches(r, img, gxy, gx, gy, sideSearchArea, r.green, match.height, r.sideBlobRadius, r.greenMedian);
		std::list<Match> pink = getMatches(r, img, gxy, gx, gy, sideSearchArea, r.pink, match.height, r.sideBlobRadius, r.pinkMedian);

		green.splice(green.end(), pink);
		filterMatches(r, green, green, r.sideBlobRadius);
		if(green.size() < 4) {
			it = centerBlobs.erase(it);
			continue; // False positive
		}

		const V2 fieldPos = r.perspective->image2field({(double)match.x, (double)match.y}, match.height);
		std::map<Match, double> orientations;
		for(const auto& sideblob : green) {
			V2 anchorPos = r.perspective->image2field({(double)sideblob.x, (double)sideblob.y}, sideblob.height);
			orientations[sideblob] = atan2(anchorPos.y - fieldPos.y, anchorPos.x - fieldPos.x);
		}

		int id = 0;
		float orientation = 0.0f;
		float score = -4.0f;
		for(const Match& a : green) {
			for(const Match& b : green) {
				if(a == b)
					continue;

				for(const Match& c : green) {
					if(a == c || b == c)
						continue;

					for(const Match& d : green) {
						if(a == d || b == d || c == d)
							continue;

						//https://www.themathdoctors.org/averaging-angles/
						const float o = atan2(
								sin(orientations[a] - patternAngles[0]) + sin(orientations[b] - patternAngles[1]) + sin(orientations[c] - patternAngles[2]) + sin(orientations[d] - patternAngles[3]),
								cos(orientations[a] - patternAngles[0]) + cos(orientations[b] - patternAngles[1]) + cos(orientations[c] - patternAngles[2]) + cos(orientations[d] - patternAngles[3])
						);
						const float s = cos(orientations[a] - o) + cos(orientations[b] - o) + cos(orientations[c] - o) + cos(orientations[d] - o);
						if(s < score)
							continue;

						score = s;
						orientation = o;
						id = patternLUT[((a.color == r.green) << 4) + ((b.color == r.green) << 3) + ((c.color == r.green) << 2) + (d.color == r.green)];
					}
				}
			}
		}

		SSL_DetectionRobot* bot = yellow ? detection->add_robots_yellow() : detection->add_robots_blue();
		bot->set_confidence(score/4.0f);
		bot->set_robot_id(id);
		bot->set_x(fieldPos.x);
		bot->set_y(fieldPos.y);
		bot->set_orientation(orientation);
		bot->set_pixel_x(imgPos.x * 2);
		bot->set_pixel_y(imgPos.y * 2);
		bot->set_height(match.height);

		it++;
	}
}

static void findBalls(const Resources& r, const std::list<Match>& orange, SSL_DetectionFrame* detection) {
	for(const Match& match : orange) {
		V2 pos = r.perspective->image2field({(double)match.x, (double)match.y}, match.height);
		SSL_DetectionBall* ball = detection->add_balls();
		ball->set_confidence(1.0f);
		//ball->set_area(0);
		ball->set_x(pos.x);
		ball->set_y(pos.y);
		//ball->set_z(0.0f);
		//TODO only RGGB
		ball->set_pixel_x(match.x * 2);
		ball->set_pixel_y(match.y * 2);
	}
}

static void updateContrast(Resources& r, Image& gxy, RLEVector& area) {
	const int fieldSize = r.mask->getRuns().size();
	const int areaSize = area.size();
	const double updateFraction = (double)areaSize / fieldSize;

	const Run& front = area.getRuns().front();
	const Run& back = area.getRuns().back();
	const CLMap<uint8_t> map = gxy.read<uint8_t>();
	//TODO dont use parts outside of field
	const double maxContrast = *std::max_element(*map + (front.x + front.y * gxy.width) * gxy.format->pixelSize(), *map + (back.x + back.length + back.y * gxy.width) * gxy.format->pixelSize()) / 255.0;
	if(DEBUG_PRINT)
		std::cout << "[Scan] contrast: " << maxContrast << std::endl;
	//TODO hardcoded factor
	r.contrast = (1.0 - updateFraction)*r.contrast + updateFraction * 0.75*maxContrast;
}

static void getAreas(Resources& r, RLEVector& field, RLEVector& other, std::vector<RLEVector>& blobs, int type) {
	for(auto& camera : r.socket->getTrackedObjects()) {
		for (const auto& object: camera.second) {
			V2 position = r.perspective->field2image({object.x, object.y, object.z});

			RLEVector area = r.perspective->getRing(position, object.z, 0.0, object.id == -1 ? r.ballRadius : 90.0);
			field.remove(area);
			other.add(area);

			if(type * 16 <= object.id && (type + 1) * 16 > object.id) { //Targeted
				RLEVector blob = r.perspective->getRing(position, object.z, 0.0, object.id == -1 ? r.ballRadius : r.centerBlobRadius);

				other.remove(blob);
				blobs.push_back(blob);
			}
		}
	}
}

std::pair<float, float> getMinAndMedian(Resources& r, Image& gxy, Image& gx, Image& gy, RLEVector& area, bool bot, RGB& color) {
	int areaSize = area.size();
	auto pos = area.scanArea(*r.arrayPool);
	auto result = r.arrayPool->acquire<float>(areaSize);
	getResult(
			r, gxy, gx, gy, color,
			bot ? r.gcSocket->defaultBotHeight : r.ballRadius,
			bot ? r.centerBlobRadius : r.ballRadius,
			*pos, *result, areaSize
	);

	auto map = result->read<float>();
	std::nth_element(*map, *map+areaSize/2, *map+areaSize);
	return {*std::min_element(*map, *map + areaSize), map[areaSize/2]};
}

void printSSR(Resources& r, Image& img, Image& gx, Image& gy, int type, RGB rgb) {
	RLEVector field = r.mask->getRuns();
	RLEVector other;
	std::vector<RLEVector> blobs;
	getAreas(r, field, other, blobs, type);

	std::pair<float, float> bestField = getMinAndMedian(r, img, gx, gy, field, type >= 0, rgb);
	std::pair<float, float> bestOther = getMinAndMedian(r, img, gx, gy, other, type >= 0, rgb);

	float bestBlob = MAXFLOAT;
	float worstBlob = 0.0;
	for(auto& blob : blobs) {
		std::pair<float, float> result = getMinAndMedian(r, img, gx, gy, blob, type >= 0, rgb);
		if(bestBlob > result.first)
			bestBlob = result.first;
		if(worstBlob < result.first)
			worstBlob = result.first;
	}

	if(blobs.empty()) {
		std::cout << "[Med " << type << "] Field:" << bestField.second << " Other:" << bestOther.second << std::endl;
		std::cout << "[Min " << type << "] Field:" << bestField.first << " Other:" << bestOther.first << std::endl;
	} else {
		std::cout << "[Med " << type << "] Field:" << bestField.second << " Other:" << bestOther.second << std::endl;
		std::cout << "[Min " << type << "] Field:" << bestField.first << " Other:" << bestOther.first << " Best:" << bestBlob << " Worst: " << worstBlob << std::endl;
		std::cout << "[SSR Med " << type << "] Field:" << (bestField.second / worstBlob) << " Best:" << (bestField.second / bestBlob) << std::endl;
		std::cout << "[SSR Min " << type << "] Field:" << (bestField.first / worstBlob) << " Other:" << (bestOther.first / worstBlob) << " Best:" << (bestField.first / bestBlob) << std::endl;
	}
}

int main(int argc, char* argv[]) {
	Resources r(YAML::LoadFile(argc > 1 ? argv[1] : "config.yml"));

	if(!r.groundTruth.empty())
		r.socket->send(GroundTruth(r.groundTruth, r.camId, getTime()).getMessage());

	/*cl::Image2D yuvBg;
	{
		double startTime = getTime();
		int error;
		//cl::Image2D bgrBg(cl::Context::getDefault(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, cl::ImageFormat(CL_RGB, CL_UNSIGNED_INT8), bg->width, bg->height, bg->width*3, *bg->read<uint8_t>(), &error);
		cl::Image2D bgrBg(cl::Context::getDefault(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGB, CL_UNSIGNED_INT8), bg->width, bg->height, 0, nullptr, &error);
		if(error != CL_SUCCESS) {
			std::cerr << "bgbgr Image creation error: " << error << std::endl;
			exit(1);
		}
		std::cout << "main " << (getTime() - startTime) * 1000.0 << " ms" << std::endl;
		error = cl::enqueueCopyBufferToImage(bg->buffer, bgrBg, 0, (cl::array<cl::size_type, 2>){0, 0}, (cl::array<cl::size_type, 2>){(cl::size_type)bg->width, (cl::size_type)bg->height});
		if(error != CL_SUCCESS) {
			std::cerr << "bgbgr copy error: " << error << std::endl;
			exit(1);
		}
		std::cout << "main " << (getTime() - startTime) * 1000.0 << " ms" << std::endl;
		yuvBg = cl::Image2D(cl::Context::getDefault(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGB, CL_UNSIGNED_INT8), bg->width, bg->height, 0, nullptr, &error);
		if(error != CL_SUCCESS) {
			std::cerr << "bgyuv Image creation error: " << error << std::endl;
			exit(1);
		}
		std::cout << "main " << (getTime() - startTime) * 1000.0 << " ms" << std::endl;
		OpenCL::wait(r.openCl->run(r.yuvkernel, cl::EnqueueArgs(cl::NDRange(bg->width, bg->height)), bgrBg, yuvBg));
	}*/

	//cv::Ptr<cv::BackgroundSubtractor> bgsub = cv::bgsegm::createBackgroundSubtractorCNT(15, true, 15*30, false); //https://sagi-z.github.io/BackgroundSubtractorCNT/doxygen/html/index.html

	uint32_t frameId = 0;
	while(true) {
		std::shared_ptr<Image> img = r.camera->readImage();
		if(img == nullptr)
			break;

		r.perspective->geometryCheck();
		//TODO Extent (min/max x/y axisparallel for 3D based search)
		r.mask->geometryCheck();

		double startTime = getTime();
		double timestamp = img->timestamp == 0 ? startTime : img->timestamp;

		//std::shared_ptr<Image> mask = std::make_shared<Image>(&PixelFormat::U8, img->width, img->height);
		//OpenCL::wait(r.openCl->run(r.bgkernel, cl::EnqueueArgs(cl::NDRange(img->width, img->height)), img->buffer, bg->buffer, mask->buffer, img->format->stride, img->format->rowStride, (uint8_t)16)); //TODO adaptive threshold
		//bgsub->apply(*img->cvRead(), *mask->cvWrite());

		//SSD von Durchschnitt in Kreis/Quadratmaske
		//Oder über MAX-Gefilterte Kantendiff
		/*int error;
		cl::Image2D bgr(cl::Context::getDefault(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, cl::ImageFormat(CL_RGB, CL_UNSIGNED_INT8), img->width, img->height, img->width*3, *img->read<uint8_t>(), &error);
		if(error != CL_SUCCESS) {
			std::cerr << "bgr Image creation error: " << error << std::endl;
			exit(1);
		}
		cl::Image2D yuv(cl::Context::getDefault(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGB, CL_UNSIGNED_INT8), img->width, img->height, 0, nullptr, &error);
		if(error != CL_SUCCESS) {
			std::cerr << "yuv Image creation error: " << error << std::endl;
			exit(1);
		}
		OpenCL::wait(r.openCl->run(r.yuvkernel, cl::EnqueueArgs(cl::NDRange(img->width, img->height)), bgr, yuv));*/

		/*Image yuv = img->toBGR();
		{
			CVMap map = yuv.cvReadWrite();
			cv::cvtColor(*map, *map, cv::COLOR_BGR2YUV);
		}
		std::shared_ptr<Image> mask = std::make_shared<Image>(&PixelFormat::U8, yuv.width, yuv.height);*/

		//Fringe issues with edges make it really bad
		//OpenCL::wait(r.openCl->run(r.bgkernel, cl::EnqueueArgs(cl::NDRange(yuv.width, yuv.height)), yuv.buffer, bg.buffer, mask->buffer, (uint8_t)16));

		//TODO background removal mask -> RLE encoding for further analysis
		//TODO idea: background subtraction on edge images with background minimization
		//TODO idea: delta images for new targets finding, else just tracking

		//std::shared_ptr<Image> mask = std::make_shared<Image>(img->format, img->width, img->height);
		//int blobsize = 11;
		//OpenCL::wait(r.openCl->run(r.diffkernel, cl::EnqueueArgs(cl::NDRange((img->width - 2*blobsize)*img->format->stride, (img->height - 2*blobsize)*img->format->rowStride)), img->buffer, mask->buffer, img->format->stride*blobsize, img->format->rowStride*blobsize));
		//cv::imwrite(img->name + ".diff.png", *mask->toBGR().cvRead());
		//break;

		if(r.perspective->getGeometryVersion()) {
			SSL_WrapperPacket wrapper;
			SSL_DetectionFrame* detection = wrapper.mutable_detection();
			detection->set_frame_number(frameId++);
			detection->set_t_capture(startTime);
			if(img->timestamp != 0)
				detection->set_t_capture_camera(img->timestamp);
			detection->set_camera_id(r.camId);

			Image bgr = img->toBGR();
			{
				CVMap map = bgr.cvReadWrite();
				cv::Mat copy;
				map->copyTo(copy);
				cv::blur(copy, *map, cv::Size(5, 5));
			}
			/*Image diff(&PixelFormat::U8, bgr.width, bgr.height, img->name);
			OpenCL::wait(r.openCl->run(r.diffkernel, cl::EnqueueArgs(cl::NDRange(diff.width-1, diff.height-1)), bgr.buffer, diff.buffer));
			cv::imwrite(img->name + ".invariant.png", *diff.cvRead());*/

			Image raw = bgr.toRGGB();
			Image gxy(raw.format, raw.width, raw.height, img->name);
			Image gx(raw.format, raw.width, raw.height, img->name);
			Image gy(raw.format, raw.width, raw.height, img->name);
			OpenCL::wait(r.openCl->run(r.diffkernel, cl::EnqueueArgs(cl::NDRange((raw.width - 1) * raw.format->stride, (raw.height - 1) * raw.format->rowStride)), raw.buffer, gxy.buffer, gx.buffer, gy.buffer, raw.format->stride, raw.format->rowStride));

			const int fieldSize = r.mask->getRuns().size();
			RLEVector scanArea;
			if(DRAW_DEBUG_IMAGES) {
				scanArea = r.mask->getRuns().getPart(0, fieldSize);
			} else {
				int parts = 60;
				int fieldStep = (fieldSize / 60);
				int fieldStart = (frameId%parts)*fieldStep;
				scanArea = r.mask->getRuns().getPart(fieldStart, fieldStart + fieldStep);
			}
			updateContrast(r, gxy, scanArea);
			getMatchesUpdateThreshold(r, bgr, gxy, gx, gy, fieldSize, scanArea, r.green, (float) r.gcSocket->defaultBotHeight, (float) r.sideBlobRadius, r.greenMedian, img->name + ".green.png");
			getMatchesUpdateThreshold(r, bgr, gxy, gx, gy, fieldSize, scanArea, r.pink, (float) r.gcSocket->defaultBotHeight, (float) r.sideBlobRadius, r.pinkMedian, img->name + ".pink.png");

			RLEVector ballArea = scanArea;
			RLEVector yellowArea = scanArea;
			RLEVector blueArea = scanArea;
			for(auto& trackedCamera : r.socket->getTrackedObjects()) {
				for(TrackingState& tracked : trackedCamera.second) {
					double timeDelta = timestamp - tracked.timestamp;
					//double timeDelta = 0.033333;
					double height = tracked.z + tracked.vz * timeDelta;
					V2 imgPos = r.perspective->field2image({
						tracked.x + tracked.vx * timeDelta,
						tracked.y + tracked.vy * timeDelta,
						height
					});

					if(imgPos.x < 0 || imgPos.y < 0 || imgPos.x >= r.perspective->getWidth() || imgPos.y >= r.perspective->getHeight()) {
						//TODO use ring part inside for tracking
						std::cout << "[Tracking] Lost out of bounds " << tracked.id << " " << timeDelta << std::endl;
						continue;
					}

					//TODO fix correct search radius (accel/decel)
					RLEVector searchArea = r.perspective->getRing(
							imgPos,
							height,
							0.0,
							std::max(r.minTrackingRadius, tracked.id != -1 ? r.maxBotAcceleration*timeDelta*timeDelta/2.0 : r.maxBallVelocity*timeDelta)
					);
					if(tracked.id == -1)
						ballArea.add(searchArea);
					else if(tracked.id < 16)
						yellowArea.add(searchArea);
					else
						blueArea.add(searchArea);
				}
			}

			std::list<Match> yellowBlobs = getMatchesUpdateThreshold(r, bgr, gxy, gx, gy, fieldSize, yellowArea, r.yellow, r.gcSocket->yellowBotHeight, (float)r.centerBlobRadius, r.yellowMedian, img->name + ".yellow.png");
			std::list<Match> blueBlobs = getMatchesUpdateThreshold(r, bgr, gxy, gx, gy, fieldSize, blueArea, r.blue, r.gcSocket->blueBotHeight, (float)r.centerBlobRadius, r.blueMedian, img->name + ".blue.png");
			std::list<Match> orangeBlobs = getMatchesUpdateThreshold(r, bgr, gxy, gx, gy, fieldSize, ballArea, r.orange, (float) r.ballRadius, (float) r.ballRadius, r.orangeMedian, img->name + ".orange.png");
			filterMatches(r, yellowBlobs, blueBlobs, r.centerBlobRadius);
			filterMatches(r, blueBlobs, yellowBlobs, r.centerBlobRadius);
			findBots(r, bgr, gxy, gx, gy, yellowBlobs, detection, true);
			findBots(r, bgr, gxy, gx, gy, blueBlobs, detection, false);
			//TODO remove all in range independent of score
			filterMatches(r, yellowBlobs, orangeBlobs, (r.botRadius+r.ballRadius)/2);
			filterMatches(r, blueBlobs, orangeBlobs, (r.botRadius+r.ballRadius)/2);
			findBalls(r, orangeBlobs, detection);

			if(DEBUG_PRINT) {
				printSSR(r, gxy, gx, gy, -1, r.orange);
				printSSR(r, gxy, gx, gy, 0, r.yellow);
				printSSR(r, gxy, gx, gy, 1, r.blue);
			}

			if(DRAW_DEBUG_IMAGES) {
				cv::imwrite(img->name + ".diff.png", *gxy.toBGR().cvRead());
				cv::imwrite(img->name + ".updateContrast.png", *bgr.cvRead());
				{
					CVMap map = gx.cvReadWrite();
					for(uchar* ptr = map->data; ptr < map->dataend; ptr++)
						*ptr = *ptr + 128;
				}
				cv::imwrite(img->name + ".gx.png", *gx.toBGR().cvRead());
			}
			if(DRAW_DEBUG_IMAGES) {
				{
					CVMap map = gy.cvReadWrite();
					for(uchar* ptr = map->data; ptr < map->dataend; ptr++)
						*ptr = *ptr + 128;
				}
				cv::imwrite(img->name + ".gy.png", *gy.toBGR().cvRead());
			}

			detection->set_t_sent(getTime());
			r.socket->send(wrapper);
			if(DRAW_DEBUG_IMAGES)
				break;
		} else if(r.socket->getGeometryVersion()) {
			geometryCalibration(r, *img);
			break;
		}

		r.rtpStreamer->sendFrame(img);
		std::cout << "[main] time " << (getTime() - startTime) * 1000.0 << " ms" << std::endl;
		std::this_thread::sleep_for(std::chrono::microseconds(33333 - (int64_t)((getTime() - startTime) * 1e6)));
	}

	return 0;
}
