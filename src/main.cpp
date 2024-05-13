#include <iostream>
#include <iomanip>
#include <opencv2/bgsegm.hpp>
#include <yaml-cpp/yaml.h>

#include "proto/ssl_vision_wrapper.pb.h"
#include "Resources.h"
#include "GroundTruth.h"
#include "calib/GeomModel.h"
#include <opencv2/video/background_segm.hpp>

#define DRAW_DEBUG_BLOBS false
#define DRAW_DEBUG_IMAGES true
#define DEBUG_PRINT false
#define RUNAWAY_PRINT false

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

struct __attribute__ ((packed)) Match {
	float x, y;
	float score;
	float height;
	RGB color;

	auto operator<=>(const Match&) const = default;
};

static void filterMatches(const Resources& r, std::list<Match>& matches, const std::list<Match>& matches2, const float radius) {
	std::erase_if(matches, [&](const Match& match) {
		return std::ranges::any_of(matches2, [&](const Match& match2) {
			if(match2.score >= match.score)
				return false;
			return dist({match.x, match.y}, {match2.x, match2.y}) < 2*radius;
		});
	});
}

static void getResult(const Resources& r, Image& gxy, Image& gx, Image& gy, RGB color, const float height, const float radius, CLArray& pos, CLArray& result, const int areaSize) {
	color.r = color.r * r.contrast;
	color.g = color.g * r.contrast;
	color.b = color.b * r.contrast;

	OpenCL::wait(r.openCl->run(r.ringkernel, cl::EnqueueArgs(cl::NDRange(areaSize)), gxy.buffer, gx.buffer, gy.buffer, pos.buffer, result.buffer, r.perspective->getClPerspective(), height, radius, color));
}

static bool outsideField(const Resources& r, V2 fieldPos) {
	//TODO additional tolerances
	return abs(fieldPos.x) > r.perspective->getFieldLength()/2.0f + r.perspective->getBoundaryWidth() ||
	       abs(fieldPos.y) > r.perspective->getFieldWidth()/2.0f + r.perspective->getBoundaryWidth();
}

static void findBots(Resources& r, std::list<Match>& centerBlobs, const std::list<Match>& greenBlobs, const std::list<Match>& pinkBlobs, SSL_DetectionFrame* detection, bool yellow) {
	double height = yellow ? r.gcSocket->yellowBotHeight : r.gcSocket->blueBotHeight;

	for(const Match& match : centerBlobs) {
		std::list<Match> green;
		for(const Match& blob : greenBlobs) {
			double distance = dist(cv::Vec2f(blob.x, blob.y), cv::Vec2f(match.x, match.y));
			//if(distance >= std::max(0.0, r.sideBlobDistance - r.minTrackingRadius/2) && distance <= r.sideBlobDistance + r.minTrackingRadius/2) {
			if(distance < 90.0f) {
				green.push_back(blob);
				green.back().color = r.green;
			}
		}
		std::list<Match> pink;
		for(const Match& blob : pinkBlobs) {
			double distance = dist(cv::Vec2f(blob.x, blob.y), cv::Vec2f(match.x, match.y));
			//if(distance >= std::max(0.0, r.sideBlobDistance - r.minTrackingRadius/2) && distance <= r.sideBlobDistance + r.minTrackingRadius/2) {
			if(distance < 90.0f) {
				pink.push_back(blob);
				pink.back().color = r.pink;
			}
		}

		green.splice(green.end(), pink);
		filterMatches(r, green, green, r.sideBlobRadius);
		if(green.size() < 4) {
			continue; // False positive
		}

		std::map<Match, double> orientations;
		for(const auto& sideblob : green) {
			orientations[sideblob] = atan2(sideblob.y - match.y, sideblob.x - match.x);
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

						float s = cos(orientations[a] - patternAngles[0] - o) + cos(orientations[b] - patternAngles[1] - o) + cos(orientations[c] - patternAngles[2] - o) + cos(orientations[d] - patternAngles[3] - o);

						//TODO "recenter" the center blob according to side blob positions
						/*s += powf((float)r.sideBlobDistance, -abs(dist(cv::Vec2f(a.x, a.y), cv::Vec2f(match.x, match.y)) - (float)r.sideBlobDistance));
						s += powf((float)r.sideBlobDistance, -abs(dist(cv::Vec2f(b.x, b.y), cv::Vec2f(match.x, match.y)) - (float)r.sideBlobDistance));
						s += powf((float)r.sideBlobDistance, -abs(dist(cv::Vec2f(c.x, c.y), cv::Vec2f(match.x, match.y)) - (float)r.sideBlobDistance));
						s += powf((float)r.sideBlobDistance, -abs(dist(cv::Vec2f(d.x, d.y), cv::Vec2f(match.x, match.y)) - (float)r.sideBlobDistance));*/

						if(s <= score)
							continue;

						score = s;
						orientation = o;
						id = patternLUT[((a.color == r.green) << 3) + ((b.color == r.green) << 2) + ((c.color == r.green) << 1) + (d.color == r.green)];
					}
				}
			}
		}

		V2 imgPos = r.perspective->field2image({match.x, match.y, r.gcSocket->maxBotHeight});
		V2 pos = r.perspective->image2field({imgPos.x, imgPos.y}, yellow ? r.gcSocket->yellowBotHeight : r.gcSocket->blueBotHeight);

		SSL_DetectionRobot* bot = yellow ? detection->add_robots_yellow() : detection->add_robots_blue();
		bot->set_confidence(score/4.0f);
		bot->set_robot_id(id);
		bot->set_x(pos.x);
		bot->set_y(pos.y);
		bot->set_orientation(orientation);
		bot->set_pixel_x(imgPos.x * 2);
		bot->set_pixel_y(imgPos.y * 2);
		bot->set_height(height);
		if(DEBUG_PRINT)
			std::cout << "Bot " << pos.x << "," << pos.y << " Y" << yellow << " " << id << " " << (orientation*180/M_PI) << "°" << std::endl;
	}
}

static void findBalls(const Resources& r, const std::list<Match>& orange, SSL_DetectionFrame* detection) {
	for(const Match& match : orange) {
		V2 imgPos = r.perspective->field2image({match.x, match.y, r.gcSocket->maxBotHeight});
		V2 pos = r.perspective->image2field({imgPos.x, imgPos.y}, r.ballRadius);

		SSL_DetectionBall* ball = detection->add_balls();
		ball->set_confidence(1.0f);
		//ball->set_area(0);
		ball->set_x(pos.x);
		ball->set_y(pos.y);
		//ball->set_z(0.0f);
		//TODO only RGGB
		ball->set_pixel_x(imgPos.x * 2);
		ball->set_pixel_y(imgPos.y * 2);
		if(DEBUG_PRINT)
			std::cout << "Ball " << pos.x << "," << pos.y << std::endl;
	}
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

//https://stackoverflow.com/questions/3018313/algorithm-to-convert-rgb-to-hsv-and-hsv-to-rgb-in-range-0-255-for-both
RGB RgbToHsv(const RGB& rgb) {
	RGB hsv;
	unsigned char rgbMin, rgbMax;

	rgbMin = rgb.r < rgb.g ? (rgb.r < rgb.b ? rgb.r : rgb.b) : (rgb.g < rgb.b ? rgb.g : rgb.b);
	rgbMax = rgb.r > rgb.g ? (rgb.r > rgb.b ? rgb.r : rgb.b) : (rgb.g > rgb.b ? rgb.g : rgb.b);

	hsv.b = rgbMax;
	if (hsv.b == 0)
	{
		hsv.r = 0;
		hsv.g = 0;
		return hsv;
	}

	hsv.g = 255 * long(rgbMax - rgbMin) / hsv.b;
	if (hsv.g == 0)
	{
		hsv.r = 0;
		return hsv;
	}

	if (rgbMax == rgb.r)
		hsv.r = 0 + 43 * (rgb.g - rgb.b) / (rgbMax - rgbMin);
	else if (rgbMax == rgb.g)
		hsv.r = 85 + 43 * (rgb.b - rgb.r) / (rgbMax - rgbMin);
	else
		hsv.r = 171 + 43 * (rgb.r - rgb.g) / (rgbMax - rgbMin);

	return hsv;
}

static void rggbDrawBlobs(const Resources& r, Image& rggb, const std::list<Match>& matches, const RGB& color) {
	auto map = rggb.readWrite<uint8_t>();
	for(const Match& match : matches) {
		V2 pos = r.perspective->field2image({match.x, match.y, r.gcSocket->maxBotHeight});
		for(int x = std::max(0, (int)pos.x-2); x < std::min(rggb.width-1, (int)pos.x+2); x++) {
			for(int y = std::max(0, (int)pos.y-2); y < std::min(rggb.height-1, (int)pos.y+2); y++) {
				map[2*x + 2*y*2*rggb.width] = color.r;
				map[2*x + 1 + 2*y*2*rggb.width] = color.g;
				map[2*x + (2*y + 1)*2*rggb.width] = color.g;
				map[2*x + 1 + (2*y + 1)*2*rggb.width] = color.b;
			}
		}
	}
}

static void bgrDrawBlobs(const Resources& r, Image& bgr, const std::list<Match>& matches, const RGB& color) {
	auto bgrMap = bgr.cvReadWrite();

	for(const Match& match : matches) {
		V2 pos = r.perspective->field2image({match.x, match.y, r.gcSocket->maxBotHeight});
		cv::drawMarker(*bgrMap, cv::Point(2*pos.x, 2*pos.y), CV_RGB(color.r, color.g, color.b), cv::MARKER_CROSS, 10);
		RGB hsv = RgbToHsv(match.color);
		cv::putText(*bgrMap, std::to_string((int)(match.score*100)) + " h" + std::to_string((int)hsv.r) + "s" + std::to_string((int)hsv.g) + "v" + std::to_string((int)hsv.b), cv::Point(2*pos.x, 2*pos.y), cv::FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(color.r, color.g, color.b));
	}
}

typedef struct __attribute__ ((packed)) RGBA {
	cl_uchar r;
	cl_uchar g;
	cl_uchar b;
	cl_uchar a;
} RGBA;

int main(int argc, char* argv[]) {
	Resources r(YAML::LoadFile(argc > 1 ? argv[1] : "config.yml"));
	cl::Kernel buf2img = r.openCl->compileFile("kernel/buf2img.cl", "-D RGGB");
	cl::Kernel sobel = r.openCl->compileFile("kernel/sobel.cl");
	cl::Kernel blur = r.openCl->compileFile("kernel/blur.cl");
	cl::Kernel matchKernel = r.openCl->compileFile("kernel/matches.cl");
	cl::Kernel houghKernel = r.openCl->compileFile("kernel/hough.cl");

	cl::Kernel perspectiveKernel = r.openCl->compileFile("kernel/perspective.cl");
	cl::Kernel colorKernel = r.openCl->compileFile("kernel/color.cl");
	cl::Kernel circleKernel = r.openCl->compileFile("kernel/circularize.cl");

	if(!r.groundTruth.empty())
		r.socket->send(GroundTruth(r.groundTruth, r.camId, getTime()).getMessage());

	//Preallocation: massive performance differences during enqueue kernel
	int rggbWidth, rggbHeight;
	std::string name;
	{
		std::shared_ptr<Image> img = r.camera->readImage();
		Image rggb = img->toRGGB();
		rggbWidth = rggb.width;
		rggbHeight = rggb.height;
		name = img->name;
	}
	CLImage clImg(rggbWidth, rggbHeight, false);

	uint32_t frameId = 0;
	while(true) {
		std::shared_ptr<Image> img = r.camera->readImage();
		if(img == nullptr)
			break;

		double startTime = getTime();
		r.socket->geometryCheck();
		r.perspective->geometryCheck(img->width, img->height);
		//TODO Extent (min/max x/y axisparallel for 3D based search)
		r.mask->geometryCheck();

		OpenCL::wait(r.openCl->run(buf2img, cl::EnqueueArgs(cl::NDRange(clImg.width, clImg.height)), img->toRGGB().buffer, clImg.image));

		//std::shared_ptr<Image> mask = std::make_shared<Image>(&PixelFormat::U8, img->width, img->height);
		//OpenCL::wait(r.openCl->run(r.bgkernel, cl::EnqueueArgs(cl::NDRange(img->width, img->height)), img->buffer, bg->buffer, mask->buffer, img->format->stride, img->format->rowStride, (uint8_t)16)); //TODO adaptive threshold
		//bgsub->apply(*img->cvRead(), *mask->cvWrite());

		//TODO background removal mask -> RLE encoding for further analysis
		//TODO idea: background subtraction on edge images with background minimization
		//TODO idea: delta images for new targets finding, else just tracking

		if(r.perspective->geometryVersion) {
			SSL_WrapperPacket wrapper;
			SSL_DetectionFrame* detection = wrapper.mutable_detection();
			detection->set_frame_number(frameId++);
			detection->set_t_capture(startTime);
			if(img->timestamp != 0)
				detection->set_t_capture_camera(img->timestamp);
			detection->set_camera_id(r.camId);

			OpenCL::wait(r.openCl->run(perspectiveKernel, cl::EnqueueArgs(cl::NDRange(r.mask->fieldSizeX, r.mask->fieldSizeY)), clImg.image, r.mask->flat->image, r.perspective->getClPerspective(), (float)r.gcSocket->maxBotHeight, r.mask->fieldScale, (float)r.mask->fieldExtentX[0], (float)r.mask->fieldExtentY[0]));
			if(DRAW_DEBUG_IMAGES) {
				cv::imwrite("img/" + img->name + ".perspective.png", r.mask->flat->read<RGBA>().cv);
			}
			OpenCL::wait(r.openCl->run(colorKernel, cl::EnqueueArgs(cl::NDRange(r.mask->fieldSizeX, r.mask->fieldSizeY)), r.mask->flat->image, r.mask->color->image));
			if(DRAW_DEBUG_IMAGES) {
				Image grayscale(&PixelFormat::U8, r.mask->fieldSizeX, r.mask->fieldSizeY, img->name);
				{
					CLMap<uint8_t> write = grayscale.write<uint8_t>();
					CLImageMap<float> read = r.mask->color->read<float>();
					for(int y = 0; y < r.mask->fieldSizeY; y++) {
						for(int x = 0; x < r.mask->fieldSizeX; x++) {
							write[x + r.mask->fieldSizeX * y] = cv::saturate_cast<uint8_t>(read[x + read.rowPitch * y] / 16.0f + 128);
						}
					}
				}
				grayscale.save(".color.png");
			}

			OpenCL::wait(r.openCl->run(circleKernel, cl::EnqueueArgs(cl::NDRange(r.mask->fieldSizeX, r.mask->fieldSizeY)), r.mask->color->image, r.mask->circ->image));
			if(DRAW_DEBUG_IMAGES) {
				Image grayscale(&PixelFormat::U8, r.mask->fieldSizeX, r.mask->fieldSizeY, img->name);
				{
					CLMap<uint8_t> write = grayscale.write<uint8_t>();
					CLImageMap<float> read = r.mask->circ->read<float>();
					for(int y = 0; y < r.mask->fieldSizeY; y++) {
						for(int x = 0; x < r.mask->fieldSizeX; x++) {
							write[x + r.mask->fieldSizeX * y] = cv::saturate_cast<uint8_t>(read[x + read.rowPitch * y]);
						}
					}
				}
				grayscale.save(".circle.png");
			}

			CLArray counter(sizeof(int));
			int maxMatches = 10000; //TODO make configurable
			CLArray matchArray(sizeof(Match)*maxMatches);
			OpenCL::wait(r.openCl->run(matchKernel, cl::EnqueueArgs(cl::NDRange(r.mask->fieldSizeX, r.mask->fieldSizeY)), r.mask->flat->image, r.mask->circ->image, matchArray.buffer, counter.buffer, (float)r.minCircularity, r.minSaturation, r.minBrightness, (int)round(r.sideBlobRadius/r.mask->fieldScale), maxMatches));
			//std::cout << "[match filtering] time " << (getTime() - startTime) * 1000.0 << " ms" << std::endl;

			std::list<Match> matches;
			{
				CLMap<Match> matchMap = matchArray.read<Match>();
				int matchAmount = 0;
				while(matchAmount < maxMatches && (matchMap[matchAmount].x != 0 || matchMap[matchAmount].y != 0 || matchMap[matchAmount].score != 0)) {
					Match& match = matchMap[matchAmount];
					match.x = match.x*r.mask->fieldScale + r.mask->fieldExtentX[0];
					match.y = match.y*r.mask->fieldScale + r.mask->fieldExtentY[0];
					matchAmount++;
				}

				if(matchAmount == maxMatches)
					std::cerr << "[blob] max blob amount reached" << std::endl;

				matches = std::list<Match>(*matchMap, *matchMap+matchAmount);
			}

			std::cout << matches.size() << std::endl;
			filterMatches(r, matches, matches, r.ballRadius);

			std::list<Match> orangeBlobs;
			std::list<Match> yellowBlobs;
			std::list<Match> blueBlobs;
			std::list<Match> greenBlobs;
			std::list<Match> pinkBlobs;
			for(const Match& match : matches) {
				RGB hsv = RgbToHsv(match.color);
				int orangeDiff = abs((int8_t)(hsv.r - r.orangeHue));
				int yellowDiff = abs((int8_t)(hsv.r - r.yellowHue));
				int blueDiff = abs((int8_t)(hsv.r - r.blueHue));
				int greenDiff = abs((int8_t)(hsv.r - r.greenHue));
				int pinkDiff = abs((int8_t)(hsv.r - r.pinkHue));
				int minDiff = std::min(std::min(orangeDiff, std::min(yellowDiff, blueDiff)), std::min(greenDiff, pinkDiff));
				if(orangeDiff == minDiff)
					orangeBlobs.push_back(match);
				else if(yellowDiff == minDiff)
					yellowBlobs.push_back(match);
				else if(blueDiff == minDiff)
					blueBlobs.push_back(match);
				else if(greenDiff == minDiff)
					greenBlobs.push_back(match);
				else
					pinkBlobs.push_back(match);
			}
			//std::cout << "[blob ordering] time " << (getTime() - startTime) * 1000.0 << " ms" << std::endl; //TODO 2.5 out of 10 ms!

			findBalls(r, orangeBlobs, detection);
			findBots(r, yellowBlobs, greenBlobs, pinkBlobs, detection, true);
			findBots(r, blueBlobs, greenBlobs, pinkBlobs, detection, false);

			if(DRAW_DEBUG_BLOBS) {
				rggbDrawBlobs(r, *img, orangeBlobs, r.orange);
				rggbDrawBlobs(r, *img, yellowBlobs, r.yellow);
				rggbDrawBlobs(r, *img, blueBlobs, r.blue);
				rggbDrawBlobs(r, *img, greenBlobs, r.green);
				rggbDrawBlobs(r, *img, pinkBlobs, r.pink);
			}

			if(DRAW_DEBUG_IMAGES) {
				Image bgr = img->toBGR();
				bgrDrawBlobs(r, bgr, orangeBlobs, r.orange);
				bgrDrawBlobs(r, bgr, yellowBlobs, r.yellow);
				bgrDrawBlobs(r, bgr, blueBlobs, r.blue);
				bgrDrawBlobs(r, bgr, greenBlobs, r.green);
				bgrDrawBlobs(r, bgr, pinkBlobs, r.pink);
				bgr.save(".matches.png");

				{
					auto bgrMap = bgr.cvReadWrite();
					for(const auto& ball : detection->balls()) {
						cv::drawMarker(*bgrMap, cv::Point(ball.pixel_x(), ball.pixel_y()), CV_RGB(r.orange.r, r.orange.g, r.orange.b), cv::MARKER_DIAMOND, 10);
					}

					for(const auto& bot : detection->robots_yellow()) {
						cv::drawMarker(*bgrMap, cv::Point(bot.pixel_x(), bot.pixel_y()), CV_RGB(r.yellow.r, r.yellow.g, r.yellow.b), cv::MARKER_DIAMOND, 10);
						cv::putText(*bgrMap, std::to_string((int)(bot.orientation()*180/M_PI)) + " " + std::to_string(bot.robot_id()), cv::Point(bot.pixel_x(), bot.pixel_y()), cv::FONT_HERSHEY_COMPLEX, 0.6, CV_RGB(255, 255, 255));
					}

					for(const auto& bot : detection->robots_blue()) {
						cv::drawMarker(*bgrMap, cv::Point(bot.pixel_x(), bot.pixel_y()), CV_RGB(r.blue.r, r.blue.g, r.blue.b), cv::MARKER_DIAMOND, 10);
						cv::putText(*bgrMap, std::to_string((int)(bot.orientation()*180.0f/M_PI)) + " " + std::to_string(bot.robot_id()), cv::Point(bot.pixel_x(), bot.pixel_y()), cv::FONT_HERSHEY_COMPLEX, 0.6, CV_RGB(255, 255, 255));
					}
				}

				bgr.save(".detections.png");
			}

			/*{
				CLImageMap<RGBA> imgMap = clImg.readWrite<RGBA>();
				CLImageMap<RGBA> bgMap = clBg.read<RGBA>();

				for(const SSL_DetectionBall& ball : detection->balls()) {
					RLEVector blob = r.perspective->getRing((V2) {ball.pixel_x()/2, ball.pixel_y()/2}, r.ballRadius, 0.0, r.ballRadius);
					for(const Run& run : blob.getRuns()) {
						for(int x = run.x; x < run.x+run.length; x++) {
							imgMap[x + run.y*imgMap.rowPitch] = {0, 0, 0, 0}; //bgMap[x + run.y*bgMap.rowPitch];
						}
					}
				}
				for(const SSL_DetectionRobot& bot : detection->robots_yellow()) {
					RLEVector blob = r.perspective->getRing((V2) {bot.pixel_x()/2, bot.pixel_y()/2}, r.gcSocket->yellowBotHeight, 0.0, r.perspective->field.max_robot_radius());
					for(const Run& run : blob.getRuns()) {
						for(int x = run.x; x < run.x+run.length; x++) {
							imgMap[x + run.y*imgMap.rowPitch] = {0, 0, 0, 0}; //bgMap[x + run.y*imgMap.rowPitch];
						}
					}
				}
				for(const SSL_DetectionRobot& bot : detection->robots_blue()) {
					RLEVector blob = r.perspective->getRing((V2) {bot.pixel_x()/2, bot.pixel_y()/2}, r.gcSocket->blueBotHeight, 0.0, r.perspective->field.max_robot_radius());
					for(const Run& run : blob.getRuns()) {
						for(int x = run.x; x < run.x+run.length; x++) {
							imgMap[x + run.y*imgMap.rowPitch] = {0, 0, 0, 0}; //bgMap[x + run.y*imgMap.rowPitch];
						}
					}
				}

				for(int y = 0; y < rggbHeight; y++) {
					for(int x = 0; x < imgMap.rowPitch; x++) {
						RGBA& n = imgMap[x + y*imgMap.rowPitch];
						const RGBA& o = bgMap[x + y*imgMap.rowPitch];
						n.r = (uint16_t)n.r*1/10 + (uint16_t)o.r*9/10;
						n.g = (uint16_t)n.g*1/10 + (uint16_t)o.g*9/10;
						n.b = (uint16_t)n.b*1/10 + (uint16_t)o.b*9/10;
						n.a = (uint16_t)n.a*1/10 + (uint16_t)o.a*9/10;
					}
				}
			}
			std::swap(clImg.image, clBg.image);*/
			//cv::imwrite("img/schubert.bg.png", clBg.read<RGBA>().cv);break;

			if(DEBUG_PRINT) {
				/*printSSR(r, gxy, gx, gy, -1, r.orange);
				printSSR(r, gxy, gx, gy, 0, r.yellow);
				printSSR(r, gxy, gx, gy, 1, r.blue);*/
			}

			detection->set_t_sent(getTime());
			r.socket->send(wrapper);
			std::cout << "[main] time " << (getTime() - startTime) * 1000.0 << " ms " << matches.size() << " blobs " << detection->balls().size() << " balls " << (detection->robots_yellow_size() + detection->robots_blue_size()) << " bots" << std::endl;
			if(DRAW_DEBUG_IMAGES)
				break;
		} else if(r.socket->getGeometryVersion()) {
			geometryCalibration(r, *img);
			break;
		}

		r.rtpStreamer->sendFrame(img);
	}

	return 0;
}
