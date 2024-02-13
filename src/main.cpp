#include <iostream>
#include <yaml-cpp/yaml.h>

#include "messages_robocup_ssl_wrapper.pb.h"
#include "Resources.h"
#include "distortion.h"


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

//const int patternLUT[16] = { 9, 15, 4, 7, 0, 3, 10, 14, 13, 11, 5, 6, 1, 2, 12, 8 };

void trackObjects(Resources& r, const double timestamp, const cl::Buffer& clBuffer, const std::vector<TrackingState>& objects, SSL_DetectionFrame* detection, std::vector<int>& filtered) {
	std::vector<int> tracked;
	for(const TrackingState& object : objects) {
		if(std::any_of(filtered.begin(), filtered.end(), [&] (int i) { return i == object.id; })) //TODO better filtering of ball areas
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
				r.openCl->run(r.sidekernel, cl::EnqueueArgs(cl::NDRange(searchAreaSize)), clBuffer, posArray->buffer, resultArray->buffer, r.perspective->getClPerspective(), height, (float)r.ballRadius, (RGB) {255, 128, 0}).wait();
			} else if(object.id < 16) {
				//r.openCl->run(r.botkernel, cl::EnqueueArgs(cl::NDRange(searchAreaSize)), clBuffer, posArray->getBuffer(), resultArray->getBuffer(), r.perspective->getClPerspective(), (float)r.defaultBotHeight, (float)r.centerBlobRadius, (RGB) {255, 255, 0}, (float)(r.sideBlobDistance - r.sideBlobRadius), (RGB) {0, 0, 0}).wait();
				r.openCl->run(r.botkernel, cl::EnqueueArgs(cl::NDRange(searchAreaSize)), clBuffer, posArray->buffer, resultArray->buffer, r.perspective->getClPerspective(), (float)r.gcSocket->yellowBotHeight, (float)r.centerBlobRadius, (RGB) {255, 255, 128}, (float)(r.sideBlobDistance - r.sideBlobRadius), (RGB) {32, 32, 32}).wait();
			} else {
				//r.openCl->run(r.botkernel, cl::EnqueueArgs(cl::NDRange(searchAreaSize)), clBuffer, posArray->getBuffer(), resultArray->getBuffer(), r.perspective->getClPerspective(), (float)r.defaultBotHeight, (float)r.centerBlobRadius, (RGB) {0, 128, 255}, (float)(r.sideBlobDistance - r.sideBlobRadius), (RGB) {0, 0, 0}).wait();
				r.openCl->run(r.botkernel, cl::EnqueueArgs(cl::NDRange(searchAreaSize)), clBuffer, posArray->buffer, resultArray->buffer, r.perspective->getClPerspective(), (float)r.gcSocket->blueBotHeight, (float)r.centerBlobRadius, (RGB) {0, 128, 255}, (float)(r.sideBlobDistance - r.sideBlobRadius), (RGB) {0, 0, 0}).wait();
			}

			CLMap<float> result = resultArray->read<float>();
			int best = std::distance(*result, std::min_element(*result, *result + searchAreaSize));

			CLMap<int> pos = posArray->read<int>();
			rawX = pos[2*best];
			rawY = pos[2*best + 1];
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
			r.openCl->run(r.sidekernel, cl::EnqueueArgs(cl::NDRange(searchAreaSize)), clBuffer, posArray->buffer, greenArray->buffer, r.perspective->getClPerspective(), height, (float)r.sideBlobRadius, (RGB) {64, 128, 48}).wait();
			//r.openCl->run(r.sidekernel, cl::EnqueueArgs(cl::NDRange(searchAreaSize)), clBuffer, posArray->getBuffer(), pinkArray->getBuffer(), r.perspective->getClPerspective(), height, (float)r.sideBlobRadius, (RGB) {255, 0, 255}).wait();
			r.openCl->run(r.sidekernel, cl::EnqueueArgs(cl::NDRange(searchAreaSize)), clBuffer, posArray->buffer, pinkArray->buffer, r.perspective->getClPerspective(), height, (float)r.sideBlobRadius, (RGB) {255, 128, 128}).wait();

			CLMap<float> green = greenArray->read<float>();
			CLMap<float> pink = pinkArray->read<float>();
			CLMap<int> pos = posArray->read<int>();
			auto bestGreen = std::min_element(*green, *green + searchAreaSize);
			auto bestPink = std::min_element(*pink, *pink + searchAreaSize);
			int anchor = ((*bestGreen < *bestPink && pattern != 0b0000) || pattern == 0b1111) ? 1 : 0;
			int anchorIndex = std::distance(anchor ? *green : *pink, anchor ? bestGreen : bestPink);
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
					auto* color = (pattern & (1 << (3-(i+j)%4))) ? *green : *pink;
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

		if(std::none_of(tracked.begin(), tracked.end(), [&] (int i) { return i == object.id; }))
			tracked.push_back(object.id);
	}

	filtered.insert(filtered.end(), tracked.begin(), tracked.end());
}

float dist(const cv::Vec2f& v1, const cv::Vec2f& v2) {
	cv::Vec2f d = v2-v1;
	return sqrtf(d.dot(d));
}

float angleDiff(const cv::Vec4f& v1, const cv::Vec4f& v2) {
	float v1a = atan2f(v1[3] - v1[1], v1[2] - v1[0]);
	float v2a = atan2f(v2[3] - v2[1], v2[2] - v2[0]);
	return abs(atan2f(sinf(v2a-v1a), cosf(v2a-v1a)));
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

void getAreas(Resources& r, RLEVector& field, RLEVector& other, std::vector<RLEVector>& blobs, int type) {
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

std::pair<float, float> getMinAndMedian(Resources& r, Image& img, RLEVector& area, bool bot, RGB& rgb, const std::string& name) {
	int areaSize = area.size();
	auto posArray = area.scanArea(*r.arrayPool);
	auto resultArray = r.arrayPool->acquire<float>(areaSize);
	int error;
	if(bot) {
		//error = r.openCl->run(r.botkernel, cl::EnqueueArgs(cl::NDRange(areaSize)), img.buffer, posArray->buffer, resultArray->buffer, r.perspective->getClPerspective(), (float)r.gcSocket->defaultBotHeight, (float)r.centerBlobRadius, rgb, (float)(r.sideBlobDistance - r.sideBlobRadius), (RGB) {0, 0, 0}).wait();
		//error = r.openCl->run(r.sidekernel, cl::EnqueueArgs(cl::NDRange(areaSize)), img.buffer, posArray->buffer, resultArray->buffer, r.perspective->getClPerspective(), (float)r.gcSocket->defaultBotHeight, (float)r.centerBlobRadius, rgb).wait();
		error = r.openCl->run(r.ringkernel, cl::EnqueueArgs(cl::NDRange(areaSize)), img.buffer, posArray->buffer, resultArray->buffer, r.perspective->getClPerspective(), (float)r.gcSocket->defaultBotHeight, (float)r.centerBlobRadius, 5.0f, rgb).wait();
	} else {
		//error = r.openCl->run(r.ballkernel, cl::EnqueueArgs(cl::NDRange(areaSize)), img.buffer, posArray->buffer, resultArray->buffer, r.perspective->getClPerspective(), (float)r.ballRadius, (float)r.ballRadius, rgb).wait();
		//error = r.openCl->run(r.sidekernel, cl::EnqueueArgs(cl::NDRange(areaSize)), img.buffer, posArray->buffer, resultArray->buffer, r.perspective->getClPerspective(), (float)r.ballRadius, (float)r.ballRadius, rgb).wait();
		error = r.openCl->run(r.ringkernel, cl::EnqueueArgs(cl::NDRange(areaSize)), img.buffer, posArray->buffer, resultArray->buffer, r.perspective->getClPerspective(), (float)r.ballRadius, (float)r.ballRadius, 5.0f, rgb).wait();
	}

	if(error != 0)
		std::cerr << "Kernel " << error << std::endl;

	auto result = resultArray->read<float>();
	auto min = *std::min_element(*result, *result+areaSize);
	/*int minId = std::distance(*result, min);
	RLEVector ring = r.perspective->getRing((V2){(double)posArray->read<int>()[2*minId], (double)posArray->read<int>()[2*minId+1]}, bot ? r.gcSocket->defaultBotHeight : r.ballRadius, (bot ? r.centerBlobRadius : r.ballRadius) - 5.0, (bot ? r.centerBlobRadius : r.ballRadius) + 5.0);
	int rr = 0;
	int g = 0;
	int b = 0;
	auto map = img.read<uint8_t>();
	for(const Run& run : ring.getRuns()) {
		for(int x = run.x; x < run.x+run.length; x++) {
			//TODO RGGB only
			rr += map[2*x + 4*run.y*img.width];
			g += map[2*x + 4*run.y*img.width + 1];
			g += map[2*x + (2*run.y + 1)*2*img.width];
			b += map[2*x + (2*run.y + 1)*2*img.width + 1];
		}
	}
	rr /= ring.size();
	g /= 2*ring.size();
	b /= ring.size();
	std::cout << name << " " << rr << " " << g << " " << b << std::endl;*/

	if(!name.empty()) {
		auto pos = posArray->read<int>();
		Image gray = img.toGrayscale();
		auto cv = gray.cvReadWrite();
		for(int i = 0; i < posArray->size/4; i+=2) {
			int x = pos[i];
			int y = pos[i+1];
			cv->data[2*x + 2*y*gray.width] = std::max(255 - (result[i/2]-min), 0.f);
			cv->data[2*x + 2*y*gray.width+1] = 255;
		}
		cv::imwrite(name, *cv);
	}

	std::nth_element(*result, *result+areaSize/2, *result+areaSize);
	return {min, result[areaSize/2]};
}

void printSSR(Resources& r, Image& img, int type, RGB rgb) {
	RLEVector field = r.mask->getRuns();
	RLEVector other;
	std::vector<RLEVector> blobs;
	getAreas(r, field, other, blobs, type);

	getMinAndMedian(r, img, r.mask->getRuns(), type >= 0, rgb, "mask" + std::to_string(type) + ".png");
	std::pair<float, float> bestField = getMinAndMedian(r, img, field, type >= 0, rgb, "field" + std::to_string(type) + ".png");
	std::pair<float, float> bestOther = getMinAndMedian(r, img, other, type >= 0, rgb, "other" + std::to_string(type) + ".png");

	float bestBlob = MAXFLOAT;
	float worstBlob = 0.0;
	for(auto& blob : blobs) {
		std::pair<float, float> result = getMinAndMedian(r, img, blob, type >= 0, rgb, ""); //"blob" + std::to_string(type) + "-" + std::to_string((unsigned long long) &blob) + ".png"
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

int halfLineWidthEstimation(const Resources& r, const Image& img) {
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

	int cameraLarge = img.width;
	int cameraSmall = img.height;
	if(cameraLarge < cameraSmall)
		std::swap(cameraSmall, cameraLarge);

	float largeRatio = cameraLarge/(float)extentLarge;
	float smallRatio = cameraSmall/(float)extentSmall;
	if(largeRatio < smallRatio)
		std::swap(smallRatio, largeRatio);

	return std::ceil(largeRatio*field.line_thickness()/2.f);
}

int main(int argc, char* argv[]) {
	Resources r(YAML::LoadFile(argc > 1 ? argv[1] : "config.yml"));

	if(!r.groundTruth.empty())
		r.socket->send(GroundTruth(r.groundTruth, r.camId, getTime()).getMessage());

	uint32_t frameId = 0;
	while(true) {
		std::shared_ptr<Image> img = r.camera->readImage();

		r.perspective->geometryCheck();
		//TODO Extent (min/max x/y axisparallel for 3D based search)
		r.mask->geometryCheck();

		double startTime = getTime();
		double timestamp = img->timestamp == 0 ? startTime : img->timestamp;

		if(r.perspective->getGeometryVersion()) {
			SSL_WrapperPacket wrapper;
			SSL_DetectionFrame* detection = wrapper.mutable_detection();
			detection->set_frame_number(frameId++);
			detection->set_t_capture(startTime);
			if(img->timestamp != 0)
				detection->set_t_capture_camera(img->timestamp);
			detection->set_camera_id(r.camId);

			std::cout << "preComp " << (getTime() - startTime) * 1000.0 << " ms" << std::endl;
			Image raw = img->toRGGB();
			std::cout << "toRGGB " << (getTime() - startTime) * 1000.0 << " ms" << std::endl;
			Image diff(raw.format, raw.width, raw.height, raw.timestamp);
			r.openCl->run(r.diffkernel, cl::EnqueueArgs(cl::NDRange((diff.width-1)*raw.format->stride, (diff.height-1)*raw.format->rowStride)), raw.buffer, diff.buffer, raw.format->stride, raw.format->rowStride).wait();
			std::cout << "diff " << (getTime() - startTime) * 1000.0 << " ms" << std::endl;
			cv::imwrite("diff.png", *diff.toBGR().cvRead());

			uint8_t diffMax;
			{
				CLMap<uint8_t> map = diff.read<uint8_t>();
				diffMax = *std::max_element(*map, *map + diff.width*diff.height*diff.format->pixelSize());
			}

			//TODO track object capability in other colorspaces
			/*std::vector<int> filtered;
			trackObjects(r, timestamp, raw.buffer, r.socket->getTrackedObjects()[r.camId], detection, filtered);
			for(auto& tracked : r.socket->getTrackedObjects()) {
				trackObjects(r, startTime, raw.buffer, tracked.second, detection, filtered);
			}*/

			//TODO thresholding (detection/search)

			//TODO hard threshold possible/advisable?
			const uint8_t diffHalf = diffMax/2;
			const uint8_t diffQuarter = diffMax/4;
			printSSR(r, diff, 0, (RGB) {diffMax, diffMax, diffQuarter});
			printSSR(r, diff, 1, (RGB) {diffQuarter, diffHalf, diffMax});
			printSSR(r, diff, -1, (RGB) {diffMax, diffHalf, diffQuarter});

			detection->set_t_sent(getTime());
			r.socket->send(wrapper);
			break;
		} else if(r.socket->getGeometryVersion()) {
		//} else {
			int halfLineWidth = halfLineWidthEstimation(r, *img); // 4, 3, 7
			std::cout << "Line width: " << halfLineWidth << std::endl;
			Image gray = img->toGrayscale();

			std::shared_ptr<Image> thresholded = std::make_shared<Image>(&PixelFormat::U8, gray.width, gray.height);
			{
				const CLMap<uint8_t> data = gray.read<uint8_t>();
				const int width = gray.width;
				int diff = 5; //TODO config
				CLMap<uint8_t> tData = thresholded->write<uint8_t>();
				for (int y = halfLineWidth; y < gray.height - halfLineWidth; y++) {
					for (int x = halfLineWidth; x < width - halfLineWidth; x++) {
						int value = data[x + y * width];
						tData[x + y * width] = (
													   (value - data[x - halfLineWidth + y * width] >
														diff &&
														value - data[x + halfLineWidth + y * width] >
														diff) ||
													   (value - data[x + (y - halfLineWidth) * width] >
														diff &&
														value - data[x + (y + halfLineWidth) * width] >
														diff)
											   ) ? 255 : 0;
					}
				}
			}

			cv::Ptr<cv::LineSegmentDetector> detector = cv::createLineSegmentDetector();
			cv::Mat4f linesMat;
			detector->detect(*thresholded->cvRead(), linesMat);

			std::list<cv::Vec4f> lines;
			for(int i = 0; i < linesMat.rows; i++)
				lines.push_back(linesMat(0, i));

			std::cout << "Line segments: " << lines.size() << std::endl;

			std::vector<std::vector<cv::Vec4f>> compoundLines;
			std::vector<cv::Vec4f> mergedLines;
			while(!lines.empty()) {
				std::vector<cv::Vec4f> compound;
				compound.push_back(lines.front());
				lines.erase(lines.cbegin());

				for(int i = 0; i < compound.size(); i++) {
					const auto& root = compound[i];
					cv::Vec2f a1(root[0], root[1]);
					cv::Vec2f b1(root[2], root[3]);
					cv::Vec4f invRoot(root[2], root[3], root[0], root[1]);

					auto lit = lines.cbegin();
					while(lit != lines.cend()) {
						cv::Vec2f a2((*lit)[0], (*lit)[1]);
						cv::Vec2f b2((*lit)[2], (*lit)[3]);
						if(
								std::min(angleDiff(root, *lit), angleDiff(invRoot, *lit)) <= 0.05 &&
								std::min(std::min(dist(a1, a2), dist(b1, b2)), std::min(dist(a1, b2), dist(b1, a2))) <= 40.0
						) {
							compound.push_back(*lit);
							lit = lines.erase(lit);
						} else {
							lit++;
						}
					}
				}

				std::sort(compound.begin(), compound.end(), [](const cv::Vec4f& v1, const cv::Vec4f& v2) { return dist(cv::Vec2f(v1[0], v1[1]), cv::Vec2f(v1[2], v1[3])) > dist(cv::Vec2f(v2[0], v2[1]), cv::Vec2f(v2[2], v2[3])); });
				compoundLines.push_back(compound);

				cv::Vec2f a(compound.front()[0], compound.front()[1]);
				cv::Vec2f b(compound.front()[2], compound.front()[3]);
				for(int i = 1; i < compound.size(); i++) {
					const auto& v = compound[i];

					cv::Vec2f c(v[0], v[1]);
					cv::Vec2f d(v[2], v[3]);

					cv::Vec2f maxA = dist(a, c) > dist(a, d) ? c : d;
					if(dist(maxA, a) > dist(a, b) && dist(maxA, a) > dist(maxA, b)) {
						a = maxA;
					}
					cv::Vec2f maxB = dist(b, c) > dist(b, d) ? c : d;
					if(dist(maxB, b) > dist(a, b) && dist(maxB, b) > dist(maxB, a)) {
						b = maxB;
					}
				}
				mergedLines.emplace_back(a[0], a[1], b[0], b[1]);
			}

			std::cout << "Compound lines: " << compoundLines.size() << std::endl;

			cv::imwrite("thresholded.png", *thresholded->cvRead());

			Image bgr = img->toBGR();
			CVMap cvBgr = bgr.cvReadWrite();
			detector->drawSegments(*cvBgr, linesMat);
			cv::imwrite("lineSegments.png", *cvBgr);

			std::vector<std::vector<Eigen::Vector2f>> l;
			for(const auto& compound : compoundLines) {
				for(int i = 1; i < compound.size(); i++) {
					cv::line(*cvBgr, {(int)compound[i-1][2], (int)compound[i-1][3]}, {(int)compound[i][0], (int)compound[i][1]}, CV_RGB(0, 0, 255));
				}

				std::vector<Eigen::Vector2f> points;
				for(const auto& segment : compound) {
					//std::cout << " " << segment[0] << "," << segment[1] << "->" << segment[2] << "," << segment[3];
					//TODO rescale to later determined focal length
					//TODO precision issues -> double?
					//points.emplace_back(segment[0]*2/img->getWidth() - 0.5f, segment[1]*2/img->getHeight() - 0.5f);
					//points.emplace_back(segment[2]*2/img->getWidth() - 0.5f, segment[3]*2/img->getHeight() - 0.5f);
					points.emplace_back(segment[0]/img->width - 0.5f, segment[1]/img->width - 0.5f);
					points.emplace_back(segment[2]/img->width - 0.5f, segment[3]/img->width - 0.5f);
				}
				//std::cout << std::endl;
				l.push_back(points);
			}
			std::cout << "Filtered compound lines: " << (l.size()/2) << std::endl;


			cv::imwrite("lines.png", *cvBgr);
			Eigen::Vector2f k = distortion(l);

			for (const auto& item : mergedLines) {
				cv::line(*cvBgr, {(int)item[0], (int)item[1]}, {(int)item[2], (int)item[3]}, CV_RGB(0, 255, 0));
			}
			cv::imwrite("lines2.png", *cvBgr);

			for(const auto& compound : compoundLines) {
				if(compound.size() == 1)
					continue;

				Eigen::Vector2f start = undistort(k, img->width, {compound.front()[0], compound.front()[1]});
				Eigen::Vector2f end = undistort(k, img->width, {compound.back()[2], compound.back()[3]});
				cv::arrowedLine(*cvBgr, {(int)start(0), (int)start(1)}, {(int)end(0), (int)end(1)}, CV_RGB(0, 255, 255));
			}
			cv::imwrite("lines3.png", *cvBgr);

			img = thresholded;
		}

		r.rtpStreamer->sendFrame(img);
		std::cout << "main " << (getTime() - startTime) * 1000.0 << " ms" << std::endl;
		std::this_thread::sleep_for(std::chrono::microseconds(33333 - (int64_t)((getTime() - startTime) * 1e6)));
	}

	return 0;
}
