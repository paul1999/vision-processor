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
#include <csignal>
#include <iostream>
#include <iomanip>
#include <opencv2/bgsegm.hpp>
#include <yaml-cpp/yaml.h>

#include "proto/ssl_vision_wrapper.pb.h"
#include "Resources.h"
#include "GroundTruth.h"
#include "calib/GeomModel.h"
#include "pattern.h"
#include "cl_kernels.h"
#include <opencv2/video/background_segm.hpp>
#include "KDTree.h"

#define DRAW_DEBUG_BLOBS true
#define DEBUG_PRINT false

struct __attribute__ ((packed)) CLMatch {
	float x, y;
	RGB color;
	RGB center;
	float circ;
	float score;

	auto operator<=>(const CLMatch&) const = default;
};

struct Match {
	Eigen::Vector2f pos;
	Eigen::Vector3i color;
	Eigen::Vector3i center;
	float circ;
	float score;

	auto operator<=>(const Match&) const = default;
};

/*static void kMeans(const std::vector<Eigen::Vector3i>& values, Eigen::Vector3i& c1, Eigen::Vector3i& c2) {
	//TODO different convergence measurement for values < 3
	if(values.size() < 3)
		return;
	Eigen::Vector3i c1backup = c1;
	Eigen::Vector3i c2backup = c2;
	//https://reasonabledeviations.com/2019/10/02/k-means-in-cpp/
	//https://www.analyticsvidhya.com/blog/2021/05/k-mean-getting-the-optimal-number-of-clusters/
	c1 = *std::min_element(values.begin(), values.end(), [&](const Eigen::Vector3i& a, const Eigen::Vector3i& b) { return (a - c1).squaredNorm() < (b - c1).squaredNorm(); });
	c2 = *std::min_element(values.begin(), values.end(), [&](const Eigen::Vector3i& a, const Eigen::Vector3i& b) { return (a - c2).squaredNorm() < (b - c2).squaredNorm(); });
	if(c1 == c2) {
		c1 = c1backup;
		c2 = c2backup;
		return;
	}

	Eigen::Vector3i oldC1 = c2;
	Eigen::Vector3i oldC2 = c1;
	int n1 = 0;
	int n2 = 0;
	while(oldC1 != c1 && oldC2 != c2) {
		Eigen::Vector3i sum1 = {0, 0, 0};
		Eigen::Vector3i sum2 = {0, 0, 0};
		n1 = 0;
		n2 = 0;
		for (const auto& value : values) {
			if((value - c1).squaredNorm() < (value - c2).squaredNorm()) {
				sum1 += value;
				n1++;
			} else {
				sum2 += value;
				n2++;
			}
		}

		if(n1 == 0 || n2 == 0) {
			std::cerr << "   N0 " << n1 << "|" << n2 << "   " << c1backup.transpose() << "|" << c2backup.transpose() << "   " << c1.transpose() << "|" << c2.transpose() << std::endl;
			c1 = c1backup;
			c2 = c2backup;
			return;
		}

		oldC1 = c1;
		oldC2 = c2;
		c1 = sum1 / n1;
		c2 = sum2 / n2;
	}

	// https://en.wikipedia.org/wiki/Silhouette_(clustering)#Simplified_Silhouette_and_Medoid_Silhouette
	float s1 = 0.0;
	float s2 = 0.0;
	for (const auto& value : values) {
		float a = (float)(value - c1).norm();
		float b = (float)(value - c2).norm();
		if(a < b) {
			s1 += (b - a) / b;
		} else {
			s2 += (a - b) / a;
		}
	}

	//TODO circularity of samples: if roughly circular: one cluster?
	//TODO if small sample size: combine multiple frames
	if(std::max(s1/(float)n1, s2/(float)n2) < 1.0f) { //TODO not working	 //TODO hardcoded value
		std::cerr << "   Skipping Update for " << n1 << "|" << n2 << "   " << c1backup.transpose() << "|" << c2backup.transpose() << "   " << c1.transpose() << "|" << c2.transpose() << std::endl;
		c1 = c1backup;
		c2 = c2backup;
	}
}*/

static void kMeans(const Eigen::Vector3i& different, const std::vector<Eigen::Vector3i>& values, Eigen::Vector3i& c1, Eigen::Vector3i& c2) {
	if(values.size() < 2)
		return;

	float inGroupDiff = INFINITY;
	float outGroupDiff = INFINITY;

	for (int i = 0; i < values.size(); i++) {
		const auto& value = values[i];
		outGroupDiff = std::min(outGroupDiff, (float)(value - different).squaredNorm());

		for (int j = i+1; j < values.size(); j++) {
			inGroupDiff = std::min(inGroupDiff, (float)(values[j] - value).squaredNorm());
		}
	}

	if(inGroupDiff > outGroupDiff) {
		std::cerr << "   Ingroup bigger than outgroup" << std::endl;
		return;
	}

	inGroupDiff = sqrtf(inGroupDiff);
	outGroupDiff = sqrtf(outGroupDiff);

	Eigen::Vector3i c1backup = c1;
	Eigen::Vector3i c2backup = c2;

	//https://reasonabledeviations.com/2019/10/02/k-means-in-cpp/
	//https://www.analyticsvidhya.com/blog/2021/05/k-mean-getting-the-optimal-number-of-clusters/
	c1 = *std::min_element(values.begin(), values.end(), [&](const Eigen::Vector3i& a, const Eigen::Vector3i& b) { return (a - c1).squaredNorm() < (b - c1).squaredNorm(); });
	c2 = *std::min_element(values.begin(), values.end(), [&](const Eigen::Vector3i& a, const Eigen::Vector3i& b) { return (a - c2).squaredNorm() < (b - c2).squaredNorm(); });
	if(c1 == c2) {
		c1 = c1backup;
		c2 = c2backup;
		return;
	}

	Eigen::Vector3i oldC1 = c2;
	Eigen::Vector3i oldC2 = c1;
	int n1 = 0;
	int n2 = 0;
	while(oldC1 != c1 && oldC2 != c2) {
		Eigen::Vector3i sum1 = {0, 0, 0};
		Eigen::Vector3i sum2 = {0, 0, 0};
		n1 = 0;
		n2 = 0;
		for (const auto& value : values) {
			if((value - c1).squaredNorm() < (value - c2).squaredNorm()) {
				sum1 += value;
				n1++;
			} else {
				sum2 += value;
				n2++;
			}
		}

		if(n1 == 0 || n2 == 0) {
			//std::cerr << "   N0 " << n1 << "|" << n2 << "   " << c1backup.transpose() << "|" << c2backup.transpose() << "   " << c1.transpose() << "|" << c2.transpose() << std::endl;
			c1 = c1backup;
			c2 = c2backup;
			return;
		}

		oldC1 = c1;
		oldC2 = c2;
		c1 = sum1 / n1;
		c2 = sum2 / n2;
	}

	float mergeRange = (outGroupDiff - inGroupDiff) / 2.0f;
	if((float)(c1 - c2).norm() < mergeRange) {
		//std::cerr << "   Skipping Update for " << c1backup.transpose() << "|" << c2backup.transpose() << "   " << c1.transpose() << "|" << c2.transpose() << std::endl;
		c1 = c1backup;
		c2 = c2backup;
	}

	/*if((c1 - c2).dot(c1backup - c2backup) <= 0) { //TODO did never trigger
		std::cerr << "   Attempted color direction inversion" << std::endl;
		c1 = c1backup;
		c2 = c2backup;
	}*/

	// https://en.wikipedia.org/wiki/Silhouette_(clustering)#Simplified_Silhouette_and_Medoid_Silhouette
	/*float s1 = 0.0;
	float s2 = 0.0;
	for (const auto& value : values) {
		float a = (float)(value - c1).norm();
		float b = (float)(value - c2).norm();
		if(a < b) {
			s1 += (b - a) / b;
		} else {
			s2 += (a - b) / a;
		}
	}

	//TODO circularity of samples: if roughly circular: one cluster?
	//TODO if small sample size: combine multiple frames
	if(std::max(s1/(float)n1, s2/(float)n2) < 1.0f) { //TODO not working	 //TODO hardcoded value
		std::cerr << "   Skipping Update for " << n1 << "|" << n2 << "   " << c1backup.transpose() << "|" << c2backup.transpose() << "   " << c1.transpose() << "|" << c2.transpose() << std::endl;
		c1 = c1backup;
		c2 = c2backup;
	}*/
}


class KDTree {
public:
	KDTree(): size(0), data(nullptr) {}
	KDTree(Match* data): data(data) {}
	KDTree(int dim, Match* data): dim(dim), data(data) {}

	void insert(Match* iData) {
		size++;
		std::unique_ptr<KDTree>& side = iData->pos[dim] < data->pos[dim] ? left : right;
		if(side != nullptr) {
			side->insert(iData);
			return;
		}

		side = std::make_unique<KDTree>((dim+1) % 2, iData);
	}

	void rangeSearch(std::vector<Match*>& values, const Eigen::Vector2f& point, const float radius) {
		if((data->pos - point).norm() <= radius)
			values.push_back(data);

		if(left != nullptr && point[dim] <= data->pos[dim] + radius)
			left->rangeSearch(values, point, radius);
		if(right != nullptr && point[dim] >= data->pos[dim] - radius)
			right->rangeSearch(values, point, radius);
	}

	inline int getSize() const { return size; }

	std::unique_ptr<KDTree> left = nullptr;
	std::unique_ptr<KDTree> right = nullptr;
	Match* data;
private:
	int dim = 0;
	int size = 1;
};

class BlobBall {
public:
	explicit BlobBall(Match& blob): blob(&blob), blobSearchRadius(0.f) {}

	BlobBall(const Resources& r, const TrackingState& tracked, const double currentTimestamp): hasTrackedBall(true) {
		auto timeDelta = (float)(currentTimestamp - tracked.timestamp);
		trackedPosition = r.perspective->model.image2field(r.perspective->model.field2image(
				Eigen::Vector3f(tracked.x, tracked.y, tracked.z) + Eigen::Vector3f(tracked.vx, tracked.vy, tracked.vz) * timeDelta
		), r.gcSocket->maxBotHeight).head<2>();
		trackedConfidence = tracked.confidence;
		blobSearchRadius = (float)r.maxBallVelocity * timeDelta;
	}

	float score() {
		//TODO
		float score = 1.0f;//blob->orangeness;

		if(hasTrackedBall)
			score *= 1 / (1 + ((pos() - trackedPosition) / 30.0f).squaredNorm()); // (30.0f) 3cm offset -> 0.5 score

		return score;
	}

	Eigen::Vector2f pos() const {
		return blob->pos;
	}

	float blobSearchRadius;
	Match* blob = nullptr;

private:
	bool hasTrackedBall = false;
	Eigen::Vector2f trackedPosition;
	float trackedConfidence = 0;
};

class BlobBot {
public:
	BlobBot(): blobSearchRadius(0.0f) {}

	explicit BlobBot(const Resources& r, Match& blob): blobSearchRadius(r.perspective->field.max_robot_radius()) {
		blobs[0] = &blob;
	}

	BlobBot(const Resources& r, const TrackingState& tracked, const double currentTimestamp): hasTrackedBot(true) {
		trackedId = tracked.id;
		auto timeDelta = (float)(currentTimestamp - tracked.timestamp);
		Eigen::Vector2f reprojectedPosition = r.perspective->model.image2field(r.perspective->model.field2image({tracked.x, tracked.y, tracked.z}), r.gcSocket->maxBotHeight).head<2>();
		trackedPosition = Eigen::Vector3f(reprojectedPosition.x(), reprojectedPosition.y(), tracked.w) + Eigen::Vector3f(tracked.vx, tracked.vy, tracked.vz) * timeDelta;
		trackedConfidence = tracked.confidence;
		//Double acceleration due to velocity determination from two frame difference
		timeDelta = std::min(timeDelta, 0.020f); //TODO hardcoded value
		blobSearchRadius = /* 0.5f * */ (float)r.maxBotAcceleration * timeDelta * timeDelta + r.perspective->field.max_robot_radius();
	}

	float score(const Resources& r) const {
		int blobAmount = 0;
		for(auto& blob : blobs)
			if(blob != nullptr)
				blobAmount++;

		if(blobAmount < 2)
			return 0.0f;

		Eigen::Vector3f p = pos();
		Eigen::Rotation2Df rotation(p.z());

		float score = 1.f;
		for(int i = 0; i < 5; i++) {
			Match* const& blob = blobs[i];
			if(blob == nullptr)
				continue;

			Eigen::Vector2f offset = (blob->pos - (p.head<2>() + rotation * patternPos[i])) / 10.0f; // (10.0f) 1cm offset -> 0.5 score
			float blobScore = 1 / (1 + offset.squaredNorm());

			if(hasTrackedBot) {
				Eigen::Vector3i blobColor;
				Eigen::Vector3i oppositeColor;
				if(i == 0) {
					blobColor = trackedId >= 16 ? r.blue : r.yellow;
					oppositeColor = trackedId >= 16 ? r.yellow : r.blue;
				} else {
					blobColor = ((patterns[trackedId % 16] >> (4-i)) & 1) ? r.green : r.pink;
					oppositeColor = ((patterns[trackedId % 16] >> (4-i)) & 1) ? r.pink : r.green;
				}

				//blobScore *= 1 - (float)(blob->color - blobColor).norm() / 443.4050f; // sqrt(3 * 256**2)
				blobScore *= ((blob->color - oppositeColor).squaredNorm() - (blob->color - blobColor).squaredNorm() > 0 ? 1.0f : 0.1f);
			}

			score = std::min(score, blobScore);
		}

		if(hasTrackedBot) {
			float rotationOffset = (p.z() - trackedPosition.z()) / M_PI; //TODO issues with wraparound?
			score *= 1 / (1 + ((p.head<2>() - trackedPosition.head<2>()) / 10.0f).squaredNorm() + rotationOffset*rotationOffset); // (10.0f) 1cm offset -> 0.5 score
			score *= std::max(blobAmount / 5.f, trackedConfidence);
		} /*else {
			score *= 0.5f;
		}*/

		return std::max(0.f, score);
	}

	inline int botId(const Resources& r) const {
		if(hasTrackedBot)
			return trackedId;

		if(blobs[0] == nullptr || blobs[1] == nullptr || blobs[2] == nullptr || blobs[3] == nullptr || blobs[4] == nullptr)
			return -1;

		Eigen::Vector3i green = r.green;
		Eigen::Vector3i pink = r.pink;
		kMeans(blobs[0]->color, {blobs[1]->color, blobs[2]->color, blobs[3]->color, blobs[4]->color}, green, pink);

		/*std::cout
		<< std::min({(blobs[1]->color - blobs[2]->color).norm(), (blobs[1]->color - blobs[3]->color).norm(), (blobs[1]->color - blobs[4]->color).norm(), (blobs[2]->color - blobs[3]->color).norm(), (blobs[2]->color - blobs[4]->color).norm(), (blobs[3]->color - blobs[4]->color).norm()})
		<< " "
		<< std::min({(blobs[0]->color - blobs[1]->color).norm(), (blobs[0]->color - blobs[2]->color).norm(), (blobs[0]->color - blobs[3]->color).norm(), (blobs[0]->color - blobs[4]->color).norm()})
		<< std::endl;*/

		return ((blobs[0]->color - r.blue).squaredNorm() < (blobs[0]->color - r.yellow).squaredNorm() ? 16 : 0) + patternLUT[
				(((blobs[1]->color - green).squaredNorm() < (blobs[1]->color - pink).squaredNorm() ? 1 : 0) << 3) +
				(((blobs[2]->color - green).squaredNorm() < (blobs[2]->color - pink).squaredNorm()? 1 : 0) << 2) +
				(((blobs[3]->color - green).squaredNorm() < (blobs[3]->color - pink).squaredNorm() ? 1 : 0) << 1) +
				((blobs[4]->color - green).squaredNorm() < (blobs[4]->color - pink).squaredNorm() ? 1 : 0)
		];
	}

	inline Eigen::Vector3f pos() const {
		float oSin = 0;
		float oCos = 0;

		for(int a = 0; a < 5; a++) {
			if(blobs[a] == nullptr)
				continue;

			for(int b = a+1; b < 5; b++) {
				if(blobs[b] == nullptr)
					continue;

				Eigen::Vector2f diff = blobs[b]->pos - blobs[a]->pos;
				float angleDelta = atan2f(diff.y(), diff.x()) - patternAnglesb2b[b*5 + a];
				oSin += sinf(angleDelta);
				oCos += cosf(angleDelta);
			}
		}

		//https://www.themathdoctors.org/averaging-angles/
		float orientation = (oSin == 0 && oCos == 0 && hasTrackedBot) ? trackedPosition[2] : atan2f(oSin, oCos);
		Eigen::Rotation2Df rotation(orientation);

		Eigen::Vector2f position(0, 0);
		int blobAmount = 0;
		for(int i = 0; i < 5; i++) {
			if(blobs[i] == nullptr)
				continue;

			blobAmount++;
			position += blobs[i]->pos - rotation * patternPos[i];
		}

		if(blobAmount == 0)
			return {INFINITY, INFINITY, orientation};

		position /= (float)blobAmount;
		return {position.x(), position.y(), orientation};
	}

	float blobSearchRadius{};
	Match* blobs[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};

	bool hasTrackedBot = false;
	Eigen::Vector3f trackedPosition;

private:
	int trackedId = 0;
	float trackedConfidence = 0;
};

static void bgrDrawBlobs(const Resources& r, Image& bgr, const std::list<CLMatch>& matches, const RGB& color) {
	auto bgrMap = bgr.cvReadWrite();

	for(const CLMatch& match : matches) {
		V2 pos = r.perspective->field2image({(float)match.x, (float)match.y, r.gcSocket->maxBotHeight});
		cv::drawMarker(*bgrMap, cv::Point(2*pos.x, 2*pos.y), CV_RGB(color.r, color.g, color.b), cv::MARKER_CROSS, 10);
		//cv::putText(*bgrMap, std::to_string((int)(match.score*100)) + " h" + std::to_string((int)hsv.r) + "s" + std::to_string((int)hsv.g) + "v" + std::to_string((int)hsv.b), cv::Point(2*pos.x, 2*pos.y), cv::FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(color.r, color.g, color.b));
	}
}

static void updateColors(Resources& r, const std::vector<std::pair<float, BlobBot>>& bestBotModels, const std::vector<Match>& ballCandidates) {
	Eigen::Vector3i oldFalseOrange = r.falseOrange;
	Eigen::Vector3i oldOrange = r.orange;
	Eigen::Vector3i oldYellow = r.yellow;
	Eigen::Vector3i oldBlue = r.blue;
	Eigen::Vector3i oldGreen = r.green;
	Eigen::Vector3i oldPink = r.pink;

	std::vector<Eigen::Vector3i> centerBlobs;
	Eigen::Vector3i pink(0, 0, 0);
	int pinkN = 0;
	Eigen::Vector3i green(0, 0, 0);
	int greenN = 0;
	for (const auto& model : bestBotModels) {
		if(model.second.blobs[0] != nullptr)
			centerBlobs.push_back(model.second.blobs[0]->color);

		int botId = model.second.botId(r) % 16;
		for(int i = 1; i < 5; i++) {
			const Match* blob = model.second.blobs[i];
			if(blob == nullptr)
				continue;

			if((patterns[botId] >> (4-i)) & 1) {
				green += blob->color;
				greenN++;
			} else {
				pink += blob->color;
				pinkN++;
			}
		}
	}

	if(pinkN > 0)
		r.pink = pink / pinkN;
	if(greenN > 0)
		r.green = green / greenN;

	kMeans(r.pink, centerBlobs, r.yellow, r.blue);

	std::vector<Eigen::Vector3i> ballBlobs;
	for (const auto& ball : ballCandidates)
		ballBlobs.push_back(ball.color);

	kMeans(r.blue, ballBlobs, r.orange, r.falseOrange);

	// Constant force to original colors
	r.falseOrange = (10*r.falseOrangeReference + 20*r.falseOrange + 70*oldFalseOrange) / 100;
	r.orange = (10*r.orangeReference + 20*r.orange + 70*oldOrange) / 100;
	r.pink = (10*r.pinkReference + 20*r.pink + 70*oldPink) / 100;
	r.green = (10*r.greenReference + 20*r.green + 70*oldGreen) / 100;
	r.yellow = (10*r.yellowReference + 20*r.yellow + 70*oldYellow) / 100;
	r.blue = (10*r.blueReference + 20*r.blue + 70*oldBlue) / 100;
}

static volatile bool noSigterm = true;
void sig_stop(int sig_num) {
	noSigterm = false;
}

int main(int argc, char* argv[]) {
	signal(SIGTERM, sig_stop);
	signal(SIGINT, sig_stop);
	Resources r(YAML::LoadFile(argc > 1 ? argv[1] : "config.yml"));

	cl::Kernel rggb2img = r.openCl->compile(kernel_rggb2img_cl, kernel_rggb2img_cl_end);
	cl::Kernel bgr2img = r.openCl->compile(kernel_bgr2img_cl, kernel_bgr2img_cl_end);
	cl::Kernel perspectiveKernel = r.openCl->compile(kernel_perspective_cl, kernel_perspective_cl_end);
	cl::Kernel colorKernel = r.openCl->compile(kernel_color_cl, kernel_color_cl_end);
	cl::Kernel satHorizontalKernel = r.openCl->compile(kernel_satHorizontal_cl, kernel_satHorizontal_cl_end);
	cl::Kernel satVerticalKernel = r.openCl->compile(kernel_satVertical_cl, kernel_satVertical_cl_end);
	cl::Kernel circleKernel = r.openCl->compile(kernel_satCircle_cl, kernel_satCircle_cl_end);
	cl::Kernel matchKernel = r.openCl->compile(kernel_matches_cl, kernel_matches_cl_end);

	while(r.waitForGeometry && !r.socket->getGeometryVersion()) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
		r.socket->geometryCheck();
	}

	uint32_t frameId = 0;
	while(noSigterm) {
		frameId++;
		std::shared_ptr<Image> img = r.camera->readImage();
		if(img == nullptr)
			break;

		double startTime = r.camera->getTime();
		r.socket->geometryCheck();
		r.perspective->geometryCheck(r.cameraAmount, img->width, img->height, r.gcSocket->maxBotHeight);

		std::shared_ptr<CLImage> clImg = r.openCl->acquire(&PixelFormat::RGBA8, img->width, img->height, img->name);
		//TODO better type switching
		OpenCL::await(img->format == &PixelFormat::RGGB8 ? rggb2img : bgr2img, cl::EnqueueArgs(cl::NDRange(clImg->width, clImg->height)), img->buffer, clImg->image);

		//std::shared_ptr<Image> mask = std::make_shared<Image>(&PixelFormat::U8, img->width, img->height);
		//OpenCL::wait(r.openCl->run(r.bgkernel, cl::EnqueueArgs(cl::NDRange(img->width, img->height)), img->buffer, bg->buffer, mask->buffer, img->format->stride, img->format->rowStride, (uint8_t)16)); //TODO adaptive threshold
		//bgsub->apply(*img->cvRead(), *mask->cvWrite());

		if(r.perspective->geometryVersion) {
			cl::NDRange visibleFieldRange(r.perspective->reprojectedFieldSize[0], r.perspective->reprojectedFieldSize[1]);
			std::shared_ptr<CLImage> flat = r.openCl->acquire(&PixelFormat::RGBA8, r.perspective->reprojectedFieldSize[0], r.perspective->reprojectedFieldSize[1], img->name);
			OpenCL::await(perspectiveKernel, cl::EnqueueArgs(visibleFieldRange), clImg->image, flat->image, r.perspective->getClPerspective(), (float)r.gcSocket->maxBotHeight, r.perspective->fieldScale, r.perspective->visibleFieldExtent[0], r.perspective->visibleFieldExtent[2]);
			//cv::GaussianBlur(flat.read<RGBA>().cv, blurred.write<RGBA>().cv, {5, 5}, 0, 0, cv::BORDER_REPLICATE);
			std::shared_ptr<CLImage> color = r.openCl->acquire(&PixelFormat::F32, r.perspective->reprojectedFieldSize[0], r.perspective->reprojectedFieldSize[1], img->name);
			OpenCL::await(colorKernel, cl::EnqueueArgs(visibleFieldRange), flat->image, color->image, (int)ceil(r.maxBlobRadius/r.perspective->fieldScale)/3);
			std::shared_ptr<CLImage> colorHor = r.openCl->acquire(&PixelFormat::F32, r.perspective->reprojectedFieldSize[0], r.perspective->reprojectedFieldSize[1], img->name);
			OpenCL::await(satHorizontalKernel, cl::EnqueueArgs(cl::NDRange(r.perspective->reprojectedFieldSize[1])), color->image, colorHor->image);
			std::shared_ptr<CLImage> colorSat = r.openCl->acquire(&PixelFormat::F32, r.perspective->reprojectedFieldSize[0], r.perspective->reprojectedFieldSize[1], img->name);
			OpenCL::await(satVerticalKernel, cl::EnqueueArgs(cl::NDRange(r.perspective->reprojectedFieldSize[0])), colorHor->image, colorSat->image);
			std::shared_ptr<CLImage> circ = r.openCl->acquire(&PixelFormat::F32, r.perspective->reprojectedFieldSize[0], r.perspective->reprojectedFieldSize[1], img->name);
			OpenCL::await(circleKernel, cl::EnqueueArgs(visibleFieldRange), colorSat->image, circ->image, (int)ceil(r.maxBlobRadius/r.perspective->fieldScale)); //TODO testing at robocup with MINBLOBRADIUS? COMPARE! , (int)floor(r.minBlobRadius/r.perspective->fieldScale)

			if(r.debugImages) {
				flat->save(".perspective.png");
				color->save(".color.png", 0.0625f, 128.f);
				circ->save(".circle.png");
			}

			CLArray counter(sizeof(cl_int)*3);
			{
				CLMap<int> counterMap = counter.write<int>();
				counterMap[0] = 0;
				counterMap[1] = 0;
				counterMap[2] = 0;
			}
			CLArray matchArray(sizeof(CLMatch) * r.maxBlobs);
			OpenCL::await(matchKernel, cl::EnqueueArgs(visibleFieldRange), flat->image, circ->image, matchArray.buffer, counter.buffer, (float)r.minCircularity, (float)r.minScore, (int)floor(r.minBlobRadius/r.perspective->fieldScale), r.maxBlobs); //TODO borked minScore and radius at robocup?
			//std::cout << "[match filtering] time " << (getTime() - startTime) * 1000.0 << " ms" << std::endl;

			std::vector<Match> matches; //Same lifetime as KDTree required
			{
				CLMap<int> counterMap = counter.read<int>();
				CLMap<CLMatch> matchMap = matchArray.read<CLMatch>();
				//std::cerr << (flat->width*flat->height - (counterMap[2] + counterMap[1] + counterMap[0])) << "circScore " << counterMap[2] << "circPeak " << counterMap[1] << "score " << counterMap[0] << std::endl;
				const int matchAmount = std::min(r.maxBlobs, counterMap[0]);
				matches.reserve(matchAmount);

				for(int i = 0; i < matchAmount; i++) {
					CLMatch& match = matchMap[i];
					matches.push_back({
						.pos = r.perspective->flat2field({match.x, match.y}),
						.color = {match.color.r, match.color.g, match.color.b},
						.center = {match.center.r, match.center.g, match.center.b},
						.circ = match.circ,
						.score = match.score
					});
				}

				if(counterMap[0] > r.maxBlobs)
					std::cerr << "[blob] max blob amount reached: " << counterMap[0] << "/" << r.maxBlobs << std::endl;
			}

			KDTree blobs = matches.empty() ? KDTree() : KDTree(&matches[0]);
			for(int i = 1; i < matches.size(); i++) {
				blobs.insert(&matches[i]);
			}

			std::vector<Match*> botBlobs;
			std::vector<std::pair<float, BlobBot>> bestBotModels;
			for (const auto& camTracked : r.socket->getTrackedObjects()) {
				for (const auto& tracked : camTracked.second) {
					if(tracked.id == -1)
						continue;

					float bestBotScore = 0.0f;
					std::unique_ptr<BlobBot> bestBot = nullptr;

					BlobBot botModel(r, tracked, startTime);
					botBlobs.clear();
					botBlobs.push_back(nullptr);
					blobs.rangeSearch(botBlobs, botModel.trackedPosition.head<2>(), botModel.blobSearchRadius);
					//TODO instead of this use mintrackingradius on each old blob
					for(Match* const& a : botBlobs) {
						botModel.blobs[0] = a;
						for(Match* const& b : botBlobs) {
							if(b != nullptr && a == b)
								continue;

							botModel.blobs[1] = b;
							for(Match* const& c : botBlobs) {
								if(c != nullptr && (a == c || b == c))
									continue;

								botModel.blobs[2] = c;
								for(Match* const& d : botBlobs) {
									if(d != nullptr && (a == d || b == d || c == d))
										continue;

									botModel.blobs[3] = d;
									for(Match* const& e : botBlobs) {
										if (e != nullptr && (a == e || b == e || c == e || d == e))
											continue;

										botModel.blobs[4] = e;
										float botScore = botModel.score(r);
										if (botScore > bestBotScore) {
											bestBot = std::make_unique<BlobBot>(botModel);
											bestBotScore = botScore;
										}
									}
								}
							}
						}
					}

					if(bestBot == nullptr)
						continue;

					bool isBestBot = true;
					Eigen::Vector2f pos = bestBot->pos().head<2>();
					for(const auto& other : bestBotModels) {
						if(other.first >= bestBotScore && (other.second.pos().head<2>() - pos).norm() < r.perspective->field.max_robot_radius() * 1.75f) { //TODO hardcoded values
							isBestBot = false;
							break;
						}
					}
					if(!isBestBot)
						continue;

					for (auto it = bestBotModels.cbegin(); it != bestBotModels.cend(); ) {
						const auto& other = *it;
						if(it->first < bestBotScore && (it->second.pos().head<2>() - pos).norm() < r.perspective->field.max_robot_radius() * 1.75f) { //TODO hardcoded values
							it = bestBotModels.erase(it);
						} else {
							it++;
						}
					}

					bestBotModels.emplace_back(bestBotScore, *bestBot);
				}
			}

			for(int i = 0; i < blobs.getSize(); i++) {
				Match& blob = matches[i];

				float bestBotScore = 0.0f;
				std::unique_ptr<BlobBot> bestBot = nullptr;

				BlobBot botModel(r, blob);
				botBlobs.clear();
				blobs.rangeSearch(botBlobs, blob.pos, botModel.blobSearchRadius);

				for(Match* const& a : botBlobs) {
					botModel.blobs[1] = a;
					for(Match* const& b : botBlobs) {
						if(a == b)
							continue;

						botModel.blobs[2] = b;
						for(Match* const& c : botBlobs) {
							if(a == c || b == c)
								continue;

							botModel.blobs[3] = c;
							for(Match* const& d : botBlobs) {
								if(a == d || b == d || c == d)
									continue;

								botModel.blobs[4] = d;
								float botScore = botModel.score(r);
								if(botScore > bestBotScore) {
									bestBot = std::make_unique<BlobBot>(botModel);
									bestBotScore = botScore;
								}
							}
						}
					}
				}

				if(bestBot == nullptr)
					continue;

				/*int botId = bestBot->botId(r);
				if(bestBotModels.contains(botId) && bestBotScore <= bestBotModels[botId].first)
					continue;*/

				bool isBestBot = true;
				Eigen::Vector2f pos = bestBot->pos().head<2>();
				for(const auto& other : bestBotModels) {
					if(other.first >= bestBotScore && (other.second.pos().head<2>() - pos).norm() < r.perspective->field.max_robot_radius() * 1.75f) { //TODO hardcoded values
						isBestBot = false;
						break;
					}
				}
				if(!isBestBot)
					continue;

				for (auto it = bestBotModels.cbegin(); it != bestBotModels.cend(); ) {
					if(it->first < bestBotScore && (it->second.pos().head<2>() - pos).norm() < r.perspective->field.max_robot_radius() * 1.75f) { //TODO hardcoded values
						it = bestBotModels.erase(it);
					} else {
						it++;
					}
				}

				bestBotModels.emplace_back(bestBotScore, *bestBot);
			}

			float botCirc = 0.0f;
			int botCircN = 0;
			std::vector<Match> ballCandidates;
			for(const auto& ball : matches) {
				bool nextToBot = false;
				for (const auto& entry : bestBotModels) {
					if((ball.pos - entry.second.pos().head<2>()).norm() < r.perspective->field.max_robot_radius() * 0.8f) {
						botCirc += ball.circ;
						botCircN++;
						nextToBot = true;
						break;
					}
				}
				if(nextToBot)
					continue;

				ballCandidates.push_back({
					.pos = r.perspective->model.image2field(r.perspective->model.field2image({ball.pos.x(), ball.pos.y(), (float)r.gcSocket->maxBotHeight}), r.perspective->field.ball_radius()).head<2>(),
					.color = ball.center,
					.circ = ball.circ,
					.score = ball.score
				});
			}
			botCirc /= (float)botCircN;

			//TODO filter bots according to color kmeans?
			updateColors(r, bestBotModels, ballCandidates);
			//std::cout << r.yellow.transpose() << "|" << r.blue.transpose() << "   " << r.green.transpose() << "|" << r.pink.transpose() << std::endl;

			//TODO area around tracked ball with reduced or alternative minCircularity?

			SSL_WrapperPacket wrapper;
			SSL_DetectionFrame* detection = wrapper.mutable_detection();
			detection->set_frame_number(frameId);
			detection->set_t_capture(startTime);
			if(img->timestamp != 0)
				detection->set_t_capture_camera(img->timestamp);
			detection->set_camera_id(r.camId);

			for (const auto& entry : bestBotModels) {
				const BlobBot& botmodel = entry.second;
				int botId = botmodel.botId(r);
				bool yellow = botId < 16;
				const Eigen::Vector3f maxPos = botmodel.pos();
				const Eigen::Vector2f imgPos = r.perspective->model.field2image({maxPos.x(), maxPos.y(), (float)r.gcSocket->maxBotHeight});
				const Eigen::Vector3f pos = r.perspective->model.image2field(imgPos, yellow ? r.gcSocket->yellowBotHeight : r.gcSocket->blueBotHeight);
				SSL_DetectionRobot* bot = yellow ? detection->add_robots_yellow() : detection->add_robots_blue();
				bot->set_confidence(entry.first);
				bot->set_robot_id(botId % 16);
				bot->set_x(pos.x());
				bot->set_y(pos.y());
				bot->set_height(pos.z());
				bot->set_orientation(maxPos.z());
				bot->set_pixel_x(imgPos.x());
				bot->set_pixel_y(imgPos.y());

				/*std::cout << entry.first << " " << botmodel.botId() << " " << std::fixed << std::setprecision(2);
				for(const Match* const& match : botmodel.blobs) {
					std::cout << (int)match->color.r << "," << (int)match->color.g << "," << (int)match->color.b << "|" << match->greenness << " ";
				}
				std::cout << std::endl;*/
			}

			botCirc /= (float)botCircN;
			for (const auto& ballCandidate : ballCandidates) {
				//TODO circ score
				//TODO more than one ball on field line unrealistic
				/*if(ballCandidate.circ < botCirc)
					continue;*/
				if((ballCandidate.color - r.falseOrange).squaredNorm() < (ballCandidate.color - r.orange).squaredNorm())
					continue;

				const Eigen::Vector2f imgPos = r.perspective->model.field2image({ballCandidate.pos.x(), ballCandidate.pos.y(), r.perspective->field.ball_radius()});
				//std::cout << pos.transpose() << " " << orangeness << " " << blob.circ << " " << blob.score << " " << (blob.circ/blob.score) << std::endl;
				SSL_DetectionBall* ball = detection->add_balls();
				ball->set_confidence(ballCandidate.circ);
				//ball->set_area(0);
				ball->set_x(ballCandidate.pos.x());
				ball->set_y(ballCandidate.pos.y());
				//ball->set_z(0.0f);
				ball->set_pixel_x(imgPos.x());
				ball->set_pixel_y(imgPos.y());
			}

			if(r.debugImages) {
				Image bgr = img->toBGR();
				bgr.save(".matches.png");

				/*{
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
				}*/

				bgr.save(".detections.png");
			}

			detection->set_t_sent(r.camera->getTime());
			r.socket->send(wrapper);
			switch(((long)(startTime/15.0) % 4)) {
			std::cout << "[main] time " << (r.camera->getTime() - startTime) * 1000.0 << " ms " << blobs.getSize() << " blobs " << detection->balls().size() << " balls " << (detection->robots_yellow_size() + detection->robots_blue_size()) << " bots" << std::endl;
				case 0:
					r.rtpStreamer->sendFrame(clImg);
					break;
				case 1:
					r.rtpStreamer->sendFrame(flat);
					break;
				case 2:
					r.rtpStreamer->sendFrame(color);
					break;
				case 3:
					r.rtpStreamer->sendFrame(circ);
					break;
			}
		} else if(r.socket->getGeometryVersion()) {
			geometryCalibration(r, *img);
		} else {
			r.rtpStreamer->sendFrame(clImg);
		}
	}

	std::cout << "Stopping vision_processor" << std::endl;
	return 0;
}
