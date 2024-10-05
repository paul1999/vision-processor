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
#include <yaml-cpp/yaml.h>
#include "Resources.h"
#include "GroundTruth.h"
#include "pattern.h"
#include "cl_kernels.h"
#include <fstream>
#include <opencv2/imgproc.hpp>

enum BlobColor {
	ORANGE,
	YELLOW,
	BLUE,
	GREEN,
	PINK,
	BOT
};

typedef struct Blob {
	Eigen::Vector3f field;
	Eigen::Vector2f flat;
	float radius;
	BlobColor color;
} Blob;

static inline Eigen::Vector2f field2flat(const Resources& r, const Eigen::Vector3f& field) {
	return r.perspective->field2flat(r.perspective->model.image2field(r.perspective->model.field2image(field), (float)r.gcSocket->maxBotHeight).head<2>());
}

static void scoreBlob(const Resources& r, const CLImageMap<float>& circMap, const CLImageMap<float>& scoreMap, const Blob& blob, float& worstBlobScore, int& blobAmount, Eigen::Vector2f& offsetSum, double& errorSum, double& errorSqSum) {
	Eigen::Vector2f maxPos;
	float maxScore = -INFINITY;
	for(int y = std::max(0, (int)floorf(blob.flat.y() - blob.radius)); y < std::min(r.perspective->reprojectedFieldSize[1], (int)ceilf(blob.flat.y() + blob.radius)); y++) {
		float xRadius = sqrtf(blob.radius*blob.radius - ((float)y - blob.flat.y())*((float)y - blob.flat.y()));
		for(int x = std::max(0, (int)floorf(blob.flat.x() - xRadius)); x < std::min(r.perspective->reprojectedFieldSize[0], (int)ceilf(blob.flat.x() + xRadius)); x++) {
			float s = scoreMap(x, y);
			if(s > maxScore) {
				float c = circMap(x, y);
				float circNegX = circMap(std::max(0, x-1), y);
				float circPosX = circMap(std::min(circMap.cv.cols-1, x+1), y);
				float circNegY = circMap(x, std::max(0, y-1));
				float circPosY = circMap(x, std::min(circMap.cv.rows-1, y+1));
				if(c > circNegX && c > circPosX && c > circNegY && c > circPosY) {
					float xdiv = circNegX - 2*c + circPosX;
					float ydiv = circNegY - 2*c + circPosY;

					//maxPos = {x, y};
					maxPos = {
							(float)x + (xdiv != 0 ? 0.5f * (circNegX - circPosX) / xdiv : 0.0f),
							(float)y + (ydiv != 0 ? 0.5f * (circNegY - circPosY) / ydiv : 0.0f)
					};
					maxScore = s;
				}
			}
		}
	}

	if(maxScore == -INFINITY)
		return;

	Eigen::Vector2f offset = r.perspective->flat2field(maxPos) - r.perspective->flat2field(blob.flat);
	float offsetNorm = offset.norm();
	blobAmount += 1;
	offsetSum += offset;
	errorSum += offsetNorm;
	errorSqSum += offsetNorm*offsetNorm;

	worstBlobScore += maxScore;
	//if(maxScore < worstBlobScore)
	//	worstBlobScore = maxScore;
}

static void scoreBot(const Resources& r, const CLImageMap<float>& circMap, const CLImageMap<float>& scoreMap, const BlobColor botColor, const SSL_DetectionRobot& bot, float& worstBlobScore, std::map<BlobColor, int>& blobAmount, std::map<BlobColor, Eigen::Vector2f>& offsetSum, std::map<BlobColor, double>& errorSum, std::map<BlobColor, double>& errorSqSum) {
	int pattern = patterns[bot.robot_id()];

	Eigen::Vector2f botOffset(0, 0);
	for(int i = 0; i < 5; i++) {
		float orientation = bot.orientation() + (float)patternAnglesb2b[5*i];
		float blobDistance = patternPos[i].norm();
		Eigen::Vector3f field(bot.x() + (float)blobDistance * cosf(orientation), bot.y() + (float)blobDistance * sinf(orientation), bot.height());
		BlobColor color = i == 0 ? botColor : ((pattern & (8 >> i)) ? GREEN : PINK);

		Eigen::Vector2f blobOffset(0, 0);
		scoreBlob(r, circMap, scoreMap, {
				.field = field,
				.flat = field2flat(r, field),
				.radius = (float)(i == 0 ? CENTER_BLOB_RADIUS : SIDE_BLOB_RADIUS) / r.perspective->fieldScale,
				.color = color
		}, worstBlobScore, blobAmount[color], blobOffset, errorSum[color], errorSqSum[color]);
		offsetSum[color] += blobOffset;
		botOffset += blobOffset / 5;
	}

	blobAmount[BOT] += 1;
	offsetSum[BOT] += botOffset;
	errorSum[BOT] += botOffset.norm();
	errorSqSum[BOT] += botOffset.squaredNorm();
}


#define SSD false
int main(int argc, char* argv[]) {
	YAML::Node configFile = YAML::LoadFile(argc > 1 ? argv[1] : "config.yml");
	Resources r(configFile);
	std::vector<SSL_DetectionFrame> groundTruth = parseGroundTruth(r.groundTruth);

	cl::Kernel rggb2img = r.openCl->compile(kernel_rggb2img_cl, kernel_rggb2img_cl_end);
	cl::Kernel bgr2img = r.openCl->compile(kernel_bgr2img_cl, kernel_bgr2img_cl_end);
	cl::Kernel perspectiveKernel = r.openCl->compile(kernel_perspective_cl, kernel_perspective_cl_end);
#if SSD
	cl::Kernel ssdKernel = r.openCl->compile(kernel_ssd_cl, kernel_ssd_cl_end);
#else
	cl::Kernel colorKernel = r.openCl->compile(kernel_color_cl, kernel_color_cl_end);
	cl::Kernel satHorizontalKernel = r.openCl->compile(kernel_satHorizontal_cl, kernel_satHorizontal_cl_end);
	cl::Kernel satVerticalKernel = r.openCl->compile(kernel_satVertical_cl, kernel_satVertical_cl_end);
	cl::Kernel circleKernel = r.openCl->compile(kernel_satCircle_cl, kernel_satCircle_cl_end);
	//cl::Kernel circleKernel = r.openCl->compile(kernel_circularize_cl, kernel_circularize_cl_end);
	cl::Kernel scoreKernel = r.openCl->compile(kernel_score_cl, kernel_score_cl_end);
#endif

	while(r.waitForGeometry && !r.socket->getGeometryVersion()) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
		r.socket->geometryCheck();
	}

	int frameId = 0;
	double imageTime = 0.0;
	double processingTime = 0.0;
	double analysisTime = 0.0;

	std::map<BlobColor, int> blobAmount;
	std::map<BlobColor, double> errorSum;
	std::map<BlobColor, double> errorSqSum;
	std::map<BlobColor, Eigen::Vector2f> offsetSum;
	offsetSum[BlobColor::ORANGE] = {0.f, 0.f};
	offsetSum[BlobColor::YELLOW] = {0.f, 0.f};
	offsetSum[BlobColor::BLUE] = {0.f, 0.f};
	offsetSum[BlobColor::GREEN] = {0.f, 0.f};
	offsetSum[BlobColor::PINK] = {0.f, 0.f};
	double blobScoreSum = 0.0;
	double percentileSum = 0.0;

	while(true) {
		double startTime = getRealTime();
		std::shared_ptr<Image> img = r.camera->readImage();
		if(img == nullptr)
			break;

		imageTime += getRealTime() - startTime;
		startTime = getRealTime();

		r.perspective->geometryCheck(img->width, img->height, r.gcSocket->maxBotHeight);

		std::shared_ptr<CLImage> clImg = r.openCl->acquire(&PixelFormat::RGBA8, img->width, img->height, img->name);
		std::shared_ptr<CLImage> flat = r.openCl->acquire(&PixelFormat::RGBA8, r.perspective->reprojectedFieldSize[0], r.perspective->reprojectedFieldSize[1], img->name);
		std::shared_ptr<CLImage> color = r.openCl->acquire(&PixelFormat::F32, r.perspective->reprojectedFieldSize[0], r.perspective->reprojectedFieldSize[1], img->name);
		std::shared_ptr<CLImage> colorHor = r.openCl->acquire(&PixelFormat::F32, r.perspective->reprojectedFieldSize[0], r.perspective->reprojectedFieldSize[1], img->name);
		std::shared_ptr<CLImage> colorSat = r.openCl->acquire(&PixelFormat::F32, r.perspective->reprojectedFieldSize[0], r.perspective->reprojectedFieldSize[1], img->name);
		std::shared_ptr<CLImage> circ = r.openCl->acquire(&PixelFormat::F32, r.perspective->reprojectedFieldSize[0], r.perspective->reprojectedFieldSize[1], img->name);
		//std::shared_ptr<CLImage> score = r.openCl->acquire(&PixelFormat::F32, r.perspective->reprojectedFieldSize[0], r.perspective->reprojectedFieldSize[1], img->name);

		cl::NDRange visibleFieldRange(r.perspective->reprojectedFieldSize[0], r.perspective->reprojectedFieldSize[1]);
		//cv::GaussianBlur(flat.read<RGBA>().cv, blurred.write<RGBA>().cv, {5, 5}, 0, 0, cv::BORDER_REPLICATE);
		OpenCL::await(img->format == &PixelFormat::RGGB8 ? rggb2img : bgr2img, cl::EnqueueArgs(cl::NDRange(clImg->width, clImg->height)), img->buffer, clImg->image);
		OpenCL::await(perspectiveKernel, cl::EnqueueArgs(visibleFieldRange), clImg->image, flat->image, r.perspective->getClPerspective(), (float)r.gcSocket->maxBotHeight, r.perspective->fieldScale, r.perspective->visibleFieldExtent[0], r.perspective->visibleFieldExtent[2]);
#if SSD
		OpenCL::await(ssdKernel, cl::EnqueueArgs(visibleFieldRange), flat->image, circ->image, r.perspective->fieldScale);
#else
		OpenCL::await(colorKernel, cl::EnqueueArgs(visibleFieldRange), flat->image, color->image, 1);//(int)ceil(r.perspective->maxBlobRadius/r.perspective->fieldScale)/3);
		OpenCL::await(satHorizontalKernel, cl::EnqueueArgs(cl::NDRange(r.perspective->reprojectedFieldSize[1])), color->image, colorHor->image);
		OpenCL::await(satVerticalKernel, cl::EnqueueArgs(cl::NDRange(r.perspective->reprojectedFieldSize[0])), colorHor->image, colorSat->image);
		OpenCL::await(circleKernel, cl::EnqueueArgs(visibleFieldRange), colorSat->image, circ->image, (int)ceil(r.perspective->maxBlobRadius/r.perspective->fieldScale));
		//OpenCL::await(circleKernel, cl::EnqueueArgs(visibleFieldRange), color->image, circ->image, (int)floor(r.perspective->minBlobRadius/r.perspective->fieldScale), (int)ceil(r.perspective->maxBlobRadius/r.perspective->fieldScale));
		//OpenCL::await(scoreKernel, cl::EnqueueArgs(visibleFieldRange), flat->image, circ->image, score->image, (float)r.minCircularity, (int)floor(r.minBlobRadius/r.perspective->fieldScale));
#endif

		processingTime += getRealTime() - startTime;
		startTime = getRealTime();

		float blobScore = 0.0;

		CLImageMap<float> circMap = circ->read<float>();
		//CLImageMap<float> scoreMap = score->read<float>();
		//CLImageMap<float> scoreMap = circ->read<float>(); //TODO RESEARCH MEMORY LEAK (NVIDIA only?)

		const SSL_DetectionFrame& detection = getCorrespondingFrame(groundTruth, ++frameId);
		for(const SSL_DetectionBall& ball : detection.balls()) {
			Eigen::Vector3f field(ball.x(), ball.y(), 30.0f); // See ssl-vision/src/app/plugins/plugin_detect_balls.h
			scoreBlob(r, circMap, circMap, {
					.field = field,
					.flat = field2flat(r, field),
					.radius = r.perspective->field.ball_radius() / r.perspective->fieldScale,
					.color = ORANGE
			}, blobScore, blobAmount[ORANGE], offsetSum[ORANGE], errorSum[ORANGE], errorSqSum[ORANGE]);
		}
		for(const SSL_DetectionRobot& bot : detection.robots_yellow())
			scoreBot(r, circMap, circMap, YELLOW, bot, blobScore, blobAmount, offsetSum, errorSum, errorSqSum);
		for(const SSL_DetectionRobot& bot : detection.robots_blue())
			scoreBot(r, circMap, circMap, BLUE, bot, blobScore, blobAmount, offsetSum, errorSum, errorSqSum);

		if(blobScore == INFINITY)
			continue;

		if(r.debugImages && frameId == 1)  {
#if SSD
			circ->save(".ssd.png", 255.0f, 0.f);
#else
			//circ->save(".circ.png", 2.0f, 128.f); // 4 dRGB, 8 YUV, 2 RGB
			//circ->save(".circ.png", 4.0f, 0.f); // 4 dRGB, 8 YUV, 2 RGB
			color->save(".color.png", 2.0f);
#endif
			//flat->save(".flat.png");
			//color->save(".color.png", 0.0625f, 128.f);
			//circ->save(".circ.png", 2.0f);
			//score->save(".score.png", 2.0f);
		}

		//int frameSize = (int)circMap.rowPitch*circ->height;
		//double worstBlobPercentile = (double)std::count_if(*circMap, *circMap + frameSize, [&](const auto& v){ return v < worstBlobScore; }) / (double)frameSize; //TODO does say nothing about clear separation
		//percentileSum += worstBlobPercentile;
		int n = circ->height*((int)circMap.rowPitch - circ->width) + (int)((circ->width*circ->height) * 0.99);
		std::nth_element(*circMap, *circMap + n, *circMap + (circMap.rowPitch*circ->height));
		percentileSum += circMap[n]; // median did not help
		blobScoreSum += blobScore;
		analysisTime += getRealTime() - startTime;
	}

	double totalError = 0.0;
	double totalSqError = 0.0;
	int totalBlobs = 0;
	for(const auto& offset : errorSum) {
		int blobs = blobAmount[offset.first];
		double stddev = sqrt(blobs*errorSqSum[offset.first] - offset.second*offset.second) / blobs;

		if(offset.first != BOT) {
			totalError += offset.second;
			totalBlobs += blobs;
			totalSqError += errorSqSum[offset.first];
		}

		std::cout << "[Blob benchmark] Avg color " << offset.first << " error: " << (offset.second / blobs) << "±" << stddev << " systematic offset: " << (offsetSum[offset.first].transpose() / blobs) << std::endl;
	}
	blobScoreSum /= totalBlobs;
	double totalStddev = sqrt(totalBlobs*totalSqError - totalError*totalError) / totalBlobs;
	std::cout << "[Blob benchmark] Total error: " << (totalError / totalBlobs) << "±" << totalStddev << " worstblob/percentile: " << blobScoreSum / (abs(blobScoreSum) + abs(percentileSum)) << std::endl;
	std::cout << "[Blob benchmark] Avg processing time: " << (processingTime / frameId) << " frame load time: " << (imageTime / frameId) << " analysis time: " << (analysisTime / frameId) << " frames: " << frameId << std::endl;

	std::cout << "[BlobMachine] " << frameId << " "
			  << totalBlobs << " " << totalError << " " << totalSqError << " "
			  << blobScoreSum << " " << percentileSum << " "
	<< blobAmount[ORANGE] << " " << errorSum[ORANGE] << " " << errorSqSum[ORANGE] << " "
	<< blobAmount[BOT] << " " << errorSum[BOT] << " " << errorSqSum[BOT] << " "
	<< totalBlobs*r.perspective->fieldScale << " " << processingTime << std::endl;
}