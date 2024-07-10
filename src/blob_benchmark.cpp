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
#include "pattern.h"
#include "cl_kernels.h"

enum BlobColor {
	ORANGE,
	YELLOW,
	BLUE,
	GREEN,
	PINK
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

static void bot2blobs(const Resources& r, const SSL_DetectionRobot& bot, const BlobColor botColor, std::vector<Blob>& blobs) {
	Eigen::Vector3f field(bot.x(), bot.y(), bot.height());
	blobs.push_back({
		.field = field,
		.flat = field2flat(r, field),
		.radius = (float)r.centerBlobRadius / r.perspective->fieldScale,
		.color = botColor
	});

	int pattern = patterns[bot.robot_id()];
	for(int i = 0; i < 4; i++) {
		float orientation = bot.orientation() + (float)patternAngles[i];
		Eigen::Vector3f field(bot.x() + (float)r.sideBlobDistance * cosf(orientation), bot.y() + (float)r.sideBlobDistance * sinf(orientation), bot.height());
		blobs.push_back({
			.field = field,
			.flat = field2flat(r, field),
			.radius = (float)r.sideBlobRadius / r.perspective->fieldScale,
			.color = (pattern & (8 >> i)) ? GREEN : PINK
		});
	}
}

#define DRAW_DEBUG_IMAGES true
#define WARN_INVISIBLE_BLOBS false

int main(int argc, char* argv[]) {
	Resources r(YAML::LoadFile(argc > 1 ? argv[1] : "config.yml"));
	std::vector<SSL_DetectionFrame> groundTruth = parseGroundTruth(r.groundTruth);

	cl::Kernel rggb2img = r.openCl->compile(kernel_rggb2img_cl, kernel_rggb2img_cl_end);
	cl::Kernel bgr2img = r.openCl->compile(kernel_bgr2img_cl, kernel_bgr2img_cl_end);
	cl::Kernel perspectiveKernel = r.openCl->compile(kernel_perspective_cl, kernel_perspective_cl_end);
	cl::Kernel colorKernel = r.openCl->compile(kernel_color_cl, kernel_color_cl_end);
	cl::Kernel circleKernel = r.openCl->compile(kernel_circularize_cl, kernel_circularize_cl_end);
	cl::Kernel scoreKernel = r.openCl->compile(kernel_score_cl, kernel_score_cl_end);

	while(r.waitForGeometry && !r.socket->getGeometryVersion()) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
		r.socket->geometryCheck();
	}

	int frameId = 0;
	long circTotal = 0;
	long circHits = 0;
	double imageTime = 0.0;
	double processingTime = 0.0;
	double analysisTime = 0.0;

	std::map<BlobColor, int> blobAmount;
	std::map<BlobColor, int> criticalOffset;
	std::map<BlobColor, double> positionError;
	std::map<BlobColor, Eigen::Vector2f> positionOffset;
	positionOffset[BlobColor::ORANGE] = {0.f, 0.f};
	positionOffset[BlobColor::YELLOW] = {0.f, 0.f};
	positionOffset[BlobColor::BLUE] = {0.f, 0.f};
	positionOffset[BlobColor::GREEN] = {0.f, 0.f};
	positionOffset[BlobColor::PINK] = {0.f, 0.f};

	//TODO Check outside peaks outside blob radius (since score next to scale)

	while(true) {
		double startTime = getTime();
		std::shared_ptr<Image> img = r.camera->readImage();
		if(img == nullptr)
			break;

		imageTime += getTime() - startTime;
		startTime = getTime();

		r.perspective->geometryCheck(r.cameraAmount, img->width, img->height, r.gcSocket->maxBotHeight);

		std::shared_ptr<CLImage> clImg = r.openCl->acquire(&PixelFormat::RGBA8, img->width, img->height, img->name);
		std::shared_ptr<CLImage> flat = r.openCl->acquire(&PixelFormat::RGBA8, r.perspective->reprojectedFieldSize[0], r.perspective->reprojectedFieldSize[1], img->name);
		std::shared_ptr<CLImage> color = r.openCl->acquire(&PixelFormat::F32, r.perspective->reprojectedFieldSize[0], r.perspective->reprojectedFieldSize[1], img->name);
		std::shared_ptr<CLImage> circ = r.openCl->acquire(&PixelFormat::F32, r.perspective->reprojectedFieldSize[0], r.perspective->reprojectedFieldSize[1], img->name);
		std::shared_ptr<CLImage> score = r.openCl->acquire(&PixelFormat::F32, r.perspective->reprojectedFieldSize[0], r.perspective->reprojectedFieldSize[1], img->name);

		cl::NDRange visibleFieldRange(r.perspective->reprojectedFieldSize[0], r.perspective->reprojectedFieldSize[1]);
		//cv::GaussianBlur(flat.read<RGBA>().cv, blurred.write<RGBA>().cv, {5, 5}, 0, 0, cv::BORDER_REPLICATE);

		OpenCL::await(scoreKernel, cl::EnqueueArgs(
				OpenCL::run(circleKernel, cl::EnqueueArgs(
						OpenCL::run(colorKernel, cl::EnqueueArgs(
								OpenCL::run(perspectiveKernel, cl::EnqueueArgs(
										OpenCL::run(img->format == &PixelFormat::RGGB8 ? rggb2img : bgr2img, cl::EnqueueArgs(cl::NDRange(clImg->width, clImg->height)), img->buffer, clImg->image),
								visibleFieldRange), clImg->image, flat->image, r.perspective->getClPerspective(), (float)r.gcSocket->maxBotHeight, r.perspective->fieldScale, r.perspective->visibleFieldExtent[0], r.perspective->visibleFieldExtent[2]),
						visibleFieldRange), flat->image, color->image, (int)ceil(r.maxBlobRadius/r.perspective->fieldScale)/3),
				visibleFieldRange), color->image, circ->image, (int)floor(r.minBlobRadius/r.perspective->fieldScale), (int)ceil(r.maxBlobRadius/r.perspective->fieldScale)),
		visibleFieldRange), flat->image, circ->image, score->image, (float)r.minCircularity, (int)floor(r.minBlobRadius/r.perspective->fieldScale));

		processingTime += getTime() - startTime;
		//std::cout << "Radius " << floor(r.minBlobRadius/r.perspective->fieldScale) << "->" << ceil(r.maxBlobRadius/r.perspective->fieldScale) << std::endl;
		startTime = getTime();

		const SSL_DetectionFrame& detection = getCorrespondingFrame(groundTruth, ++frameId);
		std::vector<Blob> blobs;
		for(const SSL_DetectionBall& ball : detection.balls()) {
			Eigen::Vector3f field(ball.x(), ball.y(), 30.0f); // See ssl-vision/src/app/plugins/plugin_detect_balls.h
			blobs.push_back({
				.field = field,
				.flat = field2flat(r, field),
				.radius = (float) r.ballRadius / r.perspective->fieldScale,
				.color = ORANGE
			});
		}
		for(const SSL_DetectionRobot& bot : detection.robots_yellow())
			bot2blobs(r, bot, YELLOW, blobs);
		for(const SSL_DetectionRobot& bot : detection.robots_blue())
			bot2blobs(r, bot, BLUE, blobs);

		float worstBlobCirc = INFINITY;
		float worstBlobScore = INFINITY;

		CLImageMap<float> circMap = circ->read<float>();
		CLImageMap<float> scoreMap = score->read<float>();
		//CLImageMap<float> scoreMap = circ->read<float>();
		for(const Blob& blob : blobs) {
			Eigen::Vector2i maxPos;
			float maxScore = -INFINITY;
			for(int y = std::max(0, (int)floorf(blob.flat.y() - blob.radius)); y < std::min(r.perspective->reprojectedFieldSize[1], (int)ceilf(blob.flat.y() + blob.radius)); y++) {
				for(int x = std::max(0, (int)floorf(blob.flat.x() - blob.radius)); x < std::min(r.perspective->reprojectedFieldSize[0], (int)ceilf(blob.flat.x() + blob.radius)); x++) {
					float s = scoreMap(x, y);
					if(s > maxScore) {
						//TOOD subpixel precision?
						maxPos = {x, y};
						maxScore = s;
					}
				}
			}

			if(maxScore == -INFINITY) {
				if(WARN_INVISIBLE_BLOBS)
					std::cerr << "Blob with color " << blob.color << " in frame " << frameId << " without score." << std::endl;
				continue;
			}

			if(DRAW_DEBUG_IMAGES && frameId == 1) {
				cv::drawMarker(flat->readWrite<RGBA>().cv, {maxPos.x(), maxPos.y()}, CV_RGB(0, 0, 255));
				cv::drawMarker(flat->readWrite<RGBA>().cv, {(int)blob.flat.x(), (int)blob.flat.y()}, CV_RGB(255, 0, 0));
			}

			Eigen::Vector2f offset = r.perspective->flat2field(maxPos.cast<float>()) - r.perspective->flat2field(blob.flat);
			float offsetNorm = offset.norm();
			blobAmount[blob.color] += 1;
			positionError[blob.color] += offsetNorm;
			positionOffset[blob.color] += offset;
			if(offsetNorm > 10.0f)
				criticalOffset[blob.color] += 1;
			if(maxScore < worstBlobScore) {
				worstBlobScore = maxScore;
				worstBlobCirc = circMap(maxPos.x(), maxPos.y());
			}
		}

		for(int y = 0; y < r.perspective->reprojectedFieldSize[1]; y++) {
			for(int x = 0; x < r.perspective->reprojectedFieldSize[0]; x++) {
				if(circMap(x, y) > 0) {
					circTotal++;

					if(circMap(x, y) > worstBlobCirc)
						circHits++;
				}
			}
		}
		//TODO max in blob area (location offset) worst blob to best blob ratio, worst blob -> how many detections
		analysisTime += getTime() - startTime;

		if(DRAW_DEBUG_IMAGES && frameId == 1)  {
			flat->save(".flat.png");
			color->save(".color.png", 0.0625f, 128.f);
			circ->save(".circ.png", 2.0f);
			//score->save(".score.png", 0.3333f, 256.f);
			score->save(".score.png", 2.0f);
		}
	}

	std::cout << "[Blob benchmark] Avg circ hits " << ((double)circHits/frameId) << " Avg circ total " << ((double)circTotal/frameId)  << std::endl;
	double totalOffset = 0.0;
	int totalAmount = 0;
	for(const auto& offset : positionError) {
		totalOffset += offset.second;
		totalAmount += blobAmount[offset.first];
		std::cout << "[Blob benchmark] Avg color: " << offset.first << " position error: " << (offset.second / blobAmount[offset.first]) << " critical offset: " << ((double)criticalOffset[offset.first] / blobAmount[offset.first]) << " systematic offset: " << (positionOffset[offset.first].transpose() / blobAmount[offset.first]) << std::endl;
	}
	std::cout << "[Blob benchmark] Avg processing time: " << (processingTime / frameId) << " frame load time: " << (imageTime / frameId) << " analysis time: " << (analysisTime / frameId) << " frames: " << frameId << std::endl;

	std::cout << "[BlobMachine] " << ((double)circTotal/frameId) << " " << ((double)circHits/frameId) << " " << (totalOffset/totalAmount) << std::endl;
}