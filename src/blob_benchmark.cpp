#include <yaml-cpp/node/parse.h>
#include "Resources.h"
#include "GroundTruth.h"
#include "pattern.h"

enum BlobColor {
	ORANGE,
	YELLOW,
	BLUE,
	GREEN,
	PINK
};

typedef struct Blob {
	Eigen::Vector2f flat;
	float radius;
	BlobColor color;
} Blob;

static inline Eigen::Vector2f field2flat(const Resources& r, float x, float y, float height) {
	return r.perspective->field2flat(r.perspective->model.image2field(r.perspective->model.field2image({x, y, height}), (float)r.gcSocket->maxBotHeight).head<2>());
}

static void bot2blobs(const Resources& r, const SSL_DetectionRobot& bot, const BlobColor botColor, std::vector<Blob>& blobs) {
	blobs.push_back({
		.flat = field2flat(r, bot.x(), bot.y(), bot.height()),
		.radius = (float)r.centerBlobRadius / r.perspective->fieldScale,
		.color = botColor
	});

	int pattern = patterns[bot.robot_id()];
	for(int i = 0; i < 4; i++) {
		Eigen::Vector2f offset(r.sideBlobDistance * cosf(bot.orientation()), r.sideBlobDistance * sinf(bot.orientation()));
		blobs.push_back({
			.flat = field2flat(r, bot.x() + (float)r.sideBlobDistance * cosf(bot.orientation()), bot.y() + (float)r.sideBlobDistance * sinf(bot.orientation()), bot.height()),
			.radius = (float)r.sideBlobRadius / r.perspective->fieldScale,
			.color = (pattern & (8 >> pattern)) ? GREEN : PINK
		});
	}
}

int main(int argc, char* argv[]) {
	Resources r(YAML::LoadFile(argc > 1 ? argv[1] : "config.yml"));
	std::vector<SSL_DetectionFrame> groundTruth = parseGroundTruth(r.groundTruth);

	cl::Kernel rggb2img = r.openCl->compileFile("kernel/rggb2img.cl");
	cl::Kernel bgr2img = r.openCl->compileFile("kernel/bgr2img.cl");
	CLImage clImg(&PixelFormat::RGBA8);

	cl::Kernel perspectiveKernel = r.openCl->compileFile("kernel/perspective.cl");
	CLImage flat(&PixelFormat::RGBA8);

	cl::Kernel colorKernel = r.openCl->compileFile("kernel/color.cl");
	CLImage color(&PixelFormat::F32);

	cl::Kernel circleKernel = r.openCl->compileFile("kernel/circularize.cl");
	CLImage circ(&PixelFormat::F32);

	CLImage score(&PixelFormat::F32);

	//cl::Kernel matchKernel = r.openCl->compileFile("kernel/matches.cl");

	while(r.waitForGeometry && !r.socket->getGeometryVersion()) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
		r.socket->geometryCheck();
	}

	int frameId = 0;
	double processingTime = 0.0;

	int blobAmount;
	double positionOffset = 0.0;

	while(true) {
		std::shared_ptr<Image> img = r.camera->readImage();
		if(img == nullptr)
			break;

		double startTime = getTime();
		r.perspective->geometryCheck(img->width, img->height, r.gcSocket->maxBotHeight);

		ensureSize(clImg, img->width, img->height, img->name);
		ensureSize(flat, r.perspective->reprojectedFieldSize[0], r.perspective->reprojectedFieldSize[1], img->name);
		ensureSize(color, r.perspective->reprojectedFieldSize[0], r.perspective->reprojectedFieldSize[1], img->name);
		ensureSize(circ, r.perspective->reprojectedFieldSize[0], r.perspective->reprojectedFieldSize[1], img->name);
		ensureSize(score, r.perspective->reprojectedFieldSize[0], r.perspective->reprojectedFieldSize[1], img->name);

		cl::NDRange visibleFieldRange(r.perspective->reprojectedFieldSize[0], r.perspective->reprojectedFieldSize[1]);
		//TODO better type switching
		OpenCL::wait(r.openCl->run(img->format == &PixelFormat::RGGB8 ? rggb2img : bgr2img, cl::EnqueueArgs(cl::NDRange(clImg.width, clImg.height)), img->buffer, clImg.image));
		OpenCL::wait(r.openCl->run(perspectiveKernel, cl::EnqueueArgs(visibleFieldRange), clImg.image, flat.image, r.perspective->getClPerspective(), (float)r.gcSocket->maxBotHeight, r.perspective->fieldScale, r.perspective->visibleFieldExtent[0], r.perspective->visibleFieldExtent[2]));
		OpenCL::wait(r.openCl->run(colorKernel, cl::EnqueueArgs(visibleFieldRange), flat.image, color.image));
		OpenCL::wait(r.openCl->run(circleKernel, cl::EnqueueArgs(visibleFieldRange), color.image, circ.image));

		//TODO scoring
		processingTime += getTime() - startTime;

		const SSL_DetectionFrame& detection = getCorrespondingFrame(groundTruth, ++frameId);
		std::vector<Blob> blobs;
		for(const SSL_DetectionBall& ball : detection.balls())
			blobs.push_back({
				.flat = field2flat(r, ball.x(), ball.y(), 30.0f), // See ssl-vision/src/app/plugins/plugin_detect_balls.h
				.radius = (float)r.ballRadius / r.perspective->fieldScale,
				.color = ORANGE
			});
		for(const SSL_DetectionRobot& bot : detection.robots_yellow())
			bot2blobs(r, bot, YELLOW, blobs);
		for(const SSL_DetectionRobot& bot : detection.robots_blue())
			bot2blobs(r, bot, BLUE, blobs);

		blobAmount += blobs.size();
		float worstBlob = INFINITY;

		CLImageMap<float> map = score.read<float>();
		for(const Blob& blob : blobs) {
			Eigen::Vector2i maxPos;
			float maxScore = -INFINITY;
			for(int y = (int)(blob.flat.y() - blob.radius); y < (int)(blob.flat.y() + blob.radius); y++) {
				for(int x = (int)(blob.flat.x() - blob.radius); x < (int)(blob.flat.x() + blob.radius); x++) {
					if(map[x + y*map.rowPitch] > maxScore) {
						//TOOD subpixel precision?
						maxPos = {x, y};
						maxScore = map[x + y*map.rowPitch];
					}
				}
			}

			positionOffset += r.perspective->fieldScale * (maxPos.cast<float>() - blob.flat).norm();
			if(maxScore < worstBlob)
				worstBlob = maxScore;
		}

		for(int y = 0; y < r.perspective->reprojectedFieldSize[1]; y++) {
			for(int x = 0; x < r.perspective->reprojectedFieldSize[0]; x++) {

			}
		}
		//TODO evaluation
		//TODO max in blob area (location offset) worst blob to best blob ratio, worst blob -> how many detections
	}

	std::cout << "[Blob benchmark] Avg processing time: " << (processingTime / frameId) << std::endl;
}