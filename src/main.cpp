#include <iostream>

#include "source/imagesource.h"
#include "rtpstreamer.h"
#include "source/spinnakersource.h"
#include "opencl.h"
#include "udpsocket.h"
#include "Perspective.h"
#include "Mask.h"

#include <yaml-cpp/yaml.h>

double getTime() {
	return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count() / 1e6;
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

	std::shared_ptr<UDPSocket> socket = std::make_shared<UDPSocket>(config["vision_ip"].as<std::string>("224.5.23.2"), config["vision_port"].as<int>(10006));
	std::shared_ptr<Perspective> perspective = std::make_shared<Perspective>(socket, config["cam_id"].as<int>(0));
	Mask mask(perspective, config["max_bot_height"].as<double>(150.0));

	std::shared_ptr<OpenCL> openCl = std::make_shared<OpenCL>();
	RTPStreamer rtpStreamer(openCl, "rtp://" + config["vision_ip"].as<std::string>("224.5.23.2") + ":" + std::to_string(config["stream_base_port"].as<int>(10100) + config["cam_id"].as<int>(0)));

	//cl::Buffer clYellowKernel = openCl->toBuffer(false, yellowKernel); //TODO is WRITE necessary at all?

	cl::Kernel kernel = openCl->compile((
#include "image2field.cl"
#include "ssd.cl"
	), "-D RGGB");
	cl::Kernel ballkernel = openCl->compile((
#include "image2field.cl"
#include "ballssd.cl"
	), "-D RGGB");
	//), "-D FILTER_WIDTH=" + std::to_string(yellowKernel->getWidth()) + " -D FILTER_HEIGHT=" + std::to_string(yellowKernel->getHeight()) + " -D STRIDE_X=" + std::to_string(img->pixelWidth()) + " -D STRIDE_Y=" + std::to_string(img->pixelHeight()));

	std::vector<int> pos;
	pos.reserve(1224*1024*2);
	for(int y = 0; y < 1024; y++) {
		for(int x = 0; x < 1224; x++) {
			pos.push_back(x);
			pos.push_back(y);
		}
	}
	std::vector<float> result;
	result.resize(pos.size()/2);
	cl::Buffer clPos(CL_MEM_USE_HOST_PTR, pos.size()*sizeof(int), pos.data());
	cl::Buffer clResult(CL_MEM_USE_HOST_PTR | CL_MEM_HOST_READ_ONLY, result.size()*sizeof(float), result.data());

	//TODO ensure pointer same
	//void* ptr = cl::enqueueMapBuffer(clYellowKernel, true, CL_MAP_READ | CL_MAP_WRITE, 0, size);
	//cl::enqueueUnmapMemObject()

	//cv::Ptr<cv::LineSegmentDetector> detector = cv::createLineSegmentDetector();

	//std::shared_ptr<Image> convResult = BufferImage::create(F32, img->getWidth(), img->getHeight());
	//cl::Buffer clConvResult = openCl->toBuffer(true, convResult);

	while(true) {
		std::shared_ptr<Image> img = camera->readImage();

		auto startTime = std::chrono::high_resolution_clock::now();
		perspective->geometryCheck();
		mask.geometryCheck();

		if(perspective->getGeometryVersion()) {
			cl::Buffer clBuffer = openCl->toBuffer(false, img);
			//yellow openCl->run(kernel, cl::EnqueueArgs(cl::NDRange(result.size())), clBuffer, clPos, clResult, perspective->getClPerspective(), 150.0f, 25.0f, (uint8_t)255, (uint8_t)255, (uint8_t)0).wait();
			//TODO issues with goal and floor outside field. Black background consideration necessary
			//blue openCl->run(kernel, cl::EnqueueArgs(cl::NDRange(result.size())), clBuffer, clPos, clResult, perspective->getClPerspective(), 150.0f, 25.0f, (uint8_t)0, (uint8_t)128, (uint8_t)255).wait();
			//ball
			openCl->run(ballkernel, cl::EnqueueArgs(cl::NDRange(result.size())), clBuffer, clPos, clResult, perspective->getClPerspective(), 21.5f, 21.5f, (uint8_t)255, (uint8_t)128, (uint8_t)0).wait();
			//std::cout << event.wait() << " " << event.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>() << std::endl;
			//TODO only if not unified memory (also rtpstreamer.cpp)
			cl::enqueueReadBuffer(clResult, true, 0, result.size()*sizeof(float), result.data());

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

			float threshold = *std::min_element(result.begin(), result.end())*1.15;
			for(int i = 0; i < result.size(); i++) {
				if(result[i] < threshold) {
					uint8_t* row0 = img->getData() + 4*img->getWidth()*pos[2*i+1] + 2*pos[2*i];
					uint8_t* row1 = row0 + 2*img->getWidth();
					row0[0] = 255;
					row0[1] = 255;
					row1[0] = 255;
					row1[1] = 0	;
				}
			}
			std::cout << "minmax " << result.size() << " " << *std::min_element(result.begin(), result.end()) << " " << *std::max_element(result.begin(), result.end()) << std::endl;
		}

		std::cout << "main " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - startTime).count() / 1000.0 << " ms" << std::endl;

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
		std::this_thread::sleep_for(std::chrono::microseconds(33333) - (std::chrono::high_resolution_clock::now() - startTime));
	}
}
