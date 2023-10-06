#include <iostream>

#include "imagesource.h"
#include "rtpstreamer.h"
#include "spinnakersource.h"
#include "messages_robocup_ssl_detection.pb.h"
#include "opencl.h"

#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

enum Stage {
	AWAITING_GEOMETRY,
	UP_AND_RUNNING
};

int main() {
	YAML::Node config = YAML::LoadFile("config.yml");
	//TODO adapt to source type

	config["source"].as<std::string>("SPINNAKER");

	/*
	//RG
	//GB
	std::shared_ptr<Image> yellowKernel = BufferImage::create(F32, 10, 10);
	auto* kData = (float*)yellowKernel->getData();
	for(int y = 0; y < yellowKernel->getHeight(); y++) {
		for(int x = 0; x < yellowKernel->getWidth(); x++) {
			kData[y*yellowKernel->getWidth() + x] = sqrt((y-5)*(y-5) + (x-5)*(x-5)) <= 4 && !(y%2 && x%2) ? 0.03 : -0.03;
			//std::cout << kData[y*yellowKernel->getWidth() + x] << " ";
		}
		//std::cout << std::endl;
	}*/

	/*
	//BGR
	std::shared_ptr<Image> yellowKernel = BufferImage::create(F32, 33, 11);
	auto* kData = (float*)yellowKernel->getData();
	for(int y = 0; y < yellowKernel->getHeight(); y++) {
		for(int x = 0; x < yellowKernel->getWidth(); x++) {
			kData[y*yellowKernel->getWidth() + x] = sqrt((y-5)*(y-5) + (x/3-5)*(x/3-5)) <= 5 && x%3 != 0 ? 0.03 : -0.02;
			//std::cout << kData[y*yellowKernel->getWidth() + x] << " ";
		}
		//std::cout << std::endl;
	}*/

	//BGR
	std::shared_ptr<Image> yellowKernel = BufferImage::create(F32, 9, 3);
	auto* kData = (float*)yellowKernel->getData();
	kData[0] = -1; kData[1] = -1; kData[2] = 0;
	kData[3] = -1; kData[4] = 0; kData[5] = 1;
	kData[6] = 0; kData[7] = 1; kData[8] = 1;

#ifdef SPINNAKER
	SpinnakerSource camera(0);
#else
	std::vector<std::string> paths;
	/*paths.emplace_back("test-data/rc2019/1/cam0/00000.png");
	paths.emplace_back("test-data/rc2019/1/cam0/00001.png");
	paths.emplace_back("test-data/rc2019/1/cam0/00002.png");
	paths.emplace_back("test-data/rc2019/1/cam0/00003.png");*/
	paths.emplace_back("test-data/rc2022/bots-balls-many-1/00000.png");
	paths.emplace_back("test-data/rc2022/bots-balls-many-1/00001.png");
	paths.emplace_back("test-data/rc2022/bots-balls-many-1/00002.png");
	paths.emplace_back("test-data/rc2022/bots-balls-many-1/00003.png");
	/*paths.emplace_back("test-data/rc2022/bots-center-ball-1-0/00000.png");
	paths.emplace_back("test-data/rc2022/bots-center-ball-1-0/00001.png");
	paths.emplace_back("test-data/rc2022/bots-center-ball-1-0/00002.png");
	paths.emplace_back("test-data/rc2022/bots-center-ball-1-0/00003.png");
	paths.emplace_back("test-data/rc2022/bots-center-ball-1-1/00000.png");
	paths.emplace_back("test-data/rc2022/bots-center-ball-1-1/00001.png");
	paths.emplace_back("test-data/rc2022/bots-center-ball-1-1/00002.png");
	paths.emplace_back("test-data/rc2022/bots-center-ball-1-1/00003.png");*/
	ImageSource camera(paths);
#endif

	std::shared_ptr<OpenCL> openCl = std::make_shared<OpenCL>();
	RTPStreamer rtpStreamer(openCl, "rtp://" + config["vision_ip"].as<std::string>("224.5.23.2") + ":" + std::to_string(config["stream_base_port"].as<int>(10100) + config["cam_id"].as<int>(0)));

	cl::Buffer clYellowKernel = openCl->toBuffer(false, yellowKernel); //TODO is WRITE necessary at all?
	//TODO softcoded stride
	cl::Kernel kernel = openCl->compile((
#include "convolution.cl"
	), "-D FILTER_WIDTH=" + std::to_string(yellowKernel->getWidth()) + " -D FILTER_HEIGHT=" + std::to_string(yellowKernel->getHeight()) + " -D STRIDE_X=" + std::to_string(3) + " -D STRIDE_Y=" + std::to_string(1));

	//cv::Ptr<cv::LineSegmentDetector> detector = cv::createLineSegmentDetector();
	std::shared_ptr<Image> img = camera.readImage();

	//std::shared_ptr<Image> convResult = BufferImage::create(F32, img->getWidth(), img->getHeight());
	//cl::Buffer clConvResult = openCl->toBuffer(true, convResult);

	while(true) {
		img = camera.readImage();
		auto startTime = std::chrono::high_resolution_clock::now();
		//cl::Buffer buffer = openCl->toBuffer(false, img);
		//cl::Event conv = openCl->run(kernel, cl::EnqueueArgs(cl::NDRange(img->getWidth()*2, img->getHeight()*2), cl::NDRange(64, 64)), buffer, clConvResult, clYellowKernel, (64+yellowKernel->getWidth()-1)*(64+yellowKernel->getHeight()-1)*sizeof(float));
		/*cl::Event conv = openCl->run(kernel, cl::EnqueueArgs(cl::NDRange(img->getWidth(), img->getHeight())), buffer, clConvResult, clYellowKernel);
		if(int e = conv.wait()) {
			std::cerr << e << " " << conv.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>() << std::endl;
		}
		//TODO only if not unified memory (also rtpstreamer.cpp)
		//cl::enqueueReadBuffer(clConvResult, true, 0, img->getWidth()*img->getHeight()*sizeof(float), convResult->getData());
		rtpStreamer.sendFrame(convResult);*/
		rtpStreamer.sendFrame(img);
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

		std::cout << "send_frame " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - startTime).count() / 1000.0 << " ms" << std::endl;
	}

	return 0;
}
