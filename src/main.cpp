#include <iostream>

#include "imagesource.h"
#include "rtpstreamer.h"
#include "spinnakersource.h"
#include "messages_robocup_ssl_detection.pb.h"
#include "opencl.h"

#include <opencv2/imgproc.hpp>
//#include <opencv2/opencv.hpp>


/*
std::shared_ptr<Image> xcorr(const std::shared_ptr<Image>& img, std::shared_ptr<Image>& kernel) {
	//TODO assert kernel i8 and >> 4
	//TODO 2d img stride
	const int strideX = 3;
	const int strideY = 1;
	const int kWidth = kernel->getWidth();
	const int kHeight = kernel->getHeight();
	const int kSize = kWidth * kHeight;
	const int width = img->getWidth();
	const int height = img->getHeight();
	const int oWidth = width - kWidth + 1;
	const int oHeight = height - kHeight + 1;

	std::vector<int8_t> buffer;
	buffer.resize(kWidth);
	std::shared_ptr<Image> output = BufferImage::create(I8, oWidth, oHeight);

	const uint8_t* data = img->getData();
	const int8_t* kData = (int8_t*)kernel->getData();
	int8_t* bData = buffer.data();
	int8_t* oData = (int8_t*)output->getData();

	for(int y = 0; y < oHeight; y++) {
		for(int x = 0; x < oWidth; x++) {
#pragma GCC ivdep
			for(int kx = 0; kx < kWidth; kx++)
				bData[kx] = 0;

			for(int ky = 0; ky < kHeight; ky++) {
#pragma GCC ivdep
				for(int kx = 0; kx < kWidth; kx++) {
					bData[kx] += (((int8_t)data[(y*strideY+ky)*width + x*strideX + kx] + (int8_t)-128) / 16) * kData[ky*kWidth + kx] / kSize;
				}
			}

			int8_t sum = 0;
#pragma GCC ivdep
			for(int kx = 0; kx < kWidth; kx++)
				sum += bData[kx];
			oData[y*oWidth + x] = sum;
		}
	}

	return output;
}
*/

int main() {
	//TODO adapt to source type

	/*
	//RG
	//GB
	std::shared_ptr<Image> yellowKernel = BufferImage::create(I8, 10, 10);
	int8_t* kData = (int8_t*)yellowKernel->getData();
	for(int y = 0; y < yellowKernel->getHeight(); y++) {
		for(int x = 0; x < yellowKernel->getWidth(); x++) {
			kData[y*yellowKernel->getWidth() + x] = sqrt((y-5)*(y-5) + (x-5)*(x-5)) <= 4 && !(y%2 && x%2) ? 127/16 : -128/16;
			std::cout << (int)kData[y*yellowKernel->getWidth() + x] << " ";
		}
		std::cout << std::endl;
	}*/

	/*
	//BGR
	std::shared_ptr<Image> yellowKernel = BufferImage::create(U8, 33, 11);
	uint8_t* kData = (uint8_t*)yellowKernel->getData();
	for(int y = 0; y < yellowKernel->getHeight(); y++) {
		for(int x = 0; x < yellowKernel->getWidth(); x++) {
			kData[y*yellowKernel->getWidth() + x] = sqrt((y-5)*(y-5) + (x/3-5)*(x/3-5)) <= 5 && x%3 != 0 ? 255 : 0;
			std::cout << (int)kData[y*yellowKernel->getWidth() + x] << " ";
		}
		std::cout << std::endl;
	}
	cv::Mat cvKernel(11, 11, CV_8UC3, (uint8_t*)yellowKernel->getData());*/

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
	RTPStreamer rtpStreamer(openCl, "rtp://224.5.23.2:10101");

	//cv::Ptr<cv::LineSegmentDetector> detector = cv::createLineSegmentDetector();

	while(true) {
		auto startTime = std::chrono::high_resolution_clock::now();
		std::shared_ptr<Image> img = camera.readImage();
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

		/*
		//TODO Folding with RGB or RAW
		// Cross-Correlation? Black background to yellow/blue foreground
		cv::Mat cvImgBGR[3];
		cv::split(cvImg, cvImgBGR);
		cv::Mat cvKernelBGR[3];
		cv::split(cvKernel, cvKernelBGR);
		cv::Mat cvCorrBGR[3];
		cv::matchTemplate(cvImgBGR[0], cvKernelBGR[0], cvCorrBGR[0], cv::TM_SQDIFF);
		cv::matchTemplate(cvImgBGR[1], cvKernelBGR[1], cvCorrBGR[1], cv::TM_SQDIFF);
		cv::matchTemplate(cvImgBGR[2], cvKernelBGR[2], cvCorrBGR[2], cv::TM_SQDIFF);
		//cv::filter2D(cvImg, cvCorr, CV_8U, cvKernel);
		cv::add(cvCorrBGR[0], cvCorrBGR[1], cvCorrBGR[0]);
		cv::add(cvCorrBGR[0], cvCorrBGR[2], cvCorrBGR[0]);
		cv::Mat cvCorrU8;
		double min, max;
		cv::minMaxLoc(cvCorrBGR[0], &min, &max);
		cvCorrBGR[0].convertTo(cvCorrU8, CV_8U, 255/(max-min), -min*255/(max-min));
		std::shared_ptr<Image> corr = std::make_shared<CVImage>(cvCorrU8, U8);
		//std::shared_ptr<Image> corr = xcorr(img, yellowKernel);
		 rtpStreamer.sendFrame(corr);*/


		std::cout << "send_frame " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - startTime).count() / 1000.0 << " ms" << std::endl;
	}

	return 0;
}
