#pragma once

#include <string>
#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>

#include "image.h"
#include "opencl.h"
#include "AlignedArray.h"


typedef struct AVCodec AVCodec;
typedef struct AVCodecContext AVCodecContext;
typedef struct AVFormatContext AVFormatContext;
typedef struct AVFrame AVFrame;
typedef struct AVStream AVStream;
typedef struct AVPacket AVPacket;


class RTPStreamer {
public:
	explicit RTPStreamer(std::shared_ptr<OpenCL> openCl, std::string uri, int framerate = 30);
	~RTPStreamer();
	void sendFrame(std::shared_ptr<Image> image);
private:
	void encoderRun();

	void allocResources();
	void freeResources();

	const std::shared_ptr<OpenCL> openCl;
	const std::string uri;
	const int framerate;
	const int frametime_us;
	int width = 0;
	int height = 0;
	const PixelFormat* format = &PixelFormat::U8;

	bool stopEncoding = false;
	std::thread encoder;

	std::shared_ptr<Image> queue = nullptr;
	std::mutex queueMutex = std::mutex();
	std::condition_variable queueSignal = std::condition_variable();
	long currentFrameId = 0;

	AVCodecContext* codecCtx = nullptr;
	AVFormatContext* fmtCtx = nullptr;
	AVStream* stream = nullptr;
	AVFrame* frame = nullptr;
	AVPacket* pkt = nullptr;

	std::unique_ptr<CLArray> buffer = nullptr;
	cl::Kernel converter;
};
