#pragma once

#include <string>
#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>

#include "image.h"


typedef struct AVCodec AVCodec;
typedef struct AVCodecContext AVCodecContext;
typedef struct AVFormatContext AVFormatContext;
typedef struct AVFrame AVFrame;
typedef struct AVStream AVStream;
typedef struct AVPacket AVPacket;


class RTPStreamer {
public:
	explicit RTPStreamer(std::string uri, int framerate = 30);
	~RTPStreamer();
	void sendFrame(std::shared_ptr<Image> image);
private:
	void encoderRun();

	void allocResources();
	void freeResources();

	const std::string uri;
	const int framerate;
	const int frametime_us;
	int width = 0;
	int height = 0;

	bool stopEncoding = false;
	std::thread encoder;

	std::queue<std::shared_ptr<Image>> queue;
	std::mutex queueMutex;
	std::condition_variable queueSignal;
	long currentFrameId = 0;

	AVCodecContext* codecCtx = nullptr;
	AVFormatContext* fmtCtx = nullptr;
	AVStream* stream = nullptr; //TODO necessary?
	AVFrame* frame = nullptr;
	AVPacket* pkt = nullptr;
};
