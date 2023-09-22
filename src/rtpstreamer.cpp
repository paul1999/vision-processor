#include "rtpstreamer.h"

#include <cstdlib>
#include <iostream>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
}


RTPStreamer::RTPStreamer(std::string uri, int framerate): uri(std::move(uri)), framerate(framerate), frametime_us(1000000 / framerate), encoder(&RTPStreamer::encoderRun, this) {}

RTPStreamer::~RTPStreamer() {
	stopEncoding = true;

	{
		std::unique_lock<std::mutex> lock(queueMutex);
		queueSignal.notify_one();
	}

	encoder.join();
	freeResources();
}

//Adopted from CC BY-SA 4.0 https://stackoverflow.com/a/61988145 Dmitrii Zabotlin
void RTPStreamer::sendFrame(std::shared_ptr<Image> image) {
	std::unique_lock<std::mutex> lock(queueMutex);

	if(queue.empty()) {
		queue.push(std::move(image));
		queueSignal.notify_one();
	}
}

//Adopted from CC BY-SA 3.0 https://stackoverflow.com/q/40825300 DankMemes and https://stackoverflow.com/q/46352604 Gaulois94
void RTPStreamer::allocResources() {
	if(codecCtx != nullptr)
		return;

	const AVCodec* codec;
	std::vector<const char*> codecNames {"h264_nvenc", "h264_qsv", "h264_vaapi", "libx264"};
	for (const auto &codecName : codecNames) {
		codec = avcodec_find_encoder_by_name(codecName);
		if(codec == nullptr)
			continue;

		codecCtx = avcodec_alloc_context3(codec);

		codecCtx->bit_rate = 1500000;
		codecCtx->width = width;
		codecCtx->height = height;
		codecCtx->time_base.num = 1;
		codecCtx->time_base.den = framerate;
		codecCtx->gop_size = framerate;
		codecCtx->max_b_frames = 0;
		codecCtx->pix_fmt = AV_PIX_FMT_NV12;
		codecCtx->codec_type = AVMEDIA_TYPE_VIDEO;

		if(strcmp(codecName, "h264_qsv") == 0) {
			av_opt_set(codecCtx->priv_data, "preset", "veryfast", 0);
		}
		if(strcmp(codecName, "libx264") == 0) {
			av_opt_set(codecCtx->priv_data, "preset", "ultrafast", 0);
			av_opt_set(codecCtx->priv_data, "tune", "zerolatency", 0);
		}

		if(avcodec_open2(codecCtx, codec, nullptr) == 0)
			break;

		av_free(codecCtx);
		codecCtx = nullptr;
	}

	if(codecCtx == nullptr) {
		std::cerr << "Failed to find suitable encoder." << std::endl;
		exit(1);
	}

	std::cout << "Using codec: " << codec->long_name << std::endl;

	fmtCtx  = avformat_alloc_context();
	const AVOutputFormat* format = av_guess_format("rtp", nullptr, nullptr);
	avformat_alloc_output_context2(&fmtCtx, format, format->name, uri.c_str());
	avio_open(&fmtCtx->pb, uri.c_str(), AVIO_FLAG_WRITE);

	stream = avformat_new_stream(fmtCtx, codec);
	avcodec_parameters_from_context(stream->codecpar, codecCtx);
	stream->time_base.num = codecCtx->time_base.num;
	stream->time_base.den = codecCtx->time_base.den;

	int write = avformat_write_header(fmtCtx, nullptr);
	if(write < 0) {
		std::cerr << "Failed to write header: " << write << std::endl;
		exit(1);
	}

	//TODO SDP file transmission
	char buf[200000];
	AVFormatContext *ac[] = { fmtCtx };
	av_sdp_create(ac, 1, buf, 20000);
	FILE* fsdp = fopen("test.sdp", "w");
	fprintf(fsdp, "%s", buf);
	fclose(fsdp);

	frame = av_frame_alloc();
	frame->format = codecCtx->pix_fmt;
	frame->width  = codecCtx->width;
	frame->height = codecCtx->height;
	av_image_alloc(frame->data, frame->linesize, frame->width, frame->height, codecCtx->pix_fmt, 32);

	pkt = av_packet_alloc();
}

void RTPStreamer::freeResources() {
	if(codecCtx != nullptr) {
		avcodec_send_frame(codecCtx, nullptr);
		av_free(codecCtx);
		codecCtx = nullptr;
	}

	if(fmtCtx != nullptr) {
		avformat_free_context(fmtCtx);
		fmtCtx = nullptr;
	}

	if(frame != nullptr) {
		av_freep(&frame->data[0]);
		av_frame_free(&frame);
		frame = nullptr;
	}

	if(pkt != nullptr) {
		av_packet_free(&pkt);
		pkt = nullptr;
	}
}

void RTPStreamer::encoderRun() {
	while(!stopEncoding) {
		std::shared_ptr<Image> image;
		{
			std::unique_lock<std::mutex> lock(queueMutex);
			while(queue.empty())
				queueSignal.wait(lock, [&]() { return !queue.empty() || stopEncoding; });

			if(stopEncoding)
				return;

			image = queue.front();
			queue.pop();
		}

		if(image->getWidth() != width || image->getHeight() != height) {
			freeResources();
			width = image->getWidth();
			height = image->getHeight();
		}

		allocResources();

		auto startTime = std::chrono::high_resolution_clock::now();
		//Convert and copy to frame
		const int width = this->width; //Necessary for gcc autovectorizer
		const int height = this->height; //Necessary for gcc autovectorizer
		if(image->getFormat() == BGR888) {
			/*cv::Mat yuv;
			cv::cvtColor(img, yuv, cv::COLOR_BGR2YUV);

			const int width = yuv.cols;
			const int height = yuv.rows;
			for(int y = 0; y < height; y++) {
				const uint8_t* __restrict yData = yuv.data + y*width*3;
				uint8_t* __restrict yFrame = frame->data[0] + y*frame->linesize[0];
				for(int x = 0; x < width; x++) {
					yFrame[x] = yData[x*3];
				}
			}
			for(int y = 0; y < height/2; y++) {
				const uint8_t* __restrict uData = yuv.data + 2*y*width*3 + 1;
				uint8_t* __restrict uFrame = frame->data[1] + y*frame->linesize[1];
				const uint8_t* __restrict vData = yuv.data + 2*y*width*3 + 2;
				uint8_t* __restrict vFrame = frame->data[2] + y*frame->linesize[2];
				for(int x = 0; x < width/2; x++) {
					uFrame[x] = uData[2*x*3];
					vFrame[x] = vData[2*x*3];
				}
			}*/
			//TODO fix autovectorization
			for (int y = 0; y < height; y++) {
				const uint8_t *__restrict imageRow = image->getData() + y * width * 3;
				uint8_t *__restrict yFrame = frame->data[0] + y * frame->linesize[0];
#pragma GCC ivdep
				for (int x = 0; x < width; x++) {
					int pos = 3 * x;
					yFrame[x] = (uint8_t) ((25 * (int16_t) imageRow[pos] + 129 * (int16_t) imageRow[pos + 1] +
											66 * (int16_t) imageRow[pos + 2]) / 256 + 16);
				}
			}

			for (int y = 0; y < height / 2; y++) {
				const uint8_t *__restrict imageRow = image->getData() + 3 * 2 * y * width;
				uint8_t *__restrict uvFrame = frame->data[1] + y * frame->linesize[1];
#pragma GCC ivdep
				for (int x = 0; x < width; x+=2) {
					int pos = 3 * x;
					uvFrame[x] = (uint8_t) ((112 * (int16_t) imageRow[pos] + -74 * (int16_t) imageRow[pos + 1] +
											-38 * (int16_t) imageRow[pos + 2]) / 256 + 128);
					uvFrame[x+1] = (uint8_t) ((-18 * (int16_t) imageRow[pos] + -94 * (int16_t) imageRow[pos + 1] +
											112 * (int16_t) imageRow[pos + 2]) / 256 + 128);
				}
			}
		} else if (image->getFormat() == RGGB8) {
			//TODO fix autovectorization
			for (int y = 0; y < height; y++) {
				const uint8_t* imageRow = image->getData() + y * width * 4;
				const uint8_t* imageRow2 = image->getData() + y * width * 4 + width*2;
				uint8_t* yFrame = frame->data[0] + y * frame->linesize[0];
#pragma GCC ivdep
				for (int x = 0; x < width; x++) {
					int pos = 2 * x;
					yFrame[x] = (uint8_t) (( 66 * (int16_t) imageRow[pos]  + 64 * (int16_t) imageRow[pos + 1] +
											 65 * (int16_t) imageRow2[pos] + 25 * (int16_t) imageRow2[pos + 1]) / 256 + 16);
				}
			}
			for (int y = 0; y < height / 2; y++) {
				const uint8_t* imageRow = image->getData() + 4 * 2 * y * width;
				const uint8_t* imageRow2 = image->getData() + 4 * 2 * y * width + width*2;
				uint8_t* uvFrame = frame->data[1] + y * frame->linesize[1];
#pragma GCC ivdep
				for (int x = 0; x < width; x+=2) {
					int pos = 2 * x;
					uvFrame[x] = (uint8_t) ((-38 * (int16_t) imageRow[pos]  + -37 * (int16_t) imageRow[pos + 1] +
											-37 * (int16_t) imageRow2[pos] + 112 * (int16_t) imageRow2[pos+1]) / 256 + 128);
					uvFrame[x+1] = (uint8_t) ((112 * (int16_t) imageRow[pos]  + -47 * (int16_t) imageRow[pos + 1] +
											-47 * (int16_t) imageRow2[pos] + -18 * (int16_t) imageRow2[pos+1]) / 256 + 128);
				}
			}
		} else if (image->getFormat() == I8) {
			for (int y = 0; y < height; y++) {
				const uint8_t* imageRow = image->getData() + y * width;
				uint8_t* yFrame = frame->data[0] + y * frame->linesize[0];
#pragma GCC ivdep
				for (int x = 0; x < width; x++) {
					yFrame[x] = (uint8_t) (imageRow[x] + 128);
				}
			}

			for (int y = 0; y < height / 2; y++) {
				uint8_t* uvFrame = frame->data[1] + y * frame->linesize[1];
				for (int x = 0; x < width; x++) {
					uvFrame[x] = 127;
				}
			}
		} else if (image->getFormat() == U8) {
			for (int y = 0; y < height; y++) {
				const uint8_t* imageRow = image->getData() + y * width;
				uint8_t* yFrame = frame->data[0] + y * frame->linesize[0];
#pragma GCC ivdep
				for (int x = 0; x < width; x++) {
					yFrame[x] = imageRow[x];
				}
			}

			for (int y = 0; y < height / 2; y++) {
				uint8_t* uvFrame = frame->data[1] + y * frame->linesize[1];
#pragma GCC ivdep
				for (int x = 0; x < width; x++) {
					uvFrame[x] = 127;
				}
			}
		} else {
			std::cerr << "Tried to send frame with unsupported format: " << image->getFormat() << std::endl;
			return;
		}

		frame->pts = currentFrameId++;
		avcodec_send_frame(codecCtx, frame);

		int status = avcodec_receive_packet(codecCtx, pkt);
		if(status == 0) {
			av_packet_rescale_ts(pkt, codecCtx->time_base, stream->time_base);
			av_interleaved_write_frame(fmtCtx, pkt);
			av_packet_unref(pkt);  // reset packet
		} else if(status == AVERROR(EAGAIN)) {
			// Encoder needs some more frames
		} else {
			std::cerr << "Encoder error: " << status << std::endl;
		}

		std::this_thread::sleep_for(std::chrono::microseconds(frametime_us) - (std::chrono::high_resolution_clock::now() - startTime));
	}
}
