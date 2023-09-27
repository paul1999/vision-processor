#include "rtpstreamer.h"

#include <cstdlib>
#include <iostream>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
}


RTPStreamer::RTPStreamer(std::shared_ptr<OpenCL> openCl, std::string uri, int framerate): openCl(std::move(openCl)), uri(std::move(uri)), framerate(framerate), frametime_us(1000000 / framerate), encoder(&RTPStreamer::encoderRun, this) {}

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
	AVOutputFormat* format = (AVOutputFormat*)av_guess_format("rtp", nullptr, nullptr);
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

	buffer = BufferImage::create(NV12, width, height);
	int uvOffset = buffer->getWidth()*buffer->getHeight();

	frame = av_frame_alloc();
	frame->format = codecCtx->pix_fmt;
	frame->width  = codecCtx->width;
	frame->height = codecCtx->height;
	//frame->extended_data = frame->data;
	frame->data[0] = buffer->getData();
	frame->data[1] = buffer->getData() + uvOffset;
	frame->linesize[0] = buffer->getWidth();
	frame->linesize[1] = buffer->getWidth();

	pkt = av_packet_alloc();

	clBuffer = cl::Buffer(CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, buffer->getWidth()*buffer->getHeight()*2, buffer->getData(), nullptr);
	switch(this->format) {
		case RGGB8:
			yConverter = openCl->compile("y", "void kernel y(global const uchar* in, global uchar* out) { int i = get_global_id(0); int o = i; int i1 = 4*i - 2*(i%" + std::to_string(width) + "); int i2 = i1 + 2*" + std::to_string(width) + ";"
											  "    out[o] = (uchar)((66*(short)in[i1] + 64*(short)in[i1+1] + 65*(short)in[i2] + 25*(short)in[i2+1]) / 256 + 16);"
											  "}");
			//TODO NV12 UV is QUARTER resolution, not HALF
			uvConverter = openCl->compile("uv", "void kernel uv(global const uchar* in, global uchar* out) { int i = 2*get_global_id(0); int o = i + " + std::to_string(uvOffset) + "; int i1 = 4*i - 2*(i%" + std::to_string(width) + "); int i2 = i1 + 2*" + std::to_string(width) + ";"
											    "    out[o] = (uchar)((-38*(short)in[i1] + -37*(short)in[i1+1] + -37*(short)in[i2] + 112*(short)in[i2+1]) / 256 + 128);"
											    "    out[o+1] = (uchar)((112*(short)in[i1] + -47*(short)in[i1+1] + -47*(short)in[i2] + -18*(short)in[i2+1]) / 256 + 128);"
											    "}");
			break;
		case BGR888:
			yConverter = openCl->compile("y", "void kernel y(global const uchar* in, global uchar* out) { int i = 3*get_global_id(0); int o = get_global_id(0);"
											  "    out[o] = (uchar)((25*(short)in[i] + 129*(short)in[i+1] + 66*(short)in[i+2]) / 256 + 16);"
											  "}");
			//TODO NV12 UV is QUARTER resolution, not HALF
			uvConverter = openCl->compile("uv", "void kernel y(global const uchar* in, global uchar* out) { int i = 6*get_global_id(0); int o = 2*get_global_id(0) + " + std::to_string(uvOffset) + ";"
												"    out[o] = (uchar)((112*(short)in[i] + -74*(short)in[i+1] + -38*(short)in[i+2]) / 256 + 128);"
												"    out[o+1] = (uchar)((-18*(short)in[i] + -94*(short)in[i+1] + 112*(short)in[i+2]) / 256 + 128);"
												"}");
			break;
		case U8:
			yConverter = openCl->compile("y", "void kernel y(global const uchar* in, global uchar* out) { int i = get_global_id(0); out[i] = in[i]; }");
			uvConverter = openCl->compile("uv", "void kernel uv(global const ushort* in, global ushort* out) { int i = get_global_id(0) + " + std::to_string(uvOffset) + "; out[i] = 0x7f7f; }");
			break;
		case I8:
			yConverter = openCl->compile("y", "void kernel y(global const char* in, global uchar* out) { int i = get_global_id(0); out[i] = (uchar)in[i] + 127; }");
			uvConverter = openCl->compile("uv", "void kernel uv(global const ushort* in, global ushort* out) { int i = get_global_id(0) + " + std::to_string(uvOffset) + "; out[i] = 0x7f7f; }");
			break;
		case NV12:
			yConverter = openCl->compile("y", "void kernel y(global const uchar* in, global uchar* out) { int i = get_global_id(0); out[i] = in[i]; }");
			uvConverter = openCl->compile("uv", "void kernel uv(global const ushort* in, global ushort* out) { int i = get_global_id(0) + " + std::to_string(uvOffset) + "; out[i] = in[i]; }");
			break;
	}
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

		if(image->getWidth() != width || image->getHeight() != height || image->getFormat() != format) {
			freeResources();
			width = image->getWidth();
			height = image->getHeight();
			format = image->getFormat();
		}

		allocResources();

		auto startTime = std::chrono::high_resolution_clock::now();
		cl::Buffer inBuffer = cl::Buffer(CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, image->getWidth()*image->getHeight()*image->pixelSize(), image->getData(), nullptr);
		cl::Event yConversion = openCl->run(yConverter, cl::EnqueueArgs(cl::NDRange(image->getWidth()*image->getHeight())), inBuffer, clBuffer);
		cl::Event uvConversion = openCl->run(uvConverter, cl::EnqueueArgs(cl::NDRange(image->getWidth()/2*image->getHeight()/2)), inBuffer, clBuffer);
		yConversion.wait();
		uvConversion.wait();
		std::cout << "frame_conversion " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - startTime).count() / 1000.0 << " ms" << std::endl;

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
