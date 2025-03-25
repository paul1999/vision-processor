/*
     Copyright 2025 Paul Bergmann

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
#ifdef DC1394

#include "dc1394driver.h"

#include <algorithm>
#include <dc1394/dc1394.h>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace std::string_literals;

namespace {
	constexpr dc1394operation_mode_t OPERATION_MODE = DC1394_OPERATION_MODE_LEGACY;
	constexpr dc1394speed_t ISO_SPEED = DC1394_ISO_SPEED_400;
	constexpr dc1394video_mode_t VIDEO_MODE = DC1394_VIDEO_MODE_FORMAT7_0;
	constexpr dc1394color_coding_t COLOR_CODING = DC1394_COLOR_CODING_RAW8;
	constexpr dc1394framerate_t FRAMERATE = DC1394_FRAMERATE_60;
	constexpr uint32_t IMAGE_WIDTH = 780;
	constexpr uint32_t IMAGE_HEIGHT = 580;

	struct CameraList {
		CameraList(dc1394_t* dc1394) {
			const auto error = dc1394_camera_enumerate(dc1394, &list);

			if (error != DC1394_SUCCESS) {
				throw std::runtime_error {
					"[DC1394] Could not enumerate cameras: "s + dc1394_error_get_string(error)
				};
			}

			std::sort(list->ids, list->ids + list->num, [](const dc1394camera_id_t& a, const dc1394camera_id_t& b) {
				return a.guid < b.guid;
			});
		}
		~CameraList() {
			dc1394_camera_free_list(list);
		}

		CameraList(const CameraList&) = delete;
		CameraList& operator=(const CameraList&) = delete;

		const dc1394camera_list_t* operator->() const {
			return list;
		}

		dc1394camera_list_t* list = nullptr;
	};

	const char* to_string(const dc1394operation_mode_t operationMode) {
		switch (operationMode) {
			case DC1394_OPERATION_MODE_LEGACY: return "1394A";
			case DC1394_OPERATION_MODE_1394B: return "1394B";
			default: return "Unknown";
		}
	}

	const char* to_string(const dc1394speed_t isoSpeed) {
		switch (isoSpeed) {
			case DC1394_ISO_SPEED_100: return "ISO Speed 100";
			case DC1394_ISO_SPEED_200: return "ISO Speed 200";
			case DC1394_ISO_SPEED_400: return "ISO Speed 400";
			case DC1394_ISO_SPEED_800: return "ISO Speed 800";
			case DC1394_ISO_SPEED_1600: return "ISO Speed 1600";
			case DC1394_ISO_SPEED_3200: return "ISO Speed 3200";
			default: return "Unknown";
		}
	}

	const char* to_string(const dc1394video_mode_t videoMode) {
		switch (videoMode) {
			case DC1394_VIDEO_MODE_FORMAT7_0: return "Format7_0";
			default: return "Unknown";
		}
	}

	const char* to_string(const dc1394color_coding_t colorCoding) {
		switch (colorCoding) {
			case DC1394_COLOR_CODING_RAW8: return "RAW8";
			default: return "Unknown";
		}
	}

	const char* to_string(const dc1394framerate_t framerate) {
		switch (framerate) {
			case DC1394_FRAMERATE_60: return "60";
			default: return "Unknown";
		}
	}
}

DC1394Driver::DC1394Driver(unsigned int id)
	: CameraDriver()
	, dc1394(dc1394_new())
{
	if (!dc1394) {
		throw std::runtime_error { "[DC1394] Could not initialize libdc1394" };
	}

	CameraList cameras { dc1394.get() };

	if (id >= cameras->num) {
		std::stringstream msg;
		msg << "[DC1394] Camera ID " << id << " out of range (" << cameras->num << " available)";
		throw std::out_of_range { msg.str() };
	}

	const dc1394camera_id_t libId = cameras->ids[id];
	std::cout << "[DC1394] Opening camera with GUID " << libId.guid << std::endl;

	camera.reset(dc1394_camera_new(dc1394.get(), libId.guid));

	if (!camera) {
		throw std::runtime_error { "[DC1394] Could not create camera" };
	}

	if (const auto error = dc1394_video_set_operation_mode(camera.get(), OPERATION_MODE); error != DC1394_SUCCESS) {
		std::stringstream msg;
		msg << "[DC1394] Could not set operation mode to " << to_string(OPERATION_MODE) << ": " << dc1394_error_get_string(error);
		throw std::runtime_error { msg.str() };
	}
	std::cout << "[DC1394] Operation mode set to " << to_string(OPERATION_MODE) << std::endl;

	if (const auto error = dc1394_video_set_iso_speed(camera.get(), ISO_SPEED); error != DC1394_SUCCESS) {
		std::stringstream msg;
		msg << "[DC1394] Could not set " << to_string(ISO_SPEED) << ": " << dc1394_error_get_string(error);
		throw std::runtime_error { msg.str() };
	}
	std::cout << "[DC1394] ISO speed set to " << to_string(ISO_SPEED) << std::endl;

	if (const auto error = dc1394_video_set_mode(camera.get(), DC1394_VIDEO_MODE_FORMAT7_0); error != DC1394_SUCCESS) {
		std::stringstream msg;
		msg << "[DC1394] Could not set video mode " << to_string(VIDEO_MODE) << ": " << dc1394_error_get_string(error);
		throw std::runtime_error { msg.str() };
	}
	std::cout << "[DC1394] Video mode set to " << to_string(VIDEO_MODE) << std::endl;

	if (const auto error = dc1394_format7_set_color_coding(camera.get(), DC1394_VIDEO_MODE_FORMAT7_0, COLOR_CODING); error != DC1394_SUCCESS) {
		std::stringstream msg;
		msg << "[DC1394] Could not set color coding to " << to_string(COLOR_CODING) << ": " << dc1394_error_get_string(error);
		throw std::runtime_error { msg.str() };
	}
	std::cout << "[DC1394] Color coding set to " << to_string(COLOR_CODING) << std::endl;

	if (const auto error = dc1394_format7_set_image_size(camera.get(), DC1394_VIDEO_MODE_FORMAT7_0, IMAGE_WIDTH, IMAGE_HEIGHT); error != DC1394_SUCCESS) {
		std::stringstream msg;
		msg << "[DC1394] Could not set image size to " << IMAGE_WIDTH << "x" << IMAGE_HEIGHT << ": " << dc1394_error_get_string(error);
		throw std::runtime_error { msg.str() };
	}
	std::cout << "[DC1394] Image size set to " << IMAGE_WIDTH << "x" << IMAGE_HEIGHT << std::endl;

	if (const auto error = dc1394_video_set_framerate(camera.get(), DC1394_FRAMERATE_60); error != DC1394_SUCCESS) {
		std::stringstream msg;
		msg << "[DC1394] Could not set framerate to " << to_string(FRAMERATE) << ": " << dc1394_error_get_string(error);
		throw std::runtime_error { msg.str() };
	}
	std::cout << "[DC1394] Framerate set to " << to_string(FRAMERATE) << std::endl;

	uint32_t bandwidth;
	if (const auto error = dc1394_video_get_bandwidth_usage(camera.get(), &bandwidth); error != DC1394_SUCCESS) {
		throw std::runtime_error {
			"[DC1394] Could not get bandwidth usage: "s + dc1394_error_get_string(error)
		};
	}
	std::cout << "[DC1394] Bandwidth usage: " << bandwidth << std::endl;

	if (const auto error = dc1394_capture_setup(camera.get(), 4, DC1394_CAPTURE_FLAGS_DEFAULT); error != DC1394_SUCCESS) {
		throw std::runtime_error {
			"[DC1394] Could not setup capture: "s + dc1394_error_get_string(error)
		};
	}
	std::cout << "[DC1394] Capture setup" << std::endl;

	if (const auto error = dc1394_video_set_transmission(camera.get(), DC1394_ON); error != DC1394_SUCCESS) {
		throw std::runtime_error {
			"[DC1394] Could not start transmission: "s + dc1394_error_get_string(error)
		};
	}
	std::cout << "[DC1394] Transmission started" << std::endl;
}

DC1394Driver::~DC1394Driver() {
	if (camera) {
		dc1394_capture_stop(camera.get());
		dc1394_video_set_transmission(camera.get(), DC1394_OFF);
	}
}

std::shared_ptr<RawImage> DC1394Driver::readImage() {
	dc1394video_frame_t* frame;
	if (auto error = dc1394_capture_dequeue(camera.get(), DC1394_CAPTURE_POLICY_WAIT, &frame); error != DC1394_SUCCESS) {
		throw std::runtime_error {
			"[DC1394] Could not dequeue frame: "s + dc1394_error_get_string(error)
		};
	}

	auto img = std::make_shared<RawImage>(&PixelFormat::RGGB8, frame->size[0], frame->size[1], frame->image);

	if (auto error = dc1394_capture_enqueue(camera.get(), frame); error != DC1394_SUCCESS) {
		throw std::runtime_error {
			"[DC1394] Could not enqueue frame: "s + dc1394_error_get_string(error)
		};
	}

	return img;
}

const PixelFormat DC1394Driver::format() {
	return PixelFormat::RGGB8;
}

double DC1394Driver::expectedFrametime() {
	return 1.0 / 49.4;
}

#endif
