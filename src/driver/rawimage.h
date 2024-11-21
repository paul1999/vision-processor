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
#pragma once


#include <string>
#include "opencl.h"


class RawImage : public CLArray {
public:
	RawImage(const RawImage& other) = default;
	RawImage(const PixelFormat* format, int width, int height): CLArray(width * height * format->pixelSize()), format(format), width(width), height(height), timestamp(0), name() {}
	RawImage(const PixelFormat* format, int width, int height, std::string name): CLArray(width * height * format->pixelSize()), format(format), width(width), height(height), timestamp(0), name(std::move(name)) {}
	RawImage(const PixelFormat* format, int width, int height, double timestamp): CLArray(width * height * format->pixelSize()), format(format), width(width), height(height), timestamp(timestamp), name() {}

	//Only use these constructors if not possible otherwise due to necessary copy (because of potential alignment mismatch for zero-copy support)
	RawImage(const PixelFormat* format, int width, int height, unsigned char* data): CLArray(data, width * height * format->pixelSize()), format(format), width(width), height(height), timestamp(0) {}
	RawImage(const PixelFormat* format, int width, int height, double timestamp, unsigned char* data): CLArray(data, width * height * format->pixelSize()), format(format), width(width), height(height), timestamp(timestamp) {}

	virtual ~RawImage() = default;

	const PixelFormat* format;
	const int width;
	const int height;
	// timestamp of 0 indicates unavailability
	const double timestamp;
	const std::string name;
};
