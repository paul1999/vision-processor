#pragma once

#include <memory>
#include "image.h"


class VideoSource {
public:
	virtual ~VideoSource() = default;

	virtual std::shared_ptr<Image> readImage() = 0;
};
