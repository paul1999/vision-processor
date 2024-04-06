#pragma once

#include "image.h"
#include "Resources.h"

float dist(const cv::Vec2f& v1, const cv::Vec2f& v2);

void geometryCalibration(const Resources& r, const Image& img);
