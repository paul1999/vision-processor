#pragma once

#include "image.h"
#include "Resources.h"

float dist(const cv::Vec2f& v1, const cv::Vec2f& v2);
void visibleFieldExtent(const Resources &r, bool withBoundary, Eigen::Vector2f &min, Eigen::Vector2f &max);
Eigen::Vector2f cv2eigen(const cv::Vec2f& v);

void geometryCalibration(const Resources& r, const Image& img);
