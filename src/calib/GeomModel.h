#pragma once

#include "image.h"
#include "Resources.h"

float dist(const cv::Vec2f& v1, const cv::Vec2f& v2);
void visibleFieldExtent(const Resources &r, bool withBoundary, Eigen::Vector2f &min, Eigen::Vector2f &max);
Eigen::Vector2f cv2eigen(const cv::Vec2f& v);

std::vector<Eigen::Vector2f> getLinePixels(const Image& thresholded);
int modelError(const Resources& r, const CameraModel& model, const std::vector<Eigen::Vector2f>& linePixels);

void geometryCalibration(const Resources& r, const Image& img);
