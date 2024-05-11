#pragma once

#include "Resources.h"


typedef std::pair<cv::Vec2f, cv::Vec2f> CVLine;
typedef std::vector<std::pair<cv::Vec2f, cv::Vec2f>> CVLines;

/* Estimates the half line width without camera model from the field size and camera amount */
int halfLineWidthEstimation(const Resources& r, const Image& img);

/* Finds line points by detecting ridges */
Image thresholdImage(const Resources& r, const Image& gray, int halfLineWidth);

/* Groups neighbouring line segments together */
std::vector<CVLines> groupLineSegments(const Resources& r, CVLines& segments);

/* Merges a list of grouped line segments to a list of lines */
CVLines mergeLineSegments(const std::vector<CVLines>& compoundLines);

/* Determines the intersection point of two lines */
cv::Vec2f lineLineIntersection(const CVLine& a, const CVLine& b);

/* Finds all intersections of the given line list inside or next to the image. */
std::vector<cv::Vec2f> lineIntersections(const CVLines& lines, int width, int height, double maxIntersectionDistance);

/* Determines the quadrilateral with the largest area for the given point set */
std::list<Eigen::Vector2f> findOuterEdges(const std::vector<cv::Vec2f>& intersections);
