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
#include <opencv2/opencv.hpp>
#include "GeomModel.h"
#include "Distortion.h"
#include "LineDetection.h"
#include "proto/ssl_vision_wrapper.pb.h"

#include <eigen3/unsupported/Eigen/LevenbergMarquardt>
//#include <opencv2/ximgproc.hpp>


float dist(const cv::Vec2f& v1, const cv::Vec2f& v2) {
	cv::Vec2f d = v2-v1;
	return sqrtf(d.dot(d));
}

void visibleFieldExtent(const Resources &r, const bool withBoundary, Eigen::Vector2f &min, Eigen::Vector2f &max) {
	return visibleFieldExtentEstimation(r.camId, r.cameraAmount, r.socket->getGeometry().field(), withBoundary, min, max);
}

Eigen::Vector2f cv2eigen(const cv::Vec2f& v) {
	return {v[0], v[1]};
}

typedef struct LineArc {
	Eigen::Vector2f center;
	float radius;
	float a1, a2;
} LineArc;

static void fieldToLines(const Resources& r, std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>>& lines, std::vector<LineArc>& arcs) {
	const SSL_GeometryFieldSize& field = r.socket->getGeometry().field();

	for(const SSL_FieldLineSegment& line : field.field_lines())
		lines.emplace_back(Eigen::Vector2f(line.p1().x(), line.p1().y()), Eigen::Vector2f(line.p2().x(), line.p2().y()));

	for(const SSL_FieldCircularArc& arc : field.field_arcs()) {
		arcs.push_back({
							   .center = {arc.center().x(), arc.center().y()},
							   .radius = arc.radius(),
							   .a1 = arc.a1(),
							   .a2 = arc.a2()
					   });
	}
}

static float sqPointLineSegmentDistance(const std::pair<Eigen::Vector2f, Eigen::Vector2f>& line, const Eigen::Vector2f& point) {
	//Adapted from Grumdrig https://stackoverflow.com/a/1501725 CC BY-SA 4.0
	const Eigen::Vector2f v = line.second - line.first;
	const Eigen::Vector2f w = point - line.first;
	const float t = std::max(0.0f, std::min(1.0f, w.dot(v) / v.dot(v)));
	const Eigen::Vector2f delta = w - t * v;
	return delta.dot(delta);
}

static float minPointModelDistance(const std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>>& lines, const std::vector<LineArc>& arcs, const Eigen::Vector2f& fieldPixel) {
	float distance = MAXFLOAT;
	for(const auto& line : lines) {
		distance = std::min(distance, sqPointLineSegmentDistance(line, fieldPixel));
	}

	distance = sqrtf(distance);

	for(const auto& arc : arcs) {
		const Eigen::Vector2f pixel2center = fieldPixel - arc.center;
		/*float angle = atan2f(pixel2center.y(), pixel2center.x());
		if(angle < 0)
			angle += 2*M_PI;*/

		//TODO cases outside of angle >= arc.a1 && angle <= arc.a2
		distance = std::min(distance, abs(sqrtf(pixel2center.dot(pixel2center)) - arc.radius));
	}

	return distance;
}

struct DirectGeometryFit : public Eigen::DenseFunctor<float> {
	const std::vector<Eigen::Vector2f>& linePixels;
	const std::vector<std::vector<Eigen::Vector2f>>& mergedPixels;
	const CameraModel& reference;
	const bool calibHeight;

	std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> lines;
	std::vector<LineArc> arcs;
	std::vector<Eigen::Vector2f> modelPoints;

	explicit DirectGeometryFit(const Resources& r, const std::vector<Eigen::Vector2f>& linePixels, const std::vector<std::vector<Eigen::Vector2f>>& mergedPixels, const CameraModel& model, const bool calibHeight): linePixels(linePixels), mergedPixels(mergedPixels), reference(model), calibHeight(calibHeight) {
		fieldToLines(r, lines, arcs);

		float stepSize = 10.f;
		for(const std::pair<Eigen::Vector2f, Eigen::Vector2f>& line : lines) {
			Eigen::Vector2f delta = line.second - line.first;
			int steps = (int)(delta.norm() / stepSize);
			delta /= (float)steps;
			for(int i = 0; i < steps; i++)
				modelPoints.emplace_back(line.first + delta*i);
		}

		for(const LineArc& arc : arcs) {
			float step = 2.f * asinf((stepSize/2.f) / arc.radius); //TODO test with fixed stepping
			for(float i = arc.a1; i <= arc.a2; i += step)
				modelPoints.emplace_back(arc.center + Eigen::Vector2f(cosf(i), sinf(i))*arc.radius);
		}

		Eigen::Vector2f extentMin;
		Eigen::Vector2f extentMax;
		visibleFieldExtent(r, true, extentMin, extentMax);
		modelPoints.erase(std::remove_if(modelPoints.begin(), modelPoints.end(), [&](const auto& item) {
			return item.x() < extentMin.x() || item.x() > extentMax.x() || item.y() < extentMin.y() || item.y() > extentMax.y();
		}), modelPoints.end());
	}

	int operator()(const InputType &x, ValueType& fvec) const {
		CameraModel model = reference;
		model.focalLength = x[0];
		model.updateEuler({x[1], x[2], x[3]});
		model.pos.x() = x[4];
		model.pos.y() = x[5];
		if(calibHeight)
			model.pos.z() = x[6];
		calibrateDistortion(mergedPixels, model);
		model.updateDerived();

		int i = 0;
		for(const Eigen::Vector2f& point : modelPoints) {
			Eigen::Vector2f image = model.field2image({point.x(), point.y(), 0.f});

			float min = MAXFLOAT;
			for(const Eigen::Vector2f& pixel : linePixels) {
				float sqNorm = (pixel - image).squaredNorm();
				if(sqNorm < min)
					min = sqNorm;
			}

			fvec[i++] = min;
		}

		/*for(const Eigen::Vector2f& pixel : linePixels) {
			Eigen::Vector2f field = model.image2field(pixel, 0.0f).head<2>();
			fvec[i++] = minPointModelDistance(lines, arcs, field);
			//fvec[i++] = sqrtf(minPointModelDistance(lines, arcs, field)); //sqrtf-model
		}*/

		return 0;
	}

	int values() const {
		return modelPoints.size();
		//return linePixels.size();
	}
};

struct EdgeGeometryFit : public Eigen::DenseFunctor<float> {
	const std::vector<std::vector<Eigen::Vector2f>>& mergedPixels;
	const std::vector<Eigen::Vector2f>& imageEdges;
	const std::list<Eigen::Vector2f>& fieldEdges;
	const CameraModel& reference;
	const bool calibHeight;

	explicit EdgeGeometryFit(const std::vector<std::vector<Eigen::Vector2f>>& mergedPixels, const std::vector<Eigen::Vector2f>& imageEdges, const std::list<Eigen::Vector2f>& fieldEdges, const CameraModel& model, const bool calibHeight): mergedPixels(mergedPixels), imageEdges(imageEdges), fieldEdges(fieldEdges), reference(model), calibHeight(calibHeight) {}

	int operator()(const InputType &x, ValueType& fvec) const {
		CameraModel model = reference;
		/*model.distortionK2 = x[0];
		model.principalPoint.x() = x[1];
		model.principalPoint.y() = x[2];*/
		model.focalLength = x[0];
		model.updateEuler({x[1], x[2], x[3]});
		model.pos.x() = x[4];
		model.pos.y() = x[5];
		if(calibHeight)
			model.pos.z() = x[6];
		calibrateDistortion(mergedPixels, model);
		model.updateDerived();

		auto iIt = imageEdges.cbegin();
		auto fIt = fieldEdges.cbegin();
		int i = 0;
		while(iIt != imageEdges.cend()) {
			const Eigen::Vector2f& f = *fIt++;
			Eigen::Vector2f error = model.field2image({f.x(), f.y(), 0.0f}) - *iIt++;
			error = error.array()*error.array();
			fvec[i++] = error.x();
			fvec[i++] = error.y();
		}

		/*for(const std::vector<Eigen::Vector2f>& distorted : lines) {
			std::vector<Eigen::Vector2f> undistorted;
			for(const Eigen::Vector2f& d : distorted)
				undistorted.push_back(model.normalizeUndistort(d));

			std::vector<float> error = lineError(undistorted);
			for(float e : error) {
				if(e == NAN)
					return -1;
				fvec(i++) = e;
			}
		}*/
		return 0;
	}

	int values() const {
		int size = 8;

		/*for (const auto& item : lines)
			size += item.size();*/

		return size;
	}
};

static bool pointAtLine(const CameraModel& model, const std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>>& lines, const std::vector<LineArc>& arcs, const float halfLineWidth, const Eigen::Vector2f& linePixel) {
	const float sqHalfLineWidth = halfLineWidth*halfLineWidth;
	const Eigen::Vector2f fieldPixel = model.image2field(linePixel, 0.0f).head<2>();
	for(const auto& line : lines) {
		if(sqPointLineSegmentDistance(line, fieldPixel) <= sqHalfLineWidth)
			return true;
	}

	for(const auto& arc : arcs) {
		const Eigen::Vector2f pixel2center = fieldPixel - arc.center;
		float angle = atan2f(pixel2center.y(), pixel2center.x());
		if(angle < 0)
			angle += 2*M_PI;

		if(abs(sqrtf(pixel2center.dot(pixel2center)) - arc.radius) <= halfLineWidth && angle >= arc.a1 && angle <= arc.a2)
			return true;
	}

	return false;
}

std::vector<Eigen::Vector2f> getLinePixels(const Image& thresholded) {
	std::vector<Eigen::Vector2f> linePixels;
	CLMap<uint8_t> data = thresholded.read<uint8_t>();
	for (int y = 0; y < thresholded.height; y++) {
		for (int x = 0; x < thresholded.width; x++) {
			if(data[x + y * thresholded.width])
				linePixels.emplace_back(x, y);
		}
	}
	return linePixels;
}

int modelError(const Resources& r, const CameraModel& model, const std::vector<Eigen::Vector2f>& linePixels) {
	const SSL_GeometryFieldSize& field = r.socket->getGeometry().field();
	const float halfLineWidth = (float)field.line_thickness() / 2.0f;

	std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> lines;
	std::vector<LineArc> arcs;
	fieldToLines(r, lines, arcs);

	int error = 0.0;
	for(const Eigen::Vector2f& linePixel : linePixels) {
		if(!pointAtLine(model, lines, arcs, halfLineWidth, linePixel))
			error += 1;
	}
	return error;
}

static void thresholdCanny(const Resources& r, const int halfLineWidth, const Image& gray, Image& thresholded) {
	cv::Canny(*gray.cvRead(), *thresholded.cvWrite(), r.fieldLineThreshold/2, r.fieldLineThreshold, halfLineWidth);
}

static void thresholdAdaMeanAnd(const Resources& r, const int halfLineWidth, const Image& bgr, Image& thresholded) {
	std::vector<cv::Mat> split_bgr(3);
	cv::split(*bgr.cvRead(), split_bgr);

	cv::Mat tb, tg, tr;
	cv::adaptiveThreshold(split_bgr[0], tb, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 20*halfLineWidth + 1, -r.fieldLineThreshold);
	cv::adaptiveThreshold(split_bgr[1], tg, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 20*halfLineWidth + 1, -r.fieldLineThreshold);
	cv::adaptiveThreshold(split_bgr[2], tr, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 20*halfLineWidth + 1, -r.fieldLineThreshold);

	{
		const CLMap<uint8_t> data = bgr.read<uint8_t>();
		CLMap<uint8_t> tData = thresholded.write<uint8_t>();
		for (int y = 0; y < bgr.height; y++) {
			for (int x = 0; x < bgr.width; x++) {
				int pos = x + y * bgr.width;
				tData[pos] = (tb.data[pos] && tb.data[pos+1] && tb.data[pos+2]) ? 255 : 0;
			}
		}
	}
}

static void thresholdAdaMedianOtsu(const int halfLineWidth, const Image& gray, Image& thresholded) {
	const int halfThresholdWidth = halfLineWidth*4+1;
	cv::Mat median;
	cv::medianBlur(*gray.cvRead(), median, 2*halfThresholdWidth+1);
	Image adaMedian(&PixelFormat::I8, gray.width, gray.height, gray.name);
	cv::subtract(*gray.cvRead(), median, *adaMedian.cvWrite(), cv::noArray(), CV_8SC1);
	{
		CVMap map = adaMedian.cvReadWrite();
		cv::max(*map, 0, *map);
		cv::Mat u8;
		map->convertTo(u8, CV_8U);
		//cv::Mat thresh;
		cv::threshold(u8, *thresholded.cvWrite(), 0.0, 255.0, cv::THRESH_BINARY + cv::THRESH_OTSU);
		//cv::ximgproc::thinning(thresh, *t.cvWrite());
	}
}

static void fieldDetection(const int halfLineWidth, const Image& img, Image& thresholded) {
	std::vector<cv::Mat> bgr(3);
	cv::split(*img.cvRead(), bgr);

	const int halfThresholdWidth = halfLineWidth*4+1;
	cv::Mat median(img.height, img.width, CV_32FC1, 0.0f);
	for(const cv::Mat& channel : bgr) {
		cv::Mat channelmedian, dx, dy, mag, angle;
		cv::medianBlur(channel, channelmedian, 2*halfThresholdWidth+1);
		//TODO not entirely optimal
		//cv::blur(channelmedian, channelmedian, {3, 3}, {-1, -1}, cv::BORDER_REPLICATE);
		cv::spatialGradient(channelmedian, dx, dy, 3, cv::BORDER_REPLICATE);
		dx.convertTo(dx, CV_32F);
		dy.convertTo(dy, CV_32F);
		cv::cartToPolar(dx, dy, mag, angle);
		median += mag;
	}

	cv::Mat u8;
	median.convertTo(u8, CV_8U);
	cv::threshold(u8, *thresholded.cvWrite(), 0.0, 255.0, cv::THRESH_BINARY + cv::THRESH_OTSU);
	//median.convertTo(*thresholded.cvWrite(), CV_8U);
}

static void thresholdAdaMedianCanny(const int halfLineWidth, const Image& gray, Image& thresholded) {
	const int halfThresholdWidth = halfLineWidth*4+1;
	cv::Mat median;
	cv::medianBlur(*gray.cvRead(), median, 2*halfThresholdWidth+1);
	Image adaMedian(&PixelFormat::I8, gray.width, gray.height, gray.name);
	cv::subtract(*gray.cvRead(), median, *adaMedian.cvWrite(), cv::noArray(), CV_8SC1);
	{
		CVMap map = adaMedian.cvReadWrite();
		cv::max(*map, 0, *map);
		cv::Mat u8;
		map->convertTo(u8, CV_8U);
		CVMap tmap = thresholded.cvWrite();
		cv::threshold(u8, *tmap, 0.0, 255.0, cv::THRESH_BINARY + cv::THRESH_OTSU);
		//cv::ximgproc::thinning(thresh, *t.cvWrite());

		for(int y = 0; y < thresholded.height; y++) {
			for(int x = 0; x < thresholded.width; x++) {
				if(map->at<uint8_t>(y, x) > 1 && tmap->at<uint8_t>(y, x) == 0)
					tmap->at<uint8_t>(y, x) = 128;
			}
		}

		int changes = 1;
		while(changes > 0) {
			changes = 0;
			for(int y = 1; y < thresholded.height-1; y++) {
				for(int x = 1; x < thresholded.width-1; x++) {
					if(tmap->at<uint8_t>(y, x) == 128 && (tmap->at<uint8_t>(y-1, x) == 255 || tmap->at<uint8_t>(y, x-1) == 255 || tmap->at<uint8_t>(y, x+1) == 255 || tmap->at<uint8_t>(y+1, x) == 255)) {
						tmap->at<uint8_t>(y, x) = 255;
						changes++;
					}
				}
			}
		}

		for(int y = 0; y < thresholded.height; y++) {
			for(int x = 0; x < thresholded.width; x++) {
				if(tmap->at<uint8_t>(y, x) == 128)
					tmap->at<uint8_t>(y, x) = 0;
			}
		}
	}
}

static void drawModel(const Resources& r, Image& thresholded, const std::vector<Eigen::Vector2f>& linePixels, const CameraModel& model) {
	std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> lines;
	std::vector<LineArc> arcs;
	fieldToLines(r, lines, arcs);

	const float halfLineWidth = (float)r.socket->getGeometry().field().line_thickness() / 2.0f;
	CLMap<uint8_t> data = thresholded.write<uint8_t>();
	for(const Eigen::Vector2f& linePixel : linePixels) {
		data[(int)linePixel.x() + (int)linePixel.y()*thresholded.width] = pointAtLine(model, lines, arcs, halfLineWidth, linePixel) ? 255 : 128;
	}
}


bool isClockwiseConvexQuadrilateral(const std::vector<Eigen::Vector2f>& vertexlist) {
	//Convexity check adapted from https://math.stackexchange.com/a/1745427 by Nominal Animal under CC BY-SA 3.0
	//Clockwise check adapted from https://stackoverflow.com/a/1165943 by Roberteo Bonvallet under CC BY-SA 3.0
	float clockwise = 0;

	float wSign = 0;

	int xSign = 0;
	int xFirstSign = 0;
	int xFlips = 0;

	int ySign = 0;
	int yFirstSign = 0;
	int yFlips = 0;

	Eigen::Vector2f curr = vertexlist.back(); //unused
	Eigen::Vector2f next = vertexlist.back();

	for(const Eigen::Vector2f& v : vertexlist) {
		Eigen::Vector2f prev = curr;
		curr = next;
		next = v;

		Eigen::Vector2f b = curr - prev;
		Eigen::Vector2f a = next - curr;

		clockwise += a[0] * (next[1]+curr[1]);

		if (a[0] > 0) {
			if (xSign == 0)
				xFirstSign = 1;
			else if (xSign < 0)
				xFlips++;

			xSign = 1;
		} else if (a[0] < 0) {
			if (xSign == 0)
				xFirstSign = -1;
			else if (xSign > 0)
				xFlips++;

			xSign = -1;
		}

		if (xFlips > 2)
			return false;

		if (a[1] > 0) {
			if (ySign == 0)
				yFirstSign = 1;
			else if (ySign < 0)
				yFlips++;

			ySign = 1;
		} else if (a[1] < 0) {
			if (ySign == 0)
				yFirstSign = -1;
			else if (ySign > 0)
				yFlips++;

			ySign = -1;
		}

		if (yFlips > 2)
			return false;

		float w = b[0]*a[1] - a[0]*b[1];
		if (wSign == 0 && w != 0)
			wSign = w;
		else if ((wSign > 0 && w < 0) || (wSign < 0 && w > 0))
			return false;
	}

	if (xSign != 0 && xFirstSign != 0 && xSign != xFirstSign)
		xFlips++;
	if (ySign != 0 && yFirstSign != 0 && ySign != yFirstSign)
		yFlips++;

	if (xFlips != 2 || yFlips != 2)
		return false;

	return clockwise < 0;
}

static void directMixedCalibration(const Resources& r, const std::vector<std::vector<Eigen::Vector2f>>& mergedPixels, const std::vector<Eigen::Vector2f>& linePixels, bool calibHeight, CameraModel& model) {
	DirectGeometryFit functor(r, linePixels, mergedPixels, model, calibHeight);
	Eigen::NumericalDiff<DirectGeometryFit> numDiff(functor);
	Eigen::LevenbergMarquardt<Eigen::NumericalDiff<DirectGeometryFit>> lm(numDiff);

	Eigen::VectorXf k(calibHeight ? 7 : 6);
	k[0] = model.focalLength;
	Eigen::Vector3f euler = model.getEuler();
	k[1] = euler.x();
	k[2] = euler.y();
	k[3] = euler.z();
	k[4] = model.pos.x();
	k[5] = model.pos.y();
	if(calibHeight)
		k[6] = model.pos.z();

	lm.minimize(k);

	if(lm.info() != Eigen::ComputationInfo::Success && lm.info() != Eigen::ComputationInfo::NoConvergence) { //xtol might be too aggressive
		std::cerr << "[Geometry calibration] Unable to find matching field model, aborting calibration for this frame. (lm.info() no success)" << std::endl;
		return;
	}

	if(calibHeight && k[6] < 0) { // camera below field
		std::cerr << "[Geometry calibration] Unable to find matching field model, aborting calibration for this frame. (camera below field)" << std::endl;
		return;
	}

	model.focalLength = k[0];
	model.updateEuler({k[1], k[2], k[3]});
	model.pos.x() = k[4];
	model.pos.y() = k[5];
	if(calibHeight)
		model.pos.z() = k[6];

	if(model.focalLength < 0) {
		model.focalLength = -k[0];
		model.f2iOrientation = Eigen::AngleAxisf(M_PI_2, Eigen::Vector3f::UnitZ()) * model.f2iOrientation;
	}
}

static bool edgeCalibration(const Resources& r, const std::vector<std::vector<Eigen::Vector2f>>& mergedPixels, const std::vector<Eigen::Vector2f>& imageEdges, bool calibHeight, CameraModel& basicModel) {
	Eigen::Vector2f extentMin;
	Eigen::Vector2f extentMax;
	visibleFieldExtent(r, false, extentMin, extentMax);
	std::list<Eigen::Vector2f> fieldEdges;
	fieldEdges.emplace_back(extentMin[0], extentMax[1]);
	fieldEdges.emplace_back(extentMax[0], extentMax[1]);
	fieldEdges.emplace_back(extentMax[0], extentMin[1]);
	fieldEdges.emplace_back(extentMin[0], extentMin[1]);
	std::cout << "[Geometry calibration] Selecting field edges: ";
	for(const Eigen::Vector2f& edge : fieldEdges)
		std::cout << edge.transpose() << " ";
	std::cout << std::endl;

	float minFastError = INFINITY;
	int minError = INT_MAX;
	CameraModel minModel;
	std::vector<Eigen::Vector2f> minEdges;

	std::vector<Eigen::Vector2f> edges(4);
	/*for(const Eigen::Vector2f& a : intersections) {
		for(const Eigen::Vector2f& b : intersections) {
			for(const Eigen::Vector2f& c : intersections) {
				for(const Eigen::Vector2f& d : intersections) {
					edges[0] = a;
					edges[1] = b;
					edges[2] = c;
					edges[3] = d;
					if(!isClockwiseConvexQuadrilateral(edges))
						continue;

					//TODO calibrate model, evaluate score
					for(int orientation = 0; orientation < 8; orientation++) {
						CameraModel model = basicModel;
						EdgeGeometryFit functor(mergedPixels, edges, fieldEdges, model, calibHeight);
						Eigen::NumericalDiff<EdgeGeometryFit> numDiff(functor);
						Eigen::LevenbergMarquardt<Eigen::NumericalDiff<EdgeGeometryFit>> lm(numDiff);

						Eigen::VectorXf k(calibHeight ? 7 : 6); //10 : 9
						/*k[0] = model.distortionK2;
						k[1] = model.principalPoint.x();
						k[2] = model.principalPoint.y();*/
	/*
						k[0] = model.focalLength;
						Eigen::Vector3f euler = model.getEuler();
						k[1] = euler.x();
						k[2] = euler.y();
						k[3] = euler.z();
						k[4] = model.pos.x();
						k[5] = model.pos.y();
						if(calibHeight)
							k[6] = model.pos.z();

						lm.minimize(k);

						if(lm.info() != Eigen::ComputationInfo::Success && lm.info() != Eigen::ComputationInfo::NoConvergence) //xtol might be too aggressive
							continue;

						if(calibHeight && k[6] < 0) // camera below field
							continue;

						/*model.distortionK2 = k[0];
						model.principalPoint.x() = k[1];
						model.principalPoint.y() = k[2];*/
						/*
						model.focalLength = k[0];
						model.updateEuler({k[1], k[2], k[3]});
						model.pos.x() = k[4];
						model.pos.y() = k[5];
						if(calibHeight)
							model.pos.z() = k[6];

						if(model.focalLength < 0) {
							model.focalLength = -k[0];
							model.f2iOrientation = Eigen::AngleAxisf(M_PI_2, Eigen::Vector3f::UnitZ()) * model.f2iOrientation;
						}

						calibrateDistortion(mergedPixels, model);
						model.updateDerived();

						//TODO use modelError to refine model further
						int error = modelError(r, model, linePixels);
						if(error < minError) {
							minError = error;
							minModel = model;
							minEdges = edges;
						}

						std::rotate(edges.begin(), std::next(edges.begin()), edges.end());
					}
				}
			}
		}
	}*/

	if(minError == INT_MAX) {
		std::cerr << "[Geometry calibration] Unable to find matching field model, aborting calibration for this frame." << std::endl;
		return false;
	}

	basicModel = minModel;
	return true;
}

void geometryCalibration(const Resources& r, const Image& img) {
	// Adapted from https://stackoverflow.com/a/25436112 by user2398029 under CC BY-SA 3.0
	// J. Immerkær, “Fast Noise Variance Estimation”, Computer Vision and Image Understanding, Vol. 64, No. 2, pp. 300-302, Sep. 1996
	Image gray = img.toGrayscale();
	cv::Mat noiseKernel = (cv::Mat_<float>(3, 3) << 1, -2, 1, -2, 4, -2, 1, -2, 1);
	cv::Mat noise;
	cv::filter2D(*gray.cvRead(), noise, -1, noiseKernel);
	float sigma = (float)cv::sum(cv::abs(noise))[0] * sqrtf(0.5f * M_PI) / (6 * gray.width * gray.height);
	std::cout << "[Geometry calibration] Noise sigma: " << sigma << std::endl;
	//0.05 (schubert) - 0.3 (default) - 0.45 (rc19)

	Image bgr = img.toBGR();
	//const int halfLineWidth = halfLineWidthEstimation(r, bgr);
	const int halfLineWidth = halfLineWidthEstimation(r, gray);
	std::cout << "[Geometry calibration] Half line width: " << halfLineWidth << std::endl;
	Image thresholded(&PixelFormat::U8, gray.width, gray.height, gray.name);

	//thresholdCanny(r, halfLineWidth, gray, thresholded);
	//thresholdImage(r, gray, halfLineWidth, thresholded);
	//thresholdAdaMeanAnd(r, halfLineWidth, bgr, thresholded);
	//thresholdAdaMedianOtsu(halfLineWidth, gray, thresholded);
	thresholdAdaMedianCanny(halfLineWidth, gray, thresholded);
	thresholded.save(".pixels.png");

	//https://docs.opencv.org/4.x/da/d7f/tutorial_back_projection.html

	//Direct calibration

	//Intrinsic line calibration
	//Extrinsic edge calibration
	// Handgiven
	// Catesian product
	// Horizontal/Vertical separated

	cv::Ptr<cv::LineSegmentDetector> detector = cv::createLineSegmentDetector();
	cv::Mat4f linesMat;
	detector->detect(*thresholded.cvRead(), linesMat);

	{
		CVMap cvBgr = bgr.cvReadWrite();
		detector->drawSegments(*cvBgr, linesMat);
	}
	//bgr.save(".linesegments.png");

	CVLines lines;
	for(int i = 0; i < linesMat.rows; i++) {
		cv::Vec4f line = linesMat(0, i);
		cv::Vec2f a = cv::Vec2f(line[0], line[1]);
		cv::Vec2f b = cv::Vec2f(line[2], line[3]);
		if(dist(a, b) >= r.minLineSegmentLength)
			lines.emplace_back(a, b);
	}
	std::cout << "[Geometry calibration] Line segments: " << lines.size() << std::endl;

	const std::vector<CVLines> compoundLines = groupLineSegments(r, lines);
	const CVLines mergedLines = mergeLineSegments(compoundLines);
	std::cout << "[Geometry calibration] Lines: " << mergedLines.size() << std::endl;

	std::vector<std::vector<Eigen::Vector2f>> mergedPixels(mergedLines.size());
	{
		const float sqHalfLineWidth = (float)(halfLineWidth*halfLineWidth);
		CLMap<uint8_t> data = thresholded.read<uint8_t>();
		for (int y = 0; y < thresholded.height; y++) {
			for (int x = 0; x < thresholded.width; x++) {
				if(data[x + y * thresholded.width]) {
					for(unsigned int i = 0; i < compoundLines.size(); i++) {
						for(const auto& segment : compoundLines[i]) {
							if(sqPointLineSegmentDistance(std::make_pair(cv2eigen(segment.first), cv2eigen(segment.second)), {x, y}) <= sqHalfLineWidth) {
								//TODO duplicate pixels at line segment border
								mergedPixels[i].emplace_back(x, y);
							}
						}
					}
				}
			}
		}
	}

	{
		CVMap cvBgr = bgr.cvReadWrite();
		for (const auto& item : mergedLines) {
			cv::line(*cvBgr, {item.first}, {item.second}, CV_RGB(0, 255, 0));
		}
		/*for (const auto& item : majorLines) {
			cv::line(*cvBgr, {item.first}, {item.second}, CV_RGB(0, 0, 255));
		}*/
	}
	bgr.save(".lines.png");

	const bool calibHeight = r.cameraHeight == 0.0;
	CameraModel model({thresholded.width, thresholded.height}, r.camId, r.cameraAmount, (float)r.cameraHeight, r.socket->getGeometry().field());
	/*if(!calibrateDistortion(mergedPixels, model))
		return;*/

	const std::vector<Eigen::Vector2f> linePixels = getLinePixels(thresholded);
	drawModel(r, thresholded, linePixels, model);
	//thresholded.save(".initial.png");

	//directMixedCalibration(r, mergedPixels, linePixels, calibHeight, model);

	model.updateDerived();

	//TODO use modelError to refine model further
	int error = modelError(r, model, linePixels);

	std::cout << "[Geometry calibration] Best model: " << model << " error " << (error/(float)linePixels.size()) << std::endl;

	SSL_WrapperPacket wrapper;
	wrapper.mutable_geometry()->CopyFrom(r.socket->getGeometry());
	wrapper.mutable_geometry()->add_calib()->CopyFrom(model.getProto(r.camId));
	//r.socket->send(wrapper);

	drawModel(r, thresholded, linePixels, model);
	//thresholded.save(".fieldlines.png");
	//thresholded.save(".otsu.png");
}