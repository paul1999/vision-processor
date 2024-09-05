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
#include "GeomModel.h"
#include "Distortion.h"
#include "LineDetection.h"
#include "proto/ssl_vision_wrapper.pb.h"

#include <eigen3/unsupported/Eigen/LevenbergMarquardt>
#include <opencv2/ximgproc.hpp>


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

struct GeometryFit : public Eigen::DenseFunctor<float> {
	const std::vector<Eigen::Vector2f>& linePixels;
	const std::vector<std::vector<Eigen::Vector2f>>& mergedPixels;
	const CameraModel& reference;
	const bool calibHeight;

	std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> lines;
	std::vector<LineArc> arcs;
	std::vector<Eigen::Vector2f> modelPoints;

	explicit GeometryFit(const Resources& r, const std::vector<Eigen::Vector2f>& linePixels, const std::vector<std::vector<Eigen::Vector2f>>& mergedPixels, const CameraModel& model, const bool calibHeight): linePixels(linePixels), mergedPixels(mergedPixels), reference(model), calibHeight(calibHeight) {
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

	const int halfThresholdWidth = halfLineWidth*4+1;
	/*cv::Mat integral;
	cv::integral(*gray.cvRead(), integral, CV_32S);
	cv::Mat paddedIntegral;
	cv::copyMakeBorder(integral, paddedIntegral, halfThresholdWidth, halfThresholdWidth, halfThresholdWidth, halfThresholdWidth, cv::BORDER_REPLICATE);
	Image adaThreshold(&PixelFormat::I8, gray.width, gray.height, gray.name);
	{
		CLMap<uint8_t> read = gray.read<uint8_t>();
		CLMap<int8_t> write = adaThreshold.write<int8_t>();
		for(int y = 0; y < gray.height; y++) {
			for(int x = 0; x < gray.width; x++) {
				write[x + y*gray.width] = cv::saturate_cast<std::int8_t>(read[x + y*gray.width] - (paddedIntegral.at<int>(y + 2*halfThresholdWidth, x + 2*halfThresholdWidth) - paddedIntegral.at<int>(y, x + 2*halfThresholdWidth) - paddedIntegral.at<int>(y + 2*halfThresholdWidth, x) + paddedIntegral.at<int>(y, x)) / (2*halfThresholdWidth * 2*halfThresholdWidth));
			}
		}
	}
	adaThreshold.save(".adathres.png");*/

	/*std::vector<cv::Mat> split_bgr(3);
	cv::split(*bgr.cvRead(), split_bgr);
	for(int i = 0; i < split_bgr.size(); i++) {
		cv::Mat median;
		cv::medianBlur(split_bgr[i], median, 2*halfThresholdWidth+1);
		Image adaMedian(&PixelFormat::I8, gray.width, gray.height, gray.name);
		cv::subtract(split_bgr[i], median, *adaMedian.cvWrite(), cv::noArray(), CV_8SC1);
		adaMedian.save(".adamedian" + std::to_string(i) + ".png");
	}*/

	cv::Mat median;
	cv::medianBlur(*gray.cvRead(), median, 2*halfThresholdWidth+1);
	Image adaMedian(&PixelFormat::I8, gray.width, gray.height, gray.name);
	cv::subtract(*gray.cvRead(), median, *adaMedian.cvWrite(), cv::noArray(), CV_8SC1);
	/*cv::Mat lut(1, 256, CV_8S);
	for(int i = 0; i < 128; i++)
		lut.at<int8_t>(i) = cv::saturate_cast<int8_t>(powf(i / 128.f, 0.641f) * 128.f);
	for(int i = 0; i < 128; i++)
		lut.at<int8_t>(i + 128) = 0;//cv::saturate_cast<int8_t>(powf((i - 128) / 128.f, 1/0.641f) * 128.f);
	{
		CVMap map = adaMedian.cvReadWrite();
		cv::LUT(*map, lut, *map);
	}*/
	//adaMedian.save(".adamedian.png");
	Image thresholded(&PixelFormat::U8, gray.width, gray.height, gray.name);
	{
		CVMap map = adaMedian.cvReadWrite();
		cv::max(*map, 0, *map);
		cv::Mat u8;
		map->convertTo(u8, CV_8U);
		//cv::Mat thresh;
		cv::threshold(u8, *thresholded.cvWrite(), 0.0, 255.0, cv::THRESH_BINARY + cv::THRESH_OTSU);
		//cv::ximgproc::thinning(thresh, *t.cvWrite());
	}
	//return;

	//https://docs.opencv.org/4.x/da/d7f/tutorial_back_projection.html

	/*std::vector<cv::Mat> split_bgr(3);
	cv::split(*bgr.cvRead(), split_bgr);

	cv::Mat tb, tg, tr;
	cv::adaptiveThreshold(split_bgr[0], tb, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 20*halfLineWidth + 1, -r.fieldLineThreshold);
	cv::adaptiveThreshold(split_bgr[1], tg, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 20*halfLineWidth + 1, -r.fieldLineThreshold);
	cv::adaptiveThreshold(split_bgr[2], tr, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 20*halfLineWidth + 1, -r.fieldLineThreshold);

	Image thresholded(&PixelFormat::U8, bgr.width, bgr.height, bgr.name);
	{
		const CLMap<uint8_t> data = bgr.read<uint8_t>();
		CLMap<uint8_t> tData = thresholded.write<uint8_t>();
		for (int y = 0; y < bgr.height; y++) {
			for (int x = 0; x < bgr.width; x++) {
				int pos = x + y * bgr.width;
				tData[pos] = (tb.data[pos] && tb.data[pos+1] && tb.data[pos+2]) ? 255 : 0;
			}
		}
	}*/
	//Image thresholded = thresholdImage(r, gray, halfLineWidth);
	//thresholded.save(".fieldlines.png");

	cv::Ptr<cv::LineSegmentDetector> detector = cv::createLineSegmentDetector();
	cv::Mat4f linesMat;
	detector->detect(*thresholded.cvRead(), linesMat);

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

	CameraModel basicModel({thresholded.width, thresholded.height}, r.camId, r.cameraAmount, (float)r.cameraHeight, r.socket->getGeometry().field());
	/*if(!calibrateDistortion(mergedPixels, basicModel))
		return;*/

	const std::vector<Eigen::Vector2f> linePixels = getLinePixels(thresholded);


	//CameraModel basicModel({thresholded.width, thresholded.height}, r.camId, r.cameraAmount, (float)r.cameraHeight, r.socket->getGeometry().field());
	const bool calibHeight = r.cameraHeight == 0.0;

	{
		const SSL_GeometryFieldSize& field = r.socket->getGeometry().field();
		const float halfLineWidth = (float)field.line_thickness() / 2.0f;

		std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> lines;
		std::vector<LineArc> arcs;
		fieldToLines(r, lines, arcs);

		CLMap<uint8_t> data = thresholded.write<uint8_t>();
		for(const Eigen::Vector2f& linePixel : linePixels) {
			data[(int)linePixel.x() + (int)linePixel.y()*thresholded.width] = pointAtLine(basicModel, lines, arcs, halfLineWidth, linePixel) ? 255 : 128;
		}
	}
	//thresholded.save(".initial.png");

	GeometryFit functor(r, linePixels, mergedPixels, basicModel, calibHeight);
	Eigen::NumericalDiff<GeometryFit> numDiff(functor);
	Eigen::LevenbergMarquardt<Eigen::NumericalDiff<GeometryFit>> lm(numDiff);

	Eigen::VectorXf k(calibHeight ? 7 : 6);
	k[0] = basicModel.focalLength;
	Eigen::Vector3f euler = basicModel.getEuler();
	k[1] = euler.x();
	k[2] = euler.y();
	k[3] = euler.z();
	k[4] = basicModel.pos.x();
	k[5] = basicModel.pos.y();
	if(calibHeight)
		k[6] = basicModel.pos.z();

	lm.minimize(k);

	if(lm.info() != Eigen::ComputationInfo::Success && lm.info() != Eigen::ComputationInfo::NoConvergence) { //xtol might be too aggressive
		std::cerr << "[Geometry calibration] Unable to find matching field model, aborting calibration for this frame. (lm.info() no success)" << std::endl;
		return;
	}

	if(calibHeight && k[6] < 0) { // camera below field
		std::cerr << "[Geometry calibration] Unable to find matching field model, aborting calibration for this frame. (camera below field)" << std::endl;
		return;
	}

	CameraModel model = basicModel;
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

	model.updateDerived();

	//TODO use modelError to refine model further
	int error = modelError(r, model, linePixels);

	std::cout << "[Geometry calibration] Best model: " << model << " error " << (error/(float)linePixels.size()) << std::endl;

	SSL_WrapperPacket wrapper;
	wrapper.mutable_geometry()->CopyFrom(r.socket->getGeometry());
	wrapper.mutable_geometry()->add_calib()->CopyFrom(model.getProto(r.camId));
	r.socket->send(wrapper);

	{
		const SSL_GeometryFieldSize& field = r.socket->getGeometry().field();
		const float halfLineWidth = (float)field.line_thickness() / 2.0f;

		std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> lines;
		std::vector<LineArc> arcs;
		fieldToLines(r, lines, arcs);

		CLMap<uint8_t> data = thresholded.write<uint8_t>();
		for(const Eigen::Vector2f& linePixel : linePixels) {
			data[(int)linePixel.x() + (int)linePixel.y()*thresholded.width] = pointAtLine(model, lines, arcs, halfLineWidth, linePixel) ? 255 : 128;
		}
	}
	//thresholded.save(".fieldlines.png");
	thresholded.save(".otsu.png");
}