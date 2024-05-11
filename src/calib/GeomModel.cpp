#include "GeomModel.h"
#include "Distortion.h"
#include "LineDetection.h"

#include <eigen3/unsupported/Eigen/LevenbergMarquardt>


float dist(const cv::Vec2f& v1, const cv::Vec2f& v2) {
	cv::Vec2f d = v2-v1;
	return sqrtf(d.dot(d));
}

void visibleFieldExtent(const Resources &r, const bool withBoundary, Eigen::Vector2f &min, Eigen::Vector2f &max) {
	return visibleFieldExtent(r.camId, r.cameraAmount, r.socket->getGeometry().field(), withBoundary, min, max);
}

Eigen::Vector2f cv2eigen(const cv::Vec2f& v) {
	return {v[0], v[1]};
}

static cv::Vec2f eigen2cv(const Eigen::Vector2f& v) {
	return {v[0], v[1]};
}

static bool estimateDistortion(const std::vector<CVLines>& compoundLines, CameraModel& model) {
	std::vector<std::vector<Eigen::Vector2f>> l;
	for(const auto& compound : compoundLines) {
		if(compound.size() == 1)
			continue;

		std::vector<Eigen::Vector2f> points;
		for(const auto& segment : compound) {
			points.emplace_back(cv2eigen(segment.first));
			points.emplace_back(cv2eigen(segment.second));
		}
		l.push_back(points);
	}

	return calibrateDistortion(l, model);
}

static cv::Vec2f undistort(const CameraModel& model, const cv::Vec2f& v) {
	return eigen2cv(model.undistort(cv2eigen(v)));
}

struct GeometryFit : public Eigen::DenseFunctor<float> {
	const std::list<Eigen::Vector2f>& imageEdges;
	const std::list<Eigen::Vector2f>& fieldEdges;
	const CameraModel& reference;
	const bool calibHeight;

	explicit GeometryFit(const std::list<Eigen::Vector2f>& imageEdges, const std::list<Eigen::Vector2f>& fieldEdges, const CameraModel& model, const bool calibHeight): imageEdges(imageEdges), fieldEdges(fieldEdges), reference(model), calibHeight(calibHeight) {}

	int operator()(const InputType &x, ValueType& fvec) const {
		CameraModel model = reference;
		model.updateFocalLength(x[0]);
		model.updateEuler({x[1], x[2], x[3]});
		model.pos.x() = x[4];
		model.pos.y() = x[5];
		if(calibHeight)
			model.pos.z() = x[6];
		model.updateDerived();

		auto iIt = imageEdges.cbegin();
		auto fIt = fieldEdges.cbegin();
		int i = 0;
		while(iIt != imageEdges.cend()) {
			const Eigen::Vector2f& f = *fIt++;
			Eigen::Vector2f error = model.field2image({f.x(), f.y(), 0.0f}) - *iIt++;
			//Eigen::Vector2f error = model.image2field(*iIt++, 0.0f).head<2>() - *(fIt++);
			error = error.array()*error.array();
			fvec[i++] = error.x();
			fvec[i++] = error.y();
		}
		return 0;
	}

	int values() const {
		return 8;
	}
};

static bool pointOnLine(const CameraModel& model, const std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>>& lines, const float halfLineWidth, const Eigen::Vector2f& linePixel) {
	const Eigen::Vector2f fieldPixel = model.image2field(linePixel, 0.0f).head<2>();
	for(const auto& line : lines) {
		const Eigen::Vector2f v = line.second - line.first;
		const Eigen::Vector2f line2pixel = fieldPixel - line.first;
		//TODO in segment?
		if(abs(v.x()*line2pixel.y() - v.y()*line2pixel.x()) / sqrtf(v.dot(v)) < halfLineWidth)
			return true;
	}

	return false;
}

static int modelError(const Resources& r, const CameraModel& model, const std::vector<Eigen::Vector2f>& linePixels) {
	const SSL_GeometryFieldSize& field = r.socket->getGeometry().field();
	const float halfLineWidth = (float)field.line_thickness() / 2.0f;

	std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> lines;
	for(const SSL_FieldLineSegment& line : field.field_lines())
		lines.emplace_back(Eigen::Vector2f(line.p1().x(), line.p1().y()), Eigen::Vector2f(line.p2().x(), line.p2().y()));

	int error = 0.0;
	for(const Eigen::Vector2f& linePixel : linePixels) {
		if(!pointOnLine(model, lines, halfLineWidth, linePixel))
			error += 1;

		//TODO arcs
	}
	return error;
}

void geometryCalibration(const Resources& r, const Image& img) {
	Image gray = img.toGrayscale();
	const int halfLineWidth = halfLineWidthEstimation(r, gray);
	std::cout << "[Geometry calibration] Half line width: " << halfLineWidth << std::endl;

	Image thresholded = thresholdImage(r, gray, halfLineWidth);

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

	CameraModel distortionModel({thresholded.width, thresholded.height}, r.camId, r.cameraAmount, r.socket->getGeometry().field());
	if(!estimateDistortion(compoundLines, distortionModel))
		return;

	CVLines majorLines;
	const float minMajorLength = std::min(img.width, img.height) * r.minMajorLineLength;
	for(const auto& line : mergedLines) {
		if(dist(line.first, line.second) >= minMajorLength)
			majorLines.emplace_back(undistort(distortionModel, line.first), undistort(distortionModel, line.second));
	}
	std::cout << "[Geometry calibration] Major lines: " << majorLines.size() << std::endl;
	if(majorLines.size() < 4) {
		std::cout << "[Geometry calibration] Less than 4 major lines, aborting calibration for this frame" << std::endl;
		return;
	}

	const std::vector<cv::Vec2f> majorIntersections = lineIntersections(majorLines, thresholded.width, thresholded.height, r.maxIntersectionDistance);
	std::cout << "[Geometry calibration] Major line intersections: " << majorIntersections.size() << std::endl;
	if(majorIntersections.size() < 4) {
		std::cout << "[Geometry calibration] Less than 4 intersections, aborting calibration for this frame" << std::endl;
		return;
	}

	std::list<Eigen::Vector2f> edges = findOuterEdges(majorIntersections);
	std::cout << "[Geometry calibration] Selecting image edges: ";
	for(const Eigen::Vector2f& edge : edges)
		std::cout << edge.transpose() << " ";
	std::cout << std::endl;

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

	std::vector<Eigen::Vector2f> linePixels;
	{
		CLMap<uint8_t> data = thresholded.read<uint8_t>();
		for (int y = 0; y < thresholded.height; y++) {
			for (int x = 0; x < thresholded.width; x++) {
				if(data[x + y * thresholded.width])
					linePixels.emplace_back(x, y);
			}
		}
	}

	const bool calibHeight = r.cameraHeight == 0.0;
	if(calibHeight) {
		distortionModel.pos.z() = r.cameraHeight;
		distortionModel.updateDerived();
	}

	int minError = INT_MAX;
	CameraModel minModel;
	for(int orientation = 0; orientation < 8; orientation++) {
		CameraModel model = distortionModel;

		GeometryFit functor(edges, fieldEdges, distortionModel, calibHeight);
		Eigen::NumericalDiff<GeometryFit> numDiff(functor);
		Eigen::LevenbergMarquardt<Eigen::NumericalDiff<GeometryFit>> lm(numDiff);

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

		if(lm.info() != Eigen::ComputationInfo::Success)
			continue;

		//TODO Fixable instead of skipping? (Wrong clockwise direction?)
		if(calibHeight && k[6] < 0) // camera below field
			continue;

		model.updateFocalLength(k[0]);
		model.updateEuler({k[1], k[2], k[3]});
		model.pos.x() = k[4];
		model.pos.y() = k[5];
		if(calibHeight)
			model.pos.z() = k[6];

		if(model.focalLength < 0) {
			model.updateFocalLength(-k[0]);
			model.f2iOrientation = Eigen::AngleAxisf(M_PI_2, Eigen::Vector3f::UnitZ()) * model.f2iOrientation;
		}

		model.updateDerived();

		//TODO use modelError to refine model further
		int error = modelError(r, model, linePixels);
		if(error < minError) {
			minError = error;
			minModel = model;
		}

		std::rotate(edges.begin(), std::next(edges.begin()), edges.end());
		if(orientation == 3) {
			//TODO better solution than double model checks
			std::reverse(edges.begin(), edges.end());
		}
	}

	if(minError == INT_MAX) {
		std::cerr << "[Geometry calibration] Unable to find matching field model, aborting calibration for this frame." << std::endl;
		return;
	}

	std::cout << "[Geometry calibration] Best model: " << minModel << " error " << (minError/(float)linePixels.size()) << std::endl;
	std::cout << "[Geometry calibration] Best model field points: ";
	for(const Eigen::Vector2f& edge : edges)
		std::cout << minModel.image2field(edge, 0.0f).head<2>().transpose() << " ";
	std::cout << std::endl;

	{
		const SSL_GeometryFieldSize& field = r.socket->getGeometry().field();
		const float halfLineWidth = (float)field.line_thickness() / 2.0f;

		std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> lines;
		for(const SSL_FieldLineSegment& line : field.field_lines())
			lines.emplace_back(Eigen::Vector2f(line.p1().x(), line.p1().y()), Eigen::Vector2f(line.p2().x(), line.p2().y()));

		CLMap<uint8_t> data = thresholded.write<uint8_t>();
		for(const Eigen::Vector2f& linePixel : linePixels) {
			data[(int)linePixel.x() + (int)linePixel.y()*thresholded.width] = pointOnLine(minModel, lines, halfLineWidth, linePixel) ? 255 : 128;

			//TODO arcs
		}
	}
	thresholded.save(".linepoints.png");

	//TODO gray/thresholded width/height might diverge from img/bgr width/height
	Image bgr = img.toBGR();
	{
		CVMap cvBgr = bgr.cvReadWrite();
		detector->drawSegments(*cvBgr, linesMat);
	}
	bgr.save(".linesegments.png");

	{
		CVMap cvBgr = bgr.cvReadWrite();
		for (const auto& item : mergedLines) {
			cv::line(*cvBgr, {item.first}, {item.second}, CV_RGB(0, 255, 0));
		}
		for (const auto& item : majorLines) {
			cv::line(*cvBgr, {item.first}, {item.second}, CV_RGB(0, 0, 255));
		}

		for (const auto& item : majorIntersections) {
			cv::drawMarker(*cvBgr, {item}, CV_RGB(0, 255, 255));
		}

		for (const auto& edge : edges) {
			cv::drawMarker(*cvBgr, {eigen2cv(edge)}, CV_RGB(0, 255, 255), cv::MARKER_DIAMOND);
		}
	}
	bgr.save(".lines.png");
	//cv::imwrite("lines2.png", *cvBgr);

	{
		CVMap cvBgr = bgr.cvReadWrite();
		for(const auto& line : mergedLines) {
			cv::Vec2f start = undistort(distortionModel, line.first);
			cv::Vec2f end = undistort(distortionModel, line.second);
			cv::arrowedLine(*cvBgr, {start}, {end}, CV_RGB(0, 255, 255));
			cv::arrowedLine(*cvBgr, {end}, {start}, CV_RGB(0, 255, 255));
		}
	}
	bgr.save(".linesundistorted.png");
	//cv::imwrite("lines3.png", *cvBgr);
}