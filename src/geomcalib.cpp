#include "geomcalib.h"
#include "distortion.h"

#include <eigen3/unsupported/Eigen/LevenbergMarquardt>


float dist(const cv::Vec2f& v1, const cv::Vec2f& v2) {
	cv::Vec2f d = v2-v1;
	return sqrtf(d.dot(d));
}

static void visibleFieldExtent(const Resources &r, const bool withBoundary, Eigen::Vector2f &min, Eigen::Vector2f &max) {
	return visibleFieldExtent(r.camId, r.cameraAmount, r.socket->getGeometry().field(), withBoundary, min, max);
}

static int halfLineWidthEstimation(const Resources& r, const Image& img) {
	Eigen::Vector2f min;
	Eigen::Vector2f max;
	visibleFieldExtent(r, true, min, max);

	Eigen::Vector2f extent = max - min;
	if(extent[0] < extent[1])
		std::swap(extent[0], extent[1]);

	Eigen::Vector2f camera(img.width, img.height);
	if(camera[0] < camera[1])
		std::swap(camera[0], camera[1]);

	Eigen::Vector2f ratio = camera.array() / extent.array();
	if(ratio[0] < ratio[1])
		std::swap(ratio[0], ratio[1]);

	const SSL_GeometryFieldSize& field = r.socket->getGeometry().field();
	return std::ceil(ratio[0] * (float)field.line_thickness()/2.0f);
}

static Image thresholdImage(const Resources& r, const Image& gray, const int halfLineWidth) {
	Image thresholded(&PixelFormat::U8, gray.width, gray.height, gray.name);

	const CLMap<uint8_t> data = gray.read<uint8_t>();
	const int width = gray.width;
	CLMap<uint8_t> tData = thresholded.write<uint8_t>();
	for (int y = halfLineWidth; y < gray.height - halfLineWidth; y++) {
		for (int x = halfLineWidth; x < width - halfLineWidth; x++) {
			int value = data[x + y * width];
			tData[x + y * width] = (
										   (value - data[x - halfLineWidth + y * width] > r.fieldLineThreshold &&
											value - data[x + halfLineWidth + y * width] > r.fieldLineThreshold) ||
										   (value - data[x + (y - halfLineWidth) * width] > r.fieldLineThreshold &&
											value - data[x + (y + halfLineWidth) * width] > r.fieldLineThreshold)
								   ) ? 255 : 0;
		}
	}

	return thresholded;
}

typedef std::pair<cv::Vec2f, cv::Vec2f> CVLine;
typedef std::vector<std::pair<cv::Vec2f, cv::Vec2f>> CVLines;

static std::vector<CVLines> groupLineSegments(const Resources& r, CVLines& segments) {
	std::vector<CVLines> compoundLines;
	while(!segments.empty()) {
		CVLines compound;
		compound.push_back(segments.front());
		segments.erase(segments.cbegin());

		for(int i = 0; i < compound.size(); i++) {
			const auto& root = compound[i];
			cv::Vec2f v1 = root.second - root.first;

			auto lit = segments.cbegin();
			while(lit != segments.cend()) {
				CVLine l = *lit;
				cv::Vec2f v2 = l.second - l.first;
				if(
						abs(acosf(abs(v2.dot(v1) / (sqrtf(v1.dot(v1)) * sqrtf(v2.dot(v2)))))) <= r.maxLineSegmentAngle &&
						// https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
						std::min(abs(v1[0]*(l.first[1] - root.first[1]) - (l.first[0] - root.first[0])*v1[1]) / sqrtf(v1.dot(v1)), abs(v1[0]*(l.second[1] - root.first[1]) - (l.second[0] - root.first[0])*v1[1]) / sqrtf(v1.dot(v1))) <= r.maxLineSegmentOffset
						) {
					compound.push_back(l);
					lit = segments.erase(lit);
				} else {
					lit++;
				}
			}
		}

		//Sort according to line length
		std::sort(compound.begin(), compound.end(), [](const CVLine& v1, const CVLine& v2) { return dist(v1.first, v1.second) > dist(v2.first, v2.second); });
		compoundLines.push_back(compound);
	}
	return compoundLines;
}

static CVLines mergeLines(const std::vector<CVLines>& compoundLines) {
	CVLines mergedLines;
	for(const auto& compound : compoundLines) {
		cv::Vec2f a = compound.front().first;
		cv::Vec2f b = compound.front().second;
		for(int i = 1; i < compound.size(); i++) {
			const auto& v = compound[i];

			const cv::Vec2f& c = v.first;
			const cv::Vec2f& d = v.second;

			cv::Vec2f max1 = a;
			cv::Vec2f max2 = b;
			float maxd = dist(a, b);

			if(dist(a, c) > maxd) {
				max1 = a;
				max2 = c;
				maxd = dist(a, c);
			}
			if(dist(a, d) > maxd) {
				max1 = a;
				max2 = d;
				maxd = dist(a, d);
			}
			if(dist(c, b) > maxd) {
				max1 = c;
				max2 = b;
				maxd = dist(c, b);
			}
			if(dist(d, b) > maxd) {
				max1 = d;
				max2 = b;
				maxd = dist(d, b);
			}
			if(dist(c, d) > maxd) {
				max1 = c;
				max2 = d;
				maxd = dist(c, d);
			}

			a = max1;
			b = max2;
		}
		mergedLines.emplace_back(a, b);
	}
	return mergedLines;
}

static Eigen::Vector2f cv2eigen(const cv::Vec2f& v) {
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

// Adapted from https://stackoverflow.com/a/7448287 by Andrey Kamaev CC BY-SA 3.0
static cv::Vec2f lineLineIntersection(const CVLine& a, const CVLine& b) {
	cv::Vec2f x = b.first - a.first;
	cv::Vec2f da = a.second - a.first;
	cv::Vec2f db = b.second - b.first;

	float cross = da[0]*db[1] - da[1]*db[0];
	if (abs(cross) < 1e-8)
		return {INFINITY, INFINITY};

	double t1 = (x[0] * db[1] - x[1] * db[0]) / cross;
	return a.first + da * t1;
}

static std::vector<cv::Vec2f> lineIntersections(const CVLines& lines, const int width, const int height, const double maxIntersectionDistance) {
	std::vector<cv::Vec2f> intersections;

	float minX = -width*maxIntersectionDistance;
	float minY = -height*maxIntersectionDistance;
	float maxX = width + width*maxIntersectionDistance;
	float maxY = height + height*maxIntersectionDistance;
	auto ita = lines.cbegin();
	while(ita != lines.cend()) {
		auto itb = ita+1;
		while(itb != lines.cend()) {
			cv::Vec2f c = lineLineIntersection(*ita, *itb);
			if(c[0] >= minX && c[1] >= minY && c[0] < maxX && c[1] < maxY)
				intersections.push_back(c);

			itb++;
		}

		ita++;
	}

	return intersections;
}

static bool inSegment(const cv::Vec2f& a, const cv::Vec2f& b, const cv::Vec2f& point) {
	return point[0] > std::min(a[0], b[0]) && point[1] > std::min(a[1], b[1]) && point[0] < std::max(a[0], b[0]) && point[1] < std::max(a[1], b[1]);
}

static cv::Vec2f undistort(const CameraModel& model, const cv::Vec2f& v) {
	return eigen2cv(model.undistort(cv2eigen(v)));
}

static std::list<Eigen::Vector2f> findOuterEdges(const std::vector<cv::Vec2f>& intersections) {
	std::list<Eigen::Vector2f> edges;
	float maxArea = 0.0f;
	for(const cv::Vec2f& a : intersections) {
		for(const cv::Vec2f& b : intersections) {
			for(const cv::Vec2f& c : intersections) {
				for(const cv::Vec2f& d : intersections) {
					if(a == b || a == c || a == d || b == c || b == d || c == d)
						continue;

					cv::Vec2f center = lineLineIntersection({a, c}, {b, d});
					if(!inSegment(a, c, center) || !inSegment(b, d, center)) // Check if ABCD is convex quadrilateral
						continue;

					//https://en.wikipedia.org/wiki/Quadrilateral#Vector_formulas
					cv::Vec2f ac = c - a;
					cv::Vec2f bd = d - b;
					float area = 0.5f * abs(ac[0] * bd[1] - bd[0] * ac[1]);
					if(area < maxArea)
						continue;

					edges.clear();
					edges.push_back(cv2eigen(a));
					edges.push_back(cv2eigen(b));
					edges.push_back(cv2eigen(c));
					edges.push_back(cv2eigen(d));
					maxArea = area;
				}
			}
		}
	}
	return edges;
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
		model.iPos.x() = x[4];
		model.iPos.y() = x[5];
		if(calibHeight)
			model.iPos.z() = x[6];
		model.updateDerived();

		auto iIt = imageEdges.cbegin();
		auto fIt = fieldEdges.cbegin();
		int i = 0;
		while(iIt != imageEdges.cend()) {
			Eigen::Vector2f error = model.image2field(*iIt++, 0.0f).head<2>() - *(fIt++);
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

static int modelError(const Resources& r, const CameraModel& model, const std::vector<Eigen::Vector2f>& linePixels) {
	const SSL_GeometryFieldSize& field = r.socket->getGeometry().field();
	const float halfLineWidth = (float)field.line_thickness() / 2.0f;

	std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> lines;
	for(const SSL_FieldLineSegment& line : field.field_lines())
		lines.emplace_back(Eigen::Vector2f(line.p1().x(), line.p1().y()), Eigen::Vector2f(line.p2().x(), line.p2().y()));

	int error = 0.0;
	for(const Eigen::Vector2f& linePixel : linePixels) {
		const Eigen::Vector2f fieldPixel = model.image2field(linePixel, 0.0f).head<2>();
		for(const auto& line : lines) {
			const Eigen::Vector2f v = line.second - line.first;
			const Eigen::Vector2f line2pixel = fieldPixel - line.first;
			if(abs(v.x()*line2pixel.y() - v.y()*line2pixel.x()) / sqrtf(v.dot(v)) < halfLineWidth)
				goto nextPoint;
		}

		//TODO arcs

		error += 1;
		nextPoint:;
	}
	return error;
}

void geometryCalibration(const Resources& r, const Image& img) {
	Image gray = img.toGrayscale();
	const int halfLineWidth = halfLineWidthEstimation(r, gray);
	std::cout << "[Geometry calibration] Half line width: " << halfLineWidth << std::endl;

	const Image thresholded = thresholdImage(r, gray, halfLineWidth);
	thresholded.save(".linepoints.png");

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
	const CVLines mergedLines = mergeLines(compoundLines);
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
		distortionModel.iPos.z() = r.cameraHeight; //TODO cameraHeight is wrong.
		distortionModel.updateDerived();
	}

	int minError = INT_MAX;
	CameraModel minModel;
	for(int orientation = 0; orientation < 4; orientation++) {
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
		k[4] = model.iPos.x();
		k[5] = model.iPos.y();
		if(calibHeight)
			k[6] = model.iPos.z();

		std::cout << lm.minimize(k) << std::endl;

		if(lm.info() != Eigen::ComputationInfo::Success) {
			std::cout << "[Geometry calibration] Levenberg-Marquandt minimization failed with code, aborting calibration for this frame: " << lm.info() << std::endl;
			return;
		}

		//TODO sanity checks

		model.updateFocalLength(k[0]);
		model.updateEuler({k[1], k[2], k[3]});
		model.iPos.x() = k[4];
		model.iPos.y() = k[5];
		if(calibHeight)
			model.iPos.z() = k[6];
		model.updateDerived();

		int error = modelError(r, model, linePixels);
		std::cout << "[Geometry calibration] Model points: ";
		for(const Eigen::Vector2f& edge : edges)
			std::cout << model.image2field(edge, 0.0f).transpose() << " ";
		std::cout << std::endl;
		std::cout << "[Geometry calibration] Model: " << model << " error " << (error/(float)linePixels.size()) << std::endl;
		if(error < minError) {
			minError = error;
			minModel = model;
		}

		std::rotate(edges.begin(), std::next(edges.begin()), edges.end());
		std::rotate(fieldEdges.begin(), std::next(fieldEdges.begin()), fieldEdges.end());
	}

	std::cout << "[Geometry calibration] Best model: " << minModel << " error " << (minError/(float)linePixels.size()) << std::endl;

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