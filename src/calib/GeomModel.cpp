#include "GeomModel.h"
#include "Distortion.h"
#include "LineDetection.h"
#include "proto/ssl_vision_wrapper.pb.h"

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


static inline float pointError(const Eigen::Vector2f& n, const Eigen::Vector2f& u, const float d0) {
	return n.transpose().dot(u) - d0;
}

static std::vector<float> lineError(const std::vector<Eigen::Vector2f>& undistorted) {
	float Ex = 0;
	float Ey = 0;
	float Exx = 0;
	float Eyy = 0;
	float Exy = 0;
	for(const Eigen::Vector2f& u : undistorted) {
		Ex += u.x();
		Ey += u.y();
		Exx += u.x()*u.x();
		Eyy += u.y()*u.y();
		Exy += u.x()*u.y();
	}
	Ex /= undistorted.size();
	Ey /= undistorted.size();
	Exx /= undistorted.size();
	Eyy /= undistorted.size();
	Exy /= undistorted.size();

	Eigen::Vector2f n;
	float d0;
	if (Exx - Ex*Ex >= Eyy - Ey*Ey) {
		float a = (Exy - Ex*Ey) / (Exx - Ex*Ex);
		float b = (Exx*Ey - Ex*Exy) / (Exx - Ex*Ex);
		n.x() = -a / sqrtf(a*a + 1);
		n.y() = 1 / sqrtf(a*a + 1);
		d0 = b / sqrtf(a*a + 1);
	} else {
		float c = (Exy - Ex*Ey) / (Eyy - Ey*Ey);
		float d = (Eyy*Ex - Ey*Exy) / (Eyy - Ey*Ey);
		n.x() = 1 / sqrtf(c*c + 1);
		n.y() = -c / sqrtf(c*c + 1);
		d0 = d / sqrtf(c*c + 1);
	}

	std::vector<float> error;
	for(const Eigen::Vector2f& u : undistorted) {
		float pe = pointError(n, u, d0);
		/*if(isnanf(pe)) {
			std::cout << "NaNP: " << std::endl << n << std::endl << u << std::endl << d0 << " " << Ex << " " << Ey << std::endl;
			exit(1);
		}*/
		error.push_back(pe);
	}
	return error;
}

struct GeometryFit : public Eigen::DenseFunctor<float> {
	const std::vector<std::vector<Eigen::Vector2f>>& lines;
	const std::vector<Eigen::Vector2f>& imageEdges;
	const std::list<Eigen::Vector2f>& fieldEdges;
	const CameraModel& reference;
	const bool calibHeight;

	explicit GeometryFit(const std::vector<std::vector<Eigen::Vector2f>>& lines, const std::vector<Eigen::Vector2f>& imageEdges, const std::list<Eigen::Vector2f>& fieldEdges, const CameraModel& model, const bool calibHeight): lines(lines), imageEdges(imageEdges), fieldEdges(fieldEdges), reference(model), calibHeight(calibHeight) {}

	int operator()(const InputType &x, ValueType& fvec) const {
		CameraModel model = reference;
		model.distortionK2 = x[0];
		model.principalPoint.x() = x[1];
		model.principalPoint.y() = x[2];
		model.focalLength = x[3];
		model.updateEuler({x[4], x[5], x[6]});
		model.pos.x() = x[7];
		model.pos.y() = x[8];
		if(calibHeight)
			model.pos.z() = x[9];
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

		for(const std::vector<Eigen::Vector2f>& distorted : lines) {
			std::vector<Eigen::Vector2f> undistorted;
			for(const Eigen::Vector2f& d : distorted)
				undistorted.push_back(model.normalizeUndistort(d));

			std::vector<float> error = lineError(undistorted);
			for(float e : error) {
				if(e == NAN)
					return -1;
				fvec(i++) = e;
			}
		}
		return 0;
	}

	int values() const {
		int size = 8;

		for (const auto& item : lines)
			size += item.size();

		return size;
	}
};

typedef struct LineArc {
	Eigen::Vector2f center;
	float radius;
	float a1, a2;
} LineArc;

static float sqPointLineSegmentDistance(const std::pair<Eigen::Vector2f, Eigen::Vector2f>& line, const Eigen::Vector2f& point) {
	//Adapted from Grumdrig https://stackoverflow.com/a/1501725 CC BY-SA 4.0
	const Eigen::Vector2f v = line.second - line.first;
	const Eigen::Vector2f w = point - line.first;
	const float t = std::max(0.0f, std::min(1.0f, w.dot(v) / v.dot(v)));
	const Eigen::Vector2f delta = w - t * v;
	return delta.dot(delta);
}

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

std::vector<Eigen::Vector2f> getLinePixels(const Image& thresholded) {
	std::vector<Eigen::Vector2f> linePixels;
	CLMap<uint8_t> data = thresholded.read<uint8_t>();
	for (int y = 0; y < thresholded.height; y++) {
		for (int x = 0; x < thresholded.width; x++) {
			if(data[x + y * thresholded.width])
				linePixels.emplace_back(x, y);
		}
	}
	return std::move(linePixels);
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
	Image gray = img.toGrayscale();
	const int halfLineWidth = halfLineWidthEstimation(r, gray);
	std::cout << "[Geometry calibration] Half line width: " << halfLineWidth << std::endl;

	// Adapted from https://stackoverflow.com/a/25436112 by user2398029 under CC BY-SA 3.0
	// J. Immerkær, “Fast Noise Variance Estimation”, Computer Vision and Image Understanding, Vol. 64, No. 2, pp. 300-302, Sep. 1996
	cv::Mat noiseKernel = (cv::Mat_<float>(3, 3) << 1, -2, 1, -2, 4, -2, 1, -2, 1);
	cv::Mat noise;
	cv::filter2D(*gray.cvRead(), noise, -1, noiseKernel);
	float sigma = cv::sum(cv::abs(noise))[0] * sqrtf(0.5f * M_PI) / (6 * gray.width * gray.height);
	std::cout << "[Geometry calibration] Noise sigma: " << sigma << std::endl;

	//0.05 (schubert) - 0.3 (default) - 0.45 (rc19)

	//https://docs.opencv.org/4.x/da/d7f/tutorial_back_projection.html

	Image t(&PixelFormat::U8, gray.width, gray.height, gray.name);
	/*cv::Mat t1, t2;
	cv::adaptiveThreshold(*gray.cvRead(), t1, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 100 + 1, r.fieldLineThreshold);
	cv::adaptiveThreshold(*gray.cvRead(), t2, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 100 + 1, -r.fieldLineThreshold);
	{
		CLMap<uint8_t> tData = t.write<uint8_t>();
		for (int y = 0; y < gray.height; y++) {
			for (int x = 0; x < gray.width; x++) {
				int pos = x + y * gray.width;
				tData[pos] = (t1.data[pos] && t2.data[pos]) ? 255 : 0;
			}
		}
	}
	t.save(".carpet.png");*/
	//return;

	Image bgr = img.toBGR();
	std::vector<cv::Mat> split_bgr(3);
	cv::split(*bgr.cvRead(), split_bgr);

	cv::Scalar mean, stddev;
	cv::meanStdDev(*bgr.cvRead(), mean, stddev);

	cv::Mat tb, tg, tr;
	cv::adaptiveThreshold(split_bgr[0], tb, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 20*halfLineWidth + 1, -r.fieldLineThreshold);
	cv::adaptiveThreshold(split_bgr[1], tg, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 20*halfLineWidth + 1, -r.fieldLineThreshold);
	cv::adaptiveThreshold(split_bgr[2], tr, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 20*halfLineWidth + 1, -r.fieldLineThreshold);

	//Image t(&PixelFormat::U8, bgr.width, bgr.height, bgr.name);
	{
		const CLMap<uint8_t> data = bgr.read<uint8_t>();
		CLMap<uint8_t> tData = t.write<uint8_t>();
		for (int y = 0; y < bgr.height; y++) {
			for (int x = 0; x < bgr.width; x++) {
				int pos = x + y * bgr.width;
				tData[pos] = (tb.data[pos] && tb.data[pos+1] && tb.data[pos+2]) ? 255 : 0;
			}
		}
	}
	t.save(".carpet.png");

	Image thresholded = thresholdImage(r, gray, halfLineWidth);
	thresholded.save(".linepoints.png");
	{
		CVMap map = thresholded.cvReadWrite();
		cv::bitwise_and(*map, *t.cvRead(), *map);
	}
	thresholded.save(".thresholded.png");

	cv::Ptr<cv::LineSegmentDetector> detector = cv::createLineSegmentDetector();
	cv::Mat4f linesMat;
	detector->detect(*thresholded.cvRead(), linesMat);

	//TODO gray/thresholded width/height might diverge from img/bgr width/height
	//Image bgr = img.toBGR();
	{
		CVMap cvBgr = bgr.cvReadWrite();
		detector->drawSegments(*cvBgr, linesMat);
	}
	bgr.save(".linesegments.png");

	CVLines lines;
	for(int i = 0; i < linesMat.rows; i++) {
		cv::Vec4f line = linesMat(0, i);
		cv::Vec2f a = cv::Vec2f(line[0], line[1]);
		cv::Vec2f b = cv::Vec2f(line[2], line[3]);
		if(dist(a, b) >= r.minLineSegmentLength)
			lines.emplace_back(a, b);
	}
	std::cout << "[Geometry calibration] Line segments: " << lines.size() << std::endl;


	//TODO region growing field and field line search (field lines, separate)
	//TODO crossings to field line net ("field" holes)
	//TODO edge estimation for 90° loops (arcs and circles?)
	//TODO reprojection error (how far are the 4 points projected)

	//TODO unified calibration
	const std::vector<CVLines> compoundLines = groupLineSegments(r, lines);
	const CVLines mergedLines = mergeLineSegments(compoundLines);
	std::cout << "[Geometry calibration] Lines: " << mergedLines.size() << std::endl;

	const std::vector<Eigen::Vector2f> linePixels = getLinePixels(thresholded);
	std::vector<std::vector<Eigen::Vector2f>> mergedPixels(mergedLines.size());
	{
		const float sqHalfLineWidth = (float)(halfLineWidth*halfLineWidth);
		CLMap<uint8_t> data = thresholded.read<uint8_t>();
		for (int y = 0; y < thresholded.height; y++) {
			for (int x = 0; x < thresholded.width; x++) {
				if(data[x + y * thresholded.width]) {
					for(int i = 0; i < compoundLines.size(); i++) {
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

	//TODO major line filter necessary? Improve?
	CVLines majorLines;
	const float minMajorLength = std::max(img.width, img.height) * r.minMajorLineLength;
	for(const auto& line : mergedLines) {
		if(dist(line.first, line.second) >= minMajorLength)
			majorLines.emplace_back(line.first, line.second);
	}
	std::cout << "[Geometry calibration] Major lines: " << majorLines.size() << std::endl;

	{
		CVMap cvBgr = bgr.cvReadWrite();
		for (const auto& item : mergedLines) {
			cv::line(*cvBgr, {item.first}, {item.second}, CV_RGB(0, 255, 0));
		}
		for (const auto& item : majorLines) {
			cv::line(*cvBgr, {item.first}, {item.second}, CV_RGB(0, 0, 255));
		}
	}
	bgr.save(".lines.png");

	if(majorLines.size() < 4) {
		std::cout << "[Geometry calibration] Less than 4 major lines, aborting calibration for this frame" << std::endl;
		return;
	}

	const std::vector<Eigen::Vector2f> intersections = lineIntersections(majorLines, thresholded.width, thresholded.height, r.maxIntersectionDistance);
	std::cout << "[Geometry calibration] Line intersections: " << intersections.size() << std::endl;

	{
		CVMap cvBgr = bgr.cvReadWrite();
		for (const auto& item : mergedLines) {
			cv::line(*cvBgr, {item.first}, {item.second}, CV_RGB(0, 255, 0));
		}
		for (const auto& item : majorLines) {
			cv::line(*cvBgr, {item.first}, {item.second}, CV_RGB(0, 0, 255));
		}

		for (const auto& item : intersections) {
			cv::drawMarker(*cvBgr, {eigen2cv(item)}, CV_RGB(0, 255, 255));
		}
	}
	bgr.save(".lines.png");

	if(intersections.size() < 4) {
		std::cout << "[Geometry calibration] Less than 4 intersections, aborting calibration for this frame" << std::endl;
		return;
	}

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

	CameraModel basicModel({thresholded.width, thresholded.height}, r.camId, r.cameraAmount, r.socket->getGeometry().field());
	const bool calibHeight = r.cameraHeight == 0.0;
	if(calibHeight) {
		basicModel.pos.z() = 5000.0f;
	} else {
		basicModel.pos.z() = r.cameraHeight;
	}
	basicModel.updateDerived();

	float minFastError = INFINITY;
	int minError = INT_MAX;
	CameraModel minModel;
	std::vector<Eigen::Vector2f> minEdges;

	std::vector<Eigen::Vector2f> edges(4);
	for(const Eigen::Vector2f& a : intersections) {
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

						GeometryFit functor(mergedPixels, edges, fieldEdges, basicModel, calibHeight);
						Eigen::NumericalDiff<GeometryFit> numDiff(functor);
						Eigen::LevenbergMarquardt<Eigen::NumericalDiff<GeometryFit>> lm(numDiff);

						Eigen::VectorXf k(calibHeight ? 10 : 9);
						k[0] = model.distortionK2;
						k[1] = model.principalPoint.x();
						k[2] = model.principalPoint.y();
						k[3] = model.focalLength;
						Eigen::Vector3f euler = model.getEuler();
						k[4] = euler.x();
						k[5] = euler.y();
						k[6] = euler.z();
						k[7] = model.pos.x();
						k[8] = model.pos.y();
						if(calibHeight)
							k[9] = model.pos.z();

						lm.minimize(k);

						if(lm.info() != Eigen::ComputationInfo::Success && lm.info() != Eigen::ComputationInfo::NoConvergence) //xtol might be too aggressive
							continue;

						if(calibHeight && k[6] < 0) // camera below field
							continue;

						model.distortionK2 = k[0];
						model.principalPoint.x() = k[1];
						model.principalPoint.y() = k[2];
						model.focalLength = k[3];
						model.updateEuler({k[4], k[5], k[6]});
						model.pos.x() = k[7];
						model.pos.y() = k[8];
						if(calibHeight)
							model.pos.z() = k[9];

						if(model.focalLength < 0) {
							model.focalLength = -k[3];
							model.f2iOrientation = Eigen::AngleAxisf(M_PI_2, Eigen::Vector3f::UnitZ()) * model.f2iOrientation;
						}

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
	}

	if(minError == INT_MAX) {
		std::cerr << "[Geometry calibration] Unable to find matching field model, aborting calibration for this frame." << std::endl;
		return;
	}

	std::cout << "[Geometry calibration] Best model: " << minModel << " error " << (minError/(float)linePixels.size()) << std::endl;
	std::cout << "[Geometry calibration] Best model field points: ";
	for(const Eigen::Vector2f& edge : minEdges)
		std::cout << minModel.image2field(edge, 0.0f).head<2>().transpose() << " ";
	std::cout << std::endl;

	SSL_WrapperPacket wrapper;
	wrapper.mutable_geometry()->CopyFrom(r.socket->getGeometry());
	wrapper.mutable_geometry()->add_calib()->CopyFrom(minModel.getProto(r.camId));
	r.socket->send(wrapper);

	{
		const SSL_GeometryFieldSize& field = r.socket->getGeometry().field();
		const float halfLineWidth = (float)field.line_thickness() / 2.0f;

		std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> lines;
		std::vector<LineArc> arcs;
		fieldToLines(r, lines, arcs);

		CLMap<uint8_t> data = thresholded.write<uint8_t>();
		for(const Eigen::Vector2f& linePixel : linePixels) {
			data[(int)linePixel.x() + (int)linePixel.y()*thresholded.width] = pointAtLine(minModel, lines, arcs, halfLineWidth, linePixel) ? 255 : 128;
		}
	}
	thresholded.save(".thresholded.png");
}