#include "geomcalib.h"



float dist(const cv::Vec2f& v1, const cv::Vec2f& v2) {
	cv::Vec2f d = v2-v1;
	return sqrtf(d.dot(d));
}

static float angleDiff(const float a1, const float a2) {
	return fabsf(atan2f(sinf(a2-a1), cosf(a2-a1)));
}

static float angleDiff(const cv::Vec4f& v1, const cv::Vec4f& v2) {
	float v1a = atan2f(v1[3] - v1[1], v1[2] - v1[0]);
	float v2a = atan2f(v2[3] - v2[1], v2[2] - v2[0]);
	return angleDiff(v1a, v2a);
}

static Eigen::Vector2f undistort(const Eigen::Vector2f& k, const int width, const Eigen::Vector2f& p) {
	Eigen::Vector2f n = p/width;
	n(0) -= 0.5f;
	n(1) -= 0.5f;

	float r2 = n(0)*n(0) + n(1)*n(1);
	float factor = 1 + k(0)*r2 + k(1)*r2*r2;

	n *= factor;
	n(0) += 0.5f;
	n(1) += 0.5f;
	return n*width;
}

static int halfLineWidthEstimation(const Resources& r, const Image& img) {
	const SSL_GeometryFieldSize& field = r.socket->getGeometry().field();
	int xSize = 1;
	int ySize = 1;
	for(int i = r.cameraAmount; i > 1; i /= 2) {
		if(field.field_length()/xSize >= field.field_width()/ySize)
			xSize *= 2;
		else
			ySize *= 2;
	}

	int xPos = 0;
	int yPos = 0;
	for(int i = r.camId % r.cameraAmount; i > 0; i--) {
		yPos++;
		if(yPos == ySize) {
			yPos = 0;
			xPos++;
		}
	}

	int extentLarge = field.field_length()/xSize + (xPos == 0 ? field.boundary_width() : 0) + (xPos == xSize-1 ? field.boundary_width() : 0);
	int extentSmall = field.field_width()/ySize + (yPos == 0 ? field.boundary_width() : 0) + (yPos == ySize-1 ? field.boundary_width() : 0);
	if(extentLarge < extentSmall)
		std::swap(extentSmall, extentLarge);

	int cameraLarge = img.width;
	int cameraSmall = img.height;
	if(cameraLarge < cameraSmall)
		std::swap(cameraSmall, cameraLarge);

	float largeRatio = cameraLarge/(float)extentLarge;
	float smallRatio = cameraSmall/(float)extentSmall;
	if(largeRatio < smallRatio)
		std::swap(smallRatio, largeRatio);

	return std::ceil(largeRatio*field.line_thickness()/2.f);
}

void geometryCalibration(const Resources& r, const Image& img) {
	int halfLineWidth = halfLineWidthEstimation(r, img); // 4, 3, 7
	std::cout << "Line width: " << halfLineWidth << std::endl;
	Image gray = img.toGrayscale();

	std::shared_ptr<Image> thresholded = std::make_shared<Image>(&PixelFormat::U8, gray.width, gray.height);
	{
		const CLMap<uint8_t> data = gray.read<uint8_t>();
		const int width = gray.width;
		int diff = 5; //TODO config
		CLMap<uint8_t> tData = thresholded->write<uint8_t>();
		for (int y = halfLineWidth; y < gray.height - halfLineWidth; y++) {
			for (int x = halfLineWidth; x < width - halfLineWidth; x++) {
				int value = data[x + y * width];
				tData[x + y * width] = (
											   (value - data[x - halfLineWidth + y * width] >
												diff &&
												value - data[x + halfLineWidth + y * width] >
												diff) ||
											   (value - data[x + (y - halfLineWidth) * width] >
												diff &&
												value - data[x + (y + halfLineWidth) * width] >
												diff)
									   ) ? 255 : 0;
			}
		}
	}

	cv::Ptr<cv::LineSegmentDetector> detector = cv::createLineSegmentDetector();
	cv::Mat4f linesMat;
	detector->detect(*thresholded->cvRead(), linesMat);

	std::list<cv::Vec4f> lines;
	for(int i = 0; i < linesMat.rows; i++)
		lines.push_back(linesMat(0, i));

	for(auto& line : lines)
		std::cout << dist(cv::Vec2f(line[0], line[1]), cv::Vec2f(line[2], line[3])) << " ";
	std::cout << std::endl;

	std::cout << "Line segments: " << lines.size() << std::endl;

	std::vector<std::vector<cv::Vec4f>> compoundLines;
	std::vector<cv::Vec4f> mergedLines;
	while(!lines.empty()) {
		std::vector<cv::Vec4f> compound;
		compound.push_back(lines.front());
		lines.erase(lines.cbegin());

		for(int i = 0; i < compound.size(); i++) {
			const auto& root = compound[i];
			cv::Vec2f a1(root[0], root[1]);
			cv::Vec2f b1(root[2], root[3]);
			cv::Vec4f invRoot(root[2], root[3], root[0], root[1]);

			auto lit = lines.cbegin();
			while(lit != lines.cend()) {
				cv::Vec4f l = *lit;
				cv::Vec2f a2(l[0], l[1]);
				cv::Vec2f b2(l[2], l[3]);
				if(
						std::min(angleDiff(root, l), angleDiff(invRoot, l)) <= 0.05 &&
						std::min(std::min(dist(a1, a2), dist(b1, b2)), std::min(dist(a1, b2), dist(b1, a2))) <= 40.0
						) {
					compound.push_back(l);
					lit = lines.erase(lit);
				} else {
					lit++;
				}
			}
		}

		std::sort(compound.begin(), compound.end(), [](const cv::Vec4f& v1, const cv::Vec4f& v2) { return dist(cv::Vec2f(v1[0], v1[1]), cv::Vec2f(v1[2], v1[3])) > dist(cv::Vec2f(v2[0], v2[1]), cv::Vec2f(v2[2], v2[3])); });
		compoundLines.push_back(compound);

		cv::Vec2f a(compound.front()[0], compound.front()[1]);
		cv::Vec2f b(compound.front()[2], compound.front()[3]);
		for(int i = 1; i < compound.size(); i++) {
			const auto& v = compound[i];

			cv::Vec2f c(v[0], v[1]);
			cv::Vec2f d(v[2], v[3]);

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
		mergedLines.emplace_back(a[0], a[1], b[0], b[1]);
	}

	std::cout << "Compound lines: " << compoundLines.size() << std::endl;

	cv::imwrite("thresholded.png", *thresholded->cvRead());

	Image bgr = img.toBGR();
	CVMap cvBgr = bgr.cvReadWrite();
	detector->drawSegments(*cvBgr, linesMat);
	cv::imwrite("lineSegments.png", *cvBgr);

	std::vector<std::vector<Eigen::Vector2f>> l;
	for(const auto& compound : compoundLines) {
		for(int i = 1; i < compound.size(); i++) {
			cv::line(*cvBgr, {(int)compound[i-1][2], (int)compound[i-1][3]}, {(int)compound[i][0], (int)compound[i][1]}, CV_RGB(0, 0, 255));
		}

		std::vector<Eigen::Vector2f> points;
		for(const auto& segment : compound) {
			//std::cout << " " << segment[0] << "," << segment[1] << "->" << segment[2] << "," << segment[3];
			//TODO rescale to later determined focal length
			//TODO precision issues -> double?
			//points.emplace_back(segment[0]*2/img->getWidth() - 0.5f, segment[1]*2/img->getHeight() - 0.5f);
			//points.emplace_back(segment[2]*2/img->getWidth() - 0.5f, segment[3]*2/img->getHeight() - 0.5f);
			points.emplace_back(segment[0]/img.width - 0.5f, segment[1]/img.width - 0.5f);
			points.emplace_back(segment[2]/img.width - 0.5f, segment[3]/img.width - 0.5f);
		}
		//std::cout << std::endl;
		l.push_back(points);
	}
	std::cout << "Filtered compound lines: " << (l.size()/2) << std::endl;


	cv::imwrite("lines.png", *cvBgr);
	Eigen::Vector2f k = distortion(l);

	for (const auto& item : mergedLines) {
		cv::line(*cvBgr, {(int)item[0], (int)item[1]}, {(int)item[2], (int)item[3]}, CV_RGB(0, 255, 0));
	}
	cv::imwrite("lines2.png", *cvBgr);

	for(const auto& compound : compoundLines) {
		if(compound.size() == 1)
			continue;

		Eigen::Vector2f start = undistort(k, img.width, {compound.front()[0], compound.front()[1]});
		Eigen::Vector2f end = undistort(k, img.width, {compound.back()[2], compound.back()[3]});
		cv::arrowedLine(*cvBgr, {(int)start(0), (int)start(1)}, {(int)end(0), (int)end(1)}, CV_RGB(0, 255, 255));
		cv::arrowedLine(*cvBgr, {(int)end(0), (int)end(1)}, {(int)start(0), (int)start(1)}, CV_RGB(0, 255, 255));
	}
	cv::imwrite("lines3.png", *cvBgr);
}