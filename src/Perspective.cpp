#include "Perspective.h"

#include <cmath>

static void updateExtent(Eigen::Vector4f& visibleFieldExtent, const Eigen::Vector3f& point) {
	if(point.x() < visibleFieldExtent[0])
		visibleFieldExtent[0] = point.x();
	if(point.x() > visibleFieldExtent[1])
		visibleFieldExtent[1] = point.x();

	if(point.y() < visibleFieldExtent[2])
		visibleFieldExtent[2] = point.y();
	if(point.y() > visibleFieldExtent[3])
		visibleFieldExtent[3] = point.y();
}

void Perspective::geometryCheck(const int width, const int height, const double maxBotHeight) {
	if(socket->getGeometryVersion() == geometryVersion && model.size.x() == width && model.size.y() == height)
		return;

	bool calibFound = false;
	for(const SSL_GeometryCameraCalibration& calib : socket->getGeometry().calib()) {
		if(calib.camera_id() == camId) {
			calibFound = true;
			model = CameraModel(calib);
			break;
		}
	}

	if(!calibFound)
		return;

	model.ensureSize({width, height});
	geometryVersion = socket->getGeometryVersion();
	field = socket->getGeometry().field();

	//update visibleFieldExtent
	Eigen::Vector2f center = model.image2field({0.0f, 0.0f}, (float)maxBotHeight).head<2>();
	visibleFieldExtent = {center.x(), center.x(), center.y(), center.y()};

	for(int x = 0; x < width; x++)
		updateExtent(visibleFieldExtent, model.image2field({(float)x, 0.0f}, (float)maxBotHeight));
	for(int x = 0; x < width; x++)
		updateExtent(visibleFieldExtent, model.image2field({(float)x, (float)height - 1.0f}, (float)maxBotHeight));

	for(int y = 0; y < height; y++)
		updateExtent(visibleFieldExtent, model.image2field({0.0f, (float)y}, (float)maxBotHeight));
	for(int y = 0; y < height; y++)
		updateExtent(visibleFieldExtent, model.image2field({(float)width - 1.0f, (float)y}, (float)maxBotHeight));

	// clamp to field boundaries
	const float halfLength = (float)getFieldLength()/2.0f + (float)getBoundaryWidth();
	const float halfWidth = (float)getFieldWidth()/2.0f + (float)getBoundaryWidth();
	visibleFieldExtent[0] = std::max(visibleFieldExtent[0], -halfLength);
	visibleFieldExtent[1] = std::min(visibleFieldExtent[1], halfLength);
	visibleFieldExtent[2] = std::max(visibleFieldExtent[2], -halfWidth);
	visibleFieldExtent[3] = std::min(visibleFieldExtent[3], halfWidth);

	//TODO optimalFieldScale
	reprojectedFieldSize = (Eigen::Vector2f(
			visibleFieldExtent[1] - visibleFieldExtent[0],
			visibleFieldExtent[3] - visibleFieldExtent[2]
	) * (1.0f/optimalFieldScale)).array().rint().cast<int>();
}

V2 Perspective::image2field(V2 pos, double height) const {
	Eigen::Vector3f p = model.image2field({pos.x, pos.y}, (float)height);
	return {p.x(), p.y()};
}

V2 Perspective::field2image(V3 pos) const {
	Eigen::Vector2f p = model.field2image({(float)pos.x, (float)pos.y, (float)pos.z});
	return {p.x(), p.y()};
}

int Perspective::getWidth() {
	return model.size.x();
}

int Perspective::getHeight() {
	return model.size.y();
}

int Perspective::getFieldLength() {
	return field.field_length();
}

int Perspective::getFieldWidth() {
	return field.field_width();
}

int Perspective::getBoundaryWidth() {
	return field.boundary_width();
}

ClPerspective Perspective::getClPerspective() const {
	const Eigen::Matrix3f& i2f = model.i2fOrientation;
	const Eigen::Matrix3f& f2i = model.f2iOrientation.toRotationMatrix();
	return {
			{model.size.x(), model.size.y()},
			1/model.focalLength,
			{model.principalPoint.x(), model.principalPoint.y()},
			model.distortionK2,
			{
				i2f(0, 0), i2f(0, 1), i2f(0, 2),
				i2f(1, 0), i2f(1, 1), i2f(1, 2),
				i2f(2, 0), i2f(2, 1), i2f(2, 2)
			},
			{model.pos.x(), model.pos.y(), model.pos.z()},
			{(field.field_length() + 2*field.boundary_width())/10, (field.field_width() + 2*field.boundary_width())/10},
			model.focalLength,
			{
				f2i(0, 0), f2i(0, 1), f2i(0, 2),
				f2i(1, 0), f2i(1, 1), f2i(1, 2),
				f2i(2, 0), f2i(2, 1), f2i(2, 2)
			}
	};
}

inline static bool inRange(V2 a, V2 b, double sqInner, double sqRadius) {
	V2 diff = {a.x - b.x, a.y - b.y};
	double sqr = diff.x*diff.x + diff.y*diff.y;
	return sqr >= sqInner && sqr <= sqRadius;
}
