#include "Perspective.h"

#include <cfloat>
#include <cmath>

void Perspective::geometryCheck(int width, int height) {
	if(socket->getGeometryVersion() == geometryVersion)
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

RLEVector Perspective::getRing(V2 pos, double height, double inner, double radius) {
	V2 root = image2field(pos, height);
	double sqInner = inner*inner;
	double sqRadius = radius*radius;
	RLEVector result;
	if(inner == 0)
		result.add(pos.x, pos.y);

	//TODO outside of perspective
	//TODO more accurate size (due to distortion)
	V2 min = pos;
	while(inRange(root, image2field({min.x-1, pos.y}, height), 0, sqRadius) && min.x > 0)
		min.x--;
	while(inRange(root, image2field({pos.x, min.y-1}, height), 0, sqRadius) && min.y > 0)
		min.y--;

	V2 max = pos;
	while(inRange(root, image2field({max.x+1, pos.y}, height), 0, sqRadius) && max.x+1 < getWidth())
		max.x++;
	while(inRange(root, image2field({pos.x, max.y+1}, height), 0, sqRadius) && max.y+1 < getHeight())
		max.y++;

	for(int x = min.x; x <= max.x; x++) {
		for(int y = min.y; y <= max.y; y++) {
			if(inRange(root, image2field({(double)x, (double)y}, height), sqInner, sqRadius))
				result.add(x, y);
		}
	}
	return result;
}
