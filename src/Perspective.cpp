#include "Perspective.h"

#include <cfloat>
#include <cmath>

static inline V3 rotation(V3 v, V3 x, V3 y, V3 z) {
	return {
		x.x * v.x + y.x * v.y + z.x * v.z,
		x.y * v.x + y.y * v.y + z.y * v.z,
		x.z * v.x + y.z * v.y + z.z * v.z
	};
}

static void quaternionToMatrix(V4 orientation, V3& rX, V3& rY, V3& rZ) {
	double qLength = sqrt(orientation.q0*orientation.q0 + orientation.q1*orientation.q1 + orientation.q2*orientation.q2 + orientation.q3*orientation.q3);
	orientation.q0 /= qLength; orientation.q1 /= qLength; orientation.q2 /= qLength; orientation.q3 /= qLength;

	double x2 = orientation.q0*orientation.q0;
	double y2 = orientation.q1*orientation.q1;
	double z2 = orientation.q2*orientation.q2;
	double xy = orientation.q0*orientation.q1;
	double xz = orientation.q0*orientation.q2;
	double yz = orientation.q1*orientation.q2;
	double wx = orientation.q3*orientation.q0;
	double wy = orientation.q3*orientation.q1;
	double wz = orientation.q3*orientation.q2;
	rX = {1.0 - 2.0 * (y2 + z2), 2.0 * (xy + wz), 2.0 * (xz - wy)};
	rY = {2.0 * (xy - wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz + wx)};
	rZ = {2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (x2 + y2)};
}

void Perspective::geometryCheck() {
	if(socket->getGeometryVersion() == geometryVersion)
		return;

	bool calibFound = false;
	for(const SSL_GeometryCameraCalibration& c : socket->getGeometry().calib()) {
		if(c.camera_id() == camId) {
			calibFound = true;
			calib = c;
			break;
		}
	}

	if(!calibFound)
		return;

	geometryVersion = socket->getGeometryVersion();
	field = socket->getGeometry().field();

	orientation = { -calib.q0(), -calib.q1(), -calib.q2(), calib.q3() };
	quaternionToMatrix(orientation, rX, rY, rZ);
	quaternionToMatrix({ calib.q0(), calib.q1(), calib.q2(), calib.q3() }, rXinv, rYinv, rZinv);

	cameraPos = rotation({calib.tx(), calib.ty(), calib.tz()}, rX, rY, rZ);
}

V2 Perspective::image2field(V2 pos, double height) {
	V2 normalized = {
			(pos.x - calib.principal_point_x()/2) / (calib.focal_length()/2),
			(pos.y - calib.principal_point_y()/2) / (calib.focal_length()/2)
	};
	//normalized = (pos-principal) / focal_length
	// distortion = 1 + a/f*a/f+b/f*b/f = 1 + a^2/f^2 + b^2/f^2
	// a/f * (1 + ka^2/f^2 + kb^2/f^2), a/f * (1 + ka^2/f^2 + kb^2/f^2)

	double distortion = 1.0 + (normalized.x*normalized.x + normalized.y*normalized.y) * calib.distortion(); //TODO inversion
	V3 camRay = rotation({distortion*normalized.x, distortion*normalized.y, 1.0}, rX, rY, rZ);

	if(camRay.z >= 0) { // Over horizon
		std::cout << "Horizon " << pos.x << " " << pos.y << std::endl;
		return { NAN, NAN };
	}

	double zFactor = (cameraPos.z + height) / camRay.z;
	V2 worldRay = {
			camRay.x*zFactor - cameraPos.x,
			camRay.y*zFactor - cameraPos.y
	};

	return worldRay;
}

V2 Perspective::field2image(V3 pos) {
	pos.x += cameraPos.x;
	pos.y += cameraPos.y;
	pos.z += cameraPos.z;

	V3 camRay = rotation(pos, rXinv, rYinv, rZinv);
	camRay.x /= camRay.z;
	camRay.y /= camRay.z;

	//Apply distortion
	if(calib.distortion() >= DBL_MIN) {
		V3 original = camRay;
		for(int i = 0; i < 100; i++) { //TODO optimize
			double r2 = camRay.x*camRay.x + camRay.y*camRay.y;
			double dr = 1 + calib.distortion()*r2;
			camRay.x = original.x/dr; camRay.y = original.y/dr;
		}
		/*double length = sqrt(camRay.x*camRay.x + camRay.y*camRay.y);
		double d = calib.distortion();
		double b = -9.0 * d * d * length + d * sqrt(d * (12.0 + 81.0 * d * length * length));
		b = (b < 0.0) ? (-pow(b, 1.0 / 3.0)) : pow(b, 1.0 / 3.0);
		double distortion = pow(2.0 / 3.0, 1.0 / 3.0) / b - b / (pow(2.0 * 3.0 * 3.0, 1.0 / 3.0) * d);
		camRay.x *= distortion;
		camRay.y *= distortion;*/
	}

	return {
		calib.focal_length()/2 * camRay.x + calib.principal_point_x()/2,
		calib.focal_length()/2 * camRay.y + calib.principal_point_y()/2
	};
}

int Perspective::getGeometryVersion() {
	return geometryVersion;
}

int Perspective::getWidth() {
	return calib.pixel_image_width()/2;
}

int Perspective::getHeight() {
	return calib.pixel_image_height()/2;
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

ClPerspective Perspective::getClPerspective() {
	return {
			{(int)calib.pixel_image_width()/2, (int)calib.pixel_image_height()/2},
			2/calib.focal_length(),
			{calib.principal_point_x()/2, calib.principal_point_y()/2},
			calib.distortion(),
			{
				(float)rX.x, (float)rY.x, (float)rZ.x,
				(float)rX.y, (float)rY.y, (float)rZ.y,
				(float)rX.z, (float)rY.z, (float)rZ.z,
			},
			{(float)cameraPos.x, (float)cameraPos.y, (float)cameraPos.z}
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
