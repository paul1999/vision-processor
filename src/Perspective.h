#pragma once

#include "proto/ssl_vision_geometry.pb.h"
#include "udpsocket.h"
#include "CameraModel.h"

struct V2 {
	double x, y;
};

struct V3{
	double x, y, z;
};

typedef struct __attribute__ ((packed)) {
	int shape[2];  // raw image shape
	float f;       // 1/(focal length)
	float p[2];    // principal point
	float d;       // distortion
	float r[9]; // rotation matrix
	float c[3];    // camera position
	int field[2]; // field size incl. boundary in cm
	float fInv;
	float rInv[9];
} ClPerspective;


class Perspective {
public:
	Perspective(std::shared_ptr<VisionSocket> socket, int camId): socket(std::move(socket)), camId(camId) {}
	void geometryCheck(int width, int height, const double maxBotHeight);

	V2 image2field(V2 pos, double height) const;
	V2 field2image(V3 pos) const;

	int getWidth();
	int getHeight();
	int getFieldLength();
	int getFieldWidth();
	int getBoundaryWidth();

	ClPerspective getClPerspective() const;

	SSL_GeometryFieldSize field;
	CameraModel model;

	Eigen::Vector4f visibleFieldExtent; // xmin, xmax, ymin, ymax
	float optimalFieldScale = 5.f; // [mm/px]
	Eigen::Vector2i reprojectedFieldSize = Eigen::Vector2i(0, 0);

	int geometryVersion = 0;

private:
	const std::shared_ptr<VisionSocket> socket;
	const int camId;
};
