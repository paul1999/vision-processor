#pragma once

#include "proto/ssl_vision_geometry.pb.h"
#include "udpsocket.h"
#include "RLEVector.h"

struct V2 {
	double x, y;
};

struct V3{
	double x, y, z;
};

struct V4{
	double q0, q1, q2, q3;
};

typedef struct __attribute__ ((packed)) {
	int shape[2];  // raw image shape
	float f;       // 1/(focal length)
	float p[2];    // principal point
	float d;       // distortion
	float r[9]; // rotation matrix
	float c[3];    // camera position
} ClPerspective;


class Perspective {
public:
	Perspective(std::shared_ptr<VisionSocket> socket, int camId): socket(std::move(socket)), camId(camId) {}
	void geometryCheck();

	V2 image2field(V2 pos, double height);
	V2 field2image(V3 pos);

	int getGeometryVersion();
	int getWidth();
	int getHeight();
	int getFieldLength();
	int getFieldWidth();
	int getBoundaryWidth();

	ClPerspective getClPerspective();
	RLEVector getRing(V2 pos, double height, double inner, double radius);

private:
	const std::shared_ptr<VisionSocket> socket;
	const int camId;

	int geometryVersion = 0;
	SSL_GeometryFieldSize field;
	SSL_GeometryCameraCalibration calib;
	V4 orientation;
	V3 cameraPos;
	V3 rX, rY, rZ;
	V3 rXinv, rYinv, rZinv;
};
