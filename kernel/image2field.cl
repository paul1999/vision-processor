#ifndef CL_VERSION_1_0
#include "clstd.h"
#endif

typedef struct __attribute__ ((packed)) {
	int shape[2];  // raw image shape
	float f;       // 1/(focal length)
	float p[2];    // principal point
	float d;       // distortion
	float r[9]; // rotation matrix
	float c[3];    // camera position
} Perspective;

typedef struct {
	float x;
	float y;
} V2;

typedef struct {
	float x;
	float y;
	float z;
} V3;

inline V2 image2field(const Perspective p, const float height, const V2 pos) {
	V2 normalized = {
			(pos.x - p.p[0]) * p.f,
			(pos.y - p.p[1]) * p.f
	};

	double distortion = 1.0 + (normalized.x*normalized.x + normalized.y*normalized.y) * p.d;
	normalized.x *= distortion;
	normalized.y *= distortion;

	V3 camRay = {
			p.r[0] * normalized.x + p.r[1] * normalized.y + p.r[2],
			p.r[3] * normalized.x + p.r[4] * normalized.y + p.r[5],
			p.r[6] * normalized.x + p.r[7] * normalized.y + p.r[8],
	};

	if(camRay.z >= 0) // Over horizon
		return (V2) { NAN, NAN };

	float zFactor = (p.c[2] + height) / camRay.z;
	return (V2) {
			camRay.x*zFactor - p.c[0],
			camRay.y*zFactor - p.c[1]
	};
}