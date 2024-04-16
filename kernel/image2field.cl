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

	float distortion = 1.0f + (normalized.x*normalized.x + normalized.y*normalized.y) * p.d;
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

inline int inRange(V2 a, V2 b, double sqRadius) {
	V2 diff = {a.x - b.x, a.y - b.y};
	double sqr = diff.x*diff.x + diff.y*diff.y;
	return sqr <= sqRadius;
}

typedef struct {
	int x;
	int y;
} I2;

typedef struct __attribute__ ((packed)) {
	uchar r;
	uchar g;
	uchar b;
} RGB;

inline void area(const Perspective perspective, const float height, const I2 pos, const V2 center, const float sqRadius, I2* min, I2* max) {
	min->x = pos.x;
	min->y = pos.y;
	max->x = pos.x;
	max->y = pos.y;

	while(inRange(center, image2field(perspective, height, (V2) {(float)min->x-1, (float)pos.y}), sqRadius))
		min->x--;
	while(inRange(center, image2field(perspective, height, (V2) {(float)max->x+1, (float)pos.y}), sqRadius))
		max->x++;
	while(inRange(center, image2field(perspective, height, (V2) {(float)pos.x, (float)min->y-1}), sqRadius))
		min->y--;
	while(inRange(center, image2field(perspective, height, (V2) {(float)pos.x, (float)max->y+1}), sqRadius))
		max->y++;
	if(min->x < 0)
		min->x = 0;
	if(min->y < 0)
		min->y = 0;
	if(max->x >= perspective.shape[0])
		max->x = perspective.shape[0];
	if(max->y >= perspective.shape[1])
		max->y = perspective.shape[1];
}


typedef struct __attribute__ ((packed)) {
	int2 shape;  // raw image shape
	float f;       // 1/(focal length)
	float2 p;    // principal point
	float d;       // distortion
	float r[9]; // rotation matrix
	float3 c;    // camera position
} CLPerspective;

inline float2 clImage2field(const Perspective p, const float height, const int2 pos) {
	float2 normalized = (convert_float2(pos) - (float2)(p.p[0], p.p[1])) * p.f;

	float distortion = 1.0f + (normalized.x*normalized.x + normalized.y*normalized.y) * p.d;
	normalized *= distortion;

	float3 camRay = (float3)(
			p.r[0] * normalized.x + p.r[1] * normalized.y + p.r[2],
			p.r[3] * normalized.x + p.r[4] * normalized.y + p.r[5],
			p.r[6] * normalized.x + p.r[7] * normalized.y + p.r[8]
	);

	if(camRay.z >= 0) // Over horizon
		return (float2)(NAN, NAN);

	float zFactor = (p.c[2] + height) / camRay.z;
	return (float2)(
			camRay.x*zFactor - p.c[0],
			camRay.y*zFactor - p.c[1]
	);
}

inline int clInRange(float2 a, float2 b, double sqRadius) {
	float2 diff = a - b;
	diff *= diff;
	return diff.x + diff.y <= sqRadius;
}

inline void clArea(const Perspective perspective, const float height, const int2 pos, const float2 center, const float sqRadius, int2* min, int2* max) {
	min->x = pos.x;
	min->y = pos.y;
	max->x = pos.x;
	max->y = pos.y;

	while(clInRange(center, clImage2field(perspective, height, (int2) (min->x-1, pos.y)), sqRadius))
		min->x--;
	while(clInRange(center, clImage2field(perspective, height, (int2) (max->x+1, pos.y)), sqRadius))
		max->x++;
	while(clInRange(center, clImage2field(perspective, height, (int2) (pos.x, min->y-1)), sqRadius))
		min->y--;
	while(clInRange(center, clImage2field(perspective, height, (int2) (pos.x, max->y+1)), sqRadius))
		max->y++;

	if(min->x < 0)
		min->x = 0;
	if(min->y < 0)
		min->y = 0;
	if(max->x >= perspective.shape[0])
		max->x = perspective.shape[0];
	if(max->y >= perspective.shape[1])
		max->y = perspective.shape[1];
}