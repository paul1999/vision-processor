#ifndef CL_VERSION_1_0
#include "clstd.h"

#include "image2field.cl"
#define RGGB
#else
#include "kernel/image2field.cl"
#endif


kernel void ssd(global const uchar* img, global const int* pos, global float* out, const Perspective perspective, const float height, const float radius, const RGB rgb) {
#ifdef RGGB
	const int xpos = 2*pos[2*get_global_id(0)];
	const int ypos = 2*pos[2*get_global_id(0)+1];
#endif

	const float sqRadius = radius*radius;
	V2 center = image2field(perspective, height, (V2) {(float)xpos, (float)ypos});

	I2 min, max;
	area(perspective, height, (I2) {xpos, ypos}, center, sqRadius, &min, &max);

	float sum = 0.0f;
	int n = 0;
	int xstep = 1 + (max.x - min.x) / 16;
	int ystep = 1 + (max.y - min.y) / 16;
	for(int y = min.y; y < max.y; y+=ystep) {
		for(int x = min.x; x < max.x; x+=xstep) {
			V2 mpos = image2field(perspective, height, (V2) {(float)x, (float)y});
			V2 diff = {mpos.x - center.x, mpos.y - center.y};
			float dist = diff.x*diff.x + diff.y*diff.y;
			if(dist <= sqRadius) {
#ifdef RGGB
				float vdiff;
				if(y%2 == 0) {
					if(x%2 == 0) {
						//sum += img[x + y*perspective.shape[0]] * r;
						vdiff = img[x + y*perspective.shape[0]] - (1-dist/sqRadius) * rgb.r;
						//vdiff = img[x + y*perspective.shape[0]] - (float)rgb.r;
					} else {
						//sum += img[x + y*perspective.shape[0]] * g;
						vdiff = img[x + y*perspective.shape[0]] - (1-dist/sqRadius) * rgb.g;
						//vdiff = img[x + y*perspective.shape[0]] - (float)rgb.g;
					}
				} else {
					if(x%2 == 0) {
						//sum += img[x + y*perspective.shape[0]] * g;
						vdiff = img[x + y*perspective.shape[0]] - (1-dist/sqRadius) * rgb.g;
						//vdiff = img[x + y*perspective.shape[0]] - (float)rgb.g;
					} else {
						//sum += img[x + y*perspective.shape[0]] * b;
						vdiff = img[x + y*perspective.shape[0]] - (1-dist/sqRadius) * rgb.b;
						//vdiff = img[x + y*perspective.shape[0]] - (float)rgb.b;
					}
				}
				sum += vdiff*vdiff;
#endif
				n++;
			}
		}
	}

	out[get_global_id(0)] = sum / n;
}
