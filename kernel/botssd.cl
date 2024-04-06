#ifndef CL_VERSION_1_0
#include "clstd.h"

#include "image2field.cl"
#define RGGB
#else
#include "kernel/image2field.cl"
#endif


kernel void ssd(global const uchar* img, global const int* pos, global float* out, const Perspective perspective, const float height, const float radius, const RGB rgb, const float bgradius, const RGB bg) {
#ifdef RGGB
	const int xpos = 2*pos[2*get_global_id(0)];
	const int ypos = 2*pos[2*get_global_id(0)+1];
#endif

	const float sqRadius = radius*radius;
	const float sqBgRadius = bgradius*bgradius;
	V2 center = image2field(perspective, height, (V2) {(float)xpos, (float)ypos});

	I2 min, max;
	area(perspective, height, (I2) {xpos, ypos}, center, sqBgRadius, &min, &max);

	float sum = 0.0f;
	int n = 0;
	int xstep = 1 + (max.x - min.x) / 16;
	int ystep = 1 + (max.y - min.y) / 16;
	for(int y = min.y; y < max.y; y+=ystep) {
		for(int x = min.x; x < max.x; x+=xstep) {
			V2 mpos = image2field(perspective, height, (V2) {(float)x, (float)y});
			V2 diff = {mpos.x - center.x, mpos.y - center.y};
			float dist = diff.x*diff.x + diff.y*diff.y;
			if(dist <= sqBgRadius) {
				float vdiff;

				if(dist <= sqRadius) {
#ifdef RGGB
					if(y%2 == 0) {
						if(x%2 == 0)
							vdiff = img[x + y*perspective.shape[0]] - (float)rgb.r;
						else
							vdiff = img[x + y*perspective.shape[0]] - (float)rgb.g;
					} else {
						if(x%2 == 0)
							vdiff = img[x + y*perspective.shape[0]] - (float)rgb.g;
						else
							vdiff = img[x + y*perspective.shape[0]] - (float)rgb.b;
					}
#endif
				} else {
#ifdef RGGB
					if(y%2 == 0) {
						if(x%2 == 0)
							vdiff = img[x + y*perspective.shape[0]] - (float)bg.r;
						else
							vdiff = img[x + y*perspective.shape[0]] - (float)bg.g;
					} else {
						if(x%2 == 0)
							vdiff = img[x + y*perspective.shape[0]] - (float)bg.g;
						else
							vdiff = img[x + y*perspective.shape[0]] - (float)bg.b;
					}
#endif
				}

				sum += vdiff*vdiff;
				n++;
			}
		}
	}

	out[get_global_id(0)] = sum / n;
}
