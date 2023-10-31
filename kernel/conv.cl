#ifndef CL_VERSION_1_0
#include "clstd.h"

#include "image2field.cl"
#define RGGB
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
	for(int y = min.y; y < max.y; y++) {
		for(int x = min.x; x < max.x; x++) {
			V2 mpos = image2field(perspective, height, (V2) {(float)x, (float)y});
			V2 diff = {mpos.x - center.x, mpos.y - center.y};
			float dist = diff.x*diff.x + diff.y*diff.y;
			if(dist <= sqBgRadius) {
				uchar reference;
				if(dist <= sqRadius) {
#ifdef RGGB
					if(y%2 == 0) {
						if(x%2 == 0)
							reference = rgb.r;
						else
							reference = rgb.g;
					} else {
						if(x%2 == 0)
							reference = rgb.g;
						else
							reference = rgb.b;
					}
#endif
				} else {
#ifdef RGGB
					if(y%2 == 0) {
						if(x%2 == 0)
							reference = rgb.r;
						else
							reference = rgb.g;
					} else {
						if(x%2 == 0)
							reference = rgb.g;
						else
							reference = rgb.b;
					}
#endif
				}

				char value = (char)img[x + y*perspective.shape[0]] - 128;
				sum += (float)value * ((char)reference - 128);
				n++;
			}
		}
	}

	out[get_global_id(0)] = sum / n;
}
