#ifndef CL_VERSION_1_0
#include "clstd.h"

#include "image2field.cl"
#define RGGB
#endif


kernel void ssd(global const uchar* img, global const int* pos, global float* out, const Perspective perspective, const float height, const float radius, const float rradius, const RGB rgb) {
#ifdef RGGB
	const int xpos = pos[2*get_global_id(0)];
	const int ypos = pos[2*get_global_id(0)+1];
#endif

	V2 center = image2field(perspective, height, (V2) {(float)xpos, (float)ypos});

	I2 minp, maxp;
	//TODO bug (position of rifts) is circle size and/or height dependent (different pattern)
	//TODO fixed radius does not affect the bug (asymmetry also not)
	//TODO color only affects effect strength
	area(perspective, height, (I2) {xpos, ypos}, center, (radius+rradius)*(radius+rradius), &minp, &maxp);
	//minp.x = max(0, xpos-9);
	//minp.y = max(0, ypos-9);
	//maxp.x = min(perspective.shape[0], xpos+9);
	//maxp.y = min(perspective.shape[1], ypos+9);
	//maxp.x = min(perspective.shape[0], xpos+10);
	//maxp.y = min(perspective.shape[1], ypos+10);

	float sum = 0.0f;
	float n = 0.0f;
	int xstep = 1;// + (maxp.x - minp.x) / 16;
	int ystep = 1;// + (maxp.y - minp.y) / 16;
	for(int y = minp.y; y < maxp.y; y+=ystep) {
		for(int x = minp.x; x < maxp.x; x+=xstep) {
			V2 mpos = image2field(perspective, height, (V2) {(float)x, (float)y});
			V2 diff = {mpos.x - center.x, mpos.y - center.y};
			float dist = native_sqrt(diff.x*diff.x + diff.y*diff.y);
			float range = fabs(dist - radius);
			if(range <= rradius) {
			//if(dist <= radius) {
#ifdef RGGB
				float vdiff;
				if(y%2 == 0) {
					if(x%2 == 0) {
						vdiff = img[x + y*perspective.shape[0]] - (float)rgb.r;
					} else {
						vdiff = img[x + y*perspective.shape[0]] - (float)rgb.g;
					}
				} else {
					if(x%2 == 0) {
						vdiff = img[x + y*perspective.shape[0]] - (float)rgb.g;
					} else {
						vdiff = img[x + y*perspective.shape[0]] - (float)rgb.b;
					}
				}
				sum += vdiff*vdiff;
#endif
				n++;
			}
		}
	}

	out[get_global_id(0)] = sum / n;
	//out[get_global_id(0)] = sum;
	//out[get_global_id(0)] = n;
	//out[get_global_id(0)] = (maxp.x - minp.x)*(maxp.y - minp.y);
}
