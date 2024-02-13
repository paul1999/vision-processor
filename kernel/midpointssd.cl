#ifndef CL_VERSION_1_0
#include "clstd.h"

#include "image2field.cl"
#define RGGB
#endif


inline void px(global const uchar* img, const Perspective perspective, const RGB rgb, const int x, const int y, float* sum, int* n) {
	if(x < 0 || y < 0 || x >= perspective.shape[0] || y >= perspective.shape[1])
		return;

	(*n)++;
	float vdiff;
#ifdef RGGB
	//TODO thicker line
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
#endif
	*sum += vdiff*vdiff;
}


kernel void ssd(global const uchar* img, global const int* pos, global float* out, const Perspective perspective, const float height, const float radius, const float rradius, const RGB rgb) {
#ifdef RGGB
	const int xpos = 2*pos[2*get_global_id(0)];
	const int ypos = 2*pos[2*get_global_id(0)+1];
#endif

	V2 center = image2field(perspective, height, (V2) {(float)xpos, (float)ypos});
	V2 offcenter = image2field(perspective, height, (V2) {(float)xpos+1, (float)ypos+1});
	V2 posdiff = {offcenter.x-center.x, offcenter.y-center.y};
	float rPerPixel = native_sqrt(posdiff.x*posdiff.x + posdiff.y*posdiff.y);
	float err = radius/rPerPixel;
	int x = round(err);
	int y = 0;
	err = err - x;

	float sum = 0.0f;
	int n = 0;
	while(x >= y) {
		px(img, perspective, rgb, xpos+x, ypos+y, &sum, &n);
		px(img, perspective, rgb, xpos+y, ypos+x, &sum, &n);
		px(img, perspective, rgb, xpos-y, ypos+x, &sum, &n);
		px(img, perspective, rgb, xpos-x, ypos+y, &sum, &n);
		px(img, perspective, rgb, xpos-x, ypos-y, &sum, &n);
		px(img, perspective, rgb, xpos-y, ypos-x, &sum, &n);
		px(img, perspective, rgb, xpos+y, ypos-x, &sum, &n);
		px(img, perspective, rgb, xpos+x, ypos-y, &sum, &n);

		if(err <= 0) {
			y += 1;
			err += 2*y + 1;
		}
		if(err > 0) {
			x -= 1;
			err += 2*x + 1;
		}
	}

	out[get_global_id(0)] = sum / n;
}
