#ifndef CL_VERSION_1_0
#include "clstd.h"

#include "image2field.cl"
#define RGGB
#endif


inline void px(global const uchar* img, const Perspective perspective, const RGB rgb, const int x, const int y, float* sum, int* n) {
	if(x < 0 || y < 0 || x >= perspective.shape[0]/2 || y >= perspective.shape[1]/2) //TODO is RGGB only
		return;

	(*n)++;
	float vdiff;
#ifdef RGGB
	vdiff = img[2*x + 2*y*perspective.shape[0]] - (float)rgb.r;
	*sum += vdiff*vdiff;
	vdiff = img[2*x+1 + 2*y*perspective.shape[0]] - (float)rgb.g;
	*sum += vdiff*vdiff;
	vdiff = img[2*x + (2*y+1)*perspective.shape[0]] - (float)rgb.g;
	*sum += vdiff*vdiff;
	vdiff = img[2*x+1 + (2*y+1)*perspective.shape[0]] - (float)rgb.b;
	*sum += vdiff*vdiff;
#endif
}

//https://www.thecrazyprogrammer.com/2016/12/bresenhams-midpoint-circle-algorithm-c-c.html
kernel void ssd(global const uchar* img, global const int* pos, global float* out, const Perspective perspective, const float height, const float radius, const RGB rgb) {
	int xpos = pos[2*get_global_id(0)]; //TODO pos as float
	int ypos = pos[2*get_global_id(0)+1];
#ifdef RGGB
	V2 center = image2field(perspective, height, (V2) {(float)2*xpos, (float)2*ypos});
	V2 offcenter = image2field(perspective, height, (V2) {(float)2*xpos+2, (float)2*ypos+2});
#endif

	V2 posdiff = {offcenter.x-center.x, offcenter.y-center.y};
	float rPerPixel = native_sqrt(posdiff.x*posdiff.x + posdiff.y*posdiff.y) * 0.5f;
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
