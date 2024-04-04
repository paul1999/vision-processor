#ifndef CL_VERSION_1_0
#include "clstd.h"

#include "image2field.cl"
#define RGGB
#else
#include "kernel/image2field.cl"
#endif


#define ADD_SQ_COSSIM(pos) vdiff = dx*(float)gx[pos] + dy*(float)gy[pos]; *cossum += fabs(vdiff) / ((float)abs[pos]*native_sqrt((float)dx*dx+dy*dy) + 0.00001);
inline void px(global const uchar* abs, global const char* gx, global const char* gy, const Perspective perspective, const RGB rgb, const int xpos, const int ypos, const int dx, const int dy, float* sum, float* cossum, int* n) {
	const int x = xpos+dx;
	const int y = ypos+dy;
	if(x < 0 || y < 0 || x >= perspective.shape[0]/2 || y >= perspective.shape[1]/2) //TODO is RGGB only
		return;

	const int pos = 2*x + 2*y*perspective.shape[0];
	(*n)++;
	float vdiff;
#ifdef RGGB
	vdiff = abs[pos] - (float)rgb.r;
	*sum += vdiff*vdiff;
	vdiff = abs[pos+1] - (float)rgb.g;
	*sum += vdiff*vdiff;
	vdiff = abs[pos+perspective.shape[0]] - (float)rgb.g;
	*sum += vdiff*vdiff;
	vdiff = abs[pos+1+perspective.shape[0]] - (float)rgb.b;
	*sum += vdiff*vdiff;

	ADD_SQ_COSSIM(pos)
	ADD_SQ_COSSIM(pos+1)
	ADD_SQ_COSSIM(pos+perspective.shape[0])
	ADD_SQ_COSSIM(pos+1+perspective.shape[0])
#endif
}

//https://www.thecrazyprogrammer.com/2016/12/bresenhams-midpoint-circle-algorithm-c-c.html
kernel void ssd(global const uchar* abs, global const char* gx, global const char* gy, global const int* pos, global float* out, const Perspective perspective, const float height, const float radius, const RGB rgb) {
	int xpos = pos[2*get_global_id(0)]; //TODO pos as float
	int ypos = pos[2*get_global_id(0)+1];
#ifdef RGGB
	V2 center = image2field(perspective, height, (V2) {(float)2*xpos, (float)2*ypos});
	//V2 offcenter = image2field(perspective, height, (V2) {(float)2*xpos+2, (float)2*ypos+2});
	V2 offcenter = image2field(perspective, height, (V2) {(float)2*xpos+2, (float)2*ypos});
#endif

	V2 posdiff = {offcenter.x-center.x, offcenter.y-center.y};
	//Field coordinates! 1 px -> mm
	//Half size should not be correct size
	float rPerPixel = native_sqrt(posdiff.x*posdiff.x + posdiff.y*posdiff.y);// * 0.5f;// * 1.25f;
	//float err = radius/rPerPixel;
	int err = round(radius/rPerPixel);
	//int x = round(err);
	int x = err;
	int y = 0;
	err = err - x;

	float sum = 0.0f;
	float cossum = 0.0f;
	int n = 0;
	while(x >= y) {
		px(abs, gx, gy, perspective, rgb, xpos, ypos, +x, +y, &sum, &cossum, &n);
		px(abs, gx, gy, perspective, rgb, xpos, ypos, -y, +x, &sum, &cossum, &n);
		px(abs, gx, gy, perspective, rgb, xpos, ypos, -x, -y, &sum, &cossum, &n);
		px(abs, gx, gy, perspective, rgb, xpos, ypos, +y, -x, &sum, &cossum, &n);
		//TODO no change in style with that change
		if(x > y && y > 0) {
			px(abs, gx, gy, perspective, rgb, xpos, ypos, +y, +x, &sum, &cossum, &n);
			px(abs, gx, gy, perspective, rgb, xpos, ypos, -x, +y, &sum, &cossum, &n);
			px(abs, gx, gy, perspective, rgb, xpos, ypos, -y, -x, &sum, &cossum, &n);
			px(abs, gx, gy, perspective, rgb, xpos, ypos, +x, -y, &sum, &cossum, &n);
		}

		if(err <= 0) {
			y += 1;
			err += 2*y + 1;
		}
		if(err > 0) {
			x -= 1;
			err += 2*x + 1;
		}
	}

	//out[get_global_id(0)] = sum / (n*cossum);
	out[get_global_id(0)] = sum / cossum;
	//out[get_global_id(0)] = cossum; //TODO cossum contains hard edges (partially depending on n, partially on other)
	//out[get_global_id(0)] = 1/cossum;
	//out[get_global_id(0)] = n/cossum;
	//out[get_global_id(0)] = rPerPixel;
	//out[get_global_id(0)] = n;
	//out[get_global_id(0)] = sum / n;
}
