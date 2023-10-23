#ifndef CL_VERSION_1_0
#include "clstd.h"

#include "image2field.cl"
#define RGGB
#endif



kernel void ssd(global const uchar* img, global const int* pos, global float* out, const Perspective perspective, const float height, const float radius, const uchar r, const uchar g, const uchar b) {
	const int xpos = pos[2*get_global_id(0)];
	const int ypos = pos[2*get_global_id(0)+1];
	const float sqRadius = radius*radius;

	V2 wpos = image2field(perspective, height, (V2) {(float)xpos, (float)ypos});

	int minx = xpos;
	int maxx = xpos;
	int miny = ypos;
	int maxy = ypos;

	bool inRange = true;
	while(inRange && minx >= 0) {
		V2 mpos = image2field(perspective, height, (V2) {(float)--minx, (float)ypos});
		V2 diff = {mpos.x - wpos.x, mpos.y - wpos.y};
		inRange = diff.x*diff.x + diff.y*diff.y <= sqRadius && !isnan(diff.x);
	}

	inRange = true;
	while(inRange && maxx <= perspective.shape[0]) {
		V2 mpos = image2field(perspective, height, (V2) {(float)++maxx, (float)ypos});
		V2 diff = {mpos.x - wpos.x, mpos.y - wpos.y};
		inRange = diff.x*diff.x + diff.y*diff.y <= sqRadius && !isnan(diff.x);
	}

	inRange = true;
	while(inRange && miny >= 0) {
		V2 mpos = image2field(perspective, height, (V2) {(float)xpos, (float)--miny});
		V2 diff = {mpos.x - wpos.x, mpos.y - wpos.y};
		inRange = diff.x*diff.x + diff.y*diff.y <= sqRadius && !isnan(diff.x);
	}

	inRange = true;
	while(inRange && maxy <= perspective.shape[1]) {
		V2 mpos = image2field(perspective, height, (V2) {(float)xpos, (float)++maxy});
		V2 diff = {mpos.x - wpos.x, mpos.y - wpos.y};
		inRange = diff.x*diff.x + diff.y*diff.y <= sqRadius && !isnan(diff.x);
	}

	minx = 2*(minx+1);
	maxx = 2*(maxx-1);
	miny = 2*(miny+1);
	maxy = 2*(maxy-1);

	float sum = 0.0f;
	int n = 0;
	for(int y = miny; y < maxy; y++) {
		for(int x = minx; x < maxx; x++) {
			V2 mpos = image2field(perspective, height, (V2) {(float)x/2, (float)y/2});
			V2 diff = {mpos.x - wpos.x, mpos.y - wpos.y};
			float rdiff = diff.x*diff.x + diff.y*diff.y;
			//TODO background modeling
			if(rdiff <= sqRadius) {
#ifdef RGGB
				float vdiff;
				if(y%2 == 0) {
					if(x%2 == 0) {
						//sum += img[x + y*perspective.shape[0]] * r;
						vdiff = img[x + y*perspective.shape[0]] - (1-rdiff/sqRadius) * r;
					} else {
						//sum += img[x + y*perspective.shape[0]] * g;
						vdiff = img[x + y*perspective.shape[0]] - (1-rdiff/sqRadius) * g;
					}
				} else {
					if(x%2 == 0) {
						//sum += img[x + y*perspective.shape[0]] * g;
						vdiff = img[x + y*perspective.shape[0]] - (1-rdiff/sqRadius) * g;
					} else {
						//sum += img[x + y*perspective.shape[0]] * b;
						vdiff = img[x + y*perspective.shape[0]] - (1-rdiff/sqRadius) * b;
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
