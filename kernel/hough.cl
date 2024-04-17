#ifndef CL_VERSION_1_0
#include "clstd.h"

#include "image2field.cl"
#else
#include "kernel/image2field.cl"
#endif

inline void px(volatile global int* votes, const Perspective p, int2 pos, const int dx, const int dy) {
	pos.x += dx;
	pos.y += dy;
	if(pos.x < 0 || pos.y < 0 || pos.x >= p.field[0] || pos.y >= p.field[1])
		return;

	const int i = pos.x + pos.y*p.field[0];
	atomic_inc(votes+i);
}

const sampler_t sampler = CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE;

kernel void hough(read_only image2d_t gx, read_only image2d_t gy, volatile global int* votes, const Perspective perspective, const float height) {
	int2 imgpos = (int2)(get_global_id(0), get_global_id(1));
	float4 gpx = convert_float4(read_imagei(gx, sampler, imgpos));
	float4 gpy = convert_float4(read_imagei(gy, sampler, imgpos));
	//TODO color gradient
	float abs2 = gpx.x*gpx.x + gpx.y*gpx.y + gpx.z*gpx.z + gpy.x*gpy.x + gpy.y*gpy.y + gpy.z*gpy.z;
	if(abs2 < 512.0f*512.0f)
		return;

	V2 pos = image2field(perspective, height, (V2) {(float)imgpos.x, (float)imgpos.y});
	pos.x = pos.x/10 + perspective.field[0]/2;
	pos.y = pos.y/10 + perspective.field[1]/2;
	int2 fieldpos = {(int)pos.x, (int)pos.y};

	//TODO voting

	int err = 0;
	int x = 3;
	int y = 0;

	while(x >= y) {
		px(votes, perspective, fieldpos, +x, +y);
		px(votes, perspective, fieldpos, -y, +x);
		px(votes, perspective, fieldpos, -x, -y);
		px(votes, perspective, fieldpos, +y, -x);
		if(x > y && y > 0) {
			px(votes, perspective, fieldpos, +y, +x);
			px(votes, perspective, fieldpos, -x, +y);
			px(votes, perspective, fieldpos, -y, -x);
			px(votes, perspective, fieldpos, +x, -y);
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
}
