#ifndef CL_VERSION_1_0
#include "clstd.h"
#endif

const sampler_t sampler = CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE;

static inline int4 read(read_only image2d_t img, int x, int y) {
	return convert_int4(read_imageui(img, sampler, (int2)(x, y)));
}

kernel void diff(read_only image2d_t img, write_only image2d_t gx, write_only image2d_t gy) {
	int2 pos = (int2)(get_global_id(0), get_global_id(1));

	//write_imagei(gx, pos, read(img, pos.x+1, pos.y) - read(img, pos.x-1, pos.y));
	//write_imagei(gy, pos, read(img, pos.x, pos.y+1) - read(img, pos.x, pos.y-1));
	write_imagei(gx, pos, read(img, pos.x+1, pos.y) - read(img, pos.x, pos.y));
	write_imagei(gy, pos, read(img, pos.x, pos.y+1) - read(img, pos.x, pos.y));
}
/*
kernel void sobel(read_only image2d_t img, write_only image2d_t gx, write_only image2d_t gy) {
	int2 pos = (int2)(get_global_id(0), get_global_id(1));

	int4 mm = read(img, pos.x-1, pos.y-1);
	int4 pm = read(img, pos.x+1, pos.y-1);
	int4 mp = read(img, pos.x-1, pos.y+1);
	int4 pp = read(img, pos.x+1, pos.y+1);

	write_imagei(gx, pos, -mm -2*read(img, pos.x-1, pos.y) -mp +pm +2*read(img, pos.x+1, pos.y) +pp);
	write_imagei(gy, pos, -mm -2*read(img, pos.x, pos.y-1) -pm +mp +2*read(img, pos.x, pos.y+1) +pp);
}
*/