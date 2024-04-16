#ifndef CL_VERSION_1_0
#include "clstd.h"
#endif

const sampler_t sampler = CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE;

static inline float4 read(read_only image2d_t img, int x, int y) {
	return convert_float4(read_imageui(img, sampler, (int2)(x, y)));
}

kernel void blur(read_only image2d_t img, write_only image2d_t blur) {
	const int2 pos = (int2)(get_global_id(0), get_global_id(1));
	write_imageui(blur, pos, convert_uint4(
				 read(img, pos.x-1, pos.y-1) * 0.0625f +
				 read(img, pos.x  , pos.y-1) * 0.1250f +
				 read(img, pos.x+1, pos.y-1) * 0.0625f +
				 read(img, pos.x-1, pos.y  ) * 0.1250f +
				 read(img, pos.x  , pos.y  ) * 0.2500f +
				 read(img, pos.x+1, pos.y  ) * 0.1250f +
				 read(img, pos.x-1, pos.y+1) * 0.0625f +
				 read(img, pos.x  , pos.y+1) * 0.1250f +
				 read(img, pos.x+1, pos.y+1) * 0.0625f
	));
}

/*kernel void blur(global const uchar* img, global char* blurred, const int xStride, const int yStride) {
	const int width = get_global_size(0);
	const int height = get_global_size(1);
	const int imgWidth = width;
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int imgPos = x + y*imgWidth;

	float v = 0.5f * img[imgPos];
	if(x >= xStride && y >= yStride)
		v += 0.25f * img[imgPos - xStride - yStride*imgWidth];
	if(x < width-xStride && y < height-yStride)
		v += 0.25f * img[imgPos + xStride + yStride*imgWidth];

	blurred[imgPos] = convert_uchar_sat(v);
}*/
