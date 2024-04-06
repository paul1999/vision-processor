#ifndef CL_VERSION_1_0
#include "clstd.h"
#endif


kernel void blur(global const uchar* img, global char* blurred, const int xStride, const int yStride) {
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
}
