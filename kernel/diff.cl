#ifndef CL_VERSION_1_0
#include "clstd.h"
#endif


kernel void diff(global const uchar* img, global uchar* out, const int xStride, const int yStride) {
	const int width = get_global_size(0);
	const int imgWidth = width+xStride;
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int imgPos = x + y*imgWidth;

	const float h = (float)img[imgPos+xStride] - img[imgPos];
	const float v = (float)img[imgPos+yStride*imgWidth] - img[imgPos];
	out[x + y*imgWidth] = convert_uchar_sat(native_sqrt(h*h + v*v));
	//out[x + y*imgWidth] = (h*h + v*v) >> 9; (with h and v as short)
}
