#ifndef CL_VERSION_1_0
#include "clstd.h"
#endif


kernel void sqridge(global const uchar* img, global uchar* out, const int xStride, int yStride) {
	const int width = get_global_size(0);
	const int imgWidth = width+2*xStride;
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	yStride *= imgWidth;
	const int center = x+xStride + y*imgWidth+yStride;

	/*if(
			img[center] > img[center-xStride] &&
			img[center] > img[center+xStride] &&
			img[center] > img[center-yStride] &&
			img[center] > img[center+yStride]
	) {
		//out[x + y*imgWidth] = img[center];
		const short h = 2*(short)img[center] - img[center-xStride] - img[center+xStride]; //TODO not only greater, but different values
		const short v = 2*(short)img[center] - img[center-yStride] - img[center+yStride];
		out[x + y*imgWidth] = convert_uchar_sat((h + v) >> 2);
	} else
		out[x + y*imgWidth] = 0;*/

	//const short h = 2*(short)img[center] - img[center-xStride] - img[center+xStride];
	//const short v = 2*(short)img[center] - img[center-yStride] - img[center+yStride];
	//out[x + y*imgWidth] = h > 0 && v > 0 ? convert_uchar_sat((h + v) >> 2) : 0;
	//out[x + y*imgWidth] = convert_uchar_sat((h + v) >> 2);

	out[x + y*imgWidth] = convert_uchar_sat((
			8*(short)img[center]
			- img[center-xStride-yStride]
			- img[center        -yStride]
			- img[center+xStride-yStride]
			- img[center-xStride]
			- img[center+xStride]
			- img[center-xStride+yStride]
			- img[center        +yStride]
			- img[center+xStride+yStride]
	) / 8);
}
