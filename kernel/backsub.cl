#ifndef CL_VERSION_1_0
#include "clstd.h"
#endif


kernel void backsub(global const uchar* img, global uchar* bg, global uchar* mask, const uchar delta) {
	const int width = get_global_size(0);
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int pos = x + y*width;
	const int imgPos = 3*pos;

	uchar m = 0;
	if(abs_diff(img[imgPos+1], bg[imgPos+1]) > delta)
		m = 255;
	if(abs_diff(img[imgPos+2], bg[imgPos+2]) > delta)
		m = 255;
	mask[pos] = m;

	if(!m) {
		bg[imgPos+1] = img[imgPos+1];
		bg[imgPos+2] = img[imgPos+2];
	}
	//bg[imgPos+1] += (img[imgPos+1]-bg[imgPos+1])/delta;
	//bg[imgPos+2] += (img[imgPos+2]-bg[imgPos+2])/delta;
}

/*kernel void backsub(global const uchar* img, global uchar* bg, global uchar* mask, const int xStride, const int yStride, const uchar delta) {
	const int width = get_global_size(0);
	const int imgWidth = width*xStride*yStride;
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int imgPos = xStride*x + y*imgWidth;

	uchar m = 0;
	for(int dy = 0; dy < yStride; dy++) {
		for(int dx = 0; dx < xStride; dx++) {
			if(abs_diff(img[imgPos + dx + dy*imgWidth], bg[imgPos + dx + dy*imgWidth]) > delta)
				m = 255;
		}
	}
	mask[x + y*width] = m;

	if(!m) {
		for(int dy = 0; dy < yStride; dy++) {
			for(int dx = 0; dx < xStride; dx++) {
				bg[imgPos + dx + dy*imgWidth] = bg[imgPos + dx + dy*imgWidth]/2 + img[imgPos + dx + dy*imgWidth]/2; //TODO background update after object detection
			}
		}
	}

	//TODO mask noise removal
}*/

/*kernel void backsub(read_only image2d_t img, read_only image2d_t bg, global uchar* mask, const int delta) {
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
	int2 pos = (int2)(get_global_id(0), get_global_id(1));
	uint4 b = read_imageui(bg, sampler, pos); // yuv
	uint4 c = read_imageui(img, sampler, pos); // yuv

	uchar m = 0;
	if(abs_diff(c.y, b.y) > delta)
		m = 255;
	if(abs_diff(c.z, b.z) > delta)
		m = 255;

	mask[pos.x + pos.y*get_global_size(0)] = m;

	//TODO background update after object detection
	//if(!m)
	//	write_imageui(bg, pos, c);
}*/