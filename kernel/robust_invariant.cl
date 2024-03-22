#ifndef CL_VERSION_1_0
#include "clstd.h"
#endif

//https://github.com/mubastan/canny/blob/master/gradient.py

//TODO BGR only
kernel void robust_invariant(global const uchar* img, global uchar* out) {
	const int width = get_global_size(0);
	const int imgWidth = width+1;
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int imgPos = 3*x + y*imgWidth*3;

	/*float gxx = 0.0;
	float gxy = 0.0;
	float gyy = 0.0;

	for(int i = 0; i < 3; i++) {
		const float dx = (float)img[i+imgPos+3] - img[imgPos];
		const float dy = (float)img[i+imgPos+3*imgWidth] - img[imgPos];
		gxx += dx*dx;
		gxy += dx*dy;
		gyy += dy*dy;
	}*/

	float b = img[imgPos];
	float g = img[imgPos+1];
	float r = img[imgPos+2];
	float bx = img[imgPos+3] - b;
	float gx = img[imgPos+3+1] - g;
	float rx = img[imgPos+3+2] - r;
	float by = img[imgPos+3*imgWidth] - b;
	float gy = img[imgPos+3*imgWidth+1] - g;
	float ry = img[imgPos+3*imgWidth+2] - r;

	float saturation = native_sqrt(2*(r*r+g*g+b*b - r*g-r*b-g*b)); // + 1e-9

	float hx = (r*(bx-gx)+g*(rx-bx)+b*(gx-rx));// / saturation; // / saturation for full invariant
	float hy = (r*(by-gy)+g*(ry-by)+b*(gy-ry));// / saturation;

	float gxx = hx*hx;
	float gxy = hx*hy;
	float gyy = hy*hy;


	//TODO smoothing?

	gxy *= 2;
	const float gxx_gyy = gxx - gyy;
	const float d = native_sqrt(gxx_gyy*gxx_gyy + gxy*gxy); // + 1e-9

	out[x + y*imgWidth] = convert_uchar_sat(native_sqrt(gxx + gyy + d)); // + 1e-9
	//out[x + y*imgWidth] = (h*h + v*v) >> 9; (with h and v as short)
}

