#ifndef CL_VERSION_1_0
#include "clstd.h"
#endif

const sampler_t sampler = CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP;

//https://github.com/mubastan/canny/blob/master/gradient.py
kernel void robust_invariant(read_only image2d_t in, write_only image2d_t out) {
	const int2 pos = (int2)(get_global_id(0), get_global_id(1));

	float4 px = convert_float4(read_imageui(in, sampler, pos));
	float4 pxx = convert_float4(read_imageui(in, sampler, (int2)(pos.x+1, pos.y))) - convert_float4(read_imageui(in, sampler, (int2)(pos.x-1, pos.y)));
	float4 pxy = convert_float4(read_imageui(in, sampler, (int2)(pos.x, pos.y+1))) - convert_float4(read_imageui(in, sampler, (int2)(pos.x, pos.y-1)));

	//float saturation = native_sqrt(2*(r*r+g*g+b*b - r*g-r*b-g*b)); // + 1e-9
	// h / saturation for full invariant

	/*float hx = px.r*(pxx.b-pxx.g) + px.g*(pxx.r-pxx.b) + px.b*(pxx.g-pxx.r);
	float hy = px.r*(pxy.b-pxy.g) + px.g*(pxy.r-pxy.b) + px.b*(pxy.g-pxy.r);

	float gxx = hx*hx;
	float gxy = hx*hy;
	float gyy = hy*hy;

	gxy *= 2;
	const float gxx_gyy = gxx - gyy;
	const float d = native_sqrt(gxx_gyy*gxx_gyy + gxy*gxy); // + 1e-9

	//Gradient direction: 0.5*arctan2(cxy, cxx_cyy)

	write_imagef(out, pos, native_sqrt(gxx + gyy + d)); // + 1e-9*/

	float4 gxx = (px.x*pxx.x + px.y*pxx.y + px.z*pxx.z)*px;
	float4 gyy = (px.x*pxy.x + px.y*pxy.y + px.z*pxy.z)*px;

	float4 acc = gxx*gyy;
	write_imagef(out, pos, native_sqrt(fabs(acc.x + acc.y + acc.z))); // + 1e-9

	//write_imagef(out, pos, native_sqrt(fabs((px.x*pxx.x + px.y*pxx.y + px.z*pxx.z)*(px.x*pxy.x + px.y*pxy.y + px.z*pxy.z)))); // + 1e-9
}

