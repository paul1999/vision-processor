#ifndef CL_VERSION_1_0
#include "clstd.h"
#endif

const sampler_t sampler = CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE;

//https://github.com/mubastan/canny/blob/master/gradient.py
kernel void robust_invariant(read_only image2d_t in, write_only image2d_t out) {
	const int2 pos = (int2)(get_global_id(0), get_global_id(1));

	float4 pxx = convert_float4(read_imageui(in, sampler, (int2)(pos.x+1, pos.y))) - convert_float4(read_imageui(in, sampler, (int2)(pos.x-1, pos.y)));
	float4 pxy = convert_float4(read_imageui(in, sampler, (int2)(pos.x, pos.y+1))) - convert_float4(read_imageui(in, sampler, (int2)(pos.x, pos.y-1)));
	/*float xScale = native_sqrt(pxx.x*pxx.x + pxx.y*pxx.y + pxx.z*pxx.z) / 32.0f;
	float yScale = native_sqrt(pxy.x*pxy.x + pxy.y*pxy.y + pxy.z*pxy.z) / 32.0f;
	pxx /= xScale;
	pxy /= yScale;*/

	pxx *= pxy;
	/*float value = pxx.x + pxx.y + pxx.z;
	write_imagef(out, pos, fabs(value) > 64.0f ? (value > 0.0f ? 2048.0f : -2048.0f) : 0.0f);*/
	/*write_imagef(out, pos, (pxx.y + pxx.z) * 16.0f);*/
	//write_imagef(out, pos, (pxx.x + pxx.y + pxx.z) * 4.0f);
	write_imagef(out, pos, (pxx.y + pxx.z) * 4.0f); // u and v channels only
}

