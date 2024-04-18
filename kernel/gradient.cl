#ifndef CL_VERSION_1_0
#include "clstd.h"

#include "image2field.cl"
#define RGGB
#else
#include "kernel/image2field.cl"
#endif

const sampler_t sampler = CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE;

inline void px(read_only image2d_t gx, read_only image2d_t gy, int2 pos, const int dx, const int dy, float* cossum) {
	pos.x += dx;
	pos.y += dy;

	float4 dgx = convert_float4(read_imagei(gx, sampler, pos));
	float4 dgy = convert_float4(read_imagei(gy, sampler, pos));

	float abspos = (float)dx*dx + (float)dy*dy;
	float4 abs2 = dgx*dgx + dgy*dgy;

	float4 vdiff = dx*dgx + dy*dgy;
	float4 csum = vdiff*vdiff / (abs2*abspos + 0.00001f);
	float2 acc = csum.xy + csum.zw;
	*cossum += acc.x + acc.y;
}

//https://www.thecrazyprogrammer.com/2016/12/bresenhams-midpoint-circle-algorithm-c-c.html
kernel void cossum(read_only image2d_t img, read_only image2d_t bg, read_only image2d_t gx, read_only image2d_t gy, global float* out, const Perspective perspective, const float height, const float radius, const int bgThreshold) {
	int2 pos = (int2)(get_global_id(0), get_global_id(1));
	/*int4 bgDiff = convert_int4(read_imageui(img, sampler, pos)) - convert_int4(read_imageui(bg, sampler, pos)); //TODO YUV für flackernde lichter
	bgDiff *= bgDiff;
	int2 bgDiffAc = bgDiff.xy + bgDiff.zw;
	if(bgDiffAc.x + bgDiffAc.y < bgThreshold) {
		out[pos.x + pos.y * get_global_size(0)] = 0.0f;
		return;
	}*/

	V2 center = image2field(perspective, height, (V2) {(float)pos.x, (float)pos.y});
	V2 offcenter = image2field(perspective, height, (V2) {(float)pos.x+1, (float)pos.y});

	V2 posdiff = {offcenter.x-center.x, offcenter.y-center.y};
	float rPerPixel = native_sqrt(posdiff.x*posdiff.x + posdiff.y*posdiff.y);
	int x = round(radius/rPerPixel);
	int y = 0;
	int err = 0;

	float cossum = 0.0f;
	int n = 0;
	while(x >= y) {
		n += 4*4;
		px(gx, gy, pos, +x, +y, &cossum);
		px(gx, gy, pos, -y, +x, &cossum);
		px(gx, gy, pos, -x, -y, &cossum);
		px(gx, gy, pos, +y, -x, &cossum);
		if(x > y && y > 0) {
			n += 4*4;
			px(gx, gy, pos, +y, +x, &cossum);
			px(gx, gy, pos, -x, +y, &cossum);
			px(gx, gy, pos, -y, -x, &cossum);
			px(gx, gy, pos, +x, -y, &cossum);
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

	/*float4 dgx = convert_float4(read_imagei(gx, sampler, pos)) / 1024.0f;
	float4 dgy = convert_float4(read_imagei(gy, sampler, pos)) / 1024.0f;
	float4 delta = dgx*dgx + dgy*dgy;
	float2 acc = delta.xy + delta.zw;*/
	out[pos.x + pos.y * get_global_size(0)] = cossum / n;// / (acc.x + acc.y + 0.00001f);
}
