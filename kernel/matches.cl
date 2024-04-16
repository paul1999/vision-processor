#ifndef CL_VERSION_1_0
#include "clstd.h"
#include "image2field.cl"
#else
#include "kernel/image2field.cl"
#endif

typedef struct __attribute__ ((packed)) {
	int2 pos;
	float score;
	float height;
	RGB color;
} Match;

const sampler_t sampler = CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE;

kernel void matches(read_only image2d_t img, global const float* circ, global Match* matches, global volatile int* counter, Perspective perspective, const float circThreshold, const uchar sThreshold, const uchar vThreshold, const float height, const float radius, const int maxMatches) {
	int2 pos = (int2)(get_global_id(0), get_global_id(1));
	const int circPos = pos.x + pos.y * get_global_size(0);
	float circScore = circ[circPos];
	if(circScore < circThreshold)
		return;

	//TODO float image
	if(pos.x > 0 && circ[circPos-1] > circScore)
		return;
	if(pos.y > 0 && circ[circPos-get_global_size(0)] > circScore)
		return;
	if(pos.x < get_global_size(0)-1 && circ[circPos+1] > circScore)
		return;
	if(pos.y < get_global_size(1)-1 && circ[circPos+get_global_size(0)] > circScore)
		return;

	pos += 1; //TODO gradient position compensation

	int n = 0;
	uint4 color = (uint4)(0, 0, 0, 0);

	{
		perspective.shape[0] /= 2;
		perspective.shape[1] /= 2;
		perspective.f *= 2;
		perspective.p[0] /= 2;
		perspective.p[1] /= 2;
		const float sqRadius = radius*radius;
		float2 center = clImage2field(perspective, height, pos);
		int2 min, max;
		//clArea(perspective, height, pos, center, sqRadius, &min, &max);
		min = pos - 5;
		max = pos + 6;

		for(int y = min.y; y < max.y; y++) {
			for(int x = min.x; x < max.x; x++) {
				int2 pxpos = (int2)(x, y);
				float2 mpos = clImage2field(perspective, height, pxpos);
				mpos -= center;
				mpos *= mpos;
				if(mpos.x + mpos.y <= sqRadius) {
					color += read_imageui(img, sampler, pxpos);
					n++;
				}
			}
		}
	}

	color.y += color.w; //TODO RGGB only
	color.y /= 2;
	color /= n;
	uchar3 hsv;

	uchar rgbMin = min(min(color.x, color.y), color.z);
	hsv.z = max(max(color.x, color.y), color.z);
	if (hsv.z == 0) {
		hsv.x = 0;
		hsv.y = 0;
	} else {
		uchar span = hsv.z - rgbMin;
		hsv.y = convert_uchar_sat(255 * convert_int(span) / hsv.z);
		if (hsv.y == 0) {
			hsv.x = 0;
		} else {
			if (hsv.z == color.x)
				hsv.x = 0 + 43 * (color.y - color.z) / span;
			else if (hsv.z == color.y)
				hsv.x = 85 + 43 * (color.z - color.x) / span;
			else
				hsv.x = 171 + 43 * (color.x - color.y) / span;
		}
	}

	if(hsv.y < sThreshold || hsv.z < vThreshold)
		return;

	int i = atomic_inc(counter);
	if(i >= maxMatches)
		return;

	global Match* match = matches + i;
	match->pos = pos;
	match->height = height;
	match->score = circScore;
	match->color.r = color.x;
	match->color.g = color.y;
	match->color.b = color.z;
}