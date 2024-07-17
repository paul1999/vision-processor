/*
     Copyright 2024 Felix Weinmann

     Licensed under the Apache License, Version 2.0 (the "License");
     you may not use this file except in compliance with the License.
     You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

     Unless required by applicable law or agreed to in writing, software
     distributed under the License is distributed on an "AS IS" BASIS,
     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     See the License for the specific language governing permissions and
     limitations under the License.
 */
#ifndef CL_VERSION_1_0
#include "clstd.h"
#endif

typedef struct __attribute__ ((packed)) {
	uchar r;
	uchar g;
	uchar b;
} RGB;

typedef struct __attribute__ ((packed)) {
	int x, y;
	RGB color;
	float orangeness;
	float yellowness;
	float blueness;
	float greenness;
	float pinkness;
} Match;

typedef struct __attribute__ ((packed)) Hues {
	uchar orange;
	uchar yellow;
	uchar blue;
	uchar green;
	uchar pink;
} Hues;

const sampler_t sampler = CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE;

kernel void matches(read_only image2d_t img, read_only image2d_t circ, global Match* matches, global volatile int* counter, const float circThreshold, const float minScore, const int radius, const struct Hues hues, const int maxMatches) {
	const int2 pos = (int2)(get_global_id(0), get_global_id(1));
	float circScore = read_imagef(circ, sampler, pos).x;
	if(circScore < circThreshold)
		return;

	//TODO subpixel position
	if(
			read_imagef(circ, sampler, (int2)(pos.x-1, pos.y)).x > circScore ||
			read_imagef(circ, sampler, (int2)(pos.x+1, pos.y)).x > circScore ||
			read_imagef(circ, sampler, (int2)(pos.x, pos.y-1)).x > circScore ||
			read_imagef(circ, sampler, (int2)(pos.x, pos.y+1)).x > circScore
	) {
		atomic_inc(counter+2);
		return;
	}

	//https://en.wikipedia.org/wiki/Standard_deviation#Rapid_calculation_methods
	int n = 0;
	uint4 s1 = (uint4)(0, 0, 0, 0);
	uint4 s2 = (uint4)(0, 0, 0, 0); // Value estimation (255*255) * (16*16) /256^4 (far in range of uint)
	{
		const int sqRadius = radius*radius;
		for(int y = -radius; y <= radius; y++) {
			for(int x = -radius; x <= radius; x++) {
				if(x*x + y*y <= sqRadius) {
					uint4 v = read_imageui(img, sampler, pos + (int2)(x, y));
					s1 += v;
					s2 += v*v;
					n++; //TODO faster by computation? -> https://mathworld.wolfram.com/GausssCircleProblem.html
				}
			}
		}
	}

	//https://en.wikipedia.org/wiki/Summed-area_table
	float4 stddev = native_sqrt((convert_float4(s2) - convert_float4(s1)*convert_float4(s1)/n) / n);

	float score = circScore / (stddev.x + stddev.y + stddev.z);
	if(score < minScore) {
		atomic_inc(counter+1);
		return;
	}

	uint4 color = s1 / n;

	//https://stackoverflow.com/questions/3018313/algorithm-to-convert-rgb-to-hsv-and-hsv-to-rgb-in-range-0-255-for-both
	uchar rgbMin = min(min(color.x, color.y), color.z);
	uchar rgbMax = max(max(color.x, color.y), color.z);
	uchar hue = 0;
	uchar span = rgbMax - rgbMin;
	if (rgbMax != 0 && span != 0) {
		if (rgbMax == color.x)
			hue = 0 + 43 * (color.y - color.z) / span;
		else if (rgbMax == color.y)
			hue = 85 + 43 * (color.z - color.x) / span;
		else
			hue = 171 + 43 * (color.x - color.y) / span;
	}

	//if(hsv.y < sThreshold || hsv.z < vThreshold)
	//	return;

	int i = atomic_inc(counter);
	if(i >= maxMatches)
		return;

	global Match* match = matches + i;
	match->x = pos.x;
	match->y = pos.y;
	match->color.r = color.x;
	match->color.g = color.y;
	match->color.b = color.z;
	match->orangeness = fabs((float)(char)(hue - hues.orange)) / 128.0f;
	match->yellowness = fabs((float)(char)(hue - hues.yellow)) / 128.0f;
	match->blueness = fabs((float)(char)(hue - hues.blue)) / 128.0f;
	match->greenness = fabs((float)(char)(hue - hues.green)) / 128.0f;
	match->pinkness = fabs((float)(char)(hue - hues.pink)) / 128.0f;
}