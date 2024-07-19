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
	float x, y;
	RGB color;
	float orangeness;
	float yellowness;
	float blueness;
	float greenness;
	float pinkness;
} Match;

const sampler_t sampler = CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE;

inline float hue2score(const int hue1, const int hue2) {
	int diff = hue1 - hue2;
	if(diff > 127) //TODO correct thresholds?
		diff -= 255;
	else if(diff < -127)
		diff += 255;

	return 1.0f - abs(diff) / 128.f;
}

kernel void matches(read_only image2d_t img, read_only image2d_t circ, global Match* matches, global volatile int* counter, const float circThreshold, const float minScore, const int radius, const int maxMatches) {
	const int2 pos = (int2)(get_global_id(0), get_global_id(1));
	float circScore = read_imagef(circ, sampler, pos).x;
	if(circScore < circThreshold)
		return;

	float circNegX = read_imagef(circ, sampler, (int2)(pos.x-1, pos.y)).x;
	float circPosX = read_imagef(circ, sampler, (int2)(pos.x+1, pos.y)).x;
	float circNegY = read_imagef(circ, sampler, (int2)(pos.x, pos.y-1)).x;
	float circPosY = read_imagef(circ, sampler, (int2)(pos.x, pos.y+1)).x;
	if(
			circNegX > circScore ||
			circPosX > circScore ||
			circNegY > circScore ||
			circPosY > circScore
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

	int4 color = convert_int4(s1 / n);

	//https://stackoverflow.com/questions/3018313/algorithm-to-convert-rgb-to-hsv-and-hsv-to-rgb-in-range-0-255-for-both
	/*uint rgbMin = min(min(color.x, color.y), color.z);
	uint rgbMax = max(max(color.x, color.y), color.z);
	uint span = rgbMax - rgbMin;
	uchar hue = 0;
	if (rgbMax != 0 && span != 0) {
		if (rgbMax == color.x)
			hue = 0 + 43 * ((uchar)color.y - (uchar)color.z) / span;
		else if (rgbMax == color.y)
			hue = 85 + 43 * ((uchar)color.z - (uchar)color.x) / span;
		else
			hue = 171 + 43 * ((uchar)color.x - (uchar)color.y) / span;
	}*/

	//if(hsv.y < sThreshold || hsv.z < vThreshold)
	//	return;

	int i = atomic_inc(counter);
	if(i >= maxMatches)
		return;

	global Match* match = matches + i;
	//https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
	match->x = pos.x + 0.5f * (circNegX - circPosX) / (circNegX - 2*circScore + circPosX);
	match->y = pos.y + 0.5f * (circNegY - circPosY) / (circNegY - 2*circScore + circPosY);
	match->color.r = color.x;
	match->color.g = color.y;
	match->color.b = color.z;
	/*match->orangeness = hue2score(hue, hues.orange);
	match->yellowness = hue2score(hue, hues.yellow);
	match->blueness = hue2score(hue, hues.blue);
	match->greenness = hue2score(hue, hues.green);
	match->pinkness = hue2score(hue, hues.pink);*/

	//color -= 128;

	match->orangeness = (color.r - color.b) / 255.0f;
	match->yellowness = (color.r - color.b) / 255.0f;
	match->greenness  = (color.g - color.r) / 255.0f;
}