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

const sampler_t sampler = CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE;

kernel void matches(read_only image2d_t img, read_only image2d_t circ, write_only image2d_t score, const float circThreshold, const int radius) {
	int2 pos = (int2)(get_global_id(0), get_global_id(1));
	float circScore = read_imagef(circ, sampler, pos).x;
	if(circScore < circThreshold) {
		write_imagef(score, pos, -INFINITY);
		return;
	}

	//TODO subpixel position
	if(
			read_imagef(circ, sampler, (int2)(pos.x-1, pos.y)).x > circScore ||
			read_imagef(circ, sampler, (int2)(pos.x+1, pos.y)).x > circScore ||
			read_imagef(circ, sampler, (int2)(pos.x, pos.y-1)).x > circScore ||
			read_imagef(circ, sampler, (int2)(pos.x, pos.y+1)).x > circScore
	) {
		write_imagef(score, pos, -INFINITY);
		return;
	}

	//https://en.wikipedia.org/wiki/Standard_deviation#Rapid_calculation_methods
	int n = 0; // https://mathworld.wolfram.com/GausssCircleProblem.html
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
					n++; //TODO faster by computation?
				}
			}
		}
	}

	uint4 color = s1 / n;
	//https://en.wikipedia.org/wiki/Summed-area_table
	float4 stddev = native_sqrt((convert_float4(s2) - convert_float4(s1)*convert_float4(s1)/n) / n);
	//float4 stddev = native_sqrt(n*convert_float4(s2) - convert_float4(s1)*convert_float4(s1)) / convert_float(n);

	/*uchar3 hsv;
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
	}*/

	//if(hsv.y < sThreshold || hsv.z < vThreshold)
	//	return;

	//write_imagef(score, pos, -(stddev.x + stddev.y + stddev.z));
	//write_imagef(score, pos, convert_float(color.x)-256.f-(stddev.x + stddev.y + stddev.z));
	write_imagef(score, pos, circScore / (stddev.x + stddev.y + stddev.z));
}