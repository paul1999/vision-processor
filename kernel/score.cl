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

//, const uchar sThreshold, const uchar vThreshold
kernel void matches(read_only image2d_t img, read_only image2d_t circ, write_only image2d_t score, const float circThreshold, const int radius) {
	int2 pos = (int2)(get_global_id(0), get_global_id(1));
	float circScore = read_imagef(circ, sampler, pos).x;
	if(circScore < circThreshold) {
		write_imagef(score, pos, -INFINITY);
		return;
	}

	//TODO subpixel position
	/*if(
			read_imagef(circ, sampler, (int2)(pos.x-1, pos.y)).x > circScore ||
			read_imagef(circ, sampler, (int2)(pos.x+1, pos.y)).x > circScore ||
			read_imagef(circ, sampler, (int2)(pos.x, pos.y-1)).x > circScore ||
			read_imagef(circ, sampler, (int2)(pos.x, pos.y+1)).x > circScore
	)
		return;*/

	int n = 0;
	uint4 color = (uint4)(0, 0, 0, 0);
	{
		const float sqRadius = radius*radius;
		for(int y = pos.y - radius; y <= pos.y + radius; y++) {
			for(int x = pos.x - radius; x <= pos.x + radius; x++) {
				int2 pxpos = (int2)(x, y);
				int2 mpos = pxpos - pos;
				mpos *= mpos;
				if(mpos.x + mpos.y <= sqRadius) {
					color += read_imageui(img, sampler, pxpos);
					n++; //TODO faster by computation
				}
			}
		}
	}

	color /= n;

	float4 stddev;
	{
		const float sqRadius = radius*radius;
		for(int y = pos.y - radius; y <= pos.y + radius; y++) {
			for(int x = pos.x - radius; x <= pos.x + radius; x++) {
				int2 pxpos = (int2)(x, y);
				int2 mpos = pxpos - pos;
				mpos *= mpos;
				if(mpos.x + mpos.y <= sqRadius) {
					float4 s = convert_float4(read_imageui(img, sampler, pxpos)) - convert_float4(color);
					s *= s;
					stddev += s;
					n++;
				}
			}
		}
	}

	stddev = native_sqrt(stddev);

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

	write_imagef(score, pos, -(stddev.x + stddev.y + stddev.z));
	//write_imagef(score, pos, convert_float(color.x)-256.f-(stddev.x + stddev.y + stddev.z));
}