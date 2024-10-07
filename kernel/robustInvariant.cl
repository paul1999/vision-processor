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

//https://github.com/mubastan/canny/blob/master/gradient.py
kernel void robust_invariant(read_only image2d_t in, write_only image2d_t out, int offset) {
	const int2 pos = (int2)(get_global_id(0), get_global_id(1));

	float4 color = convert_float4(read_imageui(in, sampler, pos));
	float4 gx = convert_float4(read_imageui(in, sampler, (int2)(pos.x+offset, pos.y))) - convert_float4(read_imageui(in, sampler, (int2)(pos.x-offset, pos.y)));
	float4 gy = convert_float4(read_imageui(in, sampler, (int2)(pos.x, pos.y+offset))) - convert_float4(read_imageui(in, sampler, (int2)(pos.x, pos.y-offset)));

	float saturation = native_sqrt(2*(color.r*color.r + color.g*color.g + color.b*color.b - color.r*color.g - color.r*color.b - color.g*color.b)); // + 1e-9

	float hx = (color.r*(gx.b-gx.g) + color.g*(gx.r-gx.b) + color.b*(gx.g-gx.r)) / saturation; // / saturation for full invariant
	float hy = (color.r*(gy.b-gy.g) + color.g*(gy.r-gy.b) + color.b*(gy.g-gy.r)) / saturation;

	float cxx = hx*hx;
	float cxy = 2*hx*hy;
	float cyy = hy*hy;

	float gxx_gyy = cxx - cyy;
	float d = native_sqrt(gxx_gyy*gxx_gyy + cxy*cxy); // + 1e-9

	write_imagef(out, pos, native_sqrt(cxx + cyy + d)); // + 1e-9; Magnitude
	//write_imagef(out, pos, 0.5*arctan2(cxy, cxx_cyy)); // + 1e-9; Direction
}

