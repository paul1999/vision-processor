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

const sampler_t sampler = CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE;

void kernel yuv2nv12(read_only image2d_t in, global uchar* out) {
	int2 pos = (int2)(get_global_id(0), get_global_id(1));
	const uint4 v = read_imageui(in, sampler, pos);

	out[pos.x + pos.y*get_global_size(0)] = v.x;

	pos /= 2;
	const int uvout = UV_OFFSET + pos.x*2 + pos.y*get_global_size(0);
	out[uvout] = v.y;
    out[uvout+1] = v.z;
}