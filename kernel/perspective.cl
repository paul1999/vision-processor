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
	int shape[2];  // raw image shape
	float f;       // 1/(focal length)
	float p[2];    // principal point
	float d;       // distortion
	float r[9]; // rotation matrix
	float c[3];    // camera position
	int field[2]; // field size incl. boundary in cm
	float fInv;
	float rInv[9];
} Perspective;

inline float2 field2image(const Perspective p, float3 fieldpos) {
	fieldpos -= (float3)(p.c[0], p.c[1], p.c[2]);

	float3 camRay = (float3)(
			p.rInv[0] * fieldpos.x + p.rInv[1] * fieldpos.y + p.rInv[2] * fieldpos.z,
			p.rInv[3] * fieldpos.x + p.rInv[4] * fieldpos.y + p.rInv[5] * fieldpos.z,
			p.rInv[6] * fieldpos.x + p.rInv[7] * fieldpos.y + p.rInv[8] * fieldpos.z
	);
	float2 camRay2 = (float2)(camRay.x / camRay.z, camRay.y / camRay.z);

	float2 camRayU = camRay2;
	for(int i = 0; i < 8; i++) {
		float2 r = camRayU*camRayU;
		float dr = 1 + p.d*(r.x + r.y);
		camRayU = camRay2/dr;
	}

	return p.fInv * camRayU	 + (float2)(p.p[0], p.p[1]);
}


const sampler_t sampler = CLK_FILTER_LINEAR | CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE;

kernel void perspective(read_only image2d_t in, write_only image2d_t out, const Perspective perspective, const float maxRobotHeight, const float fieldScale, const float fieldOffsetX, const float fieldOffsetY) {
	write_imagef(out, (int2)(get_global_id(0), get_global_id(1)), read_imagef(in, sampler, field2image(perspective, (float3)(get_global_id(0)*fieldScale + fieldOffsetX, get_global_id(1)*fieldScale + fieldOffsetY, maxRobotHeight))));
}