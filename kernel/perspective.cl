#ifndef CL_VERSION_1_0
#include "clstd.h"
#include "image2field.cl"
#else
#include "kernel/image2field.cl"
#endif

inline float2 field2image(const Perspective p, float3 fieldpos) {
	fieldpos += (float3)(p.c[0], p.c[1], p.c[2]);

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


const sampler_t sampler = CLK_FILTER_LINEAR | CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP;

kernel void buf2img(read_only image2d_t in, write_only image2d_t out, const Perspective perspective, const float maxRobotHeight, const float fieldScale, const float fieldOffsetX, const float fieldOffsetY) {
	write_imagef(out, (int2)(get_global_id(0), get_global_id(1)), read_imagef(in, sampler, field2image(perspective, (float3)(get_global_id(0)*fieldScale + fieldOffsetX, get_global_id(1)*fieldScale + fieldOffsetY, maxRobotHeight))));
}