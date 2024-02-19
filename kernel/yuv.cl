#ifndef CL_VERSION_1_0
#include "clstd.h"
#endif


kernel void toyuv(read_only image2d_t bgr, write_only image2d_t yuv) {
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
	int2 pos = (int2)(get_global_id(0), get_global_id(1));
	uint4 c = read_imageui(bgr, sampler, pos);

	write_imageui(yuv, pos, (uint4)(
			convert_uchar_sat((66*c.z + 129*c.y + 25*c.x) / 256 + 16),
			convert_uchar_sat((-38*c.z + -74*c.y + 112*c.x) / 256 + 128),
			convert_uchar_sat((112*c.z + -94*c.y + -18*c.x) / 256 + 128),
			0
	));
}
