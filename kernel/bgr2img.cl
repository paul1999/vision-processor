#ifndef CL_VERSION_1_0
#include "clstd.h"
#endif

/*float4 toYuv(float4 rgba) {
	return (float4) (
			 0.299f  *rgba.x +  0.587f  *rgba.y + 0.114f   *rgba.z,
			-0.14713f*rgba.x + -0.28886f*rgba.y + 0.436f   *rgba.z,
			 0.615f  *rgba.x + -0.51499f*rgba.y + -0.10001f*rgba.z,
			255
	);
}*/

kernel void buf2img(global const uchar* img, write_only image2d_t out) {
	int2 pos = (int2)(get_global_id(0), get_global_id(1));
	int imgpos = 3*(pos.x + pos.y*get_global_size(0));

	uchar r = img[imgpos+2];
	uchar g = img[imgpos+1];
	uchar b = img[imgpos+0];
	write_imageui(out, pos, (uint4)(
			convert_uchar_sat((66*r + 129*g + 25*b) / 256 + 16),
			convert_uchar_sat((-38*r + -74*g + 112*b) / 256 + 128),
			convert_uchar_sat((112*r + -94*g + -18*b) / 256 + 128),
			255
	));
}
