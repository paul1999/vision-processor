#ifndef CL_VERSION_1_0
#include "clstd.h"
#endif


kernel void buf2img(global const uchar* img, write_only image2d_t out) {
	int2 pos = (int2)(get_global_id(0), get_global_id(1));
	int imgpos = 2*pos.x + 2*pos.y*2*get_global_size(0);

	uchar r = img[imgpos];
	uchar g = img[imgpos+1]/2 + img[imgpos+2*get_global_size(0)]/2;
	uchar b = img[imgpos+1+2*get_global_size(0)];
	write_imageui(out, pos, (uint4)(
			convert_uchar_sat((66*r + 129*g + 25*b) / 256 + 16),
			convert_uchar_sat((-38*r + -74*g + 112*b) / 256 + 128),
			convert_uchar_sat((112*r + -94*g + -18*b) / 256 + 128),
			255
	));

	/* (gamma compensation?)
	 * 21*pow(img[imgpos], 0.45f), //R
			21*pow(img[imgpos+1]/2 + img[imgpos+2*get_global_size(0)]/2, 0.45f), //G
			21*pow(img[imgpos+1+2*get_global_size(0)], 0.45f), //B*/
}
