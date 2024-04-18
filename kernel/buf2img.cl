#ifndef CL_VERSION_1_0
#include "clstd.h"

#define RGGB
#endif


kernel void buf2img(global const uchar* img, write_only image2d_t out) {
	int2 pos = (int2)(get_global_id(0), get_global_id(1));

#ifdef RGGB
	int imgpos = 2*pos.x + 2*pos.y*2*get_global_size(0);
	write_imageui(out, pos, (uint4)(
			img[imgpos], //R
			img[imgpos+1]/2 + img[imgpos+2*get_global_size(0)]/2, //G
			img[imgpos+1+2*get_global_size(0)], //B
			255
	));
#endif
}
