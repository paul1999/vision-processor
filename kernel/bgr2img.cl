#ifndef CL_VERSION_1_0
#include "clstd.h"
#endif


kernel void buf2img(global const uchar* img, write_only image2d_t out) {
	int2 pos = (int2)(get_global_id(0), get_global_id(1));
	int imgpos = 3*(pos.x + pos.y*get_global_size(0));
	write_imageui(out, pos, (uint4)(img[imgpos+2], img[imgpos+1], img[imgpos+0], 255));
}
