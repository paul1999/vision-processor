/*
     Copyright 2025 Felix Weinmann

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


kernel void raw2quad(global const uchar* img, write_only image2d_t topleft, write_only image2d_t topright, write_only image2d_t bottomleft, write_only image2d_t bottomright) {
	const int2 pos = (int2)(get_global_id(0), get_global_id(1));
	const int row_size = 2*get_global_size(0);

	const int imgpos = 2*pos.x + 2*pos.y*row_size;
	write_imageui(topleft, pos, img[imgpos]);
	write_imageui(topright, pos, img[imgpos+1]);
	write_imageui(bottomleft, pos, img[imgpos+row_size]);
	write_imageui(bottomright, pos, img[imgpos+1+row_size]);
}
