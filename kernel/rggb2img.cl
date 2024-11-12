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


kernel void buf2img(global const uchar* img, write_only image2d_t out) {
	int2 pos = (int2)(get_global_id(0), get_global_id(1));
	int imgpos = 2*pos.x + 2*pos.y*2*get_global_size(0);

	uchar r = img[imgpos];
	uchar g = img[imgpos+1]/2 + img[imgpos+2*get_global_size(0)]/2;
	uchar b = img[imgpos+1+2*get_global_size(0)];

	/*uchar g = img[imgpos]/2 + img[imgpos+1+2*get_global_size(0)]/2;
	uchar r = img[imgpos+1];
	uchar b = img[imgpos+2*get_global_size(0)];*/

	write_imageui(out, pos, (uint4)(r, g, b, 255));
}
