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

const sampler_t sampler = CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE;

inline void px(read_only image2d_t color, int2 pos, const int dx, const int dy, float* score) {
	pos.x += dx;
	pos.y += dy;

	*score += read_imagef(color, sampler, pos).x;
}
inline void abspx(read_only image2d_t color, int2 pos, const int dx, const int dy, float* score) {
	pos.x += dx;
	pos.y += dy;

	*score += fabs(read_imagef(color, sampler, pos).x);
}

//https://www.thecrazyprogrammer.com/2016/12/bresenhams-midpoint-circle-algorithm-c-c.html
kernel void circularize(read_only image2d_t color, write_only image2d_t out, int minBlobRadius, int maxBlobRadius) {
	int2 pos = (int2)(get_global_id(0), get_global_id(1));

	float score = 0.0f;
	/*for(int i = 3; i < 8; i++) {
		int n = 0;
		int x = i;
		int err = 0;
		int y = 0;
		float ppScore = 0.0f;
		float pnScore = 0.0f;
		float npScore = 0.0f;
		float nnScore = 0.0f;

		while(x >= y) {
			n += 4;
			px(color, pos, +x, +y, &ppScore);
			px(color, pos, -y, +x, &npScore);
			px(color, pos, -x, -y, &nnScore);
			px(color, pos, +y, -x, &pnScore);
			if(x > y && y > 0) {
				n += 4;
				px(color, pos, +y, +x, &ppScore);
				px(color, pos, -x, +y, &npScore);
				px(color, pos, -y, -x, &nnScore);
				px(color, pos, +x, -y, &pnScore);
			}

			if(err <= 0) {
				y += 1;
				err += 2*y + 1;
			}
			if(err > 0) {
				x -= 1;
				err += 2*x + 1;
			}
		}

		float localScore = (ppScore + nnScore - pnScore - npScore) / n;
		if(localScore > score && ppScore > 0.0f && nnScore > 0.0f && pnScore < 0.0f && npScore < 0.0f)
			score = localScore;
	}*/

	float ppScore = 0.0f;
	float pnScore = 0.0f;
	float npScore = 0.0f;
	float nnScore = 0.0f;
	/*for(int y = 2; y < 7; y++) {
		for(int x = 2; x < 7; x++) {
			px(color, pos, -8+x, +y, &npScore);
			px(color, pos, +x, +y, &ppScore);
			px(color, pos, -8+x, -y, &nnScore);
			px(color, pos, +x, -y, &pnScore);
		}
	}
	//if(ppScore > 0.0f && nnScore > 0.0f && pnScore < 0.0f && npScore < 0.0f)
	if(fabs((ppScore/nnScore) - 1) < 0.5f && fabs((pnScore/npScore) - 1) < 0.5f)
		score = (ppScore + nnScore - pnScore - npScore) / 196.0f;*/

	/*for(int i = 0; i < 2; i++) {
		px(color, pos, -i, +i, &npScore);
		px(color, pos, +i, +i, &ppScore);
		px(color, pos, -i, -i, &nnScore);
		px(color, pos, +i, -i, &pnScore);
	}
	npScore = -npScore;
	ppScore = -ppScore;
	nnScore = -nnScore;
	pnScore = -pnScore;*/
	/*float nxDiv = 0.0f;
	float nyDiv = 0.0f;
	float pxDiv = 0.0f;
	float pyDiv = 0.0f;*/
	//for(int i = minBlobRadius; i <= maxBlobRadius; i++) {
	for(int i = 1; i <= maxBlobRadius; i++) {
		/*px(color, pos, 0, -i, &nyDiv);
		px(color, pos, -i, 0, &nxDiv);
		px(color, pos, +i, 0, &pxDiv);
		px(color, pos, 0, +i, &pyDiv);*/

		px(color, pos, -i, +i, &npScore);
		px(color, pos, +i, +i, &ppScore);
		px(color, pos, -i, -i, &nnScore);
		px(color, pos, +i, -i, &pnScore);
	}
	/*for(int y = 1; y <= maxBlobRadius; y++) {
		for(int x = 1; x <= maxBlobRadius; x++) {
			px(color, pos, -x, +y, &npScore);
			px(color, pos, +x, +y, &ppScore);
			px(color, pos, -x, -y, &nnScore);
			px(color, pos, +x, -y, &pnScore);
		}
	}*/
	//if(fabs((ppScore/nnScore) - 1) < 0.3f && fabs((pnScore/npScore) - 1) < 0.3f)
	if(ppScore > 0.0f && nnScore > 0.0f && pnScore < 0.0f && npScore < 0.0f)
		score = (ppScore + nnScore - pnScore - npScore) / (4*maxBlobRadius);

	write_imagef(out, pos, score / 4.0f);

	//float absDiff = fabs(nyDiv) + fabs(nxDiv) + fabs(pyDiv) + fabs(pxDiv);
	//write_imagef(out, pos, absDiff / 80.0f);
	//write_imagef(out, pos, 16.0f*score / absDiff);
}
