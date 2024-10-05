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

inline float ssd(read_only image2d_t img, const int2 pos, const float4 color, const float radius) {
	int n = 0;
	float4 ssd = (float4)(0, 0, 0, 0);
	for(int y = pos.y - (int)round(radius); y <= pos.y + (int)round(radius); y++) {
		float xRadius = native_sqrt(radius*radius - (y - pos.y)*(y - pos.y));
		for(int x = pos.x - (int)round(xRadius); x <= pos.x + (int)round(xRadius); x++) {
			float4 diff = convert_float4(read_imageui(img, sampler, (int2)(x, y))) - color;
			ssd += diff*diff;
			n++;
		}
	}

	return (ssd.x + ssd.y + ssd.z) / n;
}

inline float ballssd(read_only image2d_t img, const int2 pos, const float4 color, const float radius) {
	float sqRadius = radius*radius;
	int n = 0;
	float4 ssd = (float4)(0, 0, 0, 0);
	for(int y = pos.y - (int)round(radius); y <= pos.y + (int)round(radius); y++) {
		float xRadius = native_sqrt(radius*radius - (y - pos.y)*(y - pos.y));
		for(int x = pos.x - (int)round(xRadius); x <= pos.x + (int)round(xRadius); x++) {
			float dist = (x-pos.x)*(x-pos.x) + (y-pos.y)*(y-pos.y);
			float4 diff = convert_float4(read_imageui(img, sampler, (int2)(x, y))) - (1-dist/sqRadius) * color;
			ssd += diff*diff;
			n++;
		}
	}

	return (ssd.x + ssd.y + ssd.z) / n;
}

inline float botssd(read_only image2d_t img, const int2 pos, const float4 color, const float radius, const float bgRadius) {
	float sqRadius = radius*radius;
	int n = 0;
	float4 ssd = (float4)(0, 0, 0, 0);
	for(int y = pos.y - (int)round(bgRadius); y <= pos.y + (int)round(bgRadius); y++) {
		float xRadius = native_sqrt(bgRadius*bgRadius - (y - pos.y)*(y - pos.y));
		for(int x = pos.x - (int)round(xRadius); x <= pos.x + (int)round(xRadius); x++) {
			float dist = (x-pos.x)*(x-pos.x) + (y-pos.y)*(y-pos.y);
			float4 diff;
			if(dist < sqRadius) {
				diff = convert_float4(read_imageui(img, sampler, (int2)(x, y))) - color;
			} else {
				diff = convert_float4(read_imageui(img, sampler, (int2)(x, y)));
			}
			ssd += diff*diff;
			n++;
		}
	}

	return (ssd.x + ssd.y + ssd.z) / n;
}

kernel void ssd_kernel(read_only image2d_t img, write_only image2d_t out, float fieldScale) {
	int2 pos = (int2)(get_global_id(0), get_global_id(1));

	/* // RGB
	float score = ssd(img, pos, (float4)(255, 255, 0, 255), 25/fieldScale);
	score = min(score, ssd(img, pos, (float4)(0, 128, 255, 255), 25/fieldScale));
	score = min(score, ssd(img, pos, (float4)(128, 255, 128, 255), 20/fieldScale));
	score = min(score, ssd(img, pos, (float4)(255, 0, 255, 255), 20/fieldScale));
	score = min(score, ssd(img, pos, (float4)(255, 128, 0, 255), 21.5f/fieldScale));*/

	/* // dRGB
	float score = ssd(img, pos, (float4)(255, 128, 0, 255), 25/fieldScale);
	score = min(score, ssd(img, pos, (float4)(0, 128, 255, 255), 25/fieldScale));
	score = min(score, ssd(img, pos, (float4)(0, 255, 128, 255), 20/fieldScale));
	score = min(score, ssd(img, pos, (float4)(255, 0, 128, 255), 20/fieldScale));
	score = min(score, ssd(img, pos, (float4)(255, 128, 0, 255), 21.5f/fieldScale));*/

	/* // RGB RC22 optimized
	float score = ssd(img, pos, (float4)(109, 147, 119, 255), 25/fieldScale);
	score = min(score, ssd(img, pos, (float4)(65, 116, 166, 255), 25/fieldScale));
	score = min(score, ssd(img, pos, (float4)(79, 155, 155, 255), 20/fieldScale));
	score = min(score, ssd(img, pos, (float4)(127, 122, 160, 255), 20/fieldScale));
	score = min(score, ssd(img, pos, (float4)(133, 119, 96, 255), 21.5f/fieldScale));*/

	/* // RGB ballssd
	float score = ssd(img, pos, (float4)(255, 255, 0, 255), 25/fieldScale);
	score = min(score, ssd(img, pos, (float4)(0, 128, 255, 255), 25/fieldScale));
	score = min(score, ssd(img, pos, (float4)(128, 255, 128, 255), 20/fieldScale));
	score = min(score, ssd(img, pos, (float4)(255, 0, 255, 255), 20/fieldScale));
	score = min(score, ballssd(img, pos, (float4)(255, 128, 0, 255), 21.5f/fieldScale));*/

	/* // RGB botssd
	float score = botssd(img, pos, (float4)(255, 255, 0, 255), 25/fieldScale, 65/fieldScale);
	score = min(score, botssd(img, pos, (float4)(0, 128, 255, 255), 25/fieldScale, 65/fieldScale));
	score = min(score, ssd(img, pos, (float4)(128, 255, 128, 255), 20/fieldScale));
	score = min(score, ssd(img, pos, (float4)(255, 0, 255, 255), 20/fieldScale));
	score = min(score, ssd(img, pos, (float4)(255, 128, 0, 255), 21.5f/fieldScale));*/

	// RGB botssd ballssd
	float score = botssd(img, pos, (float4)(255, 255, 0, 255), 25/fieldScale, 65/fieldScale);
	score = min(score, botssd(img, pos, (float4)(0, 128, 255, 255), 25/fieldScale, 65/fieldScale));
	score = min(score, ssd(img, pos, (float4)(128, 255, 128, 255), 20/fieldScale));
	score = min(score, ssd(img, pos, (float4)(255, 0, 255, 255), 20/fieldScale));
	score = min(score, ballssd(img, pos, (float4)(255, 128, 0, 255), 21.5f/fieldScale));

	write_imagef(out, pos, (3*(255*255)-score)/(3*(255*255)));
}
