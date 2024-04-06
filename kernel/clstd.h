#pragma once

#define kernel
#define global
#define local
#define constant
#define read_write
#define read_only
#define write_only

#define CLK_NORMALIZED_COORDS_FALSE 0
#define CLK_ADDRESS_NONE 1
#define CLK_FILTER_NEAREST 2

typedef unsigned char uchar;
typedef unsigned int uint;
typedef struct { int x, y; } int2;
typedef struct { uint x, y, z, w; } uint4;
typedef int sampler_t;
typedef struct {} image2d_t;

uint4 read_imageui(image2d_t, sampler_t, int2);
void write_imageui(image2d_t, int2, uint4);

int get_global_id(int);
int get_global_size(int);

int min(int, int);
int max(int, int);
int abs_diff(int, int);
bool isnan(float);

float native_sqrt(float);
char convert_char_sat(float);
uchar convert_uchar_sat(float);
float fabs(float);
float round(float);

#define INFINITY 9999999999999.9f
#define NAN 9999999999999.9f
