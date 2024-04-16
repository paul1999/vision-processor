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
#define CLK_ADDRESS_CLAMP_TO_EDGE 1
#define CLK_FILTER_NEAREST 2

typedef unsigned char uchar;
typedef struct { uchar x, y, z; } uchar3;
typedef unsigned int uint;
typedef struct { int x, y; } int2;
typedef struct { float x, y; } float2;
typedef struct { float x, y, z; } float3;
typedef struct { float x, y, z, w; } float4;
typedef struct { int x, y, z, w; } int4;
typedef struct { uint x, y, z, w; } uint4;
typedef int sampler_t;
typedef struct {} image2d_t;

uint4 read_imageui(image2d_t, sampler_t, int2);
int4 read_imagei(image2d_t, sampler_t, int2);
void write_imageui(image2d_t, int2, uint4);
void write_imagei(image2d_t, int2, int4);

int get_global_id(int);
int get_global_size(int);

int min(int, int);
int max(int, int);
int abs_diff(int, int);
bool isnan(float);

float native_sqrt(float);
char convert_char_sat(float);
uchar convert_uchar_sat(float);
int4 convert_int4(uint4);
uint4 convert_uint4(float4);
float2 convert_float2(int2);
float4 convert_float4(int4);
float4 convert_float4(uint4);
float fabs(float);
float round(float);

int atomic_inc(volatile global int*);

#define INFINITY 9999999999999.9f
#define NAN 9999999999999.9f
