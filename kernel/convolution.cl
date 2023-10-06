#ifndef CL_VERSION_1_0
#define kernel
#define global
#define local
#define constant
typedef unsigned char uchar;
int get_global_id(int);
int get_global_size(int);

#define FILTER_WIDTH 1
#define FILTER_HEIGHT 1
#define STRIDE_X 1
#define STRIDE_Y 1
#endif

kernel void conv(global const uchar* in, global float* out, constant const float* filter) {
    if(get_global_id(0) > get_global_size(0) - FILTER_WIDTH/2 || get_global_id(1) > get_global_size(1) - FILTER_HEIGHT/2)
        return;

	const int global_linear_id = STRIDE_X*get_global_id(0) + STRIDE_Y*get_global_id(1)*STRIDE_X*get_global_size(0);
	float sum = 0.0f;
	for(int y = 0; y < FILTER_HEIGHT; y++) {
		for(int x = 0; x < FILTER_WIDTH; x++) {
			sum += (in[global_linear_id + y*STRIDE_X*get_global_size(0) + x] - 128) * filter[y*FILTER_WIDTH + x];
		}
	}
	out[get_global_id(0) + get_global_id(1)*get_global_size(0)] = sum;
}
