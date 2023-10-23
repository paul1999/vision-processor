void kernel conv(global const uchar* in, global float* out, constant float* filter, local float* cache) {
	const int local_width = get_local_size(1) + FILTER_SIZE - 1;
	const int local_linear_id = get_local_id(0) + get_local_id(1)*local_width;
	const int global_linear_id = get_global_id(0) + get_global_id(1)*get_global_size(0);
	cache[local_linear_id] = in[global_linear_id] - 128;

	if(get_global_id(0) > get_global_size(0) - FILTER_SIZE || get_global_id(1) > get_global_size(1) - FILTER_SIZE) {
		barrier(CLK_LOCAL_MEM_FENCE);
		return;
	}

	if(get_local_id(0) > get_local_size(0) - FILTER_SIZE) //TODO > or >=?
		cache[local_linear_id + FILTER_SIZE] = in[global_linear_id + FILTER_SIZE] - 128;
	if(get_local_id(1) > get_local_size(1) - FILTER_SIZE) //TODO > or >=?
		cache[local_linear_id + local_width*FILTER_SIZE] = in[global_linear_id + get_global_size(0)*FILTER_SIZE] - 128;

	barrier(CLK_LOCAL_MEM_FENCE);
	if(get_global_id(0)%2 == 1 || get_global_id(1)%2 == 1) //TODO baked STRIDE
		return;

	float sum = 0.0f;
	for(int y = 0; y < FILTER_SIZE; y++) {
		for(int x = 0; x < FILTER_SIZE; x++) {
			sum += cache[local_linear_id + y*FILTER_SIZE + x] * filter[y*FILTER_SIZE + x];
		}
	}
	out[get_global_id(0)/2 + get_global_id(1)*get_global_size(0)/4] = sum; //TODO out other width than in
}