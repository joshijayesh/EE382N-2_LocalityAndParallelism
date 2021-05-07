#pragma once


__global__
void mean_subtract(uint32_t width, uint32_t image_size, uint32_t num_images, float* data, float* A, float* mean_data);

