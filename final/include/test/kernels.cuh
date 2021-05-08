#pragma once


__global__
void mean_subtract(uint32_t width, uint32_t image_size, uint32_t num_images, float* data, float* A, float* mean_data);

__global__
void nearest_vector(int num_test_images, int num_train_images, int num_components, int num_train_per_person, float *train_projections, float *test_projections,int *predictions);

