#pragma once


#include <cuda.h>
#include "training/routine.cuh"

__global__
void mean_reduce(uint32_t width, uint32_t image_size, uint32_t num_images, float* data, float* A, float* mean_data);

__global__
void norm_squaredsum(uint32_t n, uint32_t m, float *in, float *out);

__global__
void transpose_kernel(uint32_t n, uint32_t m, float *A, float *A_t);

__global__
void matmul(uint32_t n, uint32_t m, uint32_t p, float *A, float *B, float *C);

__global__
void identity_matrix(uint32_t n, float *A);

__global__
void sort_initialize(int n, int* sort_index, float* w_1d, float* w);

__global__
void sort_vector_kernel(float* v, float*v_copy,int* sort_index,int n);

__global__
void sort_value_kernel(float* w, float* w_copy,int* sort_index,int* sort_index_copy, int n);

