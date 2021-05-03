#pragma once


#include <cuda.h>
#include "training/routine.cuh"

__global__
void mean_reduce(const DeviceConstants *pca_dev_params);

__global__
void norm_squaredsum(uint32_t n, uint32_t m, float *in, float *out);

__global__
void transpose_kernel(uint32_t n, uint32_t m, float *A, float *A_t);

__global__
void matmul(uint32_t n, uint32_t m, uint32_t p, float *A, float *B, float *C);

__global__
void identity_matrix(uint32_t n, float *A);

