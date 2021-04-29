#pragma once


#include <cuda.h>
#include "training/routine.cuh"

__global__ void mean_reduce(const DeviceConstants *pca_dev_params);

__global__
void transpose_kernel(uint32_t n, uint32_t m, float *A, float *A_t);

