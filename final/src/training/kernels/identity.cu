#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "commons.cuh"
#include "training/routine.cuh"

__global__
void identity_matrix(uint32_t n, float *A) {
    uint32_t idx_x = (blockIdx.x * 16) + threadIdx.x;
    uint32_t idx_y = (blockIdx.y * 16) + threadIdx.y;

    if(idx_x < n && idx_y < n) {
        A[idx_y * n + idx_x] = idx_x == idx_y ? 1 : 0;
    }
}

