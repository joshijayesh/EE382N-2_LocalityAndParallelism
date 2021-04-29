#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "training/routine.cuh"


__global__
void transpose_kernel(uint32_t n, uint32_t m, float *A, float *A_t) {
    __shared__ float block[TRANSPOSE_TILE][TRANSPOSE_TILE + 1];

    uint16_t idx_x = (blockIdx.x * TRANSPOSE_TILE) + threadIdx.x;
    uint16_t idx_y = (blockIdx.y * TRANSPOSE_TILE) + threadIdx.y;

    for(int j = 0; j < TRANSPOSE_TILE; j += TRANSPOSE_BLOCK_DIM_Y)
        if((idx_y + j) < m && idx_x < n)
            block[threadIdx.y + j][threadIdx.x] = A[(idx_y + j) * n + idx_x];

    __syncthreads();
    idx_x = (blockIdx.y * TRANSPOSE_TILE) + threadIdx.x;
    idx_y = (blockIdx.x * TRANSPOSE_TILE) + threadIdx.y;

    for(int j = 0; j < TRANSPOSE_TILE; j += TRANSPOSE_BLOCK_DIM_Y)
        if((idx_y + j) < n && idx_x < m)
            A_t[(idx_y + j) * m + idx_x] = block[threadIdx.x][threadIdx.y + j];
}


/* naive version (slower)
__global__
void transpose_kernel(uint32_t n, uint32_t m, float *A, float *A_t) {
    uint16_t idx_x = (blockIdx.x * TRANSPOSE_TILE) + threadIdx.x;
    uint16_t idx_y = (blockIdx.y * TRANSPOSE_TILE) + threadIdx.y;

    for(int j = 0; j < TRANSPOSE_TILE; j += TRANSPOSE_BLOCK_DIM_Y)
        if((idx_y + j) < m && idx_x < n)
            A_t[(idx_x * m) + (idx_y + j)] = A[(idx_y + j) * n + idx_x];
}
*/

