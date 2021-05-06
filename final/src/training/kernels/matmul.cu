#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "training/routine.cuh"


__global__
void matmul(uint32_t n, uint32_t m, uint32_t p, float *A, float *B, float *C) {
    __shared__ float Ab[MATMUL_TILE_DIM][MATMUL_TILE_DIM + 1];
    __shared__ float Bb[MATMUL_TILE_DIM][MATMUL_TILE_DIM + 1];

    uint32_t idx_x = (blockIdx.x * MATMUL_TILE_DIM) + threadIdx.x;
    uint32_t idx_y = (blockIdx.y * MATMUL_TILE_DIM) + threadIdx.y;

    float C_temp[4] = {0.0, 0.0, 0.0, 0.0};

    for (uint32_t tile = 0; tile < m; tile += MATMUL_TILE_DIM) {
        for(uint32_t j = 0; j < MATMUL_TILE_DIM; j += MATMUL_BLOCK_DIM_Y) {
            if(tile + threadIdx.x < m && (idx_y + j) < n)
                Ab[threadIdx.y + j][threadIdx.x] = A[(idx_y + j) * m + tile + threadIdx.x];
            else
                Ab[threadIdx.y + j][threadIdx.x] = 0.0;

            if(tile + threadIdx.y + j < m && idx_x < p)
                Bb[threadIdx.y + j][threadIdx.x] = B[(tile + threadIdx.y + j) * p + idx_x];
            else
                Bb[threadIdx.y + j][threadIdx.x] = 0.0;
        }

        __syncthreads();

        for(uint32_t j = 0; j < MATMUL_TILE_DIM; j += MATMUL_BLOCK_DIM_Y) {
            for(uint32_t i = 0; i < MATMUL_BLOCK_DIM_X; i += 1) {
                C_temp[j / MATMUL_BLOCK_DIM_Y] += Ab[threadIdx.y + j][i] * Bb[i][threadIdx.x];
            }
        }

        __syncthreads();
    }

    for(uint32_t j = 0; j < MATMUL_TILE_DIM; j += MATMUL_BLOCK_DIM_Y) {
        if((idx_y + j) < n && idx_x < p)  {
            C[(idx_y + j) * p + idx_x] = C_temp[j / MATMUL_BLOCK_DIM_Y];
        }
    }
}

