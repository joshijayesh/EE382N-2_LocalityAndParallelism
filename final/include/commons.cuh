#pragma once

#include <string>
#include <iostream>
#include <cuda.h>

#define THREADS_PER_WARP 32
#define THREADS_PER_WARP_MASK 0x1f
#define THREADS_PER_WARP_LOG 5
#define THREADS_PER_BLOCK 256
#define WARPS_PER_BLOCK 8

#define FULL_WARP_MASK 0xFFFFFFFF

#define TRANSPOSE_BLOCK_DIM_X 32
#define TRANSPOSE_BLOCK_DIM_Y 8
// Number of pixel per thread
#define TRANSPOSE_BLOCK_STRIDE 4
#define TRANSPOSE_TILE 32

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

#define MATMUL_TILE_DIM 32
#define MATMUL_BLOCK_DIM_X 32
#define MATMUL_BLOCK_DIM_Y 8

inline void CUDAERR_CHECK(cudaError_t err, std::string error_msg, int err_num) {
    if(err != cudaSuccess) {
        std::cerr << "CUDA ERROR: " << error_msg << std::endl;
        std::cerr << "CUDA ERROR: " << err << std::endl;
        exit(err_num);
    }
}
