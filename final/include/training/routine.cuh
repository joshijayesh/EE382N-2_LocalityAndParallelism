#pragma once

#include <string>
#include <stdlib.h>
#include <iostream>

#include <cuda.h>

#define THREADS_PER_WARP 32
#define THREADS_PER_WARP_MASK 0x1f
#define THREADS_PER_WARP_LOG 5
#define THREADS_PER_BLOCK 256
#define WARPS_PER_BLOCK 8
#define TRANSPOSE_BLOCK_DIM_X 32
#define TRANSPOSE_BLOCK_DIM_Y 8
// Number of pixel per thread
#define TRANSPOSE_BLOCK_STRIDE 4
#define TRANSPOSE_TILE 32

#define FULL_WARP_MASK 0xFFFFFFFF

typedef struct devConsts {
    int width;
    int height;
    int n;
    int m;
    int image_size;
    int num_images;

    float* data;
    float* A;
    float* A_t;
} DeviceConstants;

inline void CUDAERR_CHECK(cudaError_t err, std::string error_msg, int err_num) {
    if(err != cudaSuccess) {
        std::cerr << "CUDA ERROR: " << error_msg << std::endl;
        exit(err_num);
    }
}

// Maybe put this in a separate file

