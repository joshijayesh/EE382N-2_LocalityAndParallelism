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

#define FULL_WARP_MASK 0xFFFFFFFF


typedef struct devConsts {
    int width;
    int height;
    int image_size;
    int num_images;

    float* data;
    float* mean;
} DeviceConstants;

inline void CUDAERR_CHECK(cudaError_t err, std::string error_msg, int err_num) {
    if(err != cudaSuccess) {
        std::cerr << "CUDA ERROR: " << error_msg << std::endl;
        exit(err_num);
    }
}

// Maybe put this in a separate file
__constant__ DeviceConstants pca_dev_params;

