#pragma once

#include <string>
#include <stdlib.h>
#include <iostream>

#include <cuda.h>

#define CUDAMEMCPYH2D

typedef struct devConsts {
    int width;
    int height;
    int num_images;

    uint8_t* data;
} DeviceConstants;

inline void CUDAERR_CHECK(cudaError_t err, std::string error_msg, int err_num) {
    if(err != cudaSuccess) {
        std::cerr << "CUDA ERROR: " << error_msg << std::endl;
        exit(err_num);
    }
}

