#pragma once

#include <string>
#include <stdlib.h>
#include <iostream>

#define TRANSPOSE_BLOCK_DIM_X 32
#define TRANSPOSE_BLOCK_DIM_Y 8
// Number of pixel per thread
#define TRANSPOSE_BLOCK_STRIDE 4
#define TRANSPOSE_TILE 32

#define MATMUL_TILE_DIM 32
#define MATMUL_BLOCK_DIM_X 32
#define MATMUL_BLOCK_DIM_Y 8

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



