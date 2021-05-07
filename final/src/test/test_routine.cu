#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "commons.hpp"
#include "commons.cuh"
#include "pgm/pgm.hpp"

#include "test/test_routine.hpp"
#include "test/kernels.cuh"

#include "checker/routine_test.cuh"


inline void load_target(PCAMatrix matrix, float*& ptr) {
    CUDAERR_CHECK(
        cudaMalloc((void **) &ptr, sizeof(float) * matrix.n * matrix.m),
        "Unable to malloc d_data", ERR_CUDA_MALLOC);

    CUDAERR_CHECK(
        cudaMemcpy(ptr,
                   matrix.matrix,
                   sizeof(float) * matrix.n * matrix.m,
                   cudaMemcpyHostToDevice),
        "Unable to copy device constants to device!", ERR_CUDA_MEMCPY);
}


void PCATest::load_matrix(PCATextConv text_conv) {
    int device_count;

    CUDAERR_CHECK(
        cudaGetDeviceCount(&device_count),
        "Unable to read CUDA Device Count", ERR_CUDA_GET_DEVICE);

    std::cout << "Num CUDA Devices: " << device_count << std::endl;
    std::cout << height << "x" << width << std::endl;
    std::cout << "Num Images: " << num_images << std::endl;

    load_target(text_conv.mean, d_mean);
    load_target(text_conv.ev, d_train_ev);
    load_target(text_conv.wv, d_train_wv);

    CUDAERR_CHECK(
        cudaMalloc((void **) &d_data_temp, sizeof(float) * width * height * num_images),
        "Unable to malloc d_data", ERR_CUDA_MALLOC);

    CUDAERR_CHECK(
        cudaMalloc((void **) &d_data, sizeof(float) * width * height * num_images),
        "Unable to malloc d_data", ERR_CUDA_MALLOC);


    // Copy over data to the GPU
    int i = 0;
    for (PGMData img : pgm_list) {
        CUDAERR_CHECK(
            cudaMemcpy(d_data_temp + (i++ * width * height),
                       img.matrix,
                       sizeof(float) * width * height,
                       cudaMemcpyHostToDevice),
            "Unable to copy matrices to device!", ERR_CUDA_MEMCPY);
    }
}

void PCATest::mean_image() {
    // 1 warp per pixel
    uint32_t nx = (width + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 blocks2D (nx, height);
    dim3 grid2D (THREADS_PER_BLOCK, 1);

    mean_subtract<<<blocks2D, grid2D>>> (width, width * height, num_images, d_data_temp, d_data, d_mean);
    cudaDeviceSynchronize();

    #ifdef EN_CHECKER
    mean_sub_checker(width, height, pgm_list, d_data, d_mean);
    #endif
}

void PCATest::find_euclidian() {

}

void PCATest::find_confidence() {

}

void PCATest::final_image() {

}

PCATest::~PCATest() {
    if(d_mean) {
        std::cout << "Cleaning up~" << std::endl;
        cudaFree(d_mean);
        cudaFree(d_train_ev);
        cudaFree(d_train_wv);
        cudaFree(d_data);
        cudaFree(d_data_temp);
    }
}

