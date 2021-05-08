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
#include "training/kernels.cuh"

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
    num_components = text_conv.ev.n < num_components ? text_conv.ev.n : num_components;
    load_target(text_conv.ev, d_train_ev);
    load_target(text_conv.wv, d_train_wv);
    num_train_images = text_conv.wv.m;

    CUDAERR_CHECK(
        cudaMalloc((void **) &d_data_temp, sizeof(float) * width * height * num_images),
        "Unable to malloc d_data", ERR_CUDA_MALLOC);

    CUDAERR_CHECK(
        cudaMalloc((void **) &d_data, sizeof(float) * width * height * num_images),
        "Unable to malloc d_data", ERR_CUDA_MALLOC);

    CUDAERR_CHECK(
        cudaMalloc((void **) &d_results, sizeof(float) * num_components * num_images),
        "Unable to malloc d_data", ERR_CUDA_MALLOC);

    CUDAERR_CHECK(
        cudaMalloc((void **) &d_predictions, sizeof(uint32_t) * num_images),
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

    uint32_t n = num_images;
    uint32_t m = width * height;
    uint32_t p = num_components;

    dim3 block2D_2 (((n + MATMUL_TILE_DIM - 1) / MATMUL_TILE_DIM), ((p + MATMUL_TILE_DIM - 1) / MATMUL_TILE_DIM));
    dim3 grid2D_2 (MATMUL_BLOCK_DIM_X, MATMUL_BLOCK_DIM_Y);

    // Projection: Gamma = U_T * A
    matmul<<<block2D_2, grid2D_2>>> (p, m, n, d_train_ev, d_data, d_results);
    cudaDeviceSynchronize();

    #ifdef EN_CHECKER
    matmul_checker(p, m, n, d_train_ev, d_data, d_results);
    #endif
}

void PCATest::find_euclidian() {
    uint32_t blocks_x = (num_images + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    dim3 blockDim(THREADS_PER_BLOCK,1) ;
    dim3 gridDim(blocks_x,1);

    nearest_vector<<<gridDim,blockDim>>>(num_images, num_train_images, num_components, num_train_per_person,
                                         d_train_wv, d_results, d_predictions);
    cudaDeviceSynchronize();
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
        cudaFree(d_results);
        cudaFree(d_data_temp);
        cudaFree(d_predictions);
    }
}

