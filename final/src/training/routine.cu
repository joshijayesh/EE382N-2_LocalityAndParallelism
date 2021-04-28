#include <vector>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "commons.hpp"
#include "pgm/pgm.hpp"
#include "training/routine.hpp"

#include "training/routine.cuh"
#include "training/kernels.cuh"

#include "checker/routine_test.cuh"

// After some digging __constant__ is not scalable across multiple files... this thing sucks~
// Hence now just use const ptrs passed to each kernel
// __constant__ DeviceConstants pca_dev_params;


void PCARoutine::load_matrix() {
    int device_count;

    CUDAERR_CHECK(
        cudaGetDeviceCount(&device_count),
        "Unable to read CUDA Device Count", ERR_CUDA_GET_DEVICE);

    std::cout << "Num CUDA Devices: " << device_count << std::endl;

    // Allocate the matrix on the GPU
    CUDAERR_CHECK(
        cudaMalloc((void **) &d_data, sizeof(float) * width * height * num_images),
        "Unable to malloc d_data", ERR_CUDA_MALLOC);


    CUDAERR_CHECK(
        cudaMalloc((void **) &d_params, sizeof(DeviceConstants)),
        "Unable to malloc d_params", ERR_CUDA_MALLOC);
    
    // Copy over data to the GPU
    int i = 0;
    for (PGMData img : pgm_list) {
        CUDAERR_CHECK(
            cudaMemcpy(d_data + (i++ * width * height),
                       img.matrix,
                       sizeof(float) * width * height,
                       cudaMemcpyHostToDevice),
            "Unable to copy matrices to device!", ERR_CUDA_MEMCPY);
    }

    // Allocate params
    DeviceConstants params;
    params.width = width;
    params.height = height;
    params.num_images = num_images;
    params.data = d_data;
    params.image_size = width * height;

    CUDAERR_CHECK(
        cudaMemcpy(d_params,
                   &params,
                   sizeof(DeviceConstants),
                   cudaMemcpyHostToDevice),
        "Unable to copy device constants to device!", ERR_CUDA_MEMCPY);

    std::cout << "Finished GPU vars" << std::endl;
}

// Calculates mean and subtracts from each image yielding A
void PCARoutine::mean_image() {
    // 1 warp per pixel
    uint32_t nx = (width + WARPS_PER_BLOCK) / WARPS_PER_BLOCK;
    dim3 blocks2D (nx, height);
    dim3 grid2D (THREADS_PER_BLOCK, 1);

    mean_reduce<<<blocks2D, grid2D>>> ((DeviceConstants *)(d_params));
    cudaDeviceSynchronize();

    #ifdef EN_CHECKER
    mean_checker(width, height, pgm_list, d_data);
    #endif
}

void PCARoutine::transpose() {

}

void PCARoutine::matmul() {

}

PCARoutine::~PCARoutine() {
    if (d_data) {
        std::cout << "Cleaning up~" << std::endl;
        cudaFree(d_data);
        cudaFree(d_params);
    }
}

