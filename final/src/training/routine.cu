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
        cudaMalloc((void **) &d_data_temp, sizeof(float) * width * height * num_images),
        "Unable to malloc d_data", ERR_CUDA_MALLOC);

    CUDAERR_CHECK(
        cudaMalloc((void **) &d_data_transpose, sizeof(float) * width * height * num_images),
        "Unable to malloc d_data", ERR_CUDA_MALLOC);

    // cov = (width * num images)^2 -- This is hugeee!
    CUDAERR_CHECK(
        cudaMalloc((void **) &d_data_cov, sizeof(float) * (num_images * num_images)),
        "Unable to malloc d_data", ERR_CUDA_MALLOC);

    CUDAERR_CHECK(
        cudaMalloc((void **) &d_eigenvectors, sizeof(float) * (num_images * num_images)),
        "Unable to malloc d_data", ERR_CUDA_MALLOC);

    CUDAERR_CHECK(
        cudaMalloc((void **) &d_real_eigenvectors, sizeof(float) * (num_images * num_components)),
        "Unable to malloc d_data", ERR_CUDA_MALLOC);

    CUDAERR_CHECK(
        cudaMalloc((void **) &d_real_eigenvectors_transpose, sizeof(float) * (num_images * num_components)),
        "Unable to malloc d_data", ERR_CUDA_MALLOC);

    CUDAERR_CHECK(
        cudaMalloc((void **) &d_results, sizeof(float) * (num_images * num_components)),
        "Unable to malloc d_data", ERR_CUDA_MALLOC);

    CUDAERR_CHECK(
        cudaMalloc((void **) &d_params, sizeof(DeviceConstants)),
        "Unable to malloc d_params", ERR_CUDA_MALLOC);
    
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

    // Allocate params
    DeviceConstants params;
    params.width = width;
    params.height = height;
    params.n = num_images;
    params.m = height * width;
    params.num_images = num_images;
    params.data = d_data_temp;
    params.A = d_data;
    params.A_t = d_data_transpose;
    params.image_size = width * height;

    CUDAERR_CHECK(
        cudaMemcpy(d_params,
                   &params,
                   sizeof(DeviceConstants),
                   cudaMemcpyHostToDevice),
        "Unable to copy device constants to device!", ERR_CUDA_MEMCPY);

    std::cout << "Finished GPU vars" << std::endl;
    std::cout << "width = " << width << std::endl;
    std::cout << "height = " << height << std::endl;
    std::cout << "num_images = " << num_images << std::endl;
    std::cout << "n = " << params.n << std::endl;
    std::cout << "m = " << params.m << std::endl;
}

// Calculates mean and subtracts from each image yielding A
void PCARoutine::mean_image() {
    // 1 warp per pixel
    uint32_t nx = (width + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 blocks2D (nx, height);
    dim3 grid2D (THREADS_PER_BLOCK, 1);

    mean_reduce<<<blocks2D, grid2D>>> ((DeviceConstants *)(d_params));
    cudaDeviceSynchronize();

    #ifdef EN_CHECKER
    mean_checker(width, height, pgm_list, d_data);
    #endif
}

void PCARoutine::compute_covariance() {
    uint32_t n = num_images;
    uint32_t m = width * height;

    dim3 block2D (((n + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE), ((m + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE));
    dim3 grid2D (TRANSPOSE_BLOCK_DIM_X, TRANSPOSE_BLOCK_DIM_Y);

    transpose_kernel<<<block2D, grid2D>>> (n, m, d_data, d_data_transpose);
    cudaDeviceSynchronize();

    #ifdef EN_CHECKER
    transpose_checker(n, m, d_data, d_data_transpose);
    #endif

    dim3 m_block2D (((n + MATMUL_TILE_DIM - 1) / MATMUL_TILE_DIM), ((n + MATMUL_TILE_DIM - 1) / MATMUL_TILE_DIM));
    dim3 m_grid2D (MATMUL_BLOCK_DIM_X, MATMUL_BLOCK_DIM_Y);

    matmul<<<m_block2D, m_grid2D>>> (n, m, n, d_data_transpose, d_data, d_data_cov);
    identity_matrix<<<m_block2D, m_grid2D>>> (n, d_eigenvectors);
    cudaDeviceSynchronize();

    #ifdef EN_CHECKER
    matmul_checker(n, m, n, d_data_transpose, d_data, d_data_cov);
    #endif
}

void PCARoutine::pca_reduction() {
}


void PCARoutine::post_process() {
    uint32_t n = num_images;
    uint32_t m = width * height;
    uint32_t p = num_components;

    dim3 block2D (((m + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE), ((p + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE));
    dim3 grid2D (TRANSPOSE_BLOCK_DIM_X, TRANSPOSE_BLOCK_DIM_Y);

    matmul<<<block2D, grid2D>>> (m, n, p, d_data, d_eigenvectors, d_real_eigenvectors);
    cudaDeviceSynchronize();

    #ifdef EN_CHECKER
    matmul_checker(m, n, p, d_data, d_eigenvectors, d_real_eigenvectors);
    #endif

    transpose_kernel<<<block2D, grid2D>>> (p, m, d_real_eigenvectors, d_real_eigenvectors_transpose);

    #ifdef EN_CHECKER
    transpose_checker(p, m, d_real_eigenvectors, d_real_eigenvectors_transpose);
    #endif

    dim3 block2D_2 (((m + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE), ((p + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE));
    dim3 grid2D_2 (TRANSPOSE_BLOCK_DIM_X, TRANSPOSE_BLOCK_DIM_Y);

    matmul<<<block2D_2, grid2D_2>>> (p, m, n, d_real_eigenvectors_transpose, d_data, d_results);

    #ifdef EN_CHECKER
    matmul_checker(p, m, n, d_data, d_eigenvectors, d_real_eigenvectors);
    #endif
}

PCARoutine::~PCARoutine() {
    if (d_data) {
        std::cout << "Cleaning up~" << std::endl;
        cudaFree(d_data);
        cudaFree(d_data_temp);
        cudaFree(d_data_transpose);
        cudaFree(d_data_cov);
        cudaFree(d_eigenvectors);
        cudaFree(d_real_eigenvectors);
        cudaFree(d_real_eigenvectors_transpose);
        cudaFree(d_params);
        cudaFree(d_results);
    }
}

