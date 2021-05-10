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

#include "training/routine.hpp"
#include "training/routine.cuh"
#include "training/kernels.cuh"

#include "checker/routine_test.cuh"

// After some digging __constant__ is not scalable across multiple files... this thing sucks~
// Hence now just use const ptrs passed to each kernel
// __constant__ DeviceConstants pca_dev_params;


void PCATraining::load_matrix() {
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
        cudaMalloc((void **) &d_data_mean, sizeof(float) * width * height),
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
        cudaMalloc((void **) &d_eigenvectors_sorted, sizeof(float) * (num_images * num_images)),
        "Unable to malloc d_data", ERR_CUDA_MALLOC);

    CUDAERR_CHECK(
        cudaMalloc((void **) &d_real_eigenvectors, sizeof(float) * (width * height * num_components)),
        "Unable to malloc d_data", ERR_CUDA_MALLOC);

    CUDAERR_CHECK(
        cudaMalloc((void **) &d_real_eigenvectors_norm, sizeof(float) * (width * height * num_components)),
        "Unable to malloc d_data", ERR_CUDA_MALLOC);

    CUDAERR_CHECK(
        cudaMalloc((void **) &d_real_eigenvectors_transpose, sizeof(float) * (width * height * num_components)),
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

    // Allocate params -- this is unused
    DeviceConstants params;
    params.width = width;
    params.height = height;
    params.m = height * width;
    params.n = num_images;
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
void PCATraining::mean_image() {
    // 1 warp per pixel
    uint32_t nx = (width + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 blocks2D (nx, height);
    dim3 grid2D (THREADS_PER_BLOCK, 1);

    mean_reduce<<<blocks2D, grid2D>>> (width, width * height, num_images, d_data_temp, d_data, d_data_mean);
    cudaDeviceSynchronize();

    #ifdef EN_CHECKER
    mean_checker(width, height, pgm_list, d_data);
    #endif
}

void PCATraining::compute_covariance() {
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

    dim3 m_block2D_I (((n + 16 - 1) / 16), ((n + 16 - 1) / 16));
    dim3 m_grid2D_I (16, 16);
    identity_matrix<<<m_block2D_I, m_grid2D_I>>> (n, d_eigenvectors);
    cudaDeviceSynchronize();

    #ifdef EN_CHECKER
    matmul_checker(n, m, n, d_data_transpose, d_data, d_data_cov);
    #endif

    float *h_identity;

    h_identity = (float *) malloc(sizeof(float) * (n * n));

    CUDAERR_CHECK(
        cudaMemcpy(h_identity,
                   d_eigenvectors,
                   sizeof(float) * n * n,
                   cudaMemcpyDeviceToHost),
        "Unable to copy data from device!: A", ERR_CUDA_MEMCPY);

    std::ofstream file("dump_identity.txt");
    for(int i = 0; i < n; i += 1) {
        for(int j = 0; j < n; j += 1) {
            file << h_identity[i * n + j] << " ";
        }
        file << std::endl;
    }

    free(h_identity);
}

void PCATraining::sort_eigenvectors() {
    uint32_t n = num_images;
    float* v = d_eigenvectors;
    float* w = d_data_cov;

    float *h_eigenvalues;

    h_eigenvalues = (float *) malloc(sizeof(float) * (n * n));

    CUDAERR_CHECK(
        cudaMemcpy(h_eigenvalues,
                   d_data_cov,
                   sizeof(float) * n * n,
                   cudaMemcpyDeviceToHost),
        "Unable to copy data from device!: A", ERR_CUDA_MEMCPY);

    std::ofstream file("dump_eigenvalue_matrix.txt");
    for(int i = 0; i < n; i += 1) {
        for(int j = 0; j < n; j += 1) {
            file << h_eigenvalues[i * n + j] << " ";
        }
        file << std::endl;
    }

    free(h_eigenvalues);


    int* sort_index,*sort_index_copy;

    float *w_1d;
    float *w_1d_copy;
    float *v_sorted = d_eigenvectors_sorted;

    cudaMalloc((void **) &w_1d, sizeof(float)*n);
    cudaMalloc((void **) &w_1d_copy, sizeof(float)*n);

    cudaMalloc((void **) &sort_index, sizeof(int) * n);
    cudaMalloc((void **) &sort_index_copy, sizeof(int) * n);

    dim3 blockDim(256,1);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x);

    sort_initialize<<<gridDim,blockDim>>>(n,sort_index,w_1d,w);
    cudaDeviceSynchronize();

    sort_value_kernel<<<1,1>>>(w_1d,w_1d_copy,sort_index,sort_index_copy,n);
    cudaDeviceSynchronize();

    /*
    float *h_w_1d;

    h_w_1d = (float *) malloc(sizeof(float) * (n));

    CUDAERR_CHECK(
        cudaMemcpy(h_w_1d,
                   w_1d,
                   sizeof(float) * n,
                   cudaMemcpyDeviceToHost),
        "Unable to copy data from device!: A", ERR_CUDA_MEMCPY);

    std::ofstream file("dump_train_ev_sorted.txt");
    for(int i = 0; i < n; i += 1) {
        file << h_w_1d[i] << " ";
    }

    free(h_w_1d);
    */

    sort_vector_kernel<<<gridDim,blockDim>>>(v,v_sorted,sort_index,n);
    cudaDeviceSynchronize();

    float *A;

    A = (float *) malloc(sizeof(int) * (n * n));

    CUDAERR_CHECK(
        cudaMemcpy(A,
                   d_eigenvectors_sorted,
                   sizeof(int) * n * n,
                   cudaMemcpyDeviceToHost),
        "Unable to copy data from device!: A", ERR_CUDA_MEMCPY);

    std::ofstream file2("dump_sorted_ev.txt");
    for(int i = 0; i < n; i += 1) {
        for(int j = 0; j < n; j += 1) {
            file2 << A[i * n + j] << " ";
        }
        file2 << std::endl;
    }
    free(A);

    cudaFree(sort_index);
    cudaFree(sort_index_copy);
    cudaFree(w_1d);
    cudaFree(w_1d_copy);

    return;
}


void PCATraining::post_process() {
    uint32_t n = num_images;
    uint32_t m = width * height;
    uint32_t p = num_components;

    dim3 block2D (((p + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE), ((m + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE));
    dim3 grid2D (MATMUL_BLOCK_DIM_X, MATMUL_BLOCK_DIM_Y);

    // U = A * V
    matmul<<<block2D, grid2D>>> (m, n, p, d_data, d_eigenvectors_sorted, d_real_eigenvectors);
    cudaDeviceSynchronize();

    #ifdef EN_CHECKER
    matmul_checker_s(m, n, p, d_data, d_eigenvectors_sorted, d_real_eigenvectors);
    #endif

    dim3 block2D_3 (1, p);
    dim3 grid2D_3 (THREADS_PER_BLOCK, 1);

    // Normalize squared sum
    norm_squaredsum<<<block2D_3, grid2D_3>>> (m, p, d_real_eigenvectors, d_real_eigenvectors_norm);
    cudaDeviceSynchronize();

    /*
    #ifdef EN_CHECKER
    norm_squaredsum_checker(m, p, d_real_eigenvectors, d_real_eigenvectors_norm);
    #endif
    */

    float *A;

    A = (float *) malloc(sizeof(float) * (m * p));

    CUDAERR_CHECK(
        cudaMemcpy(A,
                   d_real_eigenvectors_norm,
                   sizeof(float) * m * p,
                   cudaMemcpyDeviceToHost),
        "Unable to copy data from device!: A", ERR_CUDA_MEMCPY);

    std::ofstream file("dump_re_norm.txt");
    for(int i = 0; i < m; i += 1) {
        for(int j = 0; j < p; j += 1) {
            file << A[i * p + j] << " ";
        }
        file << std::endl;
    }

    free(A);

    // Transpose for projection
    transpose_kernel<<<block2D, grid2D>>> (p, m, d_real_eigenvectors_norm, d_real_eigenvectors_transpose);
    cudaDeviceSynchronize();

    #ifdef EN_CHECKER
    transpose_checker(p, m, d_real_eigenvectors_norm, d_real_eigenvectors_transpose);
    #endif

    dim3 block2D_2 (((n + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE), ((p + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE));
    dim3 grid2D_2 (TRANSPOSE_BLOCK_DIM_X, TRANSPOSE_BLOCK_DIM_Y);

    // Projection: Gamma = U_T * A
    matmul<<<block2D_2, grid2D_2>>> (p, m, n, d_real_eigenvectors_transpose, d_data, d_results);
    cudaDeviceSynchronize();

    #ifdef EN_CHECKER
    matmul_checker(p, m, n, d_real_eigenvectors_transpose, d_data, d_results);
    #endif
}


void PCATraining::save_to_file(std::string out_file) {
    uint32_t n = num_images;
    uint32_t m = width * height;
    uint32_t p = num_components;

    float *h_real_eigenvectors_transpose;
    float *h_results;
    float *h_mean;

    h_real_eigenvectors_transpose = (float *) malloc(sizeof(float) * (p * m));
    h_results = (float *) malloc(sizeof(float) * (p * n));
    h_mean = (float *) malloc(sizeof(float) * (width * height));

    CUDAERR_CHECK(
        cudaMemcpy(h_real_eigenvectors_transpose,
                   d_real_eigenvectors_transpose,
                   sizeof(float) * p * m,
                   cudaMemcpyDeviceToHost),
        "Unable to copy data from device!: A", ERR_CUDA_MEMCPY);

    CUDAERR_CHECK(
        cudaMemcpy(h_results,
                   d_results,
                   sizeof(float) * p * n,
                   cudaMemcpyDeviceToHost),
        "Unable to copy data from device!: A", ERR_CUDA_MEMCPY);

    CUDAERR_CHECK(
        cudaMemcpy(h_mean,
                   d_data_mean,
                   sizeof(float) * width * height,
                   cudaMemcpyDeviceToHost),
        "Unable to copy data from device!: A", ERR_CUDA_MEMCPY);

    std::ofstream file(out_file);
    file << "Training Results" << std::endl;

    file << "Mean Image" << std::endl;
    file << height << "x" << width << std::endl;

    for(int i = 0; i < height; i += 1) {
        for(int j = 0; j < width; j += 1) {
            file.write(reinterpret_cast<const char *> (&h_mean[i * width + j]), sizeof(float));
        }
        file << std::endl;
    }

    file << std::endl;
    file << "EigenVectors" << std::endl;
    file << p << "x" << m << std::endl;

    for(int i = 0; i < p; i += 1) {
        for(int j = 0; j < m; j += 1) {
            file.write(reinterpret_cast<const char *> (&h_real_eigenvectors_transpose[i * m + j]), sizeof(float));
        }
        file << std::endl;
    }

    file << std::endl;
    file << "Weight Vectors" << std::endl;
    file << p << "x" << n << std::endl;

    for(int i = 0; i < p; i += 1) {
        for(int j = 0; j < n; j += 1) {
            file.write(reinterpret_cast<const char *> (&h_results[i * n + j]), sizeof(float));
        }
        file << std::endl;
    }

    file.close();
    free(h_real_eigenvectors_transpose);
    free(h_results);
    free(h_mean);
}


PCATraining::~PCATraining() {
    if (d_data) {
        std::cout << "Cleaning up~" << std::endl;
        cudaFree(d_data);
        cudaFree(d_data_mean);
        cudaFree(d_data_temp);
        cudaFree(d_data_transpose);
        cudaFree(d_data_cov);
        cudaFree(d_eigenvectors);
        cudaFree(d_eigenvectors_sorted);
	    cudaFree(d_real_eigenvectors);
	    cudaFree(d_real_eigenvectors_norm);
        cudaFree(d_real_eigenvectors_transpose);
        cudaFree(d_params);
        cudaFree(d_results);
    }
}

