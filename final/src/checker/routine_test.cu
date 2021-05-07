#include <vector>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "commons.hpp"
#include "commons.cuh"
#include "pgm/pgm.hpp"
#include "training/routine.hpp"

#include "training/routine.cuh"
#include "training/kernels.cuh"


void mean_checker(int width, int height, std::vector<PGMData> pgm_list, float* A) {
    float *data;
    bool fail = false;

    data = (float *) malloc(sizeof(float) * (width * height) * pgm_list.size());

    CUDAERR_CHECK(
        cudaMemcpy(data,
                   A,
                   sizeof(float) * width * height * pgm_list.size(),
                   cudaMemcpyDeviceToHost),
        "Unable to copy data from device!", ERR_CUDA_MEMCPY);

    for (int i = 0; i < height; i += 1) {
        for(int j = 0; j < width; j += 1) {
            int pixel = (i * width) + j;
            float sum = 0.0;
            for(PGMData img : pgm_list) {
                sum += img.matrix[pixel];
            }
            sum /= pgm_list.size();

            int k = 0;
            int row = pixel * pgm_list.size();
            for(PGMData img : pgm_list) {
                float result = data[row + k++];
                if(result != img.matrix[pixel] - sum) {
                    std::cout << "Mean compare failed px " << pixel << "! Expected " << img.matrix[pixel] - sum << " Actual " << result << std::endl;
                    fail = true;
                }
            }
        }
    }

    CERR_CHECK(!fail, "Mean checker failed!!", ERR_CHECKER_FAILED);
    std::cout << "Mean checker passed!" << std::endl;

    free(data);
}

void mean_sub_checker(int width, int height, std::vector<PGMData> pgm_list, float* A, float* mean) {
    float *data;
    float *mean_data;
    bool fail = false;

    data = (float *) malloc(sizeof(float) * (width * height) * pgm_list.size());
    mean_data = (float *) malloc(sizeof(float) * (width * height));

    CUDAERR_CHECK(
        cudaMemcpy(data,
                   A,
                   sizeof(float) * width * height * pgm_list.size(),
                   cudaMemcpyDeviceToHost),
        "Unable to copy data from device!", ERR_CUDA_MEMCPY);

    CUDAERR_CHECK(
        cudaMemcpy(mean_data,
                   mean,
                   sizeof(float) * width * height,
                   cudaMemcpyDeviceToHost),
        "Unable to copy data from device!", ERR_CUDA_MEMCPY);

    for (int i = 0; i < height; i += 1) {
        for(int j = 0; j < width; j += 1) {
            int pixel = (i * width) + j;
            float tgt = mean_data[pixel];

            int k = 0;
            int row = pixel * pgm_list.size();
            for(PGMData img : pgm_list) {
                float result = data[row + k++];
                if(result != (float)(img.matrix[pixel]) - tgt) {
                    std::cout << "Mean compare failed px " << pixel << "! Expected " << (float) (img.matrix[pixel]) - tgt << " Actual " << result << " Tgt " << tgt << std::endl;
                    fail = true;
                }
            }
        }
    }

    CERR_CHECK(!fail, "Mean checker failed!!", ERR_CHECKER_FAILED);
    std::cout << "Mean checker passed!" << std::endl;

    free(data);
    free(mean_data);
}


void transpose_checker(int width, int height, float* A, float* A_T) {
    float *data;
    float *data_T;
    bool fail = false;
    
    data = (float *) malloc(sizeof(float) * (width * height));
    data_T = (float *) malloc(sizeof(float) * (width * height)); 


    CUDAERR_CHECK(
        cudaMemcpy(data,
                   A,
                   sizeof(float) * width * height,
                   cudaMemcpyDeviceToHost),
        "Unable to copy data from device!", ERR_CUDA_MEMCPY);

    CUDAERR_CHECK(
        cudaMemcpy(data_T,
                   A_T,
                   sizeof(float) * width * height,
                   cudaMemcpyDeviceToHost),
        "Unable to copy data from device!", ERR_CUDA_MEMCPY);

    for(int n = 0; n < width * height; n += 1) {
        int i = n / height;
        int j = n % height;
        if(data_T[n] != data[width * j + i]) {
            std::cout << "Transpose failed i=" << i << ",j=" << j << "! Expected " << data[width * j + i] << " Actual " << data_T[n] << std::endl;
            fail = true;
        }
    }
    free(data);
    free(data_T);

    CERR_CHECK(!fail, "Transpose checker failed!!", ERR_CHECKER_FAILED);
    std::cout << "Transpose checker passed!" << std::endl;
}


void matmul_checker(uint32_t n, uint32_t m, uint32_t p, float *d_A, float *d_B, float *d_C) {
    float *A;
    float *B;
    float *C;
    bool fail = false;

    A = (float *) malloc(sizeof(float) * (n * m));
    B = (float *) malloc(sizeof(float) * (m * p));
    C = (float *) malloc(sizeof(float) * (n * p));

    CUDAERR_CHECK(
        cudaMemcpy(A,
                   d_A,
                   sizeof(float) * n * m,
                   cudaMemcpyDeviceToHost),
        "Unable to copy data from device!: A", ERR_CUDA_MEMCPY);

    CUDAERR_CHECK(
        cudaMemcpy(B,
                   d_B,
                   sizeof(float) * m * p,
                   cudaMemcpyDeviceToHost),
        "Unable to copy data from device! :B", ERR_CUDA_MEMCPY);

    CUDAERR_CHECK(
        cudaMemcpy(C,
                   d_C,
                   sizeof(float) * n * p,
                   cudaMemcpyDeviceToHost),
        "Unable to copy data from device! :C", ERR_CUDA_MEMCPY);

    for(int j = 0; j < n; j += 1) {
        for(int i = 0; i < p; i += 1) {
            float sum = 0.0;
            for(int k = 0; k < m; k += 1) {
                sum += A[j * m + k] * B[k * p + i];
            }

            if(C[j * p + i] != sum) {
                std::cout << "MATMUL failed i=" << i << ",j=" << j << "! Expected " << sum << " Actual " << C[j * p + i] << std::endl;
                fail = true;
                CERR_CHECK(!fail, "Matmul checker failed!!", ERR_CHECKER_FAILED);
            }

        }
    }

    free(A);
    free(B);
    free(C);

    CERR_CHECK(!fail, "Matmul checker failed!!", ERR_CHECKER_FAILED);
    std::cout << "Matmul checker passed!" << std::endl;
}

