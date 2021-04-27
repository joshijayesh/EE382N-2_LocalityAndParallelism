#include <vector>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "commons.hpp"
#include "pgm/pgm.hpp"
#include "training/routine.hpp"

#include "training/routine.cuh"

__constant__ DeviceConstants pca_dev_params;


void PCARoutine::load_matrix() {
    int device_count;

    CUDAERR_CHECK(
        cudaGetDeviceCount(&device_count),
        "Unable to read CUDA Device Count", ERR_CUDA_GET_DEVICE);

    std::cout << "Num CUDA Devices: " << device_count << std::endl;

    cudaMalloc(&d_data, sizeof(uint8_t) * width * height * num_images);

    int i = 0;
    for (PGMData img : pgm_list) {
        CUDAERR_CHECK(
            cudaMemcpy(d_data + (i++ * width * height),
                       img.matrix,
                       sizeof(uint8_t) * width * height,
                       cudaMemcpyHostToDevice),
            "Unable to copy matrices to device!", ERR_CUDA_MEMCPY);
    }

    DeviceConstants params;
    params.width = width;
    params.height = height;
    params.num_images = num_images;
    params.data = d_data;

    CUDAERR_CHECK(
        cudaMemcpyToSymbol(pca_dev_params, &params, sizeof(DeviceConstants)),
        "Unable to copy device constants to device!", ERR_CUDA_MEMCPY);
}

void PCARoutine::mean_image() {

}

void PCARoutine::subtract() {

}

void PCARoutine::transpose() {

}

void PCARoutine::matmul() {

}

PCARoutine::~PCARoutine() {
    if (d_data) {
        cudaFree(d_data);
    }
}

