#include <vector>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "commons.hpp"
#include "pgm/pgm.hpp"
#include "training/routine.hpp"

#include "training/routine.cuh"


void PCARoutine::load_matrix() {
    int device_count;

    CUDAERR_CHECK(cudaGetDeviceCount(&device_count), "Unable to read CUDA Device Count", ERR_CUDA_GET_DEVICE);
    std::cout << "Num CUDA Devices: " << device_count << std::endl;
}

void PCARoutine::mean_image() {

}

void PCARoutine::subtract() {

}

void PCARoutine::transpose() {

}

