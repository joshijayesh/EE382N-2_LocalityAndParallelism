#pragma once


#include <cuda.h>
#include "training/routine.cuh"

__global__ void mean_reduce(const DeviceConstants *pca_dev_params);

