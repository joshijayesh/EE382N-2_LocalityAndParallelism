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


void mean_checker(int width, int height, std::vector<PGMData> pgm_list, float* d_mean) {
    float *mean;

    mean = (float *) malloc(sizeof(float) * (width * height));

    CUDAERR_CHECK(
        cudaMemcpy(mean,
                   d_mean,
                   sizeof(float) * width * height,
                   cudaMemcpyDeviceToHost),
        "Unable to copy mean from device!", ERR_CUDA_MEMCPY);

    for (int i = 0; i < height; i += 1) {
        for(int j = 0; j < width; j += 1) {
            int pixel = (i * width) + j;
            float sum = 0.0;
            for(PGMData img : pgm_list) {
                sum += img.matrix[pixel];
            }
            sum /= pgm_list.size();

            if(sum != mean[pixel]) {
                std::cout << "Mean compare failed! Expected " << sum << " Actual " << mean[pixel] << std::endl;
            }
        }
    }

    free(mean);
}

