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


void mean_checker(int width, int height, std::vector<PGMData> pgm_list, float* d_data) {
    float *data;
    bool fail = false;

    data = (float *) malloc(sizeof(float) * (width * height) * pgm_list.size());

    CUDAERR_CHECK(
        cudaMemcpy(data,
                   d_data,
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
            for(PGMData img : pgm_list) {
                float result = data[pixel + (k++ * (width * height))];
                if(result != img.matrix[pixel] - sum) {
                    std::cout << "Mean compare failed px " << pixel << "! Expected " << img.matrix[pixel] - sum << " Actual " << result << std::endl;
                    fail = true;
                }
            }
        }
    }

    CERR_CHECK(!fail, "Mean checker failed!!", ERR_CHECKER_FAILED);

    free(data);
}

