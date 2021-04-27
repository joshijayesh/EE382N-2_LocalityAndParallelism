#pragma once

#include <string>
#include <stdlib.h>
#include <iostream>

#include <cuda.h>


inline void CUDAERR_CHECK(cudaError_t err, std::string error_msg, int err_num) {
    if(err != cudaSuccess) {
        std::cerr << "CUDA ERROR : " << error_msg << std::endl;
        exit(err_num);
    }
}

