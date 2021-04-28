#ifndef __COMMON_H_
#define __COMMON_H_

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>

// Comment this out to disable checker
#define EN_CHECKER

#define ERR_FAILED_OPEN_PGM 1
#define ERR_UNSUPPORTED_GRAY 2
#define ERR_ARGPARSE_FAILED 3
#define ERR_FAILED_OPEN_FILE 4
#define ERR_NO_IMGES_FOUND 5
#define ERR_IMG_DIM_MISMATCH 6
#define ERR_CUDA_GET_DEVICE 7
#define ERR_CUDA_MEMCPY 8
#define ERR_UNKNOWN_TARGET 9
#define ERR_CUDA_MALLOC 10
#define ERR_CHECKER_FAILED 11

inline void CERR_CHECK(bool check, std::string error_msg, int err_num) {
    if(!check) {
        std::cerr << "ERROR: " << error_msg << std::endl;
        exit(err_num);
    }
}

#endif // __COMMON_H_

