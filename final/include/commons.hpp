#ifndef __COMMON_H_
#define __COMMON_H_

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>

// Comment this out to disable checker
// #define EN_CHECKER

#define SUCCESS 0
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
#define ERR_INVALID_ARG 12

inline void CERR_CHECK(bool check, std::string error_msg, int err_num) {
    if(!check) {
        std::cerr << "ERROR: " << error_msg << std::endl;
        exit(err_num);
    }
}

typedef struct person {
    std::string path;
    uint32_t num_train;
    uint32_t num_test;
} Person;

#endif // __COMMON_H_

