#ifndef __MY_COMMON_H
#define __MY_COMMON_H

#include <cstdio>
#include <cstdlib>
#include <cuda.h>

#define CHECK_CUDA_ERROR(call) { cudaAssert((call), __FILE__, __LINE__); }

inline void cudaAssert(cudaError_t code, const char *file, int line) {
    if(code != cudaSuccess) {
        fprintf(stderr, "CUDA ERROR: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

#define ERROR_HOST_MALLOC_FAILED 255

#define CHECK_HOST_MALLOC(call) hostMallocCheck((call), __FILE__, __LINE__)

inline void* hostMallocCheck(void *ptr, const char *file, int line) {
    if(ptr == NULL) {
        fprintf(stderr, "HOST MALLOC FAILED: %p %s %d\n", ptr, file, line);
        exit(ERROR_HOST_MALLOC_FAILED);
    }
    return ptr;
}

#endif
