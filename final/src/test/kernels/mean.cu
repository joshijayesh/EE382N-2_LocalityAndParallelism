#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "training/routine.cuh"
#include "commons.cuh"


__global__
void mean_subtract(uint32_t width, uint32_t image_size, uint32_t num_images, float* data, float* A, float* mean_in) {
    uint16_t wid = threadIdx.x >> THREADS_PER_WARP_LOG;
    uint16_t lane = threadIdx.x & THREADS_PER_WARP_MASK;
    uint32_t idx_x = ((blockIdx.x * WARPS_PER_BLOCK) + wid);

    // 1 warp per pixel
    if(idx_x < width) {
        uint32_t pixel = (blockIdx.y * width) + idx_x;
        uint32_t stride = image_size;
        uint16_t img_num = lane;
        float *img_ptr = data + pixel + (lane * stride);

        float mean = mean_in[pixel];
        
        uint32_t A_pixel = ((blockIdx.y * width) + idx_x) * num_images;
        uint32_t A_stride = 1;
        float *A_ptr = A + A_pixel + (lane * A_stride);

        while(img_num < num_images) {
            *A_ptr = *img_ptr - mean;
            img_ptr += (stride * THREADS_PER_WARP);
            A_ptr += (A_stride * THREADS_PER_WARP);
            img_num += THREADS_PER_WARP;
        }
    }
}

