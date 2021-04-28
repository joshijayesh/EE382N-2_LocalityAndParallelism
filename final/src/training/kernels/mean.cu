#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "training/routine.cuh"


__global__
void mean_reduce() {
    uint16_t wid = threadIdx.x >> THREADS_PER_WARP_LOG;
    uint16_t lane = threadIdx.x & THREADS_PER_WARP_MASK;
    uint32_t idx_x = ((blockIdx.x * WARPS_PER_BLOCK) + wid);

    if(blockIdx.x == 1 && threadIdx.x == 0) {
        printf("bidx %d idx %d width %d\n", blockIdx.x, idx_x, pca_dev_params.width);
    }

    // 1 warp per pixel
    if(idx_x < pca_dev_params.width) {
        uint32_t pixel = (blockIdx.y * pca_dev_params.width) + idx_x;
        uint32_t stride = pca_dev_params.image_size;
        float *img_ptr = pca_dev_params.data + pixel + (lane * stride);

        uint16_t img_num = lane;
        float sum = 0.0;

        if(blockIdx.x == 1 && threadIdx.x == 0) {
            printf("bidx %d pixel %d\n", blockIdx.x,  pixel);
        }
        
        // Noted this is a bit inefficient if num_images < 32
        while(img_num < pca_dev_params.num_images) {
            sum += *img_ptr;
            img_ptr += stride;
            img_num += THREADS_PER_WARP;
        }
        
        if(blockIdx.x == 1 && threadIdx.x == 0) {
            printf("bix %d sum %f\n", blockIdx.x,  sum);
        }

        for(int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(FULL_WARP_MASK, sum, offset);
        }

        if(lane == 0) {
            sum = sum / pca_dev_params.num_images;

            *(pca_dev_params.mean + pixel) = sum;
        }
    }
}

