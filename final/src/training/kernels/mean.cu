#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "training/routine.cuh"


__global__
void mean_reduce(const DeviceConstants *pca_dev_params) {
    uint16_t wid = threadIdx.x >> THREADS_PER_WARP_LOG;
    uint16_t lane = threadIdx.x & THREADS_PER_WARP_MASK;
    uint32_t idx_x = ((blockIdx.x * WARPS_PER_BLOCK) + wid);

    // 1 warp per pixel
    if(idx_x < pca_dev_params->width) {
        uint32_t pixel = (blockIdx.y * pca_dev_params->width) + idx_x;
        uint32_t stride = pca_dev_params->image_size;
        float *img_ptr = pca_dev_params->data + pixel + (lane * stride);

        uint16_t img_num = lane;
        float sum = 0.0;

        
        // Noted this is a bit inefficient if num_images < 32
        // But aint noone runing actual training with that little training set
        while(img_num < pca_dev_params->num_images) {
            sum += *img_ptr;
            img_ptr += (stride * THREADS_PER_WARP);
            img_num += THREADS_PER_WARP;
        }
        
        for(int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(FULL_WARP_MASK, sum, offset);
        }

        if(lane == 0) {
            sum = sum / pca_dev_params->num_images;
        }

        float mean = __shfl_sync(FULL_WARP_MASK, sum, 0);

        img_num = lane;
        img_ptr = pca_dev_params->data + pixel + (lane * stride);

        uint32_t A_pixel = (blockIdx.y * pca_dev_params->n) + idx_x;
        uint32_t A_stride = pca_dev_params->width;
        float *A_ptr = pca_dev_params->A + A_pixel + (lane * A_stride);

        while(img_num < pca_dev_params->num_images) {
            *A_ptr = *img_ptr - mean;
            img_ptr += (stride * THREADS_PER_WARP);
            A_ptr += (A_stride * THREADS_PER_WARP);
            img_num += THREADS_PER_WARP;
        }
    }
}

