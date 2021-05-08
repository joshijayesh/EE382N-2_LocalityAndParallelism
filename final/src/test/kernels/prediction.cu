#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>
#include <stdlib.h>


//#include "jacobi.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#define THREADS_PER_WARP 32
#define THREADS_PER_WARP_MASK 0x1f
#define THREADS_PER_WARP_LOG 5
#define THREADS_PER_BLOCK 256
#define WARPS_PER_BLOCK 8

#define FULL_WARP_MASK 0xFFFFFFFF




__global__ void nearest_vector(int num_test_images, int num_train_images, int num_components, int num_train_per_person, float *train_projections, float *test_projections,int *predictions)
{
uint16_t wid = threadIdx.x >> THREADS_PER_WARP_LOG;
uint16_t lane = threadIdx.x & THREADS_PER_WARP_MASK;
uint32_t idx_x = ((blockIdx.x * WARPS_PER_BLOCK) + wid);


if(idx_x<num_test_images)
{
    // printf("idx_x = %d\n",idx_x );
    float *test_img_ptr = test_projections+ idx_x;
    uint16_t img_num = lane;
    float *train_img_ptr = train_projections + lane;
    uint32_t stride1 = num_train_images;
    uint32_t stride2 = num_test_images;

    float min_dst=-1;
    int min_idx;

    

    while(img_num < num_train_images) {

        float l2_dist = 0.0;
        float u;
        for(int i=0;i<num_components;i++)
        {
            u = *test_img_ptr- *train_img_ptr;
            l2_dist += u*u ;
            train_img_ptr += stride1;
            test_img_ptr += stride2;

        }

        if(min_dst<0) 
            {
                min_dst = l2_dist;
                min_idx = img_num;
            }
        else
        {
            if(l2_dist<min_dst)
            {
                min_dst = l2_dist;
                min_idx = img_num;
            }
        }

        img_num += THREADS_PER_WARP;
    
        }

        // printf("lane = %d, min_dst= %f, min_idx = %d\n",lane,min_dst,min_idx);

        for(int offset = 16; offset > 0; offset /= 2) {
            float temp1 = min_dst;
            float temp2 = __shfl_down_sync(FULL_WARP_MASK, min_dst, offset);
            int temp3 = __shfl_down_sync(FULL_WARP_MASK, min_idx, offset);
            min_dst = min(temp1,temp2);

            // printf("lane = %d, my_idx = %d, offset_idx = %d\n", lane,min_idx,temp3 );
            
            if(min_dst<temp1) min_idx = temp3;
        }

        int prediction = __shfl_sync(mask, min_idx, 0);

        int *prediction_ptr = predictions + idx_x;

        *prediction_ptr = prediction % num_train_per_person;
}




}


void prediction(int num_components,int num_train_per_person,int num_train_images, int num_test_images)
{
    float *train_projections_h, *train_projections, *test_projections_h, *test_projections;

    int *predictions_h, *predictions;


    train_projections_h = (float*)malloc(sizeof(float) * (num_components * num_train_images));
    test_projections_h = (float*)malloc(sizeof(float) * (num_components * num_test_images));
    predictions_h = (int*)malloc(sizeof(int) *  num_test_images);

    srand (time(NULL));

    // printf("Train vectors\n");
    // for(int i=0;i<num_components * num_train_images;i++)
    // {
    //     train_projections_h[i]=rand()%10;
    //     printf("%f\n",train_projections_h[i] );
    // }

    // printf("Test vectors\n");

    // for(int i=0;i<num_components * num_test_images;i++)
    // {
    //     test_projections_h[i]=rand()%10;
    //     printf("%f\n",test_projections_h[i] );
    // }

    
    cudaMalloc((void **) &train_projections, sizeof(float) * (num_components * num_train_images));

    cudaMalloc((void **) &test_projections, sizeof(float) * (num_components * num_test_images));

    cudaMalloc((void **) &predictions, sizeof(int) * (num_test_images));

    cudaMemcpy(train_projections,train_projections_h,sizeof(float) * (num_components * num_train_images),cudaMemcpyHostToDevice);
    cudaMemcpy(test_projections,test_projections_h,sizeof(float) * (num_components * num_test_images),cudaMemcpyHostToDevice);

    //each warp processes a test image
    uint32_t blocks_x = (num_test_images + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    dim3 blockDim(THREADS_PER_BLOCK,1) ;
    dim3 gridDim(blocks_x,1);

    nearest_vector<<<gridDim,blockDim>>>(num_test_images,num_train_images,num_components, num_train_per_person ,train_projections,test_projections,predictions);
    cudaDeviceSynchronize();

    cudaMemcpy(predictions_h,predictions,sizeof(int)*num_test_images,cudaMemcpyDeviceToHost);

    // printf("predictions\n");

    // for(int i=0;i<num_test_images;i++)
    // {
    //     printf("%d\n",predictions_h[i] );
    // }






}



int main(int argc,char *argv[])
{

int num_components = std::stoi(argv[1]) ;
int num_train_per_person = std::stoi(argv[2]) ;
int num_train_images = std::stoi(argv[3]);
int num_test_images = std::stoi(argv[4]);
prediction(num_components,num_train_per_person,num_train_images,num_test_images);


return 0;



}





