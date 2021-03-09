#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"
#include "saxpy.h"
#include "common_cuda.h"

__global__ void
saxpy_kernel(int N, float alpha, float* x, float* y, float* result) {

    // compute overall index from position of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N)
       result[index] = alpha * x[index] + y[index];
}

static inline
int getBlocks(long working_set_size, int threadsPerBlock) {
  // TODO: implement and use this interface if necessary  
}

void 
getArrays(int size, float **xarray, float **yarray, float **resultarray) {
  // TODO: implement and use this interface if necessary  
    *xarray      = (float *) CHECK_HOST_MALLOC(malloc(size * sizeof(float)));
    *yarray      = (float *) CHECK_HOST_MALLOC(malloc(size * sizeof(float)));
    *resultarray = (float *) CHECK_HOST_MALLOC(malloc(size * sizeof(float)));
}

void 
freeArrays(float *xarray, float *yarray, float *resultarray) {
  // TODO: implement and use this interface if necessary  
    free(xarray);
    free(yarray);
    free(resultarray);
}

void
saxpyCuda(long total_elems, float alpha, float* xarray, float* yarray, float* resultarray, int partitions) {

    const int threadsPerBlock = 512; // change this if necessary

    float *device_x;
    float *device_y;
    float *device_result;
    cudaStream_t m_streams[partitions];

    //
    // TODO: allocate device memory buffers on the GPU using
    // cudaMalloc.  The started code issues warnings on build because
    // these buffers are used in the call to saxpy_kernel below
    // without being initialized.
    //
    for (int i = 0; i < partitions; i += 1)
        cudaStreamCreate(&m_streams[i]);

    CHECK_CUDA_ERROR(cudaMalloc((void **) &device_x, total_elems * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &device_y, total_elems * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &device_result, total_elems * sizeof(float)));

    // start timing after allocation of device memory.
    double startTime = CycleTimer::currentSeconds();

    // compute number of blocks and threads per block
    // Assumption: partitions divide total_elems perfectly
    uint64_t part_elems = total_elems / partitions;
    uint32_t num_blocks = (part_elems + (threadsPerBlock - 1)) / threadsPerBlock;

    // Note this was moved from inside the for loop; does not make sense to be inside the loop

    // 
    // TODO: insert time here to begin timing only the kernel
    //
    double startGPUTime = CycleTimer::currentSeconds();

    for (int i=0; i<partitions; i++) {
        uint64_t offset = part_elems * i;
        //
        // TODO: copy input arrays to the GPU using cudaMemcpy
        //
        CHECK_CUDA_ERROR(cudaMemcpyAsync(device_x + offset, xarray + offset,
                         part_elems * sizeof(float), cudaMemcpyHostToDevice, m_streams[i]));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(device_y + offset, yarray + offset,
                         part_elems* sizeof(float), cudaMemcpyHostToDevice, m_streams[i]));
         
        // run saxpy_kernel on the GPU
        saxpy_kernel <<<num_blocks, threadsPerBlock, 0, m_streams[i]>>>
            (part_elems, alpha, device_x + offset, device_y + offset, device_result + offset);
    }
    
    for(int i = 0; i < partitions; i += 1) {
        // move it out to get more concurrency
        uint64_t offset = part_elems * i;
        //
        // TODO: copy result from GPU using cudaMemcpy
        //
        CHECK_CUDA_ERROR(cudaMemcpyAsync(resultarray + offset, device_result + offset,
                         part_elems * sizeof(float), cudaMemcpyDeviceToHost, m_streams[i]));
    }

    // NOTE -- This was moved from inside the for loop; it does not make sense to put inside the loop
    // since adding cudaDeviceSynchronize inside foor loop will kill concurrency

    // TODO: insert timer here to time only the kernel.  Since the
    // kernel will run asynchronously with the calling CPU thread, you
    // need to call cudaDeviceSynchronize() before your timer to
    // ensure the kernel running on the GPU has completed.  (Otherwise
    // you will incorrectly observe that almost no time elapses!)
    //
    cudaDeviceSynchronize();

    double endGPUTime = CycleTimer::currentSeconds();
    double timeKernel = endGPUTime - startGPUTime;

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    // end timing after result has been copied back into host memory.
    // The time elapsed between startTime and endTime is the total
    // time to copy data to the GPU, run the kernel, and copy the
    // result back to the CPU
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;

    totalTimeAvg   += overallDuration;
    timeKernelAvg  += timeKernel;

    //
    // TODO free memory buffers on the GPU
    //
    for (int i = 0; i < partitions; i += 1)
        cudaStreamDestroy(m_streams[i]);

    CHECK_CUDA_ERROR(cudaFree(device_x));
    CHECK_CUDA_ERROR(cudaFree(device_y));
    CHECK_CUDA_ERROR(cudaFree(device_result));
}

void
printCudaInfo() {

    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
