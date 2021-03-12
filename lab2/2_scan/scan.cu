#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define MAX_WARP_LEVEL 5
#define PRINT_SIZE 1024
#define THREADS_PER_BLOCK 256
#define NUM_LEVELS 10
// #define DEBUG_PRINTS

extern float toBW(int bytes, float sec);


/* Helper function to round up to a power of 2. 
 */
static inline int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__global__ void
exclusive_scan_upsweep(int* device_in, int N, int length, int* device_result, int multiplier) {
    // basic implementation for quick check -- not getting any points with this
    // multiplier defines what level of log rolling we're at...
    uint32_t index = (2 * multiplier) * (blockIdx.x * blockDim.x + threadIdx.x);

    if(index < length) {
        for(int twod = multiplier; twod < N; twod *= 2) {
            int twod1 = twod * 2;

            if((index % twod1) == 0) {  // Only the leader of each group execute
                device_result[index + twod1 - 1] += device_result[index + twod - 1];
            }

            __syncthreads();  // sync across blocks
        }
     }

     if((length == 2 * N) && index == 0) {  // last run, clear output[N-1] = 0;
        device_result[(2 * N) - 1] = 0;  // account for when length is not a multiple of N
        device_result[length - 1] = 0;
     }
}

__device__ inline uint32_t
fast_mod(const uint32_t input, const uint32_t ceil) {
    return input >= ceil ? input % ceil : input;
}

__global__ void
exclusive_scan_upsweep_opt(int* device_in, int N, int length, int* device_result, int multiplier) {
    // basic implementation for quick check -- not getting any points with this
    // multiplier defines what level of log rolling we're at...
    uint32_t curr_storage;
    uint32_t index = (multiplier << 1) * (blockIdx.x * blockDim.x + threadIdx.x);

    // assuming blockId * blockDim are > 32
    uint32_t lane = threadIdx.x & 0x1f;
    uint32_t k = 0;

    if(index < length) {
        for(int twod = multiplier; twod < N; twod <<= 1) {
            int twod1 = twod << 1;
            if(k == 0) {
                curr_storage = device_result[index + twod1 - 1] + device_result[index + twod - 1];

                for(; k < MAX_WARP_LEVEL && twod < N; k += 1) {
                    if((lane % (2 << k)) == 0) {  // need to store
                        device_result[index + twod1 - 1] = curr_storage;
                    }

                    curr_storage += __shfl_down_sync(0xffffffff, curr_storage, 1 << k);  // sync across warps

                    twod <<= 1;
                    twod1 <<= 1;
                }
                if(lane == 0) {
                    device_result[index + twod1 - 1] = curr_storage;
                }
            }
            else {
                if((index % twod1) == 0) {  // Only the leader of each group execute
                    curr_storage += device_result[index + twod1 - 1];
                    device_result[index + twod1 - 1] = curr_storage;
                }
            }

            __syncthreads();  // sync across blocks
        }
     }

     if((length == 2 * N) && index == 0) {  // last run, clear output[N-1] = 0;
        device_result[(2 * N) - 1] = 0;  // account for when length is not a multiple of N
        device_result[length - 1] = 0;
     }
}


__global__ void
exclusive_scan_downsweep(int* device_in, int N, int length, int* device_result, int multiplier) {
    // basic implementation for quick check -- not getting any points with this
    // multiplier defines what level of log rolling we're at...
    uint32_t index = (2 * multiplier) * (blockIdx.x * blockDim.x + threadIdx.x);

    // max_level to go to from N down to max_level... needs this gross stuff cuz multiplier level itself should not
    // be executed by the higher levels, as it will be covered by the lower levels
    uint32_t max_level = multiplier > 1 ? (multiplier * 2) : 1;

    if(index < length) {
        for(int twod = N; twod >= max_level; twod /= 2) {  // Note N is already / 2
            // Note max depth for each execution is log_2 threads_per_block
            int twod1 = twod * 2;

            if((index % twod1) == 0) {  // Only the leader of each group execute
                int tmp = device_result[index + twod - 1];
                device_result[index + twod - 1] = device_result[index + twod1 - 1];
                device_result[index + twod1 - 1] += tmp;
            }

            __syncthreads();  // sync across blocks
        }
     }
}

__global__ void
exclusive_scan_downsweep_opt(int* device_in, int N, int length, int* device_result, int multiplier) {
    // basic implementation for quick check -- not getting any points with this
    // multiplier defines what level of log rolling we're at...
    uint32_t index = (2 * multiplier) * (blockIdx.x * blockDim.x + threadIdx.x);

    // max_level to go to from N down to max_level... needs this gross stuff cuz multiplier level itself should not
    // be executed by the higher levels, as it will be covered by the lower levels
    uint32_t max_level = multiplier > 1 ? (multiplier * 2) : 1;
    uint32_t curr_storage = 0xffffffff;

    if(index < length) {
        for(int twod = N; twod >= max_level; twod /= 2) {  // Note N is already / 2
            // Note max depth for each execution is log_2 threads_per_block
            int twod1 = twod * 2;

            if((index % twod1) == 0) {  // Only the leader of each group execute
                if(curr_storage == 0xffffffff)
                    curr_storage = device_result[index + twod1 - 1];

                device_result[index + twod1 - 1] = curr_storage + device_result[index + twod - 1];

                if(twod == max_level) {  // last time to execute... store back
                    device_result[index + twod - 1] = curr_storage;
                }
            }

            __syncthreads();  // sync across blocks
        }
     }
}

void exclusive_scan_up(int* device_start, int length, int* device_result)
{
    /* Fill in this function with your exclusive scan implementation.
     * You are passed the locations of the input and output in device memory,
     * but this is host code -- you will need to declare one or more CUDA 
     * kernels (with the __global__ decorator) in order to actually run code
     * in parallel on the GPU.
     * Note you are given the real length of the array, but may assume that
     * both the input and the output arrays are sized to accommodate the next
     * power of 2 larger than the input.
     */

    // Not sure if this effort is necessary... but here it will try to use block level synchronization as much as
    // possible while only synchronizing the whole kernel few times in powers of threads_per_block
    // This is under the inherent assumption that device wide synchronization is pitifully slow compared to block-level
    // synchronization, so this below code will optimize it by only needing device wide synchronization ~1-3x 
    // per up/down sweeps => 2-6x total

    uint32_t threads_per_block = THREADS_PER_BLOCK; // How many do we want??
    uint32_t next_pow2 = nextPow2(length);  // Next pow2 is setup for us :)

    // Working size is half of target length, since the first execution is always N / 2 elements
    uint32_t working_size = next_pow2 >> 1;
    uint32_t num_blocks = (working_size + (threads_per_block - 1)) / threads_per_block;


    // Upsweep
    uint32_t multiplier = 1;
    uint32_t current_working_size = (threads_per_block * multiplier) > working_size ? working_size : (threads_per_block * multiplier);

    // This will break off the initial working set into working set / threads_per_block sizes and work through these
    // blocks in parallel
    // Then, proceed to combine the working set
    // Note that the intial distribution may find number of blocks that would need additional stages to combine
    // I.e. 256^3 => 256^2 => 256 => 1
    // printf("length %d working_size %d num_blocks %d\n", next_pow2, working_size, num_blocks);

    for(int i = num_blocks; ; i = (i + (threads_per_block - 1)) / threads_per_block) {
        // printf("Executing Upsweep %d N=%d Mult=%d...\n", i, current_working_size, multiplier);
        exclusive_scan_upsweep_opt<<<i, threads_per_block>>>
            (device_start, current_working_size, next_pow2, device_result, multiplier);

        multiplier *= threads_per_block;  // minor optimization to put this before cudaDeviceSync :P
        current_working_size = (threads_per_block * multiplier) > working_size ? working_size : (threads_per_block * multiplier);

        cudaDeviceSynchronize();

        if(i == 1) break;  // kinda dumb but no other way around it...
    }
}

void exclusive_scan(int* device_start, int length, int* device_result)
{
    /* Fill in this function with your exclusive scan implementation.
     * You are passed the locations of the input and output in device memory,
     * but this is host code -- you will need to declare one or more CUDA 
     * kernels (with the __global__ decorator) in order to actually run code
     * in parallel on the GPU.
     * Note you are given the real length of the array, but may assume that
     * both the input and the output arrays are sized to accommodate the next
     * power of 2 larger than the input.
     */

    // Not sure if this effort is necessary... but here it will try to use block level synchronization as much as
    // possible while only synchronizing the whole kernel few times in powers of threads_per_block
    // This is under the inherent assumption that device wide synchronization is pitifully slow compared to block-level
    // synchronization, so this below code will optimize it by only needing device wide synchronization ~1-3x 
    // per up/down sweeps => 2-6x total

    uint32_t threads_per_block = THREADS_PER_BLOCK; // How many do we want??
    uint32_t next_pow2 = nextPow2(length);  // Next pow2 is setup for us :)

    // Working size is half of target length, since the first execution is always N / 2 elements
    uint32_t working_size = next_pow2 >> 1;
    uint32_t num_blocks = (working_size + (threads_per_block - 1)) / threads_per_block;


    // Upsweep
    uint32_t multiplier = 1;
    uint32_t current_working_size = (threads_per_block * multiplier) > working_size ? working_size : (threads_per_block * multiplier);

    // This will break off the initial working set into working set / threads_per_block sizes and work through these
    // blocks in parallel
    // Then, proceed to combine the working set
    // Note that the intial distribution may find number of blocks that would need additional stages to combine
    // I.e. 256^3 => 256^2 => 256 => 1
    // printf("length %d working_size %d num_blocks %d\n", next_pow2, working_size, num_blocks);

    for(int i = num_blocks; ; i = (i + (threads_per_block - 1)) / threads_per_block) {
        // printf("Executing Upsweep %d N=%d Mult=%d...\n", i, current_working_size, multiplier);
        exclusive_scan_upsweep_opt<<<i, threads_per_block>>>
            (device_start, current_working_size, next_pow2, device_result, multiplier);
        multiplier *= threads_per_block;  // minor optimization to put this before cudaDeviceSync :P
        current_working_size = (threads_per_block * multiplier) > working_size ? working_size : (threads_per_block * multiplier);

        cudaDeviceSynchronize();

        if(i == 1) break;  // kinda dumb but no other way around it...
    }

    // Final stage of upsweep will perform clear on N-1, so need to explicitly do it

    // Downsweep
    // Continue multiplier from the level above
    // Current working size goes from same as above 1 => 256 => 256^2 => 256^3
    // Multiplier however goes the opposite way, for downsweep reasons
    multiplier /= threads_per_block;  // Needs to be on top because the for loop above will do 1 excess step of multiplier
    current_working_size = (threads_per_block * multiplier) > working_size ? working_size : (threads_per_block * multiplier);

    for(int i = 1; ; i *= threads_per_block) {
        if(i > num_blocks) i = num_blocks;
        // printf("Executing Downsweep %d N=%d Mult=%d...\n", i, current_working_size, multiplier);
        exclusive_scan_downsweep_opt<<<i, threads_per_block>>>
            (device_start, current_working_size, next_pow2, device_result, multiplier);

        multiplier /= threads_per_block;  // Needs to be on top because the for loop above will do 1 excess step of multiplier
        current_working_size = (threads_per_block * multiplier) > working_size ? working_size : (threads_per_block * multiplier);

        cudaDeviceSynchronize();

        if(i == num_blocks) break;  // Is there a way to make this look better lol
    }
}

/* This function is a wrapper around the code you will write - it copies the
 * input to the GPU and times the invocation of the exclusive_scan() function
 * above. You should not modify it.
 */
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input; 

#ifdef DEBUG_PRINTS
    printf("<<>>\n");
    for(int i = 1; i <= PRINT_SIZE; i += 1) {
        printf("%08d ", inarray[(i - 1)]);
        if(i % 8 == 0)
            printf("\n");
    }
    printf("\n");
#endif

    // We round the array sizes up to a power of 2, but elements after
    // the end of the original input are left uninitialized and not checked
    // for correctness. 
    // You may have an easier time in your implementation if you assume the 
    // array's length is a power of 2, but this will result in extra work on
    // non-power-of-2 inputs.
    int rounded_length = nextPow2(end - inarray);
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), 
               cudaMemcpyHostToDevice);

    // For convenience, both the input and output vectors on the device are
    // initialized to the input values. This means that you are free to simply
    // implement an in-place scan on the result vector if you wish.
    // If you do this, you will need to keep that fact in mind when calling
    // exclusive_scan from find_repeats.
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

#ifdef DEBUG_PRINTS
    exclusive_scan_up(device_input, end - inarray, device_result);

    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 1; i <= PRINT_SIZE; i += 1) {
        printf("%08d ", resultarray[(i - 1)]);
        if(i % 8 == 0)
            printf("\n");
    }
    printf("\n");
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), 
               cudaMemcpyHostToDevice);
#endif

    exclusive_scan(device_input, end - inarray, device_result);

    // Wait for any work left over to be completed.
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

#ifdef DEBUG_PRINTS
    for(int i = 1; i <= PRINT_SIZE; i += 1) {
        printf("%08d ", resultarray[(i - 1)]);
        if(i % 8 == 0)
            printf("\n");
    }
    printf("\n");
#endif
    return overallDuration;
}


/* Wrapper around the Thrust library's exclusive scan function
 * As above, copies the input onto the GPU and times only the execution
 * of the scan itself.
 * You are not expected to produce competitive performance to the
 * Thrust version.
 */
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    // Wait for any work left over to be completed.
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int),
               cudaMemcpyDeviceToHost);
    thrust::device_free(d_input);
    thrust::device_free(d_output);
    double overallDuration = endTime - startTime;
    return overallDuration;
}

__global__ void
find_repeats_stage1(int* device_in, int N, int* device_result) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < N - 1) {
        // write a 1 if equal, else 0 if not equal
        device_result[index] = device_in[index + 1] == device_in[index];
    } else {
        device_result[index] = 0;
    }
}

__global__ void
find_repeats_stage4(int* device_in, int N, int* device_result) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < N - 1) {
        if(device_in[index] != device_in[index + 1]) {
            device_result[device_in[index]] = index;
        }
    }
}

__global__ void
find_repeats_stage5(int* device_in, int N, int* device_result) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

    device_result[index] = device_in[index];
}

int find_repeats(int *device_input, int length, int *device_output) {
    /* Finds all pairs of adjacent repeated elements in the list, storing the
     * indices of the first element of each pair (in order) into device_result.
     * Returns the number of pairs found.
     * Your task is to implement this function. You will probably want to
     * make use of one or more calls to exclusive_scan(), as well as
     * additional CUDA kernel launches.
     * Note: As in the scan code, we ensure that allocated arrays are a power
     * of 2 in size, so you can use your exclusive_scan function with them if 
     * it requires that. However, you must ensure that the results of
     * find_repeats are correct given the original length.
     */    
    uint32_t threads_per_block = THREADS_PER_BLOCK; // How many do we want??
    uint32_t next_pow2 = nextPow2(length);  // Next pow2 is setup for us :)
    int num_repeats[1];
    // int *temp_storage;
    // cudaMalloc((void **) &temp_storage, next_pow2 * sizeof(int));

    // Stage 1:
    // Set value of the output to 1 if adjacent are equal
    uint32_t num_blocks = (next_pow2 + (threads_per_block - 1)) / threads_per_block;
    // printf("num_blocks = %d; next_power2 = %d; length = %d\n", num_blocks, next_pow2, length);
    find_repeats_stage1<<<num_blocks, threads_per_block>>>
        (device_input, length, device_output);
    cudaDeviceSynchronize();
    // printf("Finish stage1\n");

    // Stage 2:
    // Compute indexes via exclusive scan
    exclusive_scan(device_input, length, device_output);
    //printf("Finish stage2\n");

    // Stage 3:
    // Copy the last index into results
    cudaMemcpy(num_repeats, &device_output[length - 1], 1 * sizeof(int), cudaMemcpyDeviceToHost);  // last index will have the total # of repeats

    // Stage 4:
    // Properly setup the output
    find_repeats_stage4<<<num_blocks, threads_per_block>>>
        (device_output, length, device_input);
    cudaDeviceSynchronize();
    // printf("Finish stage4\n");

    find_repeats_stage5<<<num_blocks, threads_per_block>>>
        (device_input, length, device_output);

    // free(temp_storage);

    return num_repeats[0];
}

/* Timing wrapper around find_repeats. You should not modify this function.
 */
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {
    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    /*
    printf("<<>>\n");
    for(int i = 1; i <= 32; i += 1) {
        printf("%08d ", input[(i - 1)]);
    }
    printf("\n");
    */

    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    // Wait for any work left over to be completed.
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    *output_length = result;

    cudaMemcpy(output, device_output, length * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    /*
    for(int i = 1; i <= 32; i += 1) {
        printf("%08d ", output[(i - 1)]);
    }
    printf("\n");
    */

    return endTime - startTime;
}

void printCudaInfo()
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
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
