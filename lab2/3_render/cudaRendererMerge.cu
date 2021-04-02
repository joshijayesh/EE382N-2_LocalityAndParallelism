#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

#define THREADS_PER_WARP 32
#define TPW_LOG 5
#define THREADS_PER_WARP_MASK 0xFFFFFFFF
#define TPW_MASK4 0x55555555
#define TPW_MASK3 0x11111111
#define TPW_MASK2 0x01010101
#define TPW_MASK1 0x00010001

#define THREADS_PER_TB 256
#define WARPS_PER_TB 8
#define WARPS_PER_TB_MASK 0xFF
#define WPTB_MASK2 0x55
#define WPTB_MASK1 0x11

/* 4x4 config better performance in snow single but worse in large # of circles */
/*
#define SIDE_PER_THREAD 4
#define SIDE_WARP_X 32
#define SIDE_WARP_Y 16
#define BLOCK_PER_THREAD 16
// log(1024 / 32) = log(32) = 5
#define SIDE_WARP_X_PER_IMG_LOG 5
// log(64/32) = log(2) = 1
#define SIDE_WARP_X_PER_TB_LOG 1
#define SIDE_THREAD_X_PER_IMG_LOG 7
// log(1024/64) = log(16) = 4
#define SIDE_TB_X_PER_IMG_LOG 4
// log(64/4) = log(16) = 4
#define SIDE_THREAD_X_PER_TB_LOG 4

#define SIDE_TB_X 64 // TB does SIDE_WARP_X * (WARPS_PER_TB) x SIDE_WARP_Y
#define SIDE_TB_Y 64
*/

/* 2x2 config  -- better performance in large # of circles, but worse in snowsingle
*/
#define SIDE_PER_THREAD 2
#define SIDE_WARP_X 32
#define SIDE_WARP_Y 4
#define BLOCK_PER_THREAD 4
// log(1024 / 32) = log(32) = 5
#define SIDE_WARP_X_PER_IMG_LOG 5
// log(32/32) = log(1) = 0
#define SIDE_WARP_X_PER_TB_LOG 0
// log(1024/32) = log(32) = 5
#define SIDE_TB_X_PER_IMG_LOG 5
// log(64/4) = log(16) = 4

#define SIDE_TB_X 32 // TB does SIDE_WARP_X * (WARPS_PER_TB) x SIDE_WARP_Y
#define SIDE_TB_Y 32


#define CIRCLE_PER_THREAD 32 // This needs to be massaged , 32 seems to be the min pow of 2 we can use
#define CIRCLE_PER_TB 4096 // This needs to be massaged

/*
#define SIDE_PER_THREAD 16
#define SIDE_WARP_X 128
#define SIDE_WARP_Y 64
#define BLOCK_PER_THREAD 256
*/
#define noOfCirclePerThread 8192

////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
    float* circleimageData;

};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];
__constant__ int noOfCircleThreads;

int noOfCircleThreads_temp;

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// including parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}


// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}


/// Clear new space for parallel circles thread temp area
__global__ void kernelClearCircleImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    if (i >= noOfCircleThreads)
        return;

    int offset = 4 * (imageY * width + imageX);
    // float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[offset]);
    // float4 Color = *imgPtr;
    // float4 value0 = make_float4(Color.x, Color.y, Color.z, Color.w);
    float4 value1 = make_float4(r, g, b, a);
    float4 value = (i ==0) ? *(float4*)(&cuConstRendererParams.imageData[offset]) : value1;

    *(float4*)(&cuConstRendererParams.circleimageData[4*i*width*height +offset]) = value; 
    
}
// kernelAdvanceFireWorks
//
// Update the position of the fireworks (if circle is firework)
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = 3.14159;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update
        return;
    }

    // determine the fire-work center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // fire-work sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // compute distance from fire-work
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position
        // random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // travel scaled unit length
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    float* radius = cuConstRendererParams.radius;

    float cutOff = 0.5f;
    // place circle back in center after reaching threshold radisus
    if (radius[index] > cutOff) {
        radius[index] = 0.02f;
    } else {
        radius[index] += 0.01f;
    }
}


// kernelAdvanceBouncingBalls
//
// Update the positino of the balls
__global__ void kernelAdvanceBouncingBalls() {
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// move the snowflake animation forward one time step.  Updates circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // loads from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // if the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}



// shadePixelSmall -- (CUDA device code)
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ float4
shadePixelSmall(int circleIndex, float2 pixelCenter, float3 p, float rad, float4 existing) {
    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return existing;

    float3 rgb;
    float alpha;

    // there is a non-zero contribution.  Now compute the shading value

    // This conditional is in the inner loop, but it evaluates the
    // same direction for all threads so it's cost is not so
    // bad. Attempting to hoist this conditional is not a required
    // student optimization in Assignment 2
    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f-p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
        // simple: each circle has an assigned color
        int index3 = 3 * circleIndex;
        rgb = *(float3*)&(cuConstRendererParams.color[index3]);
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existing.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existing.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existing.z;
    newColor.w = oneMinusAlpha*existing.w;

    // write done a level up
    return newColor;
}


#include "circleBoxTest.cu_inl"
// #define DEBUG_PRINT_WU

// First level of Exclusive scan upsweep, only need to sync warps
// Disgusting looking but fast implementation
// This would be better if we could make it into an iterator somehow...
// I unrolled it cuz I don't wanna see that mask stuff being a memory look up
// There must be a way!
__device__ __inline__
void exclusiveScanUpsweepWarp(int &warpItem, int &tempItem, int &lane, int &wid) {
    tempItem = warpItem;

    // offset = 1
    warpItem += __shfl_down_sync(THREADS_PER_WARP, warpItem, 1);
    tempItem = __shfl_up_sync(WARPS_PER_TB_MASK, warpItem, 1);

    // offset = 2
    warpItem += __shfl_down_sync(THREADS_PER_WARP, warpItem, 2);
    tempItem = lane % 2 != 0 ? tempItem : __shfl_up_sync(TPW_MASK4, warpItem, 2);

    // offset = 4
    warpItem += __shfl_down_sync(THREADS_PER_WARP, warpItem, 4);
    tempItem = lane % 4 != 0 ? tempItem : __shfl_up_sync(TPW_MASK3, warpItem, 4);
    
    // offset = 8
    warpItem += __shfl_down_sync(THREADS_PER_WARP, warpItem, 8);
    tempItem = lane % 8 != 0 ? tempItem : __shfl_up_sync(TPW_MASK2, warpItem, 8);
    
    // offset = 16
    warpItem += __shfl_down_sync(THREADS_PER_WARP, warpItem, 16);
    tempItem = lane % 16 != 0 ? tempItem : __shfl_up_sync(TPW_MASK1, warpItem, 16);
}

__device__ __inline__
void exclusiveScanUpsweepTB(int &warpItem, int &tempItem, int &lane) {
    // Example
    // 0 1 2 3 4 5 6 7  (heldItem)
    // warp Item:
    // 1  1  5  3  9  5  13  7
    // 6  1  5  3  22 5  13  7 
    // temp Item:
    // 0  1  2  3  4  5   6  7 // init
    // 1  1  5  5  9  9  13 13
    // 6  1  6  5  22 9  22 13
    tempItem = warpItem;

    // Note:: need to unroll due to mask... is there a better way to do this??
    // This is not scalable if we change num threads per block <<

    // offset = 1
    warpItem += __shfl_down_sync(WARPS_PER_TB_MASK, warpItem, 1);
    tempItem = __shfl_up_sync(WARPS_PER_TB_MASK, warpItem, 1);

    // offset = 2
    warpItem += __shfl_down_sync(WARPS_PER_TB_MASK, warpItem, 2);
    tempItem = lane % 2 == 1 ? tempItem : __shfl_up_sync(WPTB_MASK2, warpItem, 2);

    // offset = 4 unnecessary
}

__device__ __inline__
void exclusiveScanDownsweepTB(int &warpItem, int &heldItem, int &tempItem, int &lane) {
    // Example (init warp Item from previous example)
    // 0  1  5  3  22 5 13  7 
    // Example init tempItem
    // 6  1  6  5  22 9 22 13
    // Example init heldItem
    // 0  1  2  3  4  5  6  7
    // warp Item:
    // 0  1  5  3  6  5 13  7 
    // 0  1  1  3  6  5 15  7
    // 0  0  1  3  6 10 15 21
    // temp Item:
    // 6  1  6  5 44  9 22 13
    // 1  1 10  5 15  9 26 13
    int update;

    // offset = 4
    tempItem = lane % 2 == 1 ? tempItem : __shfl_down_sync(WPTB_MASK2, tempItem, 2) + warpItem;
    update = __shfl_up_sync(WARPS_PER_TB_MASK, tempItem, 4);
    if(lane % 8 != 0) warpItem = update;  // wtf why does up_sync update me?!

    // offset = 2
    tempItem = __shfl_down_sync(WARPS_PER_TB_MASK, tempItem, 1) + warpItem;
    update = __shfl_up_sync(WARPS_PER_TB_MASK, tempItem, 2);
    if(lane % 4 != 0) warpItem = update;  // wtf why does up_sync update me?!

    // offset = 1
    update = __shfl_up_sync(WARPS_PER_TB_MASK, heldItem + warpItem, 1);
    if(lane % 2 != 0) warpItem = update;  // wtf why does up_sync update me?!
}

__device__ __inline__
void exclusiveScanDownsweepWarp(int &warpItem, int &heldItem, int &tempItem, int &lane, int &wid) {
    int update;

    // offset = 16
    tempItem = lane % 8 != 0 ? tempItem : __shfl_down_sync(TPW_MASK2, tempItem, 8) + warpItem;
    update = __shfl_up_sync(THREADS_PER_WARP, tempItem, 16);
    if(lane % 32 != 0) warpItem = update;  // wtf why does up_sync update me?!

    // offset = 8
    tempItem = lane % 4 != 0 ? tempItem : __shfl_down_sync(TPW_MASK3, tempItem, 4) + warpItem;
    update = __shfl_up_sync(THREADS_PER_WARP, tempItem, 8);
    if(lane % 16 != 0) warpItem = update;  // wtf why does up_sync update me?!

    // offset = 4
    tempItem = lane % 2 == 1 ? tempItem : __shfl_down_sync(TPW_MASK4, tempItem, 2) + warpItem;
    update = __shfl_up_sync(THREADS_PER_WARP, tempItem, 4);
    if(lane % 8 != 0) warpItem = update;  // wtf why does up_sync update me?!

    // offset = 2
    tempItem = __shfl_down_sync(THREADS_PER_WARP, tempItem, 1) + warpItem;
    update = __shfl_up_sync(THREADS_PER_WARP, tempItem, 2);
    if(lane % 4 != 0) warpItem = update;  // wtf why does up_sync update me?!

    // offset = 1
    update = __shfl_up_sync(THREADS_PER_WARP, heldItem + warpItem, 1);
    if(lane % 2 != 0) warpItem = update;  // wtf why does up_sync update me?!
}

__device__ __inline__
void exclusiveScan(int &circleIndex, int wid, int lane) {
    __shared__ int scratch[WARPS_PER_TB];
    int circleIndex2;
    int heldIndex1, heldIndex2;
    int tempIndex1, tempIndex2;
    // do my upsweep across all warps; T0 will contain final result for each warp
    heldIndex1 = circleIndex; // original index
    exclusiveScanUpsweepWarp(circleIndex, tempIndex1, lane, wid);

    if(lane == 0)
        scratch[wid] = circleIndex;
    __syncthreads();


    // first warp will handle cross TB syncs => more efficient than constantly going through shmem
    if(wid == 0 && lane < WARPS_PER_TB) { 
        circleIndex2 = scratch[lane];
        heldIndex2 = circleIndex2;

        exclusiveScanUpsweepTB(circleIndex2, tempIndex2, lane);

        if(lane == 0) circleIndex2 = 0;
        
        exclusiveScanDownsweepTB(circleIndex2, heldIndex2, tempIndex2, lane);

        scratch[lane] = circleIndex2;
    }

    __syncthreads();

    // do downscweep across all warps
    if(lane == 0)
        circleIndex = scratch[wid];

    exclusiveScanDownsweepWarp(circleIndex, heldIndex1, tempIndex1, lane, wid);
}

__device__ __inline__
void exclusiveScanUpsweepWarpOnly(int &warpItem, int &tempItem, int &lane) {
    tempItem = warpItem;

    // offset = 1
    warpItem += __shfl_down_sync(THREADS_PER_WARP, warpItem, 1);
    tempItem = __shfl_up_sync(WARPS_PER_TB_MASK, warpItem, 1);

    // offset = 2
    warpItem += __shfl_down_sync(THREADS_PER_WARP, warpItem, 2);
    tempItem = lane % 2 != 0 ? tempItem : __shfl_up_sync(TPW_MASK4, warpItem, 2);

    // offset = 4
    warpItem += __shfl_down_sync(THREADS_PER_WARP, warpItem, 4);
    tempItem = lane % 4 != 0 ? tempItem : __shfl_up_sync(TPW_MASK3, warpItem, 4);

    // offset = 8
    warpItem += __shfl_down_sync(THREADS_PER_WARP, warpItem, 8);
    tempItem = lane % 8 != 0 ? tempItem : __shfl_up_sync(TPW_MASK2, warpItem, 8);
}

__device__ __inline__
void exclusiveScanDownsweepWarpOnly(int &warpItem, int &heldItem, int &tempItem, int &lane) {
    int update;

    // offset = 16
    tempItem = lane % 8 != 0 ? tempItem : __shfl_down_sync(TPW_MASK2, tempItem, 8) + warpItem;
    update = __shfl_up_sync(THREADS_PER_WARP, tempItem, 16);
    if(lane % 32 != 0) warpItem = update;  // wtf why does up_sync update me?!

    // offset = 8
    tempItem = lane % 4 != 0 ? tempItem : __shfl_down_sync(TPW_MASK3, tempItem, 4) + warpItem;
    update = __shfl_up_sync(THREADS_PER_WARP, tempItem, 8);
    if(lane % 16 != 0) warpItem = update;  // wtf why does up_sync update me?!

    // offset = 4
    tempItem = lane % 2 == 1 ? tempItem : __shfl_down_sync(TPW_MASK4, tempItem, 2) + warpItem;
    update = __shfl_up_sync(THREADS_PER_WARP, tempItem, 4);
    if(lane % 8 != 0) warpItem = update;  // wtf why does up_sync update me?!

    // offset = 2
    tempItem = __shfl_down_sync(THREADS_PER_WARP, tempItem, 1) + warpItem;
    update = __shfl_up_sync(THREADS_PER_WARP, tempItem, 2);
    if(lane % 4 != 0) warpItem = update;  // wtf why does up_sync update me?!

    // offset = 1
    update = __shfl_up_sync(THREADS_PER_WARP, heldItem + warpItem, 1);
    if(lane % 2 != 0) warpItem = update;  // wtf why does up_sync update me?!
}

__device__ __inline__
void exclusiveScanWarpOnly(int &circleIndex, int lane) {
    int heldIndex1;
    int tempIndex1;
    // do my upsweep across all warps; T0 will contain final result for each warp
    heldIndex1 = circleIndex; // original index
    exclusiveScanUpsweepWarpOnly(circleIndex, tempIndex1, lane);

    if(lane == 0)
        circleIndex = 0;

    exclusiveScanDownsweepWarpOnly(circleIndex, heldIndex1, tempIndex1, lane);
}


// kernelRenderCirclesScan -- (CUDA device code)
//
// Scan reduction method which reduces the number of circles that each thread block needs to consider by using
// eclusive scan and walking through the remaining circles sequentially
__global__ void kernelRenderCirclesScan() {
    float3 p;
    float rad;
    float3 p_o;
    float rad_o;
    float4 existing;
    float4 original;
    __shared__ int active_circles[CIRCLE_PER_TB];
    __shared__ int total_circles;
    int total_warp_circles;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // thread ID
    int wid = tid >> TPW_LOG;  // warp ID
    int lane = tid & (THREADS_PER_WARP - 1);  // lane ID
    // assuming warps per thread is a multiple of 2 (which is true in our case, but may not always be)
    int wid_per_tb = wid & (WARPS_PER_TB - 1);

    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // Find min/max of the whole TB
    short TBMinX = (blockIdx.x * SIDE_TB_X) & (imageWidth - 1);
    short TBMaxX = TBMinX + SIDE_TB_X - 1;
    short TBMinY = ((blockIdx.x >> SIDE_TB_X_PER_IMG_LOG) * SIDE_TB_Y) & (imageWidth - 1);
    short TBMaxY = TBMinY + SIDE_TB_Y - 1;;

    float TBMinXNorm = invWidth * TBMinX;
    float TBMaxXNorm = invWidth * TBMaxX;
    float TBMinYNorm = invHeight * TBMinY;
    float TBMaxYNorm = invHeight * TBMaxY;

    // Find all the circles that are within my TB
    int m_circles_[CIRCLE_PER_THREAD];
    int circle_cnt_ = 0;
    int circle_idx_ = 0;
    int num_circles = cuConstRendererParams.numCircles;
    int circles_per_thread = (num_circles + THREADS_PER_TB- 1) / THREADS_PER_TB;  
    int circles_start = circles_per_thread * threadIdx.x;
    int circles_end = (circles_start + circles_per_thread - 1);
    circles_end = circles_end >= num_circles ? num_circles - 1 : circles_end;

    for(int circle_index = circles_start; circle_index <= circles_end; circle_index += 1) {
        int circle_index_3 = circle_index * 3;
        p = *(float3*)(&cuConstRendererParams.position[circle_index_3]);
        rad = cuConstRendererParams.radius[circle_index];
        
        // conservative finds more circles... but is WAY less computation intensitive
        if(circleInBox(p.x, p.y, rad, TBMinXNorm, TBMaxXNorm, TBMaxYNorm, TBMinYNorm))
            m_circles_[circle_cnt_++] = circle_index;
    }

    // circle_idx_ will contain the index to start from, circle_cnt_ contains how many circles to write
    circle_idx_ = circle_cnt_;
    __syncwarp();
    exclusiveScan(circle_idx_, wid_per_tb, lane);

    if(threadIdx.x == THREADS_PER_TB - 1){  // last thread to update total circles found
        total_circles = circle_idx_ + circle_cnt_;
    }

    if(circle_cnt_ != 0) {
        // Update active circles in shmem w/ all the circles that we need to consider
        // Note that will never be any thrashing here, some threads may not write anything
        int ctr = 0;
        for(int i = circle_idx_; i < circle_idx_ + circle_cnt_; i += 1) {
            active_circles[i] = m_circles_[ctr++];
        }
    }
    __syncthreads(); // all threads done updating shmem

    // well we actually want to move across the image within each warp rather than per thread... more locality
    short warpMinX = TBMinX + ((wid_per_tb * SIDE_WARP_X) & (SIDE_TB_X - 1));
    short warpMaxX = warpMinX + SIDE_WARP_X;

    short warpMinY = TBMinY + (((wid_per_tb >> SIDE_WARP_X_PER_TB_LOG) * SIDE_WARP_Y) & (SIDE_TB_Y - 1));
    short warpMaxY = warpMinY + SIDE_WARP_Y;

    int circleIndex, circleIndex3;

    // for all pixels in our current targets

    for(short pixelY = warpMinY; pixelY < warpMaxY; pixelY += 1) {
        float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + warpMinX + lane)]);

        // note we can take off 1 section of for loop since we have SIDE_WARP_X = 32...
        float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(warpMinX + lane) + 0.5f),
                                             invHeight * (static_cast<float>(pixelY) + 0.5f));

        original = *imgPtr;
        existing = original;

        for(int activeCircleIndex = 0; activeCircleIndex < total_circles; activeCircleIndex += 1) {
            // find circle from active circles
            circleIndex = active_circles[activeCircleIndex];
            circleIndex3 = circleIndex * 3;
            p = *(float3*)(&cuConstRendererParams.position[circleIndex3]);
            float rad = cuConstRendererParams.radius[circleIndex];

            // shadePixel(circleIndex, pixelCenterNorm, p, imgPtr);
            existing = shadePixelSmall(circleIndex, pixelCenterNorm, p, rad, existing);
        }

        // Write back the pixel info
        if(original.w != existing.w) {
            *imgPtr = existing;
        }
    }
}

// kernelRenderParallelCirclesScan -- (CUDA device code)
//
// Updated version of kernelRenderCirclesScan that parallelizes the number of circles that each thread needs to consider
// by number of circles per thread which is then combined via kernelShadeCircles
__global__ void kernelRenderParallelCirclesScan() {
    float3 p;
    float rad;
    float3 p_o;
    float rad_o;
    float4 existing;
    float4 original;
    int index0 = blockIdx.y * blockDim.y + threadIdx.y;         /// thread ID for parallel circles thread

    int cicle_start_id = index0*noOfCirclePerThread;

    if (cicle_start_id >= cuConstRendererParams.numCircles)
        return;

    __shared__ int active_circles[CIRCLE_PER_TB];
    __shared__ int total_circles;
    int total_warp_circles;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // thread ID
    int wid = tid >> TPW_LOG;  // warp ID
    int lane = tid & (THREADS_PER_WARP - 1);  // lane ID
    // assuming warps per thread is a multiple of 2 (which is true in our case, but may not always be)
    int wid_per_tb = wid & (WARPS_PER_TB - 1);

    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // Find min/max of the whole TB
   
    short TBMinX = (blockIdx.x * SIDE_TB_X) & (imageWidth - 1);
    short TBMaxX = TBMinX + SIDE_TB_X - 1;
    short TBMinY = ((blockIdx.x >> SIDE_TB_X_PER_IMG_LOG) * SIDE_TB_Y) & (imageWidth - 1);
    short TBMaxY = TBMinY + SIDE_TB_Y - 1;;

    float TBMinXNorm = invWidth * TBMinX;
    float TBMaxXNorm = invWidth * TBMaxX;
    float TBMinYNorm = invHeight * TBMinY;
    float TBMaxYNorm = invHeight * TBMaxY;

    // Find all the circles that are within my TB
    int m_circles_[CIRCLE_PER_THREAD];
    int circle_cnt_ = 0;
    int circle_idx_ = 0;
    int num_circles = cuConstRendererParams.numCircles;
    int circles_per_thread = (noOfCirclePerThread + THREADS_PER_TB- 1) / THREADS_PER_TB;  
    int circles_start = cicle_start_id + circles_per_thread * threadIdx.x;
    int circles_end = (circles_start + circles_per_thread - 1);
    circles_end = circles_end >= num_circles ? num_circles - 1 : circles_end;

    for(int circle_index = circles_start; circle_index <= circles_end; circle_index += 1) {
        int circle_index_3 = circle_index * 3;
        p = *(float3*)(&cuConstRendererParams.position[circle_index_3]);
        rad = cuConstRendererParams.radius[circle_index];
        
        // conservative finds more circles... but is WAY less computation intensitive
        if(circleInBox(p.x, p.y, rad, TBMinXNorm, TBMaxXNorm, TBMaxYNorm, TBMinYNorm))
            m_circles_[circle_cnt_++] = circle_index;
    }

    // circle_idx_ will contain the index to start from, circle_cnt_ contains how many circles to write
    circle_idx_ = circle_cnt_;
    __syncwarp();
    exclusiveScan(circle_idx_, wid_per_tb, lane);

    if(threadIdx.x == THREADS_PER_TB - 1){  // last thread to update total circles found
        total_circles = circle_idx_ + circle_cnt_;
    }

    if(circle_cnt_ != 0) {
        // Update active circles in shmem w/ all the circles that we need to consider
        // Note that will never be any thrashing here, some threads may not write anything
        int ctr = 0;
        for(int i = circle_idx_; i < circle_idx_ + circle_cnt_; i += 1) {
            active_circles[i] = m_circles_[ctr++];
        }
    }
    __syncthreads(); // all threads done updating shmem

    // well we actually want to move across the image within each warp rather than per thread... more locality
    short warpMinX = TBMinX + ((wid_per_tb * SIDE_WARP_X) & (SIDE_TB_X - 1));
    short warpMaxX = warpMinX + SIDE_WARP_X;

    short warpMinY = TBMinY + (((wid_per_tb >> SIDE_WARP_X_PER_TB_LOG) * SIDE_WARP_Y) & (SIDE_TB_X - 1));
    short warpMaxY = warpMinY + SIDE_WARP_Y;

    int circleIndex, circleIndex3;

    // for all pixels in our current targets
    // if (blockIdx.x == 3){
    //     printf("z%d\n",blockIdx.y);
    //     printf("z0%d\n",threadIdx.y );
    //     printf("minx %d\n", warpMinX);
    // }
    for(short pixelY = warpMinY; pixelY < warpMaxY; pixelY += 1) {
        // float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + warpMinX + lane)]);

        // note we can take off 1 section of for loop since we have SIDE_WARP_X = 32...
        float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(warpMinX + lane) + 0.5f),
                                             invHeight * (static_cast<float>(pixelY) + 0.5f));

        float4* circle_imgPtr = (float4*)(&cuConstRendererParams.circleimageData[4*index0*imageHeight*imageWidth + 4*(pixelY * imageWidth + warpMinX + lane)]);
        original = *circle_imgPtr;
        existing = original;

        for(int activeCircleIndex = 0; activeCircleIndex < total_circles; activeCircleIndex += 1) {
            // find circle from active circles
            circleIndex = active_circles[activeCircleIndex];
            circleIndex3 = circleIndex * 3;
            p = *(float3*)(&cuConstRendererParams.position[circleIndex3]);
            float rad = cuConstRendererParams.radius[circleIndex];

            // shadePixel(circleIndex, pixelCenterNorm, p, imgPtr);
            existing = shadePixelSmall(circleIndex, pixelCenterNorm, p, rad, existing);
        }

        // Write back the pixel info
        if(original.w != existing.w) {
            *circle_imgPtr = existing;
        }
    }
}


// This function combines all the output from different thread wrt to circles
__global__ void kernelShadeCircles(){
    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int loc = 4 * (imageY * width + imageX);

    float4* imagePtr = (float4*)(&cuConstRendererParams.imageData[loc]);
    float4 existingColor = *(float4*)(&cuConstRendererParams.circleimageData[loc]);
    for (int i = 1; i< noOfCircleThreads; i++){
        float4* circle_imgPtr = (float4*)(&cuConstRendererParams.circleimageData[4*i*height*width + loc]);

        float4 rgb = *circle_imgPtr;
        float alpha = rgb.w;

        // float4 existingColor = (i == 1) ? *circle_imgPtr : *imagePtr;
        existingColor.x = rgb.x + alpha * existingColor.x;
        existingColor.y = rgb.y + alpha * existingColor.y;
        existingColor.z = rgb.z + alpha * existingColor.z;
        existingColor.w = alpha*existingColor.w;
    }

    *imagePtr = existingColor;
}
////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
    cudaDeviceCircleImageData = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
        cudaFree(cudaDeviceCircleImageData);

    }
}

const Image*
CudaRenderer::getImage() {

    // need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");

    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy
    noOfCircleThreads_temp = (numCircles +noOfCirclePerThread -1)/noOfCirclePerThread;

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);
    cudaMalloc(&cudaDeviceCircleImageData,sizeof(float) * 4 * image->width * image->height * noOfCircleThreads_temp);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;
    params.circleimageData = cudaDeviceCircleImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);
    cudaMemcpyToSymbol(noOfCircleThreads, &noOfCircleThreads_temp, sizeof(int));

    // last, copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) {
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>();
    }
    cudaDeviceSynchronize();
}

void
CudaRenderer::render() {
    if(numCircles> 20000) { //numCircles > THREADS_PER_WARP) {
        // 256 threads per block is a healthy number
        dim3 blockDim(256, 1);

        // num threads needed = (imageWidth / 16) * (imageHeight / 16) = (imageWidth * imageHeight / 256)
        // int numThreads = (image->width * image->height) / 256;
        int numThreads = (image->width * image->height) / BLOCK_PER_THREAD;
        dim3 gridDim(((numThreads) + blockDim.x - 1) / blockDim.x,
            (noOfCircleThreads_temp + blockDim.y - 1) / blockDim.y);



        dim3 blockDim2(16, 16, 1);
        dim3 gridDim2((image->width + blockDim2.x - 1) / blockDim2.x,
                (image->height+ blockDim2.y - 1) / blockDim2.y,
                (noOfCircleThreads_temp + blockDim2.z - 1) / blockDim2.z);

        kernelClearCircleImage<<<gridDim2, blockDim2>>>(0.f, 0.f, 0.f, 1.f);
        cudaDeviceSynchronize();




        // kernelRenderCirclesLarge<<<gridDim, blockDim>>>();
        kernelRenderParallelCirclesScan<<<gridDim, blockDim>>>();
        cudaDeviceSynchronize();



        dim3 blockDim1(16, 16);
        dim3 gridDim1(
        (image->width + blockDim1.x - 1) / blockDim1.x,
        (image->height + blockDim1.y - 1) / blockDim1.y);
        kernelShadeCircles<<<gridDim1, blockDim1>>>();
        cudaDeviceSynchronize();
    } 
    else{
        dim3 blockDim(256, 1);
        int numThreads = (image->width * image->height) / BLOCK_PER_THREAD;
        dim3 gridDim(((numThreads) + blockDim.x - 1) / blockDim.x);

        kernelRenderCirclesScan<<<gridDim, blockDim>>>();
        cudaDeviceSynchronize();
    }

    // else {
    //     // 256 threads per block is a healthy number
    //     dim3 blockDim(256, 1);

    //     // num threads needed = (imageWidth / 16) * (imageHeight / 16) = (imageWidth * imageHeight / 256)
    //     // int numThreads = (image->width * image->height) / 256;
    //     int numThreads = (image->width * image->height) / BLOCK_PER_THREAD;
    //     dim3 gridDim(((numThreads) + blockDim.x - 1) / blockDim.x);

    //     kernelRenderCirclesSmall<<<gridDim, blockDim>>>();
    //     cudaDeviceSynchronize();
    // }
}
