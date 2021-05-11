#include <string>
#include <vector>
#include <iostream>
#include <chrono>
#include <fstream>

#include "commons.cuh"
#include "commons.cuh"
#include "training/routine.hpp"
#include "training/routine.cuh"
#include "training/kernels.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#define TOL pow(10, -12)


__global__
void identity(uint32_t n, uint32_t m, float* d) {
    uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if(y < n && x < m)
        d[y * m + x] = x == y ? 1 : 0;
}


inline void init_H(uint32_t n, float* H) {
    dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridDim((n + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (n + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

    identity<<<gridDim, blockDim>>> (n, n, H);
}


__global__
void vector_norm(uint32_t n, uint32_t full_n, float* R, float* V, float* T) {
    uint16_t wid = threadIdx.x >> THREADS_PER_WARP_LOG;
    uint16_t lane = threadIdx.x & THREADS_PER_WARP_MASK;
    uint32_t idx_x = threadIdx.x;
    float sum = 0.0;

    __shared__ float shared_sum[WARPS_PER_BLOCK];

    for(uint32_t i = idx_x; i < n; i += THREADS_PER_BLOCK) {
        float element = R[i * full_n];
        sum += element * element;
    }

    for(int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(FULL_WARP_MASK, sum, offset);
    }

    __syncthreads();

    if(lane == 0) {
        shared_sum[wid] = sum;
    }

    __syncthreads();

    if(wid == 0 && threadIdx.x < WARPS_PER_BLOCK) {
        sum = shared_sum[threadIdx.x];
        for(int offset = WARPS_PER_BLOCK / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xff, sum, offset);
        }

        if(threadIdx.x == 0) {
            sum = sqrt(sum);
            if((sum < 0 && R[0] > 0) || (sum > 0 && R[0] < 0))  // copy sign
                sum = -sum;
            shared_sum[0] = sum + R[0];
        }
    }

    __syncthreads();

    if(lane == 0) {
        sum = shared_sum[0];
    }

    sum = __shfl_sync(FULL_WARP_MASK, sum, 0);

    float tau = 0.0;

    for(uint32_t i = idx_x; i < n; i += THREADS_PER_BLOCK) {
        if(i != 0) {
            V[i] = R[i * full_n] / sum;

            tau += (V[i] * V[i]);
        } else {
            V[0] = 1;
            tau += 1;
        }
    }

    for(int offset = 16; offset > 0; offset /= 2) {
        tau += __shfl_down_sync(FULL_WARP_MASK, tau, offset);
    }

    __syncthreads();

    if(lane == 0) {
        shared_sum[wid] = tau;
    }

    __syncthreads();

    if(wid == 0 && threadIdx.x < WARPS_PER_BLOCK) {
        tau = shared_sum[threadIdx.x];
        for(int offset = WARPS_PER_BLOCK / 2; offset > 0; offset /= 2) {
            tau += __shfl_down_sync(0xff, tau, offset);
        }

        if(threadIdx.x == 0) {
            *T = 2.0 / tau;
        }
    }
}


inline void find_v(uint32_t n, uint32_t full_n, float* R, float* V, float* T) {
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim(1);

    vector_norm<<<gridDim, blockDim>>> (n, full_n, R, V, T);
    cudaDeviceSynchronize();
}

__global__
void matmul_vectors_herm(uint32_t full_n, uint32_t n, uint32_t j, float* A, float *C, float *T) {
    uint32_t idx_x = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint32_t idx_y = (blockIdx.y * blockDim.x) + threadIdx.y;

    if(idx_y < n && idx_x < n) {
        C[(idx_y + j) * full_n + j + idx_x] -= A[idx_y] * A[idx_x] * T[0];
    }
}

inline void find_herm(uint32_t n, uint32_t j, float* H, float* V, float* T) {
    uint32_t m = n - j;

    dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridDim((m + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (m + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

    matmul_vectors_herm<<<gridDim, blockDim>>> (n, m, j, V, H, T);
    cudaDeviceSynchronize();
}

inline void find_Q_R(uint32_t n, float* R, float* Q, float* H, float* R_o, float* Q_o) {
    dim3 blockDim(MATMUL_BLOCK_DIM_X, MATMUL_BLOCK_DIM_Y);
    dim3 gridDim((n + MATMUL_TILE_DIM - 1) / MATMUL_TILE_DIM, (n + MATMUL_TILE_DIM - 1) / MATMUL_TILE_DIM);

    matmul<<<gridDim, blockDim>>>(n, n, n, H, R, R_o);
    matmul<<<gridDim, blockDim>>>(n, n, n, H, Q, Q_o);
    cudaDeviceSynchronize();
}

inline void transpose_Q(uint32_t n, float* Q, float* Q_o) {
    dim3 blockDim(TRANSPOSE_BLOCK_DIM_X, TRANSPOSE_BLOCK_DIM_Y);
    dim3 gridDim((n + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE, (n + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE);

    transpose_kernel<<<gridDim, blockDim>>> (n, n, Q_o, Q);
    cudaDeviceSynchronize();
}


void qr_decomposition(uint32_t n, float* A, float* Q, float* R, float* H, float* V, float* T, float* R_o, float* Q_o) {
    cudaMemcpy(R, A, sizeof(float) * n * n, cudaMemcpyDeviceToDevice);
    init_H(n, Q);

    for(int j = 0; j < n; j += 1) {
        if(j % 2 == 0) {
            init_H(n, H);
            find_v(n - j, n, &R[j * n + j], V, T);
            find_herm(n, j, H, V, T);
            find_Q_R(n, R, Q, H, R_o, Q_o);
        } else {
            init_H(n, H);
            find_v(n - j, n, &R_o[j * n + j], V, T);
            find_herm(n, j, H, V, T);
            find_Q_R(n, R_o, Q_o, H, R, Q);
        }
    }

    if(n % 2 == 1) {
        cudaMemcpy(R, R_o, sizeof(float) * n * n, cudaMemcpyDeviceToDevice);
    } else {
        cudaMemcpy(Q_o, Q, sizeof(float) * n * n, cudaMemcpyDeviceToDevice);
    }

    transpose_Q(n, Q, Q_o);
}

inline void new_A(uint32_t n, float* R, float* Q, float* A) {
    dim3 blockDim(MATMUL_BLOCK_DIM_X, MATMUL_BLOCK_DIM_Y);
    dim3 gridDim((n + MATMUL_TILE_DIM - 1) / MATMUL_TILE_DIM, (n + MATMUL_TILE_DIM - 1) / MATMUL_TILE_DIM);

    matmul<<<gridDim, blockDim>>>(n, n, n, R, Q, A);
    cudaDeviceSynchronize();
}

__global__
void vector_norm_diag(uint32_t n, float* A, float* A_i, float* T) {
    uint16_t wid = threadIdx.x >> THREADS_PER_WARP_LOG;
    uint16_t lane = threadIdx.x & THREADS_PER_WARP_MASK;
    uint32_t idx_x = threadIdx.x;
    float sum = 0.0;

    __shared__ float shared_sum[WARPS_PER_BLOCK];

    for(uint32_t i = idx_x; i < n; i += THREADS_PER_BLOCK) {
        float element = A[i * n + i] - A_i[i * n + i];
        sum += element * element;
    }

    for(int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(FULL_WARP_MASK, sum, offset);
    }

    __syncthreads();

    if(lane == 0) {
        shared_sum[wid] = sum;
    }

    __syncthreads();

    if(wid == 0 && threadIdx.x < WARPS_PER_BLOCK) {
        sum = shared_sum[threadIdx.x];
        for(int offset = WARPS_PER_BLOCK / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xff, sum, offset);
        }

        if(threadIdx.x == 0) {
            sum = sqrt(sum);
            *T = sum;
        }
    }
}

inline void find_ev_norm(uint32_t n, float* A, float* A_i, float* Norm) {
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim(1);

    vector_norm_diag<<<gridDim, blockDim>>>(n, A, A_i, Norm);
    cudaDeviceSynchronize();
}

void qr_iteration(uint32_t n, float* A, float* Q) {
    float* d_R;
    float* d_H;
    float* d_A_i;
    float* d_V;
    float* d_T;
    float* d_R_o;
    float* d_Q_o;
    float* d_norm;
    cudaMalloc((void**) &d_R, sizeof(float) * n * n);
    cudaMalloc((void**) &d_A_i, sizeof(float) * n * n);
    cudaMalloc((void**) &d_R_o, sizeof(float) * n * n);
    cudaMalloc((void**) &d_Q_o, sizeof(float) * n * n);
    cudaMalloc((void**) &d_H, sizeof(float) * n * n);
    cudaMalloc((void**) &d_V, sizeof(float) * n);
    cudaMalloc((void**) &d_T, sizeof(float));
    cudaMalloc((void**) &d_norm, sizeof(float));

    float h_norm;

    for(int i = 0; i < n*n; i += 1) {
        cudaMemcpyAsync(d_A_i, A, sizeof(float) * n * n, cudaMemcpyDeviceToDevice);
        qr_decomposition(n, A, Q, d_R, d_H, d_V, d_T, d_R_o, d_Q_o);
        new_A(n, d_R, Q, A);
        find_ev_norm(n, A, d_A_i, d_norm);

        cudaMemcpy(&h_norm, d_norm, sizeof(float), cudaMemcpyDeviceToHost);

        if(h_norm < TOL) break;
    }

    cudaFree(d_R);
    cudaFree(d_H);
    cudaFree(d_V);
    cudaFree(d_T);
    cudaFree(d_R_o);
    cudaFree(d_Q_o);
    cudaFree(d_A_i);
    cudaFree(d_norm);
}

void QRPCA::find_eigenvectors() {
    qr_iteration(num_images, d_data_cov, d_eigenvectors);
}

