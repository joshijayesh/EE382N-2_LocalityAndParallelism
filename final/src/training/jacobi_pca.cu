#include <vector>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "commons.hpp"
#include "commons.cuh"
#include "training/routine.hpp"
#include "training/routine.cuh"

#define TOL 1.0*pow(10.0,-15.0)
#define NoOfThreads 256
#define NoofThreads_2 2*NoOfThreads


__device__ inline uint32_t nextPow2(uint32_t n){
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}


__global__ void maxElemalongx(float* A_s, int* k_s, int* l_s, int n){

    int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_y = blockIdx.y * blockDim.y + threadIdx.y;           /// here this is the row no

    if (thread_y*NoofThreads_2 + thread_x >= (n*(n-1))/2)
        return;

    int i = NoOfThreads;
    while(i > 0.9){
        if(thread_x >= i) return;
        if(fabsf(A_s[thread_y*NoofThreads_2 + thread_x +i]) > fabsf(A_s[thread_y*NoofThreads_2 + thread_x])){
            A_s[thread_y*NoofThreads_2 + thread_x] = A_s[thread_y*NoofThreads_2 + thread_x + i];
            l_s[thread_y*NoofThreads_2 + thread_x] = l_s[thread_y*NoofThreads_2 + thread_x + i];
            k_s[thread_y*NoofThreads_2 + thread_x] = k_s[thread_y*NoofThreads_2 + thread_x + i];
        }
        i = i/2;
        __syncthreads();
    }
    
    return;
}


__global__ void maxElemalongy(float* A_s, int* k_s, int* l_s, int n){

    int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
    
    int max_row = ((n*(n-1)/2) + NoofThreads_2 -1)/(NoofThreads_2);
    if (thread_x >= max_row)
        return;

    int i = nextPow2(max_row)/2;//NoOfThreads;
    while(i > 0.9){
        if(thread_x >= i) return;
        if(fabsf(A_s[(thread_x+i)*NoofThreads_2]) > fabsf(A_s[thread_x*NoofThreads_2])){
            A_s[thread_x*NoofThreads_2] = A_s[(thread_x+i)*NoofThreads_2];
            k_s[thread_x*NoofThreads_2] = k_s[(thread_x+i)*NoofThreads_2];
            l_s[thread_x*NoofThreads_2] = l_s[(thread_x+i)*NoofThreads_2];
        }
        __syncthreads();
        i = i/2;
    }

    return;
}


__global__ void kernelRotate(float* A, int k , int l, int n, double* s0, double* tau0 ) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= n) 
        return;

    double t; // tan 
    double Adiff = A[n*l +l] - A[n*k +k];
    double temp = A[n*k +l]; // float temp = A[k,l];
    if(abs(temp) < abs(Adiff)*exp10f(-38))
        t = temp/Adiff;
    else {
        float phi = Adiff/(2.0*temp);
        t = 1.0/(abs(phi) + sqrt(phi*phi + 1.0));
        if(phi < 0.0) 
            t = -t;
    }
    double c = 1.0/sqrt(t*t + 1.0); // cos
    double s = t*c;                 // sin
    double tau = s/(1.0 + c);
    *s0 = s;
    *tau0 = tau;
    
    if (idx == k || idx == l){
        A[n*k +l] = 0.0;
        A[n*k +k] = A[n*k +k] - t*temp;
        A[n*l +l] = A[n*l +l] + t*temp;
        return;
    }

    if (idx < k){
        double temp = A[n*idx + k];
        double temp2 = A[n*idx + l];
        A[n*idx + k] = temp - s*(temp2 + tau*temp);
        A[n*idx + l] = temp2+ s*(temp - tau*temp2);
    }
    else if(k < idx && idx <l){
        double temp = A[n*k + idx];
        double temp2 = A[n*idx + l];
        A[n*k + idx] = temp - s*(temp2 + tau*temp);
        A[n*idx + l] = temp2+ s*(temp - tau*temp2);
    }
    else if(l < idx && idx < n){
        double temp = A[n*k + idx];
        double temp2 = A[n*l + idx];
        A[n*k + idx] = temp - s*(temp2 + tau*temp);
        A[n*l + idx] = temp2+ s*(temp - tau*temp2);
    }
    return;
}


__global__ void kernelRotateP(float* p, int k , int l, int n, double* s0, double* tau0) {

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= n)double
        return;
    double s = *s0;
    double tau = *tau0;

    double temp = p[n*idx + k];
    double temp2 = p[n*idx + l];
    p[n*idx + k] = temp - s*(temp2 + tau*temp);
    p[n*idx + l] = temp2+ s*(temp - tau*temp2);
    return;
}

inline void init_params(float*& A_s, int*& k_s, int*& l_s, uint32_t n){
    /* Assume A/P already on device
    CUDAERR_CHECK(
        cudaMalloc((void **) &A, sizeof(float) * n*n),
        "Unable to malloc d_data", ERR_CUDA_MALLOC);
    CUDAERR_CHECK(
        cudaMalloc((void **) &p, sizeof(float) * n*n);
        "Unable to malloc d_data", ERR_CUDA_MALLOC);
    */
    CUDAERR_CHECK(
        cudaMalloc((void **) &A_s, sizeof(float) * n*n),
        "Unable to malloc d_data", ERR_CUDA_MALLOC);
    CUDAERR_CHECK(
        cudaMalloc((void **) &k_s, sizeof(int) * n*n),
        "Unable to malloc d_data", ERR_CUDA_MALLOC);
    CUDAERR_CHECK(
        cudaMalloc((void **) &l_s, sizeof(int) * n*n),
        "Unable to malloc d_data", ERR_CUDA_MALLOC);

    /*  Assume A/P already on device
    CUDAERR_CHECK(
        cudaMemcpy(A, A_h, size, cudaMemcpyHostToDevice);
        "Unable to copy matrices to device!", ERR_CUDA_MEMCPY);
    CUDAERR_CHECK(
        cudaMemcpy(p, p_h, size, cudaMemcpyHostToDevice);
        "Unable to copy matrices to device!", ERR_CUDA_MEMCPY);
    */
}


__global__ void initialize_temp_matrices(float *A , float* A_s, int *k_s, int *l_s, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= n || idy >= idx)
        return;
    
    int index = (idy*(2*n - idy - 1))/2 + idx - idy - 1;

    A_s[index] = A[idy*n + idx];
    l_s[index] = idx;
    k_s[index] = idy;

    return;
}



/* Assume p_h is already initialized as Identity matrix */
void JacobiPCA::find_eigenvectors() {
    float *A = d_data_cov;
    float *p = d_eigenvectors;
    uint32_t n = num_images;
    // printf("%s,%f\n","Tol = ",TOL);
    printf("%s\n","jacobi device side start" );

    float *A_s;
    int *k_s,*l_s;
    init_params(A_s, k_s, l_s, n);
 

    dim3 blockDim(256,1);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
                   (n + blockDim.y - 1) / blockDim.y);


    dim3 blockDim1(256,1);
    dim3 gridDim1((n + blockDim1.x - 1) / blockDim1.x);

    int max_row = ((n*(n-1)/2) + NoofThreads_2 -1)/(NoofThreads_2);
    dim3 blockDim2(NoOfThreads,1);
    dim3 gridDim2(1,max_row);

    dim3 blockDim3(NoOfThreads,1);
    dim3 gridDim3((max_row + blockDim3.x - 1) / blockDim3.x);


    int iter_max = 5*n*n;
    for(int iter = 0; iter < iter_max; iter++){
        initialize_temp_matrices<<<gridDim,blockDim>>>(A,A_s,k_s,l_s,n);
        cudaDeviceSynchronize();
        // printf("%d,%s\n",iter,"initalize" );

        maxElemalongx<<<gridDim2, blockDim2>>>(A_s,k_s,l_s,n);
        cudaDeviceSynchronize();
        maxElemalongy<<<gridDim3, blockDim3>>>(A_s,k_s,l_s,n);
        cudaDeviceSynchronize();

        float Amax =0.0;
        int k =0,l=0;
        float *ptr_Amax = &Amax;
        int *ptr_k = &k;
        int *ptr_l = &l;

        cudaMemcpy(ptr_Amax,&A_s[0],sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(ptr_k,&k_s[0],sizeof(int),cudaMemcpyDeviceToHost);
        cudaMemcpy(ptr_l,&l_s[0],sizeof(int),cudaMemcpyDeviceToHost);

        k = *ptr_k;
        l = *ptr_l;
        Amax = *ptr_Amax;
        // printf("Amax for iter - %d,%f\n", iter,Amax);
        // printf("K,l = %d,%d\n",k,l);

        if (abs(Amax) < TOL){
            break;
        }
        // printf("%s\n,","Amax checked");


        double s1 =0.0, tau1 =0.0;
        double *s0 = &s1;
        double *tau0 = &tau1;

        // printf("%s\n","kernel A rotate start" );
        kernelRotate<<<gridDim1, blockDim1>>>(A,k,l, n,s0,tau0);
        // printf("%s\n,","kernel A rotate complete" );
        kernelRotateP<<<gridDim1, blockDim1>>>(p,k,l,n,s0,tau0);
        cudaDeviceSynchronize();
        // printf("%s, %d\n,","iter complete - ", iter );

    }

    printf("%s\n","jacobi device side ends" );

    /* This can be done for debug
    size_t size = sizeof(float)* n*n;
    CUDAERR_CHECK(
        cudaMemcpy(p_h,p,size,cudaMemcpyDeviceToHost);
        "Unable to copy matrices to host!", ERR_CUDA_MEMCPY);
    CUDAERR_CHECK(
        cudaMemcpy(A_h,A,size,cudaMemcpyDeviceToHost);
        "Unable to copy matrices to host!", ERR_CUDA_MEMCPY);

    for(int io =0; io<n*n;io++)
        printf("%f,",p_h[io]);

    printf("%s\n,","Eigen values are" );
    for(int io =0; io<n;io++)
        printf("%f,",A_h[io*n + io]);
    */

    // need to leave it on device, host doesn't consume it yet
    // cudaFree(A);
    // cudaFree(p);
    cudaFree(A_s);
    cudaFree(l_s);
    cudaFree(k_s);
}


