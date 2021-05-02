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

#define TOL 1.0*pow(10.0,-10.0)
#define NoOfThreads 256

////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

__device__ inline int nextPow2(int n){
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}


__device__ void maxElem003( float* A_s, int* k_s, int* l_s, int n, int i,int k){

    int idx = k;

    if (idx >= (n-i))
        return;

    if(fabsf(A_s[(idx +i)*n+idx+i+1]) > fabsf(A_s[idx*(n+1)+1])){
        A_s[idx*(n+1)+1] = A_s[(idx +i)*n+idx+i+1];
        k_s[idx*(n+1)+1] = k_s[(idx +i)*n+idx+i+1];
        l_s[idx*(n+1)+1] = l_s[(idx +i)*n+idx+i+1];
    }

    return;
}


__device__ void maxElem001( float* A_s, int* k_s, int* l_s, int n, int i, int y, int k){

    int m = k; //blockIdx.x * blockDim.x + threadIdx.x
    int idx =  m + y+1;

    if (idx >= (n-i))
        return;

    if(fabsf(A_s[y*n + idx +i]) > fabsf(A_s[y*n + idx])){
        A_s[y*n + idx] = A_s[y*n + idx +i];
        // k_s[y*n + idx] = k_s[y*n + idx +i];
        l_s[y*n + idx] = l_s[y*n + idx +i];
    }
    k_s[y*n + idx] = y;
    return;

}



__global__ void maxElemalongx(float* A_s, int* k_s, int* l_s, int n){

    // int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_y = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_y >= n)
        return;

    // int no_of_threads = 256;
    dim3 blockDim(NoOfThreads, 1);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x);
    dim3 gridDim2((((n + NoOfThreads -1)/NoOfThreads) + blockDim.x - 1) / blockDim.x);

    int i = nextPow2(n- thread_y -1)/2;//NoOfThreads;
    while(i > 0.9){
        for (int k = 0;k<i;k++){
            maxElem001(A_s,k_s,l_s,n, i, thread_y,k);
        }
        i = i/2;
    }

    for(int io =0; io<n;io++)
        printf(" thread_y, %d, %d, %f\n", thread_y, k_s[thread_y + io], A_s[thread_y + io]);
    
    return;
}

__global__ void maxElemalongy(float* A_s, int* k_s, int* l_s, int n){


    int i = nextPow2(n-1)/2;//NoOfThreads;
    while(i > 0.9){
        for (int k = 0;k<i;k++){
            maxElem003(A_s,k_s,l_s,n, i,k);
        }
        i = i/2;
    }

    // for(int io =0; io<n*n;io++)
    //     printf(" %d, %f\n", l_s[io], A_s[io]);

    return;
}



__global__ void kernelRotate(float* A, int k0 , int l0, int n, float* s0, float* tau0 ) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= n) 
        return;
    // printf("%s\n","enter");
    int k = k0;
    int l = l0;
    float t; // tan 
    float Adiff = A[n*l +l] - A[n*k +k];
    float temp = A[n*k +l]; // float temp = A[k,l];
    if(abs(temp) < abs(Adiff)*exp10f(-10))
        t = temp/Adiff;
    else {
        float phi = Adiff/(2.0*temp);
        t = 1.0/(abs(phi) + sqrt(phi*phi + 1.0));
        if(phi < 0.0) 
            t = -t;
    }
    // printf("%s\n","check 1");
    float c = 1.0/sqrt(t*t + 1.0); // cos
    float s = t*c;                 // sin
    float tau = s/(1.0 + c);
    *s0 = s;
    *tau0 = tau;
    

    // printf("%s\n","check 2");
    if (idx == k || idx == l){
        A[n*k +l] = 0.0;
        A[n*k +k] = A[n*k +k] - t*temp;
        A[n*l +l] = A[n*l +l] + t*temp;
        return;
    }

    if (idx < k){
        float temp = A[n*idx + k];
        float temp2 = A[n*idx + l];
        A[n*idx + k] = temp - s*(temp2 + tau*temp);
        A[n*idx + l] = temp2+ s*(temp - tau*temp2);
    }
    else if(k < idx && idx <l){
        float temp = A[n*k + idx];
        float temp2 = A[n*idx + l];
        A[n*k + idx] = temp - s*(temp2 + tau*temp);
        A[n*idx + l] = temp2+ s*(temp - tau*temp2);
    }
    else if(l < idx && idx < n){
        float temp = A[n*k + idx];
        float temp2 = A[n*l + idx];
        A[n*k + idx] = temp - s*(temp2 + tau*temp);
        A[n*l + idx] = temp2+ s*(temp - tau*temp2);
    }
    // printf("%s\n","check 3");
    return;
}


__global__ void kernelRotateP(float* p, int k0 , int l0, int n, float* s0, float* tau0) {

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= n)
        return;
    float s = *s0;
    float tau = *tau0;

    int k = k0;
    int l = l0;
    float temp = p[n*idx + k];
    float temp2 = p[n*idx + l];
    p[n*idx + k] = temp - s*(temp2 + tau*temp);
    p[n*idx + l] = temp2+ s*(temp - tau*temp2);
    // p[n*k + idx] = p[n*idx + k];
    // p[n*l + idx] = p[n*idx + l];
    return;
}


void initialize_identity(float *I, int size){
    memset(I, 0, sizeof(float)*size*size);
    for (int i = 0; i < size; i++)
        I[i * size + i] = 1.0;

    return;
}

__global__ void initialize_l_s(int* l_s,int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n)
        return;


    for(int i = 0; i< n;i++){
        l_s[idx*n + i] = i;
        // printf("idx %d, %d\n",idx, l_s[idx*n +i]);
    }
    return;
}


__global__ void initialize_k_s(int* k_s,int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n)
        return;

    for(int i = 0; i< n;i++){
        k_s[idx*n + i] = idx;
    }
    return;
}

__global__ void initialize_A_s(float *A ,float* A_s,int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n)
        return;

    for(int i = 0; i< n;i++){
        A_s[idx*n + i] = A[idx*n + i];
        // printf("idx %d %f,",idx,A[idx*n+i]);
    }
    return;
}


void kernelRotateFull(float *A_h, float *p_h, int n) {
    // n = no of training images
    // int maxRot = 5*(512*512) // Set limit on number of rotations
    // p = identity(n)*1.0 // Initialize transformation matrix
    // for i in range(maxRot): //Jacobi rotation loop
    //     Amax,k,l = maxElem(A)
    //     if Amax < tol: 
    //         return diagonal(A),p
    //     rotate(A,p,k,l)

    // float *A_h;
    // float *p_h;
    // int n=3;
    // float *A_s_h;
    // int *k_s_h,*l_s_h;
  
   // *A = {{1,2},{3,4}}; 
    /// method for max element


    initialize_identity(p_h,n);


    // for(int io =0; io<n*n;io++)
    //     printf("%f,",A_s_h[io]);

    printf("%s,%f\n","Tol = ",TOL);

    for(int io =0; io<n*n;io++)
        printf("%f,",A_h[io]);

    printf("%s\n","host side" );


    float *A;
    float *p;
    float *A_s;
    int *k_s,*l_s;

    cudaMalloc((void **) &A, sizeof(float) * n*n);
    cudaMalloc((void **) &p, sizeof(float) * n*n);
    cudaMalloc((void **) &A_s, sizeof(float) * n*n);
    cudaMalloc((void **) &k_s, sizeof(int) * n*n);
    cudaMalloc((void **) &l_s, sizeof(int) * n*n);

    size_t size = sizeof(float)* n*n;
    cudaMemcpy(A,A_h,size,cudaMemcpyHostToDevice);
    cudaMemcpy(p,p_h,size,cudaMemcpyHostToDevice);


    dim3 blockDim1(256,1);
    dim3 gridDim1((n + blockDim1.x - 1) / blockDim1.x);
                  // (n + blockDim1.y - 1) / blockDim1.y);
    initialize_k_s<<<gridDim1,blockDim1>>>(k_s,n);



    int iter_max = 5*n*n;
    for(int iter = 0; iter < iter_max; iter++){
        initialize_l_s<<<gridDim1,blockDim1>>>(l_s,n);
        initialize_A_s<<<gridDim1,blockDim1>>>(A,A_s,n);
        cudaDeviceSynchronize();
        // printf("%d,%s\n",iter,"initalize" );


        maxElemalongx<<<gridDim1, blockDim1>>>(A_s,k_s,l_s,n);
        cudaDeviceSynchronize();

        maxElemalongy<<<1,1>>>(A_s,k_s,l_s,n);
        // printf("%s\n,","max complete");

        float Amax =0.0;
        int k =0,l=0;
        float *ptr_Amax = &Amax;
        int *ptr_k = &k;
        int *ptr_l = &l;

        cudaMemcpy(ptr_Amax,&A_s[1],sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(ptr_k,&k_s[1],sizeof(int),cudaMemcpyDeviceToHost);
        cudaMemcpy(ptr_l,&l_s[1],sizeof(int),cudaMemcpyDeviceToHost);

        k = *ptr_k;
        l = *ptr_l;
        Amax = *ptr_Amax;
        printf("Amax for iter - %d,%f\n", iter,Amax);
        printf("K,l = %d,%d\n",k,l);


        if (Amax < TOL)
            break;
        // printf("%s\n,","Amax checked");

        dim3 blockDim(256, 1);
        dim3 gridDim((n + blockDim.x - 1) / blockDim.x);

        float s1 =0.0, tau1 =0.0;
        float *s0 = &s1;
        float *tau0 = &tau1;

        // printf("%s\n","kernel A rotate start" );
        kernelRotate<<<gridDim, blockDim>>>(A,k,l, n,s0,tau0);
        // printf("%s\n,","kernel A rotate complete" );
        kernelRotateP<<<gridDim, blockDim>>>(p,k,l,n,s0,tau0);
        cudaDeviceSynchronize();
        // printf("%s, %d\n,","iter complete - ", iter );

    }
    printf("%s\n","host side ends" );

    cudaMemcpy(p_h,p,size,cudaMemcpyDeviceToHost);
    cudaMemcpy(A_h,A,size,cudaMemcpyDeviceToHost);
    for(int io =0; io<n*n;io++)
        printf("%f,",p_h[io]);

    printf("%s\n,","Eigen values are" );
    for(int io =0; io<n;io++)
        printf("%f,",A_h[io*n + io]);




    cudaFree(A);
    cudaFree(A_s);
    cudaFree(p);
    cudaFree(l_s);
    cudaFree(k_s);

    return;




}




////////////////////////////////////////////////////////////////////////////////////////


int main(){


    float *A, *p;
    int n=3;
    A = (float*)malloc(n*n*sizeof(float));
    p = (float*)malloc(n*n*sizeof(float));

    // FILE *fp;
   // char buff[255];

   // fp = fopen("test.txt", "r"); 
   // fscanf(fp, "%f", A);
   // for (int i =0; i<n*n;i++)
   //      printf("%d,",A[i] );
    // *A = 1;
    // A[1] = 2;
    // A[2] = 3;
    // A[3] = 2;
    // A[4] = 4;
    // A[5] = 6;
    // A[6] = 3;
    // A[7] = 6;
    // A[8] = 9;
    //  *A_h = 3;
    // A_h[1] = 2;
    // A_h[2] = 2;
    // A_h[3] = 1;
    kernelRotateFull(A,p,n);
    printf("%s\n","done" );

    // fclose(fp);

    delete[] A;
    delete[] p;

}
