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


using namespace std;


__device__ inline int nextPow2(int n)
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


// __global__ void compare(float* w, int* sort_index, int n)

// {
// int idx = blockIdx.x*blockDim.x + threadIdx.x;


// }

// __global__ void merge(float* w, int* sort_index)

// {



// }


__device__ void merge(float* w,float* w_copy,int* sort_index,int* sort_index_copy, int start,int n1,int n2)


{


for(int i=0;i<n2;i++)
{
    int r=start+i;
    w_copy[r] = w[r];
    sort_index_copy[r]  = sort_index[r];
}


int i=0,j=n1,k=0;
int r,s,t;
while(i<n1 && j<n2)
{
    r=start+i;
    s=start+j;
    t=start+k;
    if(w_copy[r]<w_copy[s])
        {
            w[t] = w_copy[s];
            sort_index[t] = sort_index_copy[s];
            j++;
        }
        else
        {
            w[t] = w_copy[r];
            sort_index[t] = sort_index_copy[r];
            i++;
        }

        k++;


}

while(i<n1)
{
    r=start+i;
    t=start+k;

    w[t] = w_copy[r];
    sort_index[t] = sort_index_copy[r];
    i++;
    k++;
}

while(j<n2)
{
    s=start+j;
    t=start+k;

    w[t] = w_copy[s];
    sort_index[t] = sort_index_copy[s];
    j++;
    k++;
}




}

__device__ void merge_sort(float* w,float* w_copy,int* sort_index,int* sort_index_copy, int start,int n)


{

if(n==1) return;

else
{
merge_sort(w, w_copy, sort_index,sort_index_copy,start,n/2);
merge_sort(w, w_copy,sort_index,sort_index_copy,start+n/2,n-n/2);
merge(w,w_copy,sort_index,sort_index_copy,start,n/2,n);

return;

}



}




__global__ void sort_value_kernel(float* w, float* w_copy,int* sort_index,int* sort_index_copy, int n)

{

     printf("Entered sort_value_kernel\n");


merge_sort(w, w_copy,sort_index,sort_index_copy,0,n);



}

__global__ void sort_vector_kernel(float* v, float*v_copy,int* sort_index,int n)

{
   

int row = blockIdx.x * blockDim.x + threadIdx.x;

if(row>=n) return;

 printf("Entered sort_vector_kernel\n");

for(int i=0;i<n;i++)
{
    v_copy[row*n+i] = v[row*n+sort_index[i]];
}


}




void sort_eigenvectors(int n) {
   

int* sort_index,*sort_index_h,*sort_index_copy;

float *w,*v,*w_h,*v_h,*v_copy;

float *w_copy;

dim3 blockDim(256,1);
dim3 gridDim((n + blockDim.x - 1) / blockDim.x);


w_h = (float*)malloc(n*n*sizeof(float));

v_h = (float*)malloc(n*n*sizeof(int));

sort_index_h = (int*)malloc(n*sizeof(int));

for(int i=0;i<n;i++)
        {
            sort_index_h[i]=i;
            w_h[i] = i;

        }




cudaMalloc((void **) &w, sizeof(float)*n);
cudaMalloc((void **) &w_copy, sizeof(float)*n);

cudaMalloc((void **) &v, sizeof(float) * n*n);
cudaMalloc((void **) &v_copy, sizeof(float) * n*n);

cudaMalloc((void **) &sort_index, sizeof(int) * n);
cudaMalloc((void **) &sort_index_copy, sizeof(int) * n);

size_t size = sizeof(float)* n*n;

cudaMemcpy(w,w_h,sizeof(float)* n,cudaMemcpyHostToDevice);
cudaMemcpy(v,v_h,size,cudaMemcpyHostToDevice);
cudaMemcpy(sort_index,sort_index_h,sizeof(int)*n,cudaMemcpyHostToDevice);


sort_value_kernel<<<1,1>>>(w,w_copy,sort_index,sort_index_copy,n);

cudaDeviceSynchronize();

sort_vector_kernel<<<gridDim,blockDim>>>(v,v_copy,sort_index,n);

cudaDeviceSynchronize();

cudaMemcpy(w_h,w,sizeof(float)* n,cudaMemcpyDeviceToHost);
cudaMemcpy(v_h,v_copy,size,cudaMemcpyDeviceToHost);

for(int i=0;i<n;i++)
{
    printf("%f\n",w_h[i] );
}



    delete[] w_h;
    delete[] v_h;
    delete[] sort_index_h;


    cudaFree(sort_index);
    cudaFree(sort_index_copy);
    cudaFree(w);
    cudaFree(w_copy);
    cudaFree(v);
    cudaFree(v_copy);

    return;



}



int main(int argc,char* argv[]){

    sort_eigenvectors(stoi(argv[1]));
    return 0;
}
