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



merge_sort(w, w_copy,sort_index,sort_index_copy,0,n);



}

__global__ void sort_vector_kernel(float* v, float*v_copy,int* sort_index,int n)

{
   

int row = blockIdx.x * blockDim.x + threadIdx.x;

if(row>=n) return;

for(int i=0;i<n;i++)
{
    v_copy[row*n+i] = v[row*n+sort_index[i]];
}


}


__global__ void sort_initialize(int n, int* sort_index, float* w_1d, float* w)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;

if(idx>=n) return;
sort_index[idx] = idx;
w_1d[idx] = w[idx*(n+1)];

}





