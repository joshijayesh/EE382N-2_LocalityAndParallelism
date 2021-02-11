#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h> 
#include "matmul_defaults.h"
#include <immintrin.h>

#define NUM_ADDL_ARGS 0

extern int M,N,P;


//// right now it is assumed that N=M=P
// calculate C = AxB
void matmul_cache_oblivious(float **A, float **B, float **C,int M,int N,int P,int offset_m,int offset_n,int offset_p){
    int LOOP_UNROLL_CONSTANT = 2;
    __m512 sum[2] __attribute__ ((aligned (AVX_ALIGNMENT)));

    if(M+N+P<=96)  //expecting 16x16 matrices
    {
        int i,j,k;
        int i_u;
        __m512 b;
        for(int j=0;j<N;j++)
        {
            // __m512 sum = _mm512_load_ps(&C[offset_n+j][offset_m]);
            for(i_u = 0; i_u < LOOP_UNROLL_CONSTANT; i_u++){
                sum[i_u] = _mm512_load_ps(&C[offset_n+j][offset_m + i_u * NUM_FLOATS_PER_AVX]);
            }
            for(int k=0;k<P;k++)
            {
                // sum+= A[offset_m+i][offset_p+k]*B[offset_p+k][offset_n+j];
                b = _mm512_set1_ps(B[offset_n + j][offset_p + k]);
                for(i_u = 0; i_u < LOOP_UNROLL_CONSTANT; i_u++) {
                    sum[i_u] = _mm512_add_ps(sum[i_u],
                            _mm512_mul_ps(
                                _mm512_load_ps(&A[offset_m+k][offset_p + i_u * NUM_FLOATS_PER_AVX]),
                                b));
                }
            }		
            // C[offset_m+i][offset_n+j]+=sum;
            for(i_u = 0; i_u < LOOP_UNROLL_CONSTANT; i_u++){
                _mm512_storeu_ps(&C[offset_n+j][offset_m + i_u * NUM_FLOATS_PER_AVX], sum[i_u]);
            }
            // _mm512_store_ps(&C[offset_n+j][offset_m], sum);
        }
    }

    else
    {
        matmul_cache_oblivious(A,B,C,M/2,N/2,P/2,offset_m,offset_n,offset_p);
        matmul_cache_oblivious(A,B,C,M/2,N/2,P-P/2,offset_m,offset_n,offset_p+P/2);

        matmul_cache_oblivious(A,B,C,M/2,N-N/2,P/2,offset_m,offset_n+N/2,offset_p);
        matmul_cache_oblivious(A,B,C,M/2,N-N/2,P-P/2,offset_m,offset_n+N/2,offset_p+P/2);

        matmul_cache_oblivious(A,B,C,M-M/2,N/2,P/2,offset_m+M/2,offset_n,offset_p);
        matmul_cache_oblivious(A,B,C,M-M/2,N/2,P-P/2,offset_m+M/2,offset_n,offset_p+P/2);

        matmul_cache_oblivious(A,B,C,M-M/2,N-N/2,P/2,offset_m+M/2,offset_n+N/2,offset_p);
        matmul_cache_oblivious(A,B,C,M-M/2,N-N/2,P-P/2,offset_m+M/2,offset_n+N/2,offset_p+P/2);
    }



   return;
}


void matmul_basic(float **A, float **B, float **C){
    float sum;
    int     i;
    int     j;
    int     k;

    for (i=0; i<M; i++) {
        // for each row of C
        for (j=0; j<N; j++) {
            // for each column of C
            sum = 0.0f; // temporary value
            for (k=0; k<P; k++) {
                // dot product of row from A and column from B
                sum += A[i][k]*B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

// function to allocate a matrix on the heap
// creates an mXn matrix and returns the pointer.
//
// the matrices are in row-major order.
void create_matrix(float*** A, int m, int n) {
    float **T = 0;
    int i;

    T = (float**)aligned_alloc(AVX_ALIGNMENT, m*sizeof(float*));
    for ( i=0; i<m; i++ ) {
         T[i] = (float*)aligned_alloc(AVX_ALIGNMENT, n*sizeof(float));
    }
    *A = T;
}

int main(int argc, char *argv[]) {
    float** A;
    float** B;
    float** C;
    // float** D;

    // M=atoi(argv[1]);
    // N=atoi(argv[2]);
    // P=atoi(argv[3]);

    get_args(NUM_ADDL_ARGS, "", argc, argv);

    create_matrix(&A, M, P);
    create_matrix(&B, P, N);
    create_matrix(&C, M, N);
    // create_matrix(&D, M, N);


    // assume some initialization of A and B
    // think of this as a library where A and B are
    // inputs in row-major format, and C is an output
    // in row-major.

    for (int i=0; i<M; i++) {

        for (int j=0; j<N; j++) {
          A[i][j]= i + j+1;
          C[i][j]=0.0f;
          // printf("%f  ",A[i][j] );

        }
        // printf("\n");
    }
//    printf("\n");

    for (int i=0; i<N; i++) {

        for (int j=0; j<P; j++) {

            B[i][j]= i - j+1;
            // printf("%f  ",B[i][j] );
        }
        // printf("\n");
    }
    // printf("\n");
 
    matmul_cache_oblivious(A, B, C,M,N,P,0,0,0);
    // matmul_basic(A,B,D);

    // bool equal=true;

    //  for (int i=0; i<M; i++) {

    //     for (int j=0; j<N; j++) {
    //     printf("%f  ",C[i][j] );

    //     }
    //     printf("\n");
    // }

    // printf("\n");

    // for (int i=0; i<M; i++) {

    //     for (int j=0; j<N; j++) {
    //       if(C[i][j]!=D[i][j]) {equal=false;break;}

    //     }
    // }

    // printf("%d\n",equal);


    return (0);
}
