#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h> 
#include <matmul_defaults.h>

#define NUM_ADDL_ARGS 0

extern int M,N,P;


//// right now it is assumed that N=M=P
// calculate C = AxB
void matmul_cache_oblivious(float **A, float **B, float **C,int M,int N,int P,int offset_m,int offset_n,int offset_p){


   if(M+N+P<=48){  //expecting 16x16 matrices
	int i,j,k;
	for(int i=0;i<M;i++){
		for(int j=0;j<N;j++){
			float sum=0;
		    for(int k=0;k<P;k++){
		    	sum+= A[offset_m+i][offset_p+k]*B[offset_n+j][offset_p+k];
		    }		
			C[offset_m+i][offset_n+j]+=sum;
		}
	 }
    }

    else{
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


// void matmul_basic(float **A, float **B, float **C){
//     float sum;
//     int     i;
//     int     j;
//     int     k;

//     for (i=0; i<M; i++) {
//         // for each row of C
//         for (j=0; j<N; j++) {
//             // for each column of C
//             sum = 0.0f; // temporary value
//             for (k=0; k<P; k++) {
//                 // dot product of row from A and column from B
//                 sum += A[i][k]*B[k][j];
//             }
//             C[i][j] = sum;
//         }
//     }
// }

// function to allocate a matrix on the heap
// creates an mXn matrix and returns the pointer.
//
// the matrices are in row-major order.
void create_matrix(float*** A, int m, int n) {
    float **T = 0;
    int i;

    T = (float**)malloc( m*sizeof(float*));
    for ( i=0; i<m; i++ ) {
         T[i] = (float*)malloc(n*sizeof(float));
    }
    *A = T;
}


void transpose_matrix(float **A, float **B,int m, int n){
  for (int i =0;i <m;i++){
    for (int j =0; j<n;j++){
      B[j][i] = A[i][j];
    }
  }
}

int main(int argc, char *argv[]) {
    float** A;
    float** B, **D;
    float** C;
    // float** D;

    // M=atoi(argv[1]);
    // N=atoi(argv[2]);
    // P=atoi(argv[3]);

    get_args(NUM_ADDL_ARGS, "", argc, argv);

    create_matrix(&A, M, P);
    create_matrix(&B, P, N);
    create_matrix(&C, M, N);
    create_matrix(&D, N, P);


    // assume some initialization of A and B
    // think of this as a library where A and B are
    // inputs in row-major format, and C is an output
    // in row-major.

    transpose_matrix(B, D,P,N);

    matmul_cache_oblivious(A, D, C,M,N,P,0,0,0);
  


    return (0);
}
