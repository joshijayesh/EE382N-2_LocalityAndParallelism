#include <stdlib.h>
#include <stdio.h>
#include <matmul_defaults.h>

#define NUM_ADDL_ARGS 0

extern int N, M, P;

// calculate C = AxB
void matmul_new(float **A, float **B, float **C) {
  float sum;
  int   i;
  int   j;
  int   k;

  for (i=0; i<M; i++) {
    // for each row of C
    for (j=0; j<N; j++) {
      // for each column of C
      sum = 0.0f; // temporary value
      for (k=0; k<P; k++) {
        // dot product of row from A and column from B
        sum += A[i][k]*B[j][k];
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

int main() {
  float** A;
  float** B, **D;
  float** C;

    get_args(NUM_ADDL_ARGS, "", argc, argv);

  create_matrix(&A, M, P);
  create_matrix(&B, P, N);
  create_matrix(&C, M, N);
  create_matrix(&D, N, P);

  transpose_matrix(B, D,P,N);
  // assume some initialization of A and B
  // think of this as a library where A and B are
  // inputs in row-major format, and C is an output
  // in row-major.

  matmul_new(A, D, C);

  return (0);
}