#include <stdlib.h>
#include <stdio.h>
#include <matmul_defaults.h>

#define ARG_B1_INDEX NUM_ARGS + 0
#define ARG_B2_INDEX NUM_ARGS + 1
#define ARG_B3_INDEX NUM_ARGS + 2
#define NUM_ADDL_ARGS 1

extern int N, M, P;
int B1, B2, B3;

// calculate C = AxB
void matmul(float **A, float **B, float **C){
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

  T = (float**)malloc( m*sizeof(float*));
  for ( i=0; i<m; i++ ) {
     T[i] = (float*)malloc(n*sizeof(float));
  }
  *A = T;
}

int main(int argc, char *argv[]) {
  float** A;
  float** B;
  float** C;

  get_args(NUM_ADDL_ARGS, "B1", argc, argv);
  B1 = atoi(argv[ARG_B1_INDEX]);

  create_matrix(&A, M, P);
  create_matrix(&B, P, N);
  create_matrix(&C, M, N);

  // assume some initialization of A and B
  // think of this as a library where A and B are
  // inputs in row-major format, and C is an output
  // in row-major.

  matmul(A, B, C);

  return (0);
}
