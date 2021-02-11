#include <stdlib.h>
#include <stdio.h>
#include <matmul_defaults.h>
#include <immintrin.h>

#define ARG_B1_INDEX NUM_ARGS + 0
#define ARG_B2_INDEX NUM_ARGS + 1
#define ARG_B3_INDEX NUM_ARGS + 2
#define NUM_ADDL_ARGS 3

extern int N, M, P;
// Note: set B1/B2/B3 to N to removing tiling at those levels
int B1, B2, B3;  // ASSUMPTION: B1/B2/B3 divides N/M/P perfectly

// calculate C = AxB
void matmul(float **A, float **B, float **C){
    int     i, i_B1, i_B2, i_B3, i_u;
    int     j, j_B1, j_B2, j_B3;
    int     k, k_B1, k_B2, k_B3;
    __m512  b;
    __m512 sum[2] __attribute__ ((aligned (AVX_ALIGNMENT)));

    int LOOP_UNROLL_CONSTANT = 2;

    // L3 Tiling
    for (i_B3=0; i_B3 < M; i_B3 += B3) {
        for (j_B3=0; j_B3 < N; j_B3 += B3) {
            for (k_B3=0; k_B3 < P; k_B3 += B3) {

                // L2 Tiling
                for (i_B2=i_B3; i_B2 < (i_B3 + B3); i_B2 += B2) {
                    for (j_B2=j_B3; j_B2 < (j_B3 + B3); j_B2 += B2) {
                        for (k_B2=k_B3; k_B2 < (k_B3 + B3); k_B2 += B2) {

                            // L1 Tiling
                            for (i_B1=i_B2; i_B1 < (i_B2 + B2); i_B1 += B1) {
                                for (j_B1=j_B2; j_B1 < (j_B2 + B2); j_B1 += B1) {
                                    for (k_B1=k_B2; k_B1 < (k_B2 + B2); k_B1 += B1) {

                                        // inner block
                                        for (i=i_B1; i<(i_B1 + B1); i+=NUM_FLOATS_PER_AVX * LOOP_UNROLL_CONSTANT) {
                                            // for each row of C
                                            for (j=j_B1; j<(j_B1 + B1); j++) {
                                                // for each column of C
                                                
                                                for(i_u = 0; i_u < LOOP_UNROLL_CONSTANT; i_u++){
                                                    sum[i_u] = _mm512_load_ps(&C[j][i_u * NUM_FLOATS_PER_AVX + i]);
                                                }
                                                for (k=k_B1; k<(k_B1 + B1); k++) {
                                                    // dot product of row from A and column from B
                                                    // sum += A[i][k] * B[k][j];
                                                    b = _mm512_set1_ps(B[k][j]);
                                                    for(i_u = 0; i_u < LOOP_UNROLL_CONSTANT; i_u++) {
                                                        sum[i_u] = _mm512_add_ps(sum[i_u],
                                                                _mm512_mul_ps(
                                                                    _mm512_load_ps(&A[k][i_u * NUM_FLOATS_PER_AVX + i]),
                                                                    b));
                                                    }
                                                }
                                                for(i_u = 0; i_u < LOOP_UNROLL_CONSTANT; i_u++){
                                                    _mm512_storeu_ps(&C[j][(i_u * NUM_FLOATS_PER_AVX) + i], sum[i_u]);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
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

    get_args(NUM_ADDL_ARGS, "B1 B2 B3", argc, argv);
    B1 = atoi(argv[ARG_B1_INDEX]);
    B2 = atoi(argv[ARG_B2_INDEX]);
    B3 = atoi(argv[ARG_B3_INDEX]);

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
