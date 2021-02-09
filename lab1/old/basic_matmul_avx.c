#include <stdlib.h>
#include <stdio.h>
#include <matmul_defaults.h>
#include <immintrin.h>

#define NUM_ADDL_ARGS 0

extern int N, M, P;

// calculate C = AxB
void matmul(float *A, float *B, float *C){
    int     i, i_u;
    int     j;
    int     k;
    __m512  b;
    __m512 sum[4] __attribute__ ((aligned (AVX_ALIGNMENT)));

    // Expect N > 16
    int LOOP_UNROLL_CONSTANT = N >= 64 ? 4 : N / 16;
    
    for (i=0; i<M; i += NUM_FLOATS_PER_AVX * LOOP_UNROLL_CONSTANT) {
        // for each row of C
        for (j=0; j<N; j++) {
            // for each column of C
            for(i_u = 0; i_u < LOOP_UNROLL_CONSTANT; i_u++){
                sum[i_u] = _mm512_set1_ps(0.0f);
            }

            for (k=0; k<P; k++) {
                // dot product of row from A and column from B
                // sum += A[i + (k * N)]*B[k + (j * P)];
                b = _mm512_set1_ps(B[k + (j * P)]);
                for(i_u = 0; i_u < LOOP_UNROLL_CONSTANT; i_u++) {
                    sum[i_u] = _mm512_add_ps(sum[i_u],
                                _mm512_mul_ps(
                                    _mm512_load_ps(&A[(i + i_u * NUM_FLOATS_PER_AVX) + (k * N)]),
                                    b));
                }
            }

            for(i_u = 0; i_u < LOOP_UNROLL_CONSTANT; i_u++){
                _mm512_store_ps(&C[(i + i_u * NUM_FLOATS_PER_AVX) + (j *P)], sum[i_u]);
            }
        }
    }
}

// function to allocate a matrix on the heap
// creates an mXn matrix and returns the pointer.
//
// the matrices are in row-major order.
void create_matrix(float** A, int m, int n) {
    *A = (float*) aligned_alloc(AVX_ALIGNMENT, m * n * sizeof(float));
}

int main(int argc, char *argv[]) {
    float* A;
    float* B;
    float* C;

    get_args(NUM_ADDL_ARGS, "", argc, argv);

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

