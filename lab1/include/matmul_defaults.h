#ifndef BASIC_MATMUL_H
#define BASIC_MATMUL_H

#define ARG_N_INDEX 1
#define ARG_M_INDEX 2
#define ARG_P_INDEX 3

#define NUM_ARGS 4

#define ERROR_INVALID_ARGS 1

#define AVX_LEVEL 512
#define NUM_FLOATS_PER_AVX AVX_LEVEL / 32  // 32 = size of float in bits
#define AVX_ALIGNMENT NUM_FLOATS_PER_AVX * 4  // 4 = size of float in bytes

int N, M, P;

void usage(const char* addl_str) {
    fprintf(stderr, "usage: basicmatmul N M P ");
    fprintf(stderr, addl_str);
    fprintf(stderr, "\n");
    exit(ERROR_INVALID_ARGS);
}

void get_args(int additional_args, const char* addl_str, int argc, char *argv[]) {
  if(argc != NUM_ARGS + additional_args) {
      usage(addl_str);
  }

  N = atoi(argv[ARG_N_INDEX]);
  M = atoi(argv[ARG_M_INDEX]);
  P = atoi(argv[ARG_P_INDEX]);
}

#endif

