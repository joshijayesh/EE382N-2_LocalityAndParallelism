#ifndef BASIC_MATMUL_H
#define BASIC_MATMUL_H

#define ARG_N_INDEX 1
#define ARG_M_INDEX 2
#define ARG_P_INDEX 3

#define NUM_ARGS 4

#define ERROR_INVALID_ARGS 1

int N, M, P;

void usage() {
    fprintf(stderr, "usage: basicmatmul N M P\n");
    exit(ERROR_INVALID_ARGS);
}

void get_args(int argc, char *argv[]) {
  if(argc != NUM_ARGS) {
      usage();
  }

  N = atoi(argv[ARG_N_INDEX]);
  M = atoi(argv[ARG_M_INDEX]);
  P = atoi(argv[ARG_P_INDEX]);
}

#endif

