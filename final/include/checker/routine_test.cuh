#pragma once

#include <vector>

#include "pgm/pgm.hpp"

void mean_checker(int width, int height, std::vector<PGMData> pmg_list, float* d_mean);

void mean_sub_checker(int width, int height, std::vector<PGMData> pmg_list, float* A, float *d_mean);

void transpose_checker(int width, int height, float* A, float* A_T);

void matmul_checker(uint32_t n, uint32_t m, uint32_t p, float *A, float *B, float *C);

void matmul_checker_s(uint32_t n, uint32_t m, uint32_t p, float *A, float *B, float *C);

void norm_squaredsum_checker(uint32_t m, uint32_t p, float *d_real_eigenvectors, float *d_real_eigenvectors_norm);
