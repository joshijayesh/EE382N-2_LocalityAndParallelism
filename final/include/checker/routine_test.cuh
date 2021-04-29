#pragma once

#include <vector>

#include "pgm/pgm.hpp"

void mean_checker(int width, int height, std::vector<PGMData> pmg_list, float* d_mean);

void transpose_checker(int width, int height, float* A, float* A_T);
