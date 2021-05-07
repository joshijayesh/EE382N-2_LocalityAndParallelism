#pragma once

#include <string>
#include <stdlib.h>
#include <iostream>

typedef struct devConsts {
    int width;
    int height;
    int n;
    int m;
    int image_size;
    int num_images;

    float* data;
    float* A;
    float* A_t;
} DeviceConstants;



