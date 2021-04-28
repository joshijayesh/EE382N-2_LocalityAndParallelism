#ifndef __PGM_PGM_H_
#define __PGM_PGM_H_

#include <string>

typedef struct _PGMData {
    int row;
    int col;
    int max_gray;
    float* matrix;
} PGMData;

PGMData read_PGM(std::string file_name);

#endif // __PGM_PGM_H

