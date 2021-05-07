#pragma once

#include <vector>

#include "pgm/pgm.hpp"

typedef struct matrix {
    uint32_t n;
    uint32_t m;
    float *matrix;
} PCAMatrix;

typedef struct pCATextConv {
    PCAMatrix mean;
    PCAMatrix ev;
    PCAMatrix wv;
} PCATextConv;

class PCATest {
    public:
        PCATest(std::vector<PGMData> pl, uint32_t num_components):
            pgm_list(pl),
            width(pl.front().col),
            height(pl.front().row),
            num_images(pl.size()),
            num_components(num_components)
        {}

        void load_matrix(PCATextConv);
        void mean_image();
        void find_euclidian();
        void find_confidence();
        void final_image();

        ~PCATest();

    protected:
        std::vector<PGMData> pgm_list;
        int width;
        int height;
        int num_images;
        uint32_t num_components;

        float *d_mean;
        float *d_train_ev;
        float *d_train_wv;
        float *d_data;
        float *d_data_temp;
};

