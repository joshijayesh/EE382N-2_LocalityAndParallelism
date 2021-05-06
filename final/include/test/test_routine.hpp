#pragma once

#include <vector>

#include "pgm/pgm.hpp"

class PCATest {
    public:
        PCATest(std::vector <PGMData> pl, uint32_t num_components):
            pgm_list(pl),
            width(pl.front().col),
            height(pl.front().row),
            num_images(pl.size()),
            num_components(num_components)
        {}

        void load_matrix();
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
};

