#ifndef __TRAINING_ROUTINE_HPP_
#define __TRAINING_ROUTINE_HPP_

#include <vector>

#include "pgm/pgm.hpp"

class PCARoutine {
    public:
        PCARoutine(std::vector<PGMData> pl) :
            pgm_list(pl),
            width(pl.front().col),
            height(pl.front().row),
            num_images(pl.size())
        {}

        void load_matrix();
        void mean_image();
        void compute_covariance();
        virtual void find_eigenvectors()=0;

        virtual ~PCARoutine();
    protected:
        std::vector<PGMData> pgm_list;
        int width;
        int height;
        int num_images;

        float *d_data_temp;
        float *d_data;
        float *d_data_transpose;
        float *d_data_cov;
        void **d_params;
};

class JacobiPCA: public PCARoutine {
    public:
        JacobiPCA(std::vector<PGMData> pl) :
            PCARoutine(pl)
        {};

        void find_eigenvectors();

        ~JacobiPCA() {};
};

#endif  // __TRAINING_ROUTINE_HPP_

