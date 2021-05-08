#ifndef __TRAINING_ROUTINE_HPP_
#define __TRAINING_ROUTINE_HPP_

#include <vector>

#include "pgm/pgm.hpp"

class PCATraining {
    public:
        PCATraining(std::vector<PGMData> pl, uint32_t num_components) :
            pgm_list(pl),
            width(pl.front().col),
            height(pl.front().row),
            num_images(pl.size()),
            num_components(num_components)
        {}

        void load_matrix();
        void mean_image();
        void compute_covariance();
        virtual void find_eigenvectors()=0;
        void sort_eigenvectors();
        void post_process();
        void save_to_file(std::string);

        virtual ~PCATraining();
    protected:
        std::vector<PGMData> pgm_list;
        int width;
        int height;
        int num_images;
        uint32_t num_components;

        float *d_data_temp;
        float *d_data;
        float *d_data_mean;
        float *d_data_transpose;
        float *d_data_cov;
        float *d_eigenvectors;
        float *d_eigenvalues;
        float *d_eigenvectors_sorted;

        float *d_real_eigenvectors;

        float *d_real_eigenvectors_norm;
        float *d_real_eigenvectors_transpose;
        float *d_results;

        void **d_params;
};

class JacobiPCA: public PCATraining {
    public:
        JacobiPCA(std::vector<PGMData> pl, uint32_t num_components) :
            PCATraining(pl, num_components)
        {};

        void find_eigenvectors();

        ~JacobiPCA() {};
};

#endif  // __TRAINING_ROUTINE_HPP_

