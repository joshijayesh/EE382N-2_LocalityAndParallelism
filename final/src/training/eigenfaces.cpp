#include <vector>
#include <map>
#include <string>
#include <functional>

#include "commons.hpp"
#include "pgm/pgm.hpp"
#include "training/routine.hpp"


PCARoutine* find_class(std::string target, std::vector<PGMData> pgm_list, uint32_t num_components) {
    if (target == "jacobi") return new JacobiPCA(pgm_list, num_components);
    else CERR_CHECK(false, "Unknown target: " + target, ERR_UNKNOWN_TARGET);

    // This is just to avoid warning... unless we want to default to this?
    return new JacobiPCA(pgm_list, num_components);
}

// TODO: make customizable target
void launch_training(std::vector<PGMData> pgm_list, uint32_t num_components) {
    PCARoutine* routine = find_class("jacobi", pgm_list,
                                     num_components > pgm_list.size() ? pgm_list.size() : num_components);
    routine->load_matrix();
    routine->mean_image();
    /*
    routine->compute_covariance();
    routine->find_eigenvectors();
    routine->sort_eigenvectors();
    routine->post_process();
    */
    delete routine;
}

