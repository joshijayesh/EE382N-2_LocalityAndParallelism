#include <vector>
#include <map>
#include <string>
#include <functional>

#include "commons.hpp"
#include "pgm/pgm.hpp"
#include "training/routine.hpp"


PCARoutine* find_class(std::string target, std::vector<PGMData> pgm_list) {
    if (target == "jacobi") return new JacobiPCA(pgm_list);
    else CERR_CHECK(false, "Unknown target: " + target, ERR_UNKNOWN_TARGET);

    // This is just to avoid warning... unless we want to default to this?
    return new JacobiPCA(pgm_list);
}

// TODO: make customizable target
void launch_training(std::vector<PGMData> pgm_list) {
    PCARoutine* routine = find_class("jacobi", pgm_list);
    routine->load_matrix();
    routine->mean_image();
    routine->compute_covariance();
    delete routine;
}

