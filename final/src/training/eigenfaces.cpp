#include <vector>

#include "pgm/pgm.hpp"
#include "training/routine.hpp"


void launch_training(std::vector<PGMData> pgm_list) {
    PCARoutine routine (pgm_list);
    routine.load_matrix();
}

