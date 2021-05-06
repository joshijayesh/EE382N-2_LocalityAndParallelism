#include <vector>
#include <map>
#include <string>
#include <functional>

#include "commons.hpp"
#include "pgm/pgm.hpp"
#include "test/test_pca.hpp"
#include "test/test_routine.hpp"


void launch_test(std::vector<PGMData> pgm_list, std::string input, uint32_t num_components) {
    PCATest* routine = new PCATest(pgm_list, num_components);
    
    routine->load_matrix();
    routine->mean_image();
    routine->find_euclidian();
    routine->find_confidence();
    routine->final_image();

    delete routine;
}

