#ifndef __TRAINING_EIGENFACES_HPP_
#define __TRAINING_EIGENFACES_HPP_

#include <vector>
#include <string>

#include "pgm/pgm.hpp"


void launch_training(std::vector<PGMData>, std::vector<std::string>, uint32_t, std::string);

#endif //__TRAINING_EIGENFACES_HPP_

