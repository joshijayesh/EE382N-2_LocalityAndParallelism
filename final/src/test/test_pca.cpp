#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <string>
#include <functional>

#include "commons.hpp"
#include "pgm/pgm.hpp"
#include "test/test_pca.hpp"
#include "test/test_routine.hpp"


PCAMatrix parse_matrix(std::ifstream& file) {
    PCAMatrix matrix;
    char toss;

    file >> matrix.n >> toss >> matrix.m;
    matrix.matrix = (float *) malloc(sizeof(float) * matrix.n * matrix.m);
    file.read(&toss, 1);

    for(uint32_t i = 0; i < matrix.n; i += 1) {
        for(uint32_t j = 0; j < matrix.m; j += 1) {
            file.read(reinterpret_cast<char *> (&matrix.matrix[i * matrix.m + j]), sizeof(float));
        }
        file.read(&toss, 1);
    }

    return matrix;
}

PCATextConv parse_input(std::string input) {
    PCATextConv text;
    std::ifstream file(input);
    std::string line;

    while(std::getline(file, line)) {
        if(line == "Mean Image"){
            text.mean = parse_matrix(file);
        }
        if(line == "EigenVectors"){
            text.ev = parse_matrix(file);
        }
        if(line == "Weight Vectors"){
            text.wv = parse_matrix(file);
        }
    }

    file.close();

    return text;
}


void launch_test(std::vector<PGMData> pgm_list, std::vector<Person> persons, std::string input, uint32_t num_components) {
    uint32_t num_train = persons.front().num_train;

    PCATest* routine = new PCATest(pgm_list, num_components > pgm_list.size() ? pgm_list.size() : num_components, num_train);
    PCATextConv text_conv = parse_input(input);
    
    routine->load_matrix(text_conv);
    routine->mean_image();
    routine->find_euclidian();
    routine->find_confidence();
    routine->final_image();

    delete routine;
}

