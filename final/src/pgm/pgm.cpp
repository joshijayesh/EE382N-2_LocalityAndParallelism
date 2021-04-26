#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <stdio.h>

#include "pgm/pgm.hpp"
#include "commons.hpp"


void SkipComments(std::ifstream &fp) {
    char ch;
    char line[100];

    while(true) {
        fp.get(ch);
        if(ch != EOF && isspace(ch)) break;
    }

    if (ch == '#') {
        fp.get(line, sizeof(line));
        SkipComments(fp);
    } else {
        fp.seekg (-1, fp.cur);
    }
}

PGMData read_PGM(std::string file_name) {
    char version[3];
    int i, j;
    char lo;
    PGMData data;

    std::ifstream pgm_file(file_name);

    // Read Version
    pgm_file.get(version, sizeof(version));
    CERR_CHECK(std::string(version) == "P5", "Unknown magic value on file " + file_name, ERR_FAILED_OPEN_PGM);
    SkipComments(pgm_file);

    // Read Row/Col
    pgm_file >> data.col;
    SkipComments(pgm_file);
    pgm_file >> data.row;
    SkipComments(pgm_file);
    pgm_file >> data.max_gray;
    CERR_CHECK(data.max_gray == 255, "Max Gray is not 255! Unsupported at the moment", ERR_UNSUPPORTED_GRAY);

    char c;
    pgm_file.get(c);

    // Read matrix
    data.matrix = (uint8_t *) malloc(sizeof(uint8_t) * (data.row * data.col));
    for(i = 0; i < data.row; i += 1) {
        for(j = 0; j < data.col; j += 1) {
            pgm_file.get(lo);
            data.matrix[(i * data.col) + j] = lo;
        }
    }

    // fclose(pgm_file);
    pgm_file.close();
    return data;
}

/*
int main(int argc, const char *argv[]) {
    PGMData tmp = read_PGM(argv[1]);

    std::cout << tmp.row << "x" << tmp.col << std::endl;
    printf("%x\n%x\n%x\n%x\n", tmp.matrix[0], tmp.matrix[1], tmp.matrix[2], tmp.matrix[3]);
    std::cout << "Complete" << std::endl;
}*/

