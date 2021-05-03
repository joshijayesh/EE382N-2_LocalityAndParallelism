#include <string>
#include <iostream>
#include <stdio.h>
#include <sys/stat.h>
#include <dirent.h>
#include <vector>

#include "commons.hpp"
#include "pgm/pgm.hpp"
#include "cxxopts.hpp"
#include "training/eigenfaces.hpp"


void parse(std::string src, struct stat s, std::vector<PGMData> &pgm_list, uint32_t max_size) {
    CERR_CHECK(s.st_mode & (S_IFREG | S_IFDIR), "Unknown file " + src, ERR_FAILED_OPEN_FILE);

    if(s.st_mode & S_IFDIR) {
        struct dirent *entry;
        struct stat s_new;
        DIR *dp;

        dp = opendir(src.c_str());
        CERR_CHECK(dp != NULL, "Unable to open dir " + src, ERR_FAILED_OPEN_FILE);

        while ((entry = readdir(dp))) {
            std::string name = src + "/" + entry->d_name;
            std::string basename = entry->d_name;
            if(basename == "." || basename == "..") continue;
            CERR_CHECK(stat(name.c_str(), &s_new) == 0, "Unable to stat " + name, ERR_FAILED_OPEN_FILE);
            parse(name, s_new, pgm_list, max_size);

            if(pgm_list.size() > max_size) return;
        }

        closedir(dp);
    } else if (s.st_mode & S_IFREG) {
        if(src.size() > 4 && src.substr(src.size() - 4) == ".pgm") {
            PGMData pgm_data = read_PGM(src);
            pgm_list.push_back(pgm_data);
        }
    }
}

void verify(std::vector<PGMData> &pgm_list) {
    CERR_CHECK(pgm_list.size() > 0, "Found no PGMs! Need at least something to work with :/", ERR_NO_IMGES_FOUND);
    int row = 0;
    int col = 0;

    for(PGMData img : pgm_list) {
        if(row == 0) {
            row = img.row;
            col = img.col;
        } else {
            CERR_CHECK(img.row == row, "Rows Mismatch " + std::to_string(row) + " vs " + std::to_string(img.row),
                       ERR_IMG_DIM_MISMATCH);
            CERR_CHECK(img.col == col, "Cols Mismatch " + std::to_string(col) + " vs " + std::to_string(img.col),
                       ERR_IMG_DIM_MISMATCH);
        }
    }
}

void start(std::string src, uint32_t max_size, uint32_t num_components) {
    struct stat s;
    std::vector<PGMData> pgm_list = {};

    CERR_CHECK(stat(src.c_str(), &s) == 0, "Unable to stat " + src, ERR_FAILED_OPEN_FILE);
    CERR_CHECK(s.st_mode & S_IFDIR, "SRC needs to be a directory!", ERR_FAILED_OPEN_FILE);

    parse(src, s, pgm_list, max_size);

    std::cout << "Number of PGMs found " << pgm_list.size() << std::endl;

    verify(pgm_list);

    launch_training(pgm_list, num_components);
}


int main(int argc, char* argv[]) {
    std::string src;
    int max_pgms;
    int num_components;
    cxxopts::Options options("final", "Face recognition on GPU using PCA");

    options.add_options()
        ("s,src", "Main directory containing .pgm files", cxxopts::value<std::string>())
        ("n,num_pgms", "Max number of files to load", cxxopts::value<int>()->default_value("100000"))
        ("k,num_components", "Number of components to compute", cxxopts::value<int>()->default_value("100000"))
        ("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if(result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    try {
        src = result["src"].as<std::string>();
    } catch (cxxopts::option_has_no_value_exception) {
        std::cout << "Parse Error: " << "Src is required!" << std::endl << std::endl;
        std::cout << options.help() << std::endl;
        exit(0);
    }

    max_pgms = result["num_pgms"].as<int>();
    num_components = result["num_components"].as<int>();

    start(src, (uint32_t) max_pgms, (uint32_t) num_components);
}

