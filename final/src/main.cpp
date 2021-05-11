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
#include "test/test_pca.hpp"


Person make_person(std::string path, uint32_t num_train, uint32_t num_test) {
    Person m_p = {path, num_train, num_test};
    return m_p;
}

uint32_t cnt_pgms(std::string src) {
    uint32_t cnt = 0;
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

        if(name.size() > 4 && name.substr(name.size() - 4) == ".pgm") {
            cnt += 1;
        }
    }
    return cnt;
}


bool parse(std::string src, struct stat s, std::vector<PGMData> &pgm_list, std::vector<Person> &pgm_ordering,
           float test_train_split, std::string target) {
    CERR_CHECK(s.st_mode & (S_IFREG | S_IFDIR), "Unknown file " + src, ERR_FAILED_OPEN_FILE);

    if(s.st_mode & S_IFDIR) {
        struct dirent *entry;
        struct stat s_new;
        DIR *dp;

        dp = opendir(src.c_str());
        CERR_CHECK(dp != NULL, "Unable to open dir " + src, ERR_FAILED_OPEN_FILE);

        uint32_t cnt = 0;
        uint32_t total_pgms = cnt_pgms(src);
        uint32_t num_train = 0;
        uint32_t num_test = 0;

        // std::cout << src << " cnt = " << total_pgms << std::endl;

        while ((entry = readdir(dp))) {
            std::string name = src + "/" + entry->d_name;
            std::string basename = entry->d_name;
            if(basename == "." || basename == "..") continue;
            CERR_CHECK(stat(name.c_str(), &s_new) == 0, "Unable to stat " + name, ERR_FAILED_OPEN_FILE);
            
            if(target == "train") {
                parse(name, s_new, pgm_list, pgm_ordering, test_train_split, target);
                num_train += 1;

                if(total_pgms != 0 && ((float) (cnt + 1) / total_pgms) >= test_train_split)
                    break;
            } else {
                if(total_pgms == 0 || ((float) (cnt) / total_pgms) >= test_train_split) {
                    parse(name, s_new, pgm_list, pgm_ordering, test_train_split, target);
                    num_test += 1;
                } else
                    num_train += 1;
            }
            cnt += 1;
        }

        closedir(dp);

        if(total_pgms != 0) {
            std::cout << src << std::endl;
            pgm_ordering.push_back(make_person(src, num_train, num_test));
        }

        return false;
    } else if (s.st_mode & S_IFREG) {
        std::cout << src << std::endl;
        if(src.size() > 4 && src.substr(src.size() - 4) == ".pgm") {
            PGMData pgm_data = read_PGM(src);
            pgm_list.push_back(pgm_data);

            return true;
        }
        return false;
    }
    return false;
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

void start(std::string src, std::string dest, std::string target, std::string input, float test_train_split,
          uint32_t num_components, std::string algorithm) {
    struct stat s;
    std::vector<PGMData> pgm_list = {};
    std::vector<Person> pgm_ordering = {};

    CERR_CHECK(stat(src.c_str(), &s) == 0, "Unable to stat " + src, ERR_FAILED_OPEN_FILE);
    CERR_CHECK(s.st_mode & S_IFDIR, "SRC needs to be a directory!", ERR_FAILED_OPEN_FILE);

    parse(src, s, pgm_list, pgm_ordering, test_train_split, target);

    std::cout << "Number of PGMs found " << pgm_list.size() << std::endl;

    verify(pgm_list);

    if(target == "train")
        launch_training(pgm_list, num_components, dest, algorithm);
    else
        launch_test(pgm_list, pgm_ordering, input, num_components);
}


int main(int argc, char* argv[]) {
    std::string src;
    std::string dest;
    std::string target;
    std::string input;
    std::string algorithm;
    float test_train_split;
    int num_components;
    cxxopts::Options options("final", "Face recognition on GPU using PCA");

    options.add_options()
        ("s,src", "Main directory containing .pgm files", cxxopts::value<std::string>())
        ("d,dest", "Dest file containing for output", cxxopts::value<std::string>())
        ("i,input", "Input file for test (undefined for train)", cxxopts::value<std::string>()->default_value("jumble"))
        ("k,num_components", "Number of components to compute", cxxopts::value<int>()->default_value("100000"))
        ("t,target", "Choose target as 'test' or 'train'", cxxopts::value<std::string>()->default_value("train"))
        ("a,algorithm", "Choose algorithm to run, default jacobi", cxxopts::value<std::string>()->default_value("jacobi"))
        ("m,train_test_split", "Specify train-test split", cxxopts::value<int>()->default_value("100"))
        ("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if(result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(ERR_INVALID_ARG);
    }

    try {
        src = result["src"].as<std::string>();
        dest = result["dest"].as<std::string>();
    } catch (cxxopts::option_has_no_value_exception) {
        std::cout << "Parse Error: " << "Src & Dest is required!" << std::endl << std::endl;
        std::cout << options.help() << std::endl;
        exit(ERR_INVALID_ARG);
    }
    target = result["target"].as<std::string>();
    CERR_CHECK(target == "train" || target == "test", "target must be either 'train' or 'test'", ERR_INVALID_ARG);

    test_train_split = (float) (result["train_test_split"].as<int>()) / 100.0;
    CERR_CHECK(0 <= test_train_split && 1.0 >= test_train_split, "test_train_split must be within [0, 1.0]", ERR_INVALID_ARG);

    algorithm = result["algorithm"].as<std::string>();
    CERR_CHECK(algorithm == "jacobi" || algorithm == "qr", "algorithm must be either 'jacobi' or 'qr'", ERR_INVALID_ARG);

    input = result["input"].as<std::string>();

    num_components = result["num_components"].as<int>();

    start(src, dest, target, input, test_train_split, (uint32_t) num_components, algorithm);
}

