#include <vector>
#include <map>
#include <string>
#include <functional>
#include <chrono>

#include "commons.hpp"
#include "pgm/pgm.hpp"
#include "training/routine.hpp"


PCATraining* find_class(std::string target, std::vector<PGMData> pgm_list, uint32_t num_components) {
    if (target == "jacobi") return new JacobiPCA(pgm_list, num_components);
    if (target == "qr") return new QRPCA(pgm_list, num_components);
    else CERR_CHECK(false, "Unknown target: " + target, ERR_UNKNOWN_TARGET);

    // This is just to avoid warning... unless we want to default to this?
    return new JacobiPCA(pgm_list, num_components);
}

// TODO: make customizable target
void launch_training(std::vector<PGMData> pgm_list, uint32_t num_components,
                     std::string out_file, std::string algorithm) {
    PCATraining* routine = find_class(algorithm, pgm_list,
                                     num_components > pgm_list.size() ? pgm_list.size() : num_components);
    routine->load_matrix();

    auto start = std::chrono::high_resolution_clock::now();
    routine->mean_image();
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Time Mean Image: " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " us" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    routine->compute_covariance();
    stop = std::chrono::high_resolution_clock::now();
    std::cout << "Time Covariance: " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " us" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    routine->find_eigenvectors();
    stop = std::chrono::high_resolution_clock::now();
    std::cout << "Time Find EV: " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " us" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    routine->sort_eigenvectors();
    stop = std::chrono::high_resolution_clock::now();
    std::cout << "Time Sort EV: " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " us" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    routine->post_process();
    stop = std::chrono::high_resolution_clock::now();
    std::cout << "Time Post Process: " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " us" << std::endl;

    routine->save_to_file(out_file);
    delete routine;
}

