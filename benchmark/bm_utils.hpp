#ifndef BM_UTILS_HPP
#define BM_UTILS_HPP

#include <string>
#include "../lib/algos/SimilaritySearchAlgorithm.hpp"
#include "../commons/test_bm_utils.hpp" 
#include "../commons/paramSetup.hpp"

void runSSTBenchmark(
    daisy::SimilaritySearchAlgorithm* search,
    const std::string& dataset_path,
    const std::string& query_path,
    int num_thread,
    size_t k
);

#endif 