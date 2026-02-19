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

void runSSTBenchmarkSetup(
    daisy::SimilaritySearchAlgorithm* search,
    const std::string& dataset_path,
    const std::string& query_path,
    int num_thread,
    size_t k,
    float*& query_out,
    daisy::idx_t*& I_out,
    float*& D_out,
    daisy::idx_t& n_query_out
);

#endif 