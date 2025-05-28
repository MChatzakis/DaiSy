#ifndef BM_UTILS_HPP
#define BM_UTILS_HPP

#include <string>
#include "../lib/algos/SimilaritySearchAlgorithm.hpp"
#include "../commons/test_bm_utils.hpp" 
#include "../commons/paramSetup.hpp"

/**
 * @brief Run brute force search benchmark: load data, build index, search.
 *        No correctness checks or assertions.
 * 
 * @param search Pointer to similarity search algorithm instance
 * @param dataset_path Path to database binary
 * @param query_path Path to query binary
 * @param num_thread Number of threads
 * @param k Number of neighbors to search for
 */
void runSSTBenchmark(
    diNoLib::SimilaritySearchAlgorithm* search,
    const std::string& dataset_path,
    const std::string& query_path,
    int num_thread,
    size_t k
);

#endif // BM_UTILS_HPP