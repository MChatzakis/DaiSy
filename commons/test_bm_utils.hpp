#ifndef TEST_BM_UTILS_HPP
#define TEST_BM_UTILS_HPP

#include <string>
#include "../lib/algos/SimilaritySearchAlgorithm.hpp"

bool isclose(daisy::idx_t a, daisy::idx_t b);

bool isclose(double a, double b, double rtol = 1e-5, double atol = 1e-8);

float *readFile(const std::string &filepath, size_t &outSize);

std::string pathToFilename(std::string path);

bool parseFilenameForConfig(const std::string &filename,
                            const std::string &prefix,
                            daisy::idx_t &dim,
                            daisy::idx_t &n_database,
                            daisy::idx_t &n_query,
                            daisy::idx_t &k);

void compareWithGroundTruth(const std::string &pathI,
                            const std::string &pathD,
                            const daisy::idx_t *I,
                            float *D,
                            daisy::idx_t n_query,
                            daisy::idx_t k,
                            double rtol = 1e-2,
                            double atol = 1e-8);

void assert_eq(size_t a, size_t b, const std::string &msg);

void add_failure(const std::string &msg);

struct SSTestConfig
{
    std::string name;
    std::string dataset_path;
    std::string query_path;
    std::string gt_I_prefix;
    std::string gt_D_prefix;
    int thread_count;
    int k_value;
    int query_limit = 0;  // 0 = use full query file; >0 = use first query_limit queries
};

#endif 