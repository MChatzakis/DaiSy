#include "paramSetup.hpp"

// Define global constants
const char *astro_data = "../data/astronomy.data.len256.size50000.znorm.bin";
const char *astro_query = "../data/astronomy.query.len256.size100.znorm.bin";
const char *astro_name = "AstronomyData_q100";
const char *astro_gt_data = "../tests/groundtruth/Indices/bruteForce_gtFAISS_I_astronomy_len256_size50000_q100_k";
const char *astro_gt_query = "../tests/groundtruth/Distances/bruteForce_gtFAISS_D_astronomy_len256_size50000_q100_k";

const char *random_data = "../data/random.data.randwalk.len96.size200000.znorm.bin";
const char *random_query = "../data/random.query.randwalk.len96.size1000.bin";
const char *random_name = "RandomWalkData_q1000";
const char *random_gt_data = "../tests/groundtruth/Indices/bruteForce_gtFAISS_I_random_len96_size200000_q1000_k";
const char *random_gt_query = "../tests/groundtruth/Distances/bruteForce_gtFAISS_D_random_len96_size200000_q1000_k";

// DTW-specific groundtruth paths
const char *astro_gt_dtw_data = "../tests/groundtruth/Indices/bruteForce_gtDTW_I_astronomy_len256_size50000_q100_k";
const char *astro_gt_dtw_query = "../tests/groundtruth/Distances/bruteForce_gtDTW_D_astronomy_len256_size50000_q100_k";
const char *random_gt_dtw_data = "../tests/groundtruth/Indices/bruteForce_gtDTW_I_random_len96_size200000_q1000_k";
const char *random_gt_dtw_query = "../tests/groundtruth/Distances/bruteForce_gtDTW_D_random_len96_size200000_q1000_k";

std::vector<SSTestConfig> generate_configs(
    const char *name,
    const char *data,
    const char *query,
    const char *gt_data,
    const char *gt_query)
{
    std::vector<SSTestConfig> configs;
    for (int threads : {1, 4, 8})
    {
        for (int k : {1, 10, 100})
        {
            configs.push_back({name, data, query, gt_data, gt_query, threads, k});
        }
    }
    return configs;
}

const std::vector<SSTestConfig> test_configs = []
{
    std::vector<SSTestConfig> configs;

    auto astro_configs = generate_configs(astro_name, astro_data, astro_query, astro_gt_data, astro_gt_query);
    auto random_configs = generate_configs(random_name, random_data, random_query, random_gt_data, random_gt_query);

    configs.insert(configs.end(), astro_configs.begin(), astro_configs.end());
    configs.insert(configs.end(), random_configs.begin(), random_configs.end());

    return configs;
}();

const std::vector<SSTestConfig> test_configs_dtw = []
{
    std::vector<SSTestConfig> configs;

    auto astro_configs_dtw = generate_configs(astro_name, astro_data, astro_query, astro_gt_dtw_data, astro_gt_dtw_query);
    auto random_configs_dtw = generate_configs(random_name, random_data, random_query, random_gt_dtw_data, random_gt_dtw_query);

    configs.insert(configs.end(), astro_configs_dtw.begin(), astro_configs_dtw.end());
    configs.insert(configs.end(), random_configs_dtw.begin(), random_configs_dtw.end());

    return configs;
}();
