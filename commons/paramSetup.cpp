#include "paramSetup.hpp"
#include <string>
#include <cstring>
#include <vector>

#ifndef PROJECT_ROOT_DIR
#error "PROJECT_ROOT_DIR must be defined by CMake"
#endif

// Helper function to construct absolute paths from project root
static std::string make_absolute_path(const char* relative_path) {
    std::string root(PROJECT_ROOT_DIR);
    std::string rel(relative_path);
    // Remove leading "../" from relative path
    while (rel.substr(0, 3) == "../") {
        rel = rel.substr(3);
    }
    return root + "/" + rel;
}

// Store paths as static strings to ensure they persist
static std::string astro_data_str = make_absolute_path("data/astronomy.data.len256.size50000.znorm.bin");
static std::string astro_query_str = make_absolute_path("data/astronomy.query.len256.size100.znorm.bin");
static std::string astro_gt_data_str = make_absolute_path("tests/groundtruth/Indices/bruteForce_gtFAISS_I_astronomy_len256_size50000_q100_k");
static std::string astro_gt_query_str = make_absolute_path("tests/groundtruth/Distances/bruteForce_gtFAISS_D_astronomy_len256_size50000_q100_k");

static std::string random_data_str = make_absolute_path("data/random.data.randwalk.len96.size200000.znorm.bin");
static std::string random_query_str = make_absolute_path("data/random.query.randwalk.len96.size1000.bin");
static std::string random_gt_data_str = make_absolute_path("tests/groundtruth/Indices/bruteForce_gtFAISS_I_random_len96_size200000_q1000_k");
static std::string random_gt_query_str = make_absolute_path("tests/groundtruth/Distances/bruteForce_gtFAISS_D_random_len96_size200000_q1000_k");

static std::string astro_gt_dtw_data_str = make_absolute_path("tests/groundtruth/Indices/bruteForce_gtDTW_I_astronomy_len256_size50000_q100_k");
static std::string astro_gt_dtw_query_str = make_absolute_path("tests/groundtruth/Distances/bruteForce_gtDTW_D_astronomy_len256_size50000_q100_k");
static std::string random_gt_dtw_data_str = make_absolute_path("tests/groundtruth/Indices/bruteForce_gtDTW_I_random_len96_size200000_q1000_k");
static std::string random_gt_dtw_query_str = make_absolute_path("tests/groundtruth/Distances/bruteForce_gtDTW_D_random_len96_size200000_q1000_k");

const char *astro_data = astro_data_str.c_str();
const char *astro_query = astro_query_str.c_str();
const char *astro_name = "AstronomyData_q100";
const char *astro_gt_data = astro_gt_data_str.c_str();
const char *astro_gt_query = astro_gt_query_str.c_str();

const char *random_data = random_data_str.c_str();
const char *random_query = random_query_str.c_str();
const char *random_name = "RandomWalkData_q1000";
const char *random_gt_data = random_gt_data_str.c_str();
const char *random_gt_query = random_gt_query_str.c_str();

const char *astro_gt_dtw_data = astro_gt_dtw_data_str.c_str();
const char *astro_gt_dtw_query = astro_gt_dtw_query_str.c_str();
const char *random_gt_dtw_data = random_gt_dtw_data_str.c_str();
const char *random_gt_dtw_query = random_gt_dtw_query_str.c_str();

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

std::vector<SSTestConfig> generate_configs_custom(
    const char *name,
    const char *data,
    const char *query,
    const char *gt_data,
    const char *gt_query,
    std::vector<int> thread_counts,
    std::vector<int> k_values)
{
    std::vector<SSTestConfig> configs;
    for (int threads : thread_counts)
    {
        for (int k : k_values)
        {
            configs.push_back({name, data, query, gt_data, gt_query, threads, k});
        }
    }
    return configs;
}

// Large datasets path (no ground truth - use empty strings)
static const char *seismic_data = "/mnt/hddhelp/mchatzakis/similarity-search-datasets/data_size100M_seismic_len256_znorm.bin";
static const char *seismic_query = "/mnt/hddhelp/mchatzakis/similarity-search-datasets/queries_ctrl100_seismic_len256_znorm.bin";
static const char *astro270M_data = "/mnt/hddhelp/mchatzakis/similarity-search-datasets/data_size270M_astronomy_len256_znorm.bin";
static const char *astro270M_query = "/mnt/hddhelp/mchatzakis/similarity-search-datasets/queries_ctrl100_astronomy_len256_znorm.bin";

const std::vector<SSTestConfig> test_configs_large = []
{
    std::vector<SSTestConfig> configs;
    auto seismic_configs = generate_configs_custom(
        "Seismic100M", seismic_data, seismic_query, "", "", {48}, {100});
    auto astro_configs = generate_configs_custom(
        "Astronomy270M", astro270M_data, astro270M_query, "", "", {64}, {10});
    configs.insert(configs.end(), seismic_configs.begin(), seismic_configs.end());
    configs.insert(configs.end(), astro_configs.begin(), astro_configs.end());
    return configs;
}();

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

const std::vector<SSTestConfig> test_configs_random_light = []
{
    std::vector<SSTestConfig> configs;
    for (int threads : {1})
    {
        for (int k : {100})
        {
            configs.push_back({random_name, random_data, random_query, random_gt_data, random_gt_query, threads, k});
        }
    }
    return configs;
}();

const std::vector<SSTestConfig> test_configs_astro_only = []
{
    std::vector<SSTestConfig> configs;
    auto astro_configs = generate_configs(astro_name, astro_data, astro_query, astro_gt_data, astro_gt_query);
    configs.insert(configs.end(), astro_configs.begin(), astro_configs.end());
    return configs;
}();

