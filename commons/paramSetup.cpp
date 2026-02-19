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
            configs.push_back({name, data, query, gt_data, gt_query, threads, k, 0});
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
            configs.push_back({name, data, query, gt_data, gt_query, threads, k, 0});
        }
    }
    return configs;
}

static const char *seismic_data = "/mnt/hddhelp/mchatzakis/similarity-search-datasets/data_size100M_seismic_len256_znorm.bin";
static const char *seismic_query = "/mnt/hddhelp/mchatzakis/similarity-search-datasets/queries_ctrl10000_seismic_len256_znorm.bin";
static const char *deep100m_data = "/mnt/hddhelp/mchatzakis/datasets/processed/DEEP100M/base.100M.fvecs";
static const char *deep100m_query = "/mnt/hddhelp/mchatzakis/datasets/processed/DEEP100M/query.10K.fvecs";

// Solo DEEP e Seismic, q=100, k=1,10,100,1000 (indici 0-3 DEEP, 4-7 Seismic)
const std::vector<SSTestConfig> test_configs_deep_seismic_astro270m = []
{
    std::vector<SSTestConfig> configs;
    auto push = [&](const char* name, const char* data, const char* query, int k_val, int q_limit) {
        configs.push_back({name, data, query, "", "", 48, k_val, q_limit});
    };
    for (int k_val : {1, 10, 100, 1000}) {
        push("DEEP100M", deep100m_data, deep100m_query, k_val, 100);
    }
    for (int k_val : {1, 10, 100, 1000}) {
        push("Seismic100M", seismic_data, seismic_query, k_val, 100);
    }
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
            configs.push_back({random_name, random_data, random_query, random_gt_data, random_gt_query, threads, k, 0});
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

