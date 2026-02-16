#ifndef PARAM_SETUP_HPP
#define PARAM_SETUP_HPP

#include <vector>
#include "test_bm_utils.hpp"

extern const char *astro_data;
extern const char *astro_query;
extern const char *astro_name;
extern const char *astro_gt_data;
extern const char *astro_gt_query;

extern const char *random_data;
extern const char *random_query;
extern const char *random_name;
extern const char *random_gt_data;
extern const char *random_gt_query;

extern const char *astro_gt_dtw_data;
extern const char *astro_gt_dtw_query;
extern const char *random_gt_dtw_data;
extern const char *random_gt_dtw_query;

std::vector<SSTestConfig> generate_configs(
    const char *name,
    const char *data,
    const char *query,
    const char *gt_data,
    const char *gt_query);

std::vector<SSTestConfig> generate_configs_custom(
    const char *name,
    const char *data,
    const char *query,
    const char *gt_data,
    const char *gt_query,
    std::vector<int> thread_counts,
    std::vector<int> k_values = {1, 10, 100});

extern const std::vector<SSTestConfig> test_configs;
extern const std::vector<SSTestConfig> test_configs_large;
extern const std::vector<SSTestConfig> test_configs_dtw;
extern const std::vector<SSTestConfig> test_configs_random_light;
extern const std::vector<SSTestConfig> test_configs_random_only;
extern const std::vector<SSTestConfig> test_configs_astro_only;

#endif
