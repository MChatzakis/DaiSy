#ifndef PARAM_SETUP_HPP
#define PARAM_SETUP_HPP

#include <vector>
#include "test_bm_utils.hpp"

// Declare global paths
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

// Function declaration
std::vector<SSTestConfig> generate_configs(
    const char* name,
    const char* data,
    const char* query,
    const char* gt_data,
    const char* gt_query
);

// Test config list declaration
extern const std::vector<SSTestConfig> test_configs;

#endif // PARAM_SETUP_HPP
