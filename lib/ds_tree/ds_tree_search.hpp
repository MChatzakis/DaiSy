#pragma once

#include "ds_tree_node.hpp"
#include "../algos/DataSource.hpp"

#include <vector>
#include <queue>
#include <utility>
#include <cfloat>
#include <cstdio>

namespace daisy {

struct HerculesKnnResult {
    float  distance   = FLT_MAX;
    idx_t  series_idx = 0;
};

// min-heap: smallest lower bound pops first
using HerculesPQ = std::priority_queue<
    std::pair<float, HerculesNode *>,
    std::vector<std::pair<float, HerculesNode *>>,
    std::greater<std::pair<float, HerculesNode *>>>;

float l2sq(const float *a, const float *b, int dim, float bound);

void knn_bounded_insert(std::vector<HerculesKnnResult> &knn, idx_t k,
                        idx_t series_idx, float dist);

float calculate_node_min_distance(HerculesNode *node, const float *query);

void approximate_knn_search(const float *query, int dim,
                             HerculesPQ &pq, FILE *raw_file,
                             std::vector<float> &ts_buf,
                             std::vector<HerculesKnnResult> &knn,
                             idx_t k, int max_leaves);

} // namespace daisy
