#pragma once

#include "node.hpp"
#include "../DataSource.hpp"
#include "../../isax/iSAXTypes.hpp"

#include <vector>
#include <queue>
#include <utility>
#include <cstdio>

namespace daisy {

struct HerculesKnnResult {
    float distance = FLT_MAX;
    idx_t series_idx = 0;
};

// min-heap: smallest lower bound pops first
using HerculesPQ = std::priority_queue<
    std::pair<float, HerculesNode *>,
    std::vector<std::pair<float, HerculesNode *> >,
    std::greater<std::pair<float, HerculesNode *> > >;
// node + segment inizialization
HerculesNode *create_child_node(HerculesNode *parent);
bool node_init_segments(HerculesNode *node, const std::vector<int>& split_points);
SegmentSketch *init_segment_sketches(int num_child_segments);

// stats + segment utilities
float calc_mean(float *series, int start, int end);
bool calc_split_points(std::vector<int>& points, int ts_length, int segment_size);
void calc_mean_stdev(float *series, int start, int end, float *mean, float *stdev);
int get_segment_start(const std::vector<int>& points, int idx);
int get_segment_end(const std::vector<int>& points, int idx);
int get_segment_length(const std::vector<int>& points, int i);

// PLA-based lower bounds
bool series_segment_sketch_do_sketch(SegmentSketch *series_segment_sketch,
                                     float *series, int fromIdx, int toIdx);
bool node_segment_sketch_update_sketch(SegmentSketch *node_segment_sketch,
                                       float *series, int fromIdx, int toIdx);
bool calc_hs_split_points(std::vector<int>& hs_split_points,
                          const std::vector<int>& split_points, int min_length);
float range_calc(SegmentSketch sketch, int len);

// node splitting
bool node_split_policy_route_to_left(HerculesNode *node, float *series,
                                     SegmentSketch *series_segment_sketch);
void split_node_evaluate_policies(HerculesNode *node, int *hs_split_point); // hs_split_point < 0 means vertical-only split
bool split_node_create_children(HerculesNode *node,
                                std::vector<int>& child_node_points);
void split_node(HerculesNode *node, SegmentSketch *sketch,
                const float *database, int dim);
bool update_node_statistics(HerculesNode *node, float *timeseries);

// build the index
HerculesNode *hercules_index_build(const float *database, int n, int dim,
                                   int leaf_size, int init_segments,
                                   int num_build_threads = 1);
HerculesNode *hercules_index_build(DataSource *data_source, int leaf_size, int init_segments,
                                   int num_build_threads = 1);
bool hercules_index_write(HerculesNode *root, const float *database,
                          int dim, int leaf_size, int init_segments,
                          int paa_segments, int sax_cardinality, int sax_bit_cardinality,
                          const char *root_dir);

// search + queries
void destroy_tree(HerculesNode *node); // caller must own the tree
float calculate_node_min_distance(HerculesNode *node, const float *query);
void approximate_knn_search(const float *query, int dim,
                             HerculesPQ &pq, FILE *raw_file,
                             std::vector<float> &ts_buf,
                             std::vector<HerculesKnnResult> &knn,
                             idx_t k, int max_leaves);
void hercules_knn_search(HerculesNode *root, const float *query, int dim,
                          idx_t k, idx_t *I, float *D,
                          const char *root_dir, float epsilon, int approx_leaves,
                          int paa_segments, sax_type sax_bit_cardinality, int sax_cardinality,
                          float eapca_th, float sax_th, int n_series,
                          int num_query_threads = 1);

} 
