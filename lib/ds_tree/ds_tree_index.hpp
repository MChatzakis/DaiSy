#pragma once

#include "ds_tree_node.hpp"
#include "../algos/DataSource.hpp"

#include <vector>
#include <climits>
#include <pthread.h>

namespace daisy {

// ---- segment / sketch utilities ----
float calc_mean(float *series, int start, int end);
void  calc_mean_stdev(float *series, int start, int end, float *mean, float *stdev);
bool  calc_split_points(std::vector<int> &points, int ts_length, int segment_size);
bool  calc_hs_split_points(std::vector<int> &hs_split_points,
                            const std::vector<int> &split_points, int min_length);

int get_segment_start(const std::vector<int> &points, int idx);
int get_segment_end  (const std::vector<int> &points, int idx);
int get_segment_length(const std::vector<int> &points, int i);

bool  series_segment_sketch_do_sketch(SegmentSketch *series_segment_sketch,
                                      float *series, int fromIdx, int toIdx);
bool  node_segment_sketch_update_sketch(SegmentSketch *node_segment_sketch,
                                        float *series, int fromIdx, int toIdx);
bool  update_node_statistics(HerculesNode *node, float *timeseries);
float range_calc(SegmentSketch sketch, int len);

// ---- node initialisation ----
HerculesNode    *create_child_node(HerculesNode *parent);
bool             node_init_segments(HerculesNode *node, const std::vector<int> &split_points);
SegmentSketch   *init_segment_sketches(int num_child_segments);

// ---- splitting ----
bool node_split_policy_route_to_left(HerculesNode *node, float *series,
                                     SegmentSketch *series_segment_sketch);
void split_node_evaluate_policies(HerculesNode *node, int *hs_split_point);
bool split_node_create_children(HerculesNode *node, std::vector<int> &child_node_points);
void split_node(HerculesNode *node, SegmentSketch *sketch,
                const float *database, int dim);

// ---- parallel-insert worker data (Linux / pthreads only) ----
#ifndef __APPLE__
struct HerculesInsertWorkerData {
    HerculesInsertWorkerData *threads_data;
    HerculesNode             *root;
    const float              *database_orig;

    pthread_barrier_t *db_barrier;
    pthread_barrier_t *continue_barrier;
    pthread_barrier_t *flush_barrier;

    unsigned int buffer_max;
    int          buffer_counter;
    int          continue_handshake;
    bool         is_flusher;
    int          num_insert_workers;
    int          thread_id;

    float       **double_buffer;
    unsigned int *db_size;
    unsigned int *db_counter;
    bool         *finished;
    int          *flush_counter;
    int          *flush_order;
    unsigned int  initial_db_size;
    unsigned int *db_global_start;

    int dim;
    int leaf_size;
};
#endif // __APPLE__

// ---- index build / destroy ----
HerculesNode *hercules_index_build(const float *database, int n, int dim,
                                   int leaf_size, int init_segments,
                                   int num_build_threads = 1);
HerculesNode *hercules_index_build(DataSource *data_source, int leaf_size,
                                   int init_segments, int num_build_threads = 1);
void destroy_tree(HerculesNode *node);

} // namespace daisy
