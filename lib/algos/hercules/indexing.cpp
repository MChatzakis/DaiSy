#include "../hhercules/indexing.hpp"
#include "../../isax/SAX.hpp"

#include <cmath>
#include <cfloat>
#include <climits>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include <string>
#include <atomic>
#include <pthread.h>
#include <sys/stat.h>

namespace daisy
{

    void calc_mean_stdev(float *series, int start, int end, float *mean, float *stdev)
    {
        float sum_x_squares = 0, sum_x = 0; //sum of x's and sum of x squares
        int count_x;

        *stdev = 0;
        *mean = 0;

        if (start >= end){
            fprintf(stderr, "error in stdev start >= end\n");
        } else {
            count_x = end - start; //size of the series

            for (int i = start; i < end; i++){
                sum_x += series[i];
                sum_x_squares += series[i] * series[i];
            }

            *mean = sum_x / count_x;
            sum_x_squares -= ((sum_x * sum_x) / count_x);

            //DO WE REALLY NEED SQRT???
            *stdev = sqrtf(sum_x_squares / count_x);
        }
    }

    float calc_mean(float *series, int start, int end)
    {
        float mean = 0;

        if (start >= end){
            fprintf(stderr, "error start > end \n");
        } else {
            for (int i = start; i < end; i++)
                mean += series[i];
            mean /= (end - start);
        }

        return mean;
    }

    int get_segment_start(const std::vector<int>& points, int idx)
    {
        if (idx == 0)
            return 0;
        else
            return points[idx - 1];
    }

    int get_segment_end(const std::vector<int>& points, int idx)
    {
        return points[idx];
    }

    int get_segment_length(const std::vector<int>& points, int i)
    {
        if (i == 0)
            return points[i];
        else
            return points[i] - points[i - 1];
    }

    bool series_segment_sketch_do_sketch(SegmentSketch *series_segment_sketch,
                                         float *series, int fromIdx, int toIdx)
    {
        calc_mean_stdev(series, fromIdx, toIdx, &series_segment_sketch->indicators[0], &series_segment_sketch->indicators[1]);
        return true;
    }

    bool node_segment_sketch_update_sketch(SegmentSketch *node_segment_sketch,
                                           float *series, int fromIdx, int toIdx)
    {
        SegmentSketch series_segment_sketch;

        if (!series_segment_sketch_do_sketch(&series_segment_sketch, series, fromIdx, toIdx)){
            fprintf(stderr,"Error in node_init_segments(): Could not calculate the series segment \
                    sketch.\n");
            return false;
        }

        node_segment_sketch->indicators[0] = fmaxf(node_segment_sketch->indicators[0], series_segment_sketch.indicators[0]);
        node_segment_sketch->indicators[1] = fminf(node_segment_sketch->indicators[1], series_segment_sketch.indicators[0]);
        node_segment_sketch->indicators[2] = fmaxf(node_segment_sketch->indicators[2], series_segment_sketch.indicators[1]);
        node_segment_sketch->indicators[3] = fminf(node_segment_sketch->indicators[3], series_segment_sketch.indicators[1]);
        node_segment_sketch->num_indicators = 4;

        return true;
    }

    bool update_node_statistics(HerculesNode *node, float *timeseries)
    {
        //update vertical node_segment_sketch
        for (int i = 0; i < (int)node->node_points.size(); i++){
            int from = get_segment_start(node->node_points, i);
            int to = get_segment_end(node->node_points, i);

            if (!node_segment_sketch_update_sketch(&node->node_segment_sketches[i], timeseries, from, to)){
                fprintf(stderr, "Error in dstree_index.c: Could not update vertical sketch for node segment.\n");
                return false;
            }
        }

        //update horizontal node_segment_sketch
        for (int i = 0; i < (int)node->hs_node_points.size(); i++){
            if (!node_segment_sketch_update_sketch(&node->hs_node_segment_sketches[i], timeseries,
                                                   get_segment_start(node->hs_node_points, i),
                                                   get_segment_end(node->hs_node_points, i))){
                fprintf(stderr, "Error in dstree_index.c: Could not update horizontal sketch for node segment.\n");
                return false;
            }
        }

        ++node->node_size;
        return true;
    }

    bool calc_hs_split_points(std::vector<int>& hs_split_points,
                              const std::vector<int>& split_points, int min_length)
    {
        int c = 0;
        int segment_size = (int)split_points.size();

        hs_split_points.resize(segment_size * 2);

        for (int i = 0; i < segment_size; i++){
            int length = split_points[i]; //i==0
            if (i > 0)
                length = split_points[i] - split_points[i - 1];

            if (length >= min_length * 2){
                int start = 0;
                if (i > 0)
                    start = split_points[i - 1];
                hs_split_points[c++] = start + (length / 2);
            }
            hs_split_points[c++] = split_points[i];
        }

        hs_split_points.resize(c);
        return true;
    }

    bool calc_split_points(std::vector<int>& points, int ts_length, int segment_size)
    {
        int avg_length = ts_length / segment_size;

        points.resize(segment_size);
        for (int i = 0; i < segment_size; ++i)
            points[i] = (i + 1) * avg_length;

        //set the last one
        points[segment_size - 1] = ts_length;

        return true;
    }

    bool node_init_segments(HerculesNode *node, const std::vector<int>& split_points)
    {
        node->node_points = split_points;

        int min_length = 1; //minimum length of new segment = 1

        calc_hs_split_points(node->hs_node_points, node->node_points, min_length);

        node->node_segment_sketches.resize(node->node_points.size());
        node->hs_node_segment_sketches.resize(node->hs_node_points.size());

        return true;
    }

    SegmentSketch *init_segment_sketches(int num_child_segments)
    {
        SegmentSketch *child_node_segment_sketches = new SegmentSketch[num_child_segments];
        return child_node_segment_sketches;
    }

    HerculesNode *create_child_node(HerculesNode *parent)
    {
        HerculesNode *child_node = new HerculesNode();

        if (child_node == nullptr){
            fprintf(stderr,"Error in dstree_node_split.c: Could not initialize \
                    child node of parent.\n");
            return nullptr;
        }

        child_node->parent = parent;
        child_node->level = parent->level + 1;
        child_node->is_leaf = true;

        return child_node;
    }

    static bool mean_node_segment_split_policy_split(node_split_policy *policy,
                                                      SegmentSketch sketch,
                                                      SegmentSketch *ret)
    {
        const int num_splits = 2; //default split into 2 node

        float max_mean = sketch.indicators[0];
        float min_mean = sketch.indicators[1];

        policy->indicator_split_idx = 0; //mean based split
        policy->indicator_split_val = (max_mean + min_mean) / 2;

        ret[0].num_indicators = sketch.num_indicators;
        ret[1].num_indicators = sketch.num_indicators;

        for (int i = 0; i < num_splits; i++){
            for (int j = 0; j < ret[i].num_indicators; ++j)
                ret[i].indicators[j] = sketch.indicators[j];
        }

        /* make sure indicator split idx is 0 */
        ret[0].indicators[1] = policy->indicator_split_val;
        ret[1].indicators[0] = policy->indicator_split_val;

        return true;
    }

    static bool stdev_node_segment_split_policy_split(node_split_policy *policy,
                                                       SegmentSketch sketch,
                                                       SegmentSketch *ret)
    {
        const int num_splits = 2; //default split into 2 node

        float max_stdev = sketch.indicators[2];
        float min_stdev = sketch.indicators[3];

        policy->indicator_split_idx = 1; //stdev based split
        policy->indicator_split_val = (float)(max_stdev + min_stdev) / 2;

        ret[0].num_indicators = sketch.num_indicators;
        ret[1].num_indicators = sketch.num_indicators;

        for (int i = 0; i < num_splits; i++){
            for (int j = 0; j < ret[i].num_indicators; ++j)
                ret[i].indicators[j] = sketch.indicators[j];
        }

        /* make sure indicator split idx is 1 */
        ret[0].indicators[2] = policy->indicator_split_val;
        ret[1].indicators[3] = policy->indicator_split_val;

        return true;
    }

    static int get_hs_split_point(const std::vector<int>& points, int from, int to)
    {
        if (std::binary_search(points.begin(), points.end(), to))
            return from;
        return to;
    }

    float range_calc(SegmentSketch sketch, int len)
    {
        float mean_width = sketch.indicators[0] - sketch.indicators[1];
        float stdev_upper = sketch.indicators[2];
        float stdev_lower = sketch.indicators[3];

        return len * (mean_width * mean_width + stdev_upper * stdev_upper);
    }

    bool node_split_policy_route_to_left(HerculesNode *node, float *series,
                                         SegmentSketch *series_segment_sketch)
    {
        struct node_split_policy *policy = node->split_policy;
        bool route_to_left = false;

        if (!series_segment_sketch_do_sketch(series_segment_sketch, series, policy->split_from, policy->split_to)){
            fprintf(stderr,"Error in dstree_node_split.c: Could not calculate the series \
                    segment sketch .\n");
        }

        if (series_segment_sketch->indicators[policy->indicator_split_idx] < policy->indicator_split_val)
            route_to_left = true;

        return route_to_left;
    }

    void split_node_evaluate_policies(HerculesNode *node, int *hs_split_point)
    {
        node_split_policy curr_node_split_policy;
        float max_diff_value = (FLT_MAX * (-1));
        float avg_children_range_value = 0;
        *hs_split_point = -1;
        const int num_child_segments = 2; //by default split to two subsegments

        for (int i = 0; i < (int)node->node_points.size(); ++i){
            SegmentSketch curr_node_segment_sketch = node->node_segment_sketches[i];

            //This is the QoS of this segment. QoS is the estimation quality evaluated as =
            //QoS = segment_length * (max_mean_min_mean) * ((max_mean_min_mean) +
            // (max_stdev * max_stdev))
            //The smaller the QoS, the more effective the bounds are for similarity
            //estimation

            float node_range_value = range_calc(curr_node_segment_sketch,
                                                get_segment_length(node->node_points, i));

            //for every split policy (j=0: mean, j=1: stdev)
            for (int j = 0; j < 2; ++j){
                node_split_policy curr_node_segment_split_policy;
                curr_node_segment_split_policy.indicator_split_idx = j;

                SegmentSketch *child_node_segment_sketches = init_segment_sketches(num_child_segments);

                if (j == 0)
                    mean_node_segment_split_policy_split(&curr_node_segment_split_policy,
                                                         curr_node_segment_sketch,
                                                         child_node_segment_sketches);
                else
                    stdev_node_segment_split_policy_split(&curr_node_segment_split_policy,
                                                          curr_node_segment_sketch,
                                                          child_node_segment_sketches);

                float range_values[num_child_segments];
                for (int k = 0; k < num_child_segments; ++k){
                    SegmentSketch child_node_segment_sketch = child_node_segment_sketches[k];
                    range_values[k] = range_calc(child_node_segment_sketch,
                                                 get_segment_length(node->node_points, i));
                }

                //diff_value represents the splitting benefit
                //B = QoS(N) - (QoS_leftNode + QoS_rightNode)/2
                //the higher the diff_value, the better is the splitting

                avg_children_range_value = calc_mean(range_values, 0, num_child_segments);
                float diff_value = node_range_value - avg_children_range_value;

                if (diff_value > max_diff_value){
                    max_diff_value = diff_value;
                    curr_node_split_policy.split_from = get_segment_start(node->node_points, i);
                    curr_node_split_policy.split_to = get_segment_end(node->node_points, i);
                    curr_node_split_policy.indicator_split_idx = curr_node_segment_split_policy.indicator_split_idx;
                    curr_node_split_policy.indicator_split_val = curr_node_segment_split_policy.indicator_split_val;
                }
                delete[] child_node_segment_sketches;
            }
        }

        //add trade-off for horizontal split
        max_diff_value = max_diff_value * 2;

        for (int i = 0; i < (int)node->hs_node_points.size(); ++i){
            SegmentSketch curr_hs_node_segment_sketch = node->hs_node_segment_sketches[i];
            float node_range_value = range_calc(curr_hs_node_segment_sketch,
                                                get_segment_length(node->hs_node_points, i));

            //for every split policy (j=0: mean, j=1: stdev)
            for (int j = 0; j < 2; ++j){
                node_split_policy curr_hs_node_segment_split_policy;
                curr_hs_node_segment_split_policy.indicator_split_idx = j;

                SegmentSketch *child_node_segment_sketches = init_segment_sketches(num_child_segments);

                if (j == 0)
                    mean_node_segment_split_policy_split(&curr_hs_node_segment_split_policy,
                                                         curr_hs_node_segment_sketch,
                                                         child_node_segment_sketches);
                else
                    stdev_node_segment_split_policy_split(&curr_hs_node_segment_split_policy,
                                                          curr_hs_node_segment_sketch,
                                                          child_node_segment_sketches);

                float range_values[num_child_segments];
                for (int k = 0; k < num_child_segments; ++k){
                    SegmentSketch child_node_segment_sketch = child_node_segment_sketches[k];
                    range_values[k] = range_calc(child_node_segment_sketch,
                                                 get_segment_length(node->hs_node_points, i));
                }

                avg_children_range_value = calc_mean(range_values, 0, num_child_segments);
                float diff_value = node_range_value - avg_children_range_value;

                if (diff_value > max_diff_value){
                    max_diff_value = diff_value;
                    curr_node_split_policy.split_from = get_segment_start(node->hs_node_points, i);
                    curr_node_split_policy.split_to = get_segment_end(node->hs_node_points, i);
                    curr_node_split_policy.indicator_split_idx = curr_hs_node_segment_split_policy.indicator_split_idx;
                    curr_node_split_policy.indicator_split_val = curr_hs_node_segment_split_policy.indicator_split_val;
                    *hs_split_point = get_hs_split_point(node->node_points,
                                                          curr_node_split_policy.split_from,
                                                          curr_node_split_policy.split_to);
                }

                delete[] child_node_segment_sketches;
            }
        }

        delete node->split_policy;
        node->split_policy = new node_split_policy();
        node->split_policy->split_from = curr_node_split_policy.split_from;
        node->split_policy->split_to = curr_node_split_policy.split_to;
        node->split_policy->indicator_split_idx = curr_node_split_policy.indicator_split_idx;
        node->split_policy->indicator_split_val = curr_node_split_policy.indicator_split_val;
    }

    bool split_node_create_children(HerculesNode *node,
                                    std::vector<int>& child_node_points)
    {
        if (!node->is_leaf){
            fprintf(stderr,"Error in dstree_node_split.c: Trying to split a node \
                    that is not a leaf.\n");
            return false;
        } else {
            node->left_child = create_child_node(node);
            if (node->left_child == nullptr){
                fprintf(stderr,"Error in dstree_node_split.c: Left child not \
                        initialized properly.\n");
                return false;
            }

            if (!node_init_segments(node->left_child, child_node_points)){
                fprintf(stderr,"Error in dstree_node_split.c: Could not initialize \
                        segments for left child.\n");
                return false;
            }

            node->left_child->is_leaf = true;
            node->left_child->is_left = true;

            node->right_child = create_child_node(node);
            if (node->right_child == nullptr){
                fprintf(stderr,"Error in dstree_node_split.c: Right child not \
                        initialized properly.\n");
                return false;
            }

            if (!node_init_segments(node->right_child, child_node_points)){
                fprintf(stderr,"Error in dstree_node_split.c: Could not initialize \
                        segments for right child.\n");
                return false;
            }

            node->right_child->is_leaf = true;
        }

        return true;
    }

    void split_node(HerculesNode *node, SegmentSketch *sketch,
                    const float *database, int dim)
    {
        int hs_split_point = -1;
        std::vector<int> child_node_points;

        split_node_evaluate_policies(node, &hs_split_point);

        if (hs_split_point < 0){
            child_node_points = node->node_points; //children will have the same number of segments as parent
        } else {
            child_node_points = node->node_points; //children will have one additional segment than the parent
            child_node_points.push_back(hs_split_point);
            std::sort(child_node_points.begin(), child_node_points.end());
        }

        if (!split_node_create_children(node, child_node_points)){
            fprintf(stderr,"Error in dstree_index.c: could not split node.\n");
            return;
        }

        for (int idx = 0; idx < (int)node->series_indices.size(); ++idx){
            float *ts = const_cast<float *>(database + (long)node->series_indices[idx] * dim);
            SegmentSketch series_segment_sketch;
            if (node_split_policy_route_to_left(node, ts, &series_segment_sketch)){
                if (!update_node_statistics(node->left_child, ts)){
                    fprintf(stderr,"Error in dstree_index.c: could not update \
                             statistics at left child.\n");
                    return;
                }
                node->left_child->series_indices.push_back(node->series_indices[idx]);
            } else {
                if (!update_node_statistics(node->right_child, ts)){
                    fprintf(stderr,"Error in dstree_index.c: could not update \
                             statistics at right child.\n");
                    return;
                }
                node->right_child->series_indices.push_back(node->series_indices[idx]);
            }
        }

        node->is_leaf = false;
        node->series_indices.clear();
    }

    // pthread_barrier_t is available on Linux (glibc) but not on macOS.
#ifndef __APPLE__

    struct HerculesInsertWorkerData {
        HerculesInsertWorkerData *threads_data; // array of all workers (for coordinator)
        HerculesNode *root;
        const float *database_orig; // full original database (for split_node)

        pthread_barrier_t *db_barrier;
        pthread_barrier_t *continue_barrier;
        pthread_barrier_t *flush_barrier;

        unsigned int buffer_max;
        int buffer_counter;
        int continue_handshake;
        bool is_flusher;
        int num_insert_workers;
        int thread_id;

        float **double_buffer;
        unsigned int *db_size;
        unsigned int *db_counter;
        bool *finished;
        int *flush_counter;
        int *flush_order;
        unsigned int initial_db_size;
        unsigned int *db_global_start; // global start index per buffer slot

        int dim;
        int leaf_size;
    };

    static void hercules_parallel_insert(HerculesNode *root, float *timeseries,
                                          int series_global_idx, const float *database_orig,
                                          int leaf_size, int dim)
    {
        HerculesNode *node = root;
        SegmentSketch sketch;

        // Phase 1: unlocked traversal to leaf
        while (!node->is_leaf){
            if (node_split_policy_route_to_left(node, timeseries, &sketch))
                node = node->left_child;
            else
                node = node->right_child;
        }

        // Lock the candidate leaf
        pthread_mutex_lock(&node->lock_data);

        // Phase 2: verify still leaf; re-route if a concurrent split changed it
        while (!node->is_leaf){
            pthread_mutex_unlock(&node->lock_data);
            while (!node->is_leaf){
                if (node_split_policy_route_to_left(node, timeseries, &sketch))
                    node = node->left_child;
                else
                    node = node->right_child;
            }
            pthread_mutex_lock(&node->lock_data);
        }

        update_node_statistics(node, timeseries);
        node->series_indices.push_back(series_global_idx);

        if (node->node_size >= (unsigned int)leaf_size)
            split_node(node, &sketch, database_orig, dim);

        pthread_mutex_unlock(&node->lock_data);
    }

    static void hercules_flush_worker(void *transferdata)
    {
        HerculesInsertWorkerData *d = (HerculesInsertWorkerData *)transferdata;

        if (d->buffer_counter >= (int)d->buffer_max)
            __sync_fetch_and_add(d->flush_counter, 1);

        __sync_fetch_and_add(&d->continue_handshake, 1);
        pthread_barrier_wait(d->continue_barrier);
        __sync_fetch_and_sub(&d->continue_handshake, 1);

        if (__sync_fetch_and_add(d->flush_order, 0) != 0){
            int buffer_counter = d->buffer_counter;
            pthread_barrier_wait(d->flush_barrier);
            __sync_fetch_and_sub(&d->buffer_counter, buffer_counter);
        }
    }

    static void hercules_flush_coordinator(void *transferdata)
    {
        HerculesInsertWorkerData *d = (HerculesInsertWorkerData *)transferdata;
        int num_threads = d->num_insert_workers;
        int worker_id = d->thread_id;

        __sync_fetch_and_add(&d->threads_data[worker_id].continue_handshake, 1);

        for (int j = 0; j < num_threads; j++){
            while (__sync_fetch_and_add(&d->threads_data[j].continue_handshake, 0) != 1)
                ;
        }

        if (__sync_fetch_and_add(d->flush_counter, 0) >= INT_MAX ||
            d->buffer_counter >= (int)d->buffer_max){
            __sync_fetch_and_add(d->flush_order, 1);
        }

        int num_full_buffers = *d->flush_counter;
        __sync_fetch_and_sub(d->flush_counter, num_full_buffers);

        pthread_barrier_wait(d->continue_barrier);
        __sync_fetch_and_sub(&d->threads_data[worker_id].continue_handshake, 1);

        if (__sync_fetch_and_add(d->flush_order, 0) != 0){
            // flush_index_to_disk omitted (in-memory build)
            __sync_fetch_and_sub(d->flush_order, 1);

            pthread_barrier_wait(d->flush_barrier);

            int buffer_counter = d->buffer_counter;
            __sync_fetch_and_sub(&d->buffer_counter, buffer_counter);
        }
    }

    static void *hercules_insert_worker(void *transferdata)
    {
        HerculesInsertWorkerData *d = (HerculesInsertWorkerData *)transferdata;
        int toggle = 0;

        while (!d->finished[toggle]){
            if (d->buffer_counter < (int)d->buffer_max){
                unsigned int ts_id = __sync_fetch_and_add(&d->db_counter[toggle], 1);
                while (ts_id < d->db_size[toggle]){
                    unsigned int offset = ts_id * (unsigned int)d->dim;
                    float *timeseries = d->double_buffer[toggle] + offset;
                    int global_idx = (int)(d->db_global_start[toggle] + ts_id);

                    hercules_parallel_insert(d->root, timeseries, global_idx,
                                              d->database_orig, d->leaf_size, d->dim);

                    ts_id = __sync_fetch_and_add(&d->db_counter[toggle], 1);
                }
            }

            pthread_barrier_wait(d->db_barrier);

            if (d->is_flusher)
                hercules_flush_coordinator(transferdata);
            else
                hercules_flush_worker(transferdata);

            toggle = 1 - toggle;
        }

        return nullptr;
    }

#endif 

    HerculesNode *hercules_index_build(const float *database, int n, int dim,
                                       int leaf_size, int init_segments,
                                       int num_build_threads)
    {
        std::vector<int> split_points;
        if (!calc_split_points(split_points, dim, init_segments)){
            fprintf(stderr,"Error in hercules_index_build: Could not calculate split points.\n");
            return nullptr;
        }

        HerculesNode *root = new HerculesNode();
        if (!node_init_segments(root, split_points)){
            fprintf(stderr,"Error in hercules_index_build: Could not initialize root segments.\n");
            delete root;
            return nullptr;
        }

        if (num_build_threads <= 1){
            for (int i = 0; i < n; ++i){
                float *ts = const_cast<float *>(database + (long)i * dim);

                HerculesNode *node = root;
                while (!node->is_leaf){
                    SegmentSketch sketch;
                    if (node_split_policy_route_to_left(node, ts, &sketch))
                        node = node->left_child;
                    else
                        node = node->right_child;
                }

                update_node_statistics(node, ts);
                node->series_indices.push_back(i);

                if (node->node_size >= (unsigned int)leaf_size){
                    SegmentSketch sketch;
                    split_node(node, &sketch, database, dim);
                }
            }

            return root;
        }

        // parallel path 
#ifndef __APPLE__
        int num_workers = num_build_threads - 1;

        unsigned int initial_db_size = (unsigned int)n;

        pthread_barrier_t db_barrier, continue_barrier, flush_barrier;
        pthread_barrier_init(&db_barrier, NULL, (unsigned int)num_build_threads);
        pthread_barrier_init(&continue_barrier, NULL, (unsigned int)num_workers);
        pthread_barrier_init(&flush_barrier, NULL, (unsigned int)num_workers);

        unsigned int db_size[2] = {0, 0};
        unsigned int db_counter[2] = {0, 0};
        unsigned int db_global_start[2] = {0, 0};
        bool finished[2] = {false, false};
        int flush_counter = 0;
        int flush_order = 0;

        int toggle = 0;

        db_size[toggle] = (unsigned int)std::min((unsigned long long)initial_db_size,
                                                  (unsigned long long)n);

        float **double_buffer = (float **)calloc(2, sizeof(float *));
        double_buffer[0] = (float *)malloc(sizeof(float) * (size_t)dim * initial_db_size);
        double_buffer[1] = (float *)malloc(sizeof(float) * (size_t)dim * initial_db_size);

        // load first batch into double_buffer[toggle] (memcpy replaces fread)
        memcpy(double_buffer[toggle], database,
               sizeof(float) * (size_t)dim * db_size[toggle]);
        db_global_start[toggle] = 0;
        toggle = 1 - toggle;

        HerculesInsertWorkerData *input_data =
            (HerculesInsertWorkerData *)malloc(sizeof(HerculesInsertWorkerData) * num_workers);

        for (int i = 0; i < num_workers; i++){
            input_data[i].threads_data = input_data;
            input_data[i].root = root;
            input_data[i].database_orig = database;
            input_data[i].db_barrier = &db_barrier;
            input_data[i].continue_barrier = &continue_barrier;
            input_data[i].flush_barrier = &flush_barrier;
            input_data[i].buffer_max = (unsigned int)INT_MAX;
            input_data[i].buffer_counter = 0;
            input_data[i].continue_handshake = 0;
            input_data[i].is_flusher = (i == 0);
            input_data[i].num_insert_workers = num_workers;
            input_data[i].thread_id = i;
            input_data[i].double_buffer = double_buffer;
            input_data[i].db_size = db_size;
            input_data[i].db_counter = db_counter;
            input_data[i].finished = finished;
            input_data[i].flush_counter = &flush_counter;
            input_data[i].flush_order = &flush_order;
            input_data[i].initial_db_size = initial_db_size;
            input_data[i].db_global_start = db_global_start;
            input_data[i].dim = dim;
            input_data[i].leaf_size = leaf_size;
        }

        pthread_t *threadid = (pthread_t *)malloc(sizeof(pthread_t) * num_workers);

        for (int j = 0; j < num_workers; j++)
            pthread_create(&threadid[j], NULL, hercules_insert_worker, (void *)&input_data[j]);

        // coordinator loop (with initial_db_size==n the loop body never runs)
        for (unsigned long long i = (unsigned long long)db_size[1 - toggle];
             i < (unsigned long long)n;
             i += (unsigned long long)db_size[toggle]){
            db_size[toggle] = (unsigned int)std::min((unsigned long long)initial_db_size,
                                                      (unsigned long long)n - i);
            memcpy(double_buffer[toggle], database + (long)i * dim,
                   sizeof(float) * (size_t)dim * db_size[toggle]);
            db_global_start[toggle] = (unsigned int)i;
            db_counter[toggle] = 0;
            toggle = 1 - toggle;
            pthread_barrier_wait(&db_barrier);
        }

        // signal workers all data has been processed
        finished[toggle] = true;
        pthread_barrier_wait(&db_barrier);

        for (int j = 0; j < num_workers; j++)
            pthread_join(threadid[j], NULL);

        free(double_buffer[0]);
        free(double_buffer[1]);
        free(double_buffer);
        free(input_data);
        free(threadid);

        pthread_barrier_destroy(&db_barrier);
        pthread_barrier_destroy(&continue_barrier);
        pthread_barrier_destroy(&flush_barrier);

        return root;
#else
        // fall through to sequential path (pthread_barrier_t unavailable on macOS)
        for (int i = 0; i < n; ++i){
            float *ts = const_cast<float *>(database + (long)i * dim);

            HerculesNode *node = root;
            while (!node->is_leaf){
                SegmentSketch sketch;
                if (node_split_policy_route_to_left(node, ts, &sketch))
                    node = node->left_child;
                else
                    node = node->right_child;
            }

            update_node_statistics(node, ts);
            node->series_indices.push_back(i);

            if (node->node_size >= (unsigned int)leaf_size){
                SegmentSketch sketch;
                split_node(node, &sketch, database, dim);
            }
        }

        return root;
#endif 
    }

    HerculesNode *hercules_index_build(DataSource *ds, int leaf_size, int init_segments, int num_build_threads)
    {
        int n = (int)ds->getTotalRecords();
        int dim = (int)ds->getDim();
        const float *buf = ds->rawPointer();
        if (buf)
            return hercules_index_build(buf, n, dim, leaf_size, init_segments, num_build_threads);
        std::vector<float> tmp((size_t)n * dim);
        ds->reset();
        for (int i = 0; i < n; i++)
            ds->nextRecord(tmp.data() + (size_t)i * dim);
        return hercules_index_build(tmp.data(), n, dim, leaf_size, init_segments, num_build_threads);
    }

    void destroy_tree(HerculesNode *node)
    {
        if (node == nullptr)
            return;
        destroy_tree(node->left_child);
        destroy_tree(node->right_child);
        delete node;
    }

    // file_pos is a record index (not a byte offset).
    static uint64_t assign_file_pos(HerculesNode *node, uint64_t pos, unsigned long &leaf_count)
    {
        if (!node->is_leaf){
            pos = assign_file_pos(node->left_child, pos, leaf_count);
            pos = assign_file_pos(node->right_child, pos, leaf_count);
        } else {
            node->file_pos = pos;
            pos += node->node_size;
            ++leaf_count;
        }
        return pos;
    }

    static bool write_leaf_data(HerculesNode *node, const float *database, int dim,
                                 int paa_segments, int sax_cardinality, int sax_bit_cardinality,
                                 FILE *raw_file, FILE *sims_file)
    {
        if (!node->is_leaf){
            if (!write_leaf_data(node->left_child, database, dim, paa_segments, sax_cardinality, sax_bit_cardinality, raw_file, sims_file))
                return false;
            if (!write_leaf_data(node->right_child, database, dim, paa_segments, sax_cardinality, sax_bit_cardinality, raw_file, sims_file))
                return false;
            return true;
        }

        int ts_values_per_segment = dim / paa_segments;
        std::vector<sax_type> sax(paa_segments);

        for (int idx = 0; idx < (int)node->series_indices.size(); idx++){
            float *ts = const_cast<float *>(database + (long)node->series_indices[idx] * dim);

            fwrite(ts, sizeof(float), dim, raw_file);

            sax_from_ts(ts, sax.data(), ts_values_per_segment, paa_segments,
                        sax_cardinality, sax_bit_cardinality);
            fwrite(sax.data(), sizeof(sax_type), paa_segments, sims_file);
        }

        return true;
    }

    // writes is_leaf + level first, then recurses into children (internal), then node data.
    static bool write_tree_node(HerculesNode *node, FILE *file)
    {
        unsigned char is_leaf = (unsigned char)node->is_leaf;
        fwrite(&is_leaf, sizeof(unsigned char), 1, file);
        fwrite(&node->level, sizeof(unsigned int), 1, file);

        if (!node->is_leaf){
            if (!write_tree_node(node->left_child, file)) return false;
            if (!write_tree_node(node->right_child, file)) return false;

            fwrite(&node->node_size, sizeof(unsigned int), 1, file);

            if (node->split_policy == nullptr){
                fprintf(stderr, "Error in write_tree_node: internal node has null split_policy.\n");
                return false;
            }
            fwrite(node->split_policy, sizeof(node_split_policy), 1, file);

            int num_node_points = (int)node->node_points.size();
            fwrite(&num_node_points, sizeof(int), 1, file);
            fwrite(node->node_points.data(), sizeof(int), num_node_points, file);
            for (int i = 0; i < num_node_points; i++){
                fwrite(&node->node_segment_sketches[i].num_indicators, sizeof(int), 1, file);
                fwrite(node->node_segment_sketches[i].indicators, sizeof(float),
                       node->node_segment_sketches[i].num_indicators, file);
            }
        } else {
            fwrite(&node->node_size, sizeof(unsigned int), 1, file);

            int num_node_points = (int)node->node_points.size();
            fwrite(&num_node_points, sizeof(int), 1, file);
            fwrite(node->node_points.data(), sizeof(int), num_node_points, file);
            for (int i = 0; i < num_node_points; i++){
                fwrite(&node->node_segment_sketches[i].num_indicators, sizeof(int), 1, file);
                fwrite(node->node_segment_sketches[i].indicators, sizeof(float),
                       node->node_segment_sketches[i].num_indicators, file);
            }

            fwrite(&node->file_pos, sizeof(uint64_t), 1, file);
        }

        return true;
    }

    bool hercules_index_write(HerculesNode *root, const float *database,
                              int dim, int leaf_size, int init_segments,
                              int paa_segments, int sax_cardinality, int sax_bit_cardinality,
                              const char *root_dir)
    {
        mkdir(root_dir, 0755); //ignore error if directory already exists

        std::string raw_path = std::string(root_dir) + "/leaves_raw.idx";
        std::string sims_path = std::string(root_dir) + "/leaves_sims.idx";
        std::string tree_path = std::string(root_dir) + "/root.idx";

        FILE *raw_file = fopen(raw_path.c_str(), "wb");
        FILE *sims_file = fopen(sims_path.c_str(), "wb");
        FILE *tree_file = fopen(tree_path.c_str(), "wb");

        if (raw_file == nullptr || sims_file == nullptr || tree_file == nullptr){
            fprintf(stderr, "Error in hercules_index_write: could not open output files in %s\n", root_dir);
            if (raw_file) fclose(raw_file);
            if (sims_file) fclose(sims_file);
            if (tree_file) fclose(tree_file);
            return false;
        }

        // write placeholder header to root.idx (leaf_count backfilled after traversal)
        unsigned long leaf_count = 0;
        unsigned int timeseries_size = (unsigned int)dim;
        unsigned int init_seg = (unsigned int)init_segments;
        unsigned int max_leaf = (unsigned int)leaf_size;

        fwrite(&leaf_count, sizeof(unsigned long), 1, tree_file);
        fwrite(&timeseries_size, sizeof(unsigned int), 1, tree_file);
        fwrite(&init_seg, sizeof(unsigned int), 1, tree_file);
        fwrite(&paa_segments, sizeof(int), 1, tree_file);
        fwrite(&sax_bit_cardinality, sizeof(int), 1, tree_file);
        fwrite(&max_leaf, sizeof(unsigned int), 1, tree_file);

        //assign file_pos to each leaf (post-order)
        assign_file_pos(root, 0, leaf_count);

        //write leaf data to leaves_raw.idx and leaves_sims.idx (post-order)
        if (!write_leaf_data(root, database, dim, paa_segments, sax_cardinality,
                             sax_bit_cardinality, raw_file, sims_file)){
            fprintf(stderr, "Error in hercules_index_write: could not write leaf data.\n");
            fclose(raw_file); fclose(sims_file); fclose(tree_file);
            return false;
        }

        // write tree structure to root.idx (post-order)
        if (!write_tree_node(root, tree_file)){
            fprintf(stderr, "Error in hercules_index_write: could not write tree node.\n");
            fclose(raw_file); fclose(sims_file); fclose(tree_file);
            return false;
        }

        // backfill leaf_count at byte 0 of root.idx
        fseek(tree_file, 0L, SEEK_SET);
        fwrite(&leaf_count, sizeof(unsigned long), 1, tree_file);

        fclose(raw_file);
        fclose(sims_file);
        fclose(tree_file);

        return true;
    }

    struct LeafCandidate {
        HerculesNode *node;
        float lb;
    };

    struct SeriesCandidate {
        uint64_t record_idx;
        idx_t series_idx;
        float lb_sax;
    };

    struct HerculesCSWorkerData {
        const std::vector<LeafCandidate> *lclist;
        std::atomic<unsigned int> *cs_idx;
        const std::vector<sax_type> *sax_cache;
        const float *query_paa;
        const sax_type *max_sax_cardinalities;
        sax_type sax_bit_cardinality;
        int sax_cardinality;
        int paa_segments;
        float mindist_sqrt;
        const std::vector<HerculesKnnResult> *knn;
        idx_t k;
        std::vector<SeriesCandidate> local_sclist;
    };

    struct HerculesCRWorkerData {
        const std::vector<SeriesCandidate> *sclist;
        std::atomic<unsigned int> *cr_idx;
        const float *query;
        int dim;
        idx_t k;
        const char *raw_path;
        std::vector<HerculesKnnResult> *knn;
        pthread_rwlock_t *lock_bsf;
    };

    // L2-squared with early-exit bound
    static float l2sq(const float *a, const float *b, int dim, float bound)
    {
        float sum = 0;
        for (int i = 0; i < dim; i++){
            float diff = a[i] - b[i];
            sum += diff * diff;
            if (sum >= bound) return sum;
        }
        return sum;
    }

    static void knn_bounded_insert(std::vector<HerculesKnnResult> &knn, idx_t k,
                                    idx_t series_idx, float dist)
    {
        // overwrite last slot (highest distance), then insertion-sort ascending
        knn[k - 1].distance = dist;
        knn[k - 1].series_idx = series_idx;

        for (idx_t i = 1; i < k; i++){
            idx_t j = i;
            while (j > 0 && knn[j - 1].distance > knn[j].distance){
                std::swap(knn[j], knn[j - 1]);
                --j;
            }
        }
    }

    static int count_leaves(HerculesNode *node)
    {
        if (node->is_leaf) return 1;
        return count_leaves(node->left_child) + count_leaves(node->right_child);
    }

    float calculate_node_min_distance(HerculesNode *node, const float *query)
    {
        float sum = 0;
        int num_points = (int)node->node_points.size();

        std::vector<float> mean_per_segment(num_points);
        std::vector<float> stdev_per_segment(num_points);

        for (int i = 0; i < num_points; i++)
            calc_mean_stdev(const_cast<float *>(query),
                            get_segment_start(node->node_points, i),
                            get_segment_end(node->node_points, i),
                            &mean_per_segment[i], &stdev_per_segment[i]);

        for (int i = 0; i < num_points; i++){
            float temp_dist = 0;
            float temp = 0;

            if ((stdev_per_segment[i] - node->node_segment_sketches[i].indicators[2]) *
                (stdev_per_segment[i] - node->node_segment_sketches[i].indicators[3]) > 0){
                temp = fminf(fabsf(stdev_per_segment[i] - node->node_segment_sketches[i].indicators[2]),
                             fabsf(stdev_per_segment[i] - node->node_segment_sketches[i].indicators[3]));
                temp_dist += temp * temp;
            }

            if ((mean_per_segment[i] - node->node_segment_sketches[i].indicators[0]) *
                (mean_per_segment[i] - node->node_segment_sketches[i].indicators[1]) > 0){
                temp = fminf(fabsf(mean_per_segment[i] - node->node_segment_sketches[i].indicators[0]),
                             fabsf(mean_per_segment[i] - node->node_segment_sketches[i].indicators[1]));
                temp_dist += temp * temp;
            }

            sum += temp_dist * (float)get_segment_length(node->node_points, i);
        }

        return sum;
    }

    void approximate_knn_search(const float *query, int dim,
                                 HerculesPQ &pq, FILE *raw_file,
                                 std::vector<float> &ts_buf,
                                 std::vector<HerculesKnnResult> &knn,
                                 idx_t k, int max_leaves)
    {
        int loaded_leaves = 0;
        while (!pq.empty()){
            std::pair<float, HerculesNode *> top = pq.top();
            pq.pop();
            float lb = top.first;
            HerculesNode *node = top.second;

            float kth_bsf = knn[k - 1].distance;
            if (lb > kth_bsf)
                break;

            if (node->is_leaf){
                ts_buf.resize((size_t)node->node_size * dim);
                fseek(raw_file, (long)node->file_pos * dim * (long)sizeof(float), SEEK_SET);
                fread(ts_buf.data(), sizeof(float), (size_t)node->node_size * dim, raw_file);

                for (unsigned int idx = 0; idx < node->node_size; idx++){
                    kth_bsf = knn[k - 1].distance;
                    float dist = l2sq(query, ts_buf.data() + (size_t)idx * dim, dim, kth_bsf);
                    if (dist < kth_bsf)
                        knn_bounded_insert(knn, k, (idx_t)node->series_indices[idx], dist);
                }

                ++loaded_leaves;
                if (loaded_leaves >= max_leaves)
                    break;
            } else {
                kth_bsf = knn[k - 1].distance;
                float child_lb;

                child_lb = calculate_node_min_distance(node->left_child, query);
                if (child_lb < kth_bsf)
                    pq.push(std::make_pair(child_lb, node->left_child));

                child_lb = calculate_node_min_distance(node->right_child, query);
                if (child_lb < kth_bsf)
                    pq.push(std::make_pair(child_lb, node->right_child));
            }
        }
    }

    static void hercules_skip_sequential_scan(const std::vector<LeafCandidate> &lclist,
                                              const float *query, int dim,
                                              FILE *raw_file,
                                              std::vector<float> &ts_buf,
                                              std::vector<HerculesKnnResult> &knn,
                                              idx_t k, float epsilon)
    {
        for (size_t i = 0; i < lclist.size(); i++){
            HerculesNode *leaf = lclist[i].node;
            float kth_bsf = knn[k - 1].distance;

            if (lclist[i].lb <= kth_bsf / (1.0f + epsilon)){
                ts_buf.resize((size_t)leaf->node_size * dim);
                fseek(raw_file, (long)leaf->file_pos * dim * (long)sizeof(float), SEEK_SET);
                fread(ts_buf.data(), sizeof(float), (size_t)leaf->node_size * dim, raw_file);

                for (unsigned int idx = 0; idx < leaf->node_size; idx++){
                    kth_bsf = knn[k - 1].distance;
                    float dist = l2sq(query, ts_buf.data() + (size_t)idx * dim, dim, kth_bsf);
                    if (dist < kth_bsf)
                        knn_bounded_insert(knn, k, (idx_t)leaf->series_indices[idx], dist);
                }
            }
        } }

    static void hercules_find_candidate_series(const std::vector<LeafCandidate> &lclist,
                                               const std::vector<sax_type> &sax_cache,
                                               const std::vector<float> &query_paa,
                                               const std::vector<sax_type> &max_sax_cardinalities,
                                               sax_type sax_bit_cardinality,
                                               int sax_cardinality,
                                               int paa_segments,
                                               float mindist_sqrt,
                                               float kth_bsf,
                                               std::vector<SeriesCandidate> &sclist)
    {
        for (size_t ci = 0; ci < lclist.size(); ci++){
            HerculesNode *leaf = lclist[ci].node;

            if (lclist[ci].lb <= kth_bsf){
                for (unsigned int i = 0; i < leaf->node_size; i++){
                    uint64_t idx = leaf->file_pos + i;
                    sax_type *sax = const_cast<sax_type *>(&sax_cache[idx * paa_segments]);

                    float mindist = minidist_paa_to_isax(
                        const_cast<float *>(query_paa.data()),
                        sax,
                        const_cast<sax_type *>(max_sax_cardinalities.data()),
                        sax_bit_cardinality,
                        sax_cardinality,
                        paa_segments,
                        MINVAL,
                        MAXVAL,
                        mindist_sqrt);
                    if (mindist <= kth_bsf){
                        SeriesCandidate sc;
                        sc.record_idx = idx;
                        sc.series_idx = (idx_t)leaf->series_indices[i];
                        sc.lb_sax = mindist;
                        sclist.push_back(sc);
                    }
                }
            }
        }
    }

    static void *hercules_cs_worker(void *arg)
    {
        HerculesCSWorkerData *d = (HerculesCSWorkerData *)arg;

        while (true){
            unsigned int ci = d->cs_idx->fetch_add(1u, std::memory_order_relaxed);
            if (ci >= (unsigned int)d->lclist->size()) break;

            const LeafCandidate &lc = (*d->lclist)[ci];
            float kth_bsf = (*d->knn)[d->k - 1].distance;

            if (lc.lb <= kth_bsf){
                HerculesNode *leaf = lc.node;
                for (unsigned int i = 0; i < leaf->node_size; i++){
                    uint64_t idx = leaf->file_pos + i;
                    sax_type *sax = const_cast<sax_type *>(&(*d->sax_cache)[idx * d->paa_segments]);
                    float mindist = minidist_paa_to_isax(
                        const_cast<float *>(d->query_paa),
                        sax,
                        const_cast<sax_type *>(d->max_sax_cardinalities),
                        d->sax_bit_cardinality,
                        d->sax_cardinality,
                        d->paa_segments,
                        MINVAL, MAXVAL,
                        d->mindist_sqrt);
                    if (mindist <= kth_bsf){
                        SeriesCandidate sc;
                        sc.record_idx = idx;
                        sc.series_idx = (idx_t)leaf->series_indices[i];
                        sc.lb_sax = mindist;
                        d->local_sclist.push_back(sc);
                    }
                }
            }
        }
        return nullptr;
    }

    static void *hercules_cr_worker(void *arg)
    {
        HerculesCRWorkerData *d = (HerculesCRWorkerData *)arg;
        FILE *raw_file = fopen(d->raw_path, "rb");
        if (raw_file == nullptr) return nullptr;

        std::vector<float> ts_buf(d->dim);

        while (true){
            unsigned int t = d->cr_idx->fetch_add(1u, std::memory_order_relaxed);
            if (t >= (unsigned int)d->sclist->size()) break;

            const SeriesCandidate &sc = (*d->sclist)[t];
            float kth_bsf = (*d->knn)[d->k - 1].distance;

            if (sc.lb_sax <= kth_bsf){
                fseek(raw_file, (long)sc.record_idx * d->dim * (long)sizeof(float), SEEK_SET);
                fread(ts_buf.data(), sizeof(float), d->dim, raw_file);
                float dist = l2sq(d->query, ts_buf.data(), d->dim, kth_bsf);
                if (dist < kth_bsf){
                    pthread_rwlock_wrlock(d->lock_bsf);
                    kth_bsf = (*d->knn)[d->k - 1].distance;
                    if (dist < kth_bsf)
                        knn_bounded_insert(*d->knn, d->k, sc.series_idx, dist);
                    pthread_rwlock_unlock(d->lock_bsf);
                }
            }
        }

        fclose(raw_file);
        return nullptr;
    }

    void hercules_knn_search(HerculesNode *root, const float *query, int dim,
                              idx_t k, idx_t *I, float *D,
                              const char *root_dir, float epsilon, int approx_leaves,
                              int paa_segments, sax_type sax_bit_cardinality, int sax_cardinality,
                              float eapca_th, float sax_th, int n_series,
                              int num_query_threads)
    {
        std::string raw_path = std::string(root_dir) + "/leaves_raw.idx";
        FILE *raw_file = fopen(raw_path.c_str(), "rb");
        if (raw_file == nullptr){
            fprintf(stderr, "hercules_knn_search: cannot open %s\n", raw_path.c_str());
            return;
        }

        std::vector<HerculesKnnResult> knn(k);
        std::vector<float> ts_buf;

        HerculesPQ pq;
        pq.push(std::make_pair(calculate_node_min_distance(root, query), root));

        // approximate phase
        approximate_knn_search(query, dim, pq, raw_file, ts_buf, knn, k, approx_leaves);

        // exact candidate collection → LCList
        std::vector<LeafCandidate> lclist;

        while (!pq.empty()){
            std::pair<float, HerculesNode *> top = pq.top();
            pq.pop();
            float lb = top.first;
            HerculesNode *node = top.second;

            float kth_bsf = knn[k - 1].distance;
            if (lb > kth_bsf / (1.0f + epsilon))
                break;

            if (node->is_leaf){
                LeafCandidate c;
                c.node = node;
                c.lb = lb;
                lclist.push_back(c);
            } else {
                kth_bsf = knn[k - 1].distance;
                float child_lb;

                child_lb = calculate_node_min_distance(node->left_child, query);
                if (child_lb < kth_bsf / (1.0f + epsilon))
                    pq.push(std::make_pair(child_lb, node->left_child));

                child_lb = calculate_node_min_distance(node->right_child, query);
                if (child_lb < kth_bsf / (1.0f + epsilon))
                    pq.push(std::make_pair(child_lb, node->right_child));
            }
        }

        // sort LCList by file_pos ascending (sequential I/O)
        std::sort(lclist.begin(), lclist.end(),
                  [](const LeafCandidate &a, const LeafCandidate &b){
                      return a.node->file_pos < b.node->file_pos;
                  });

        // 1st level threshold: if EAPCA pruned too few leaves -> skip to sequential scan
        int total_leaves = count_leaves(root);
        float first_level_pruning = 1.0f - (float)lclist.size() / (float)total_leaves;

        if (first_level_pruning < eapca_th){
            hercules_skip_sequential_scan(lclist, query, dim, raw_file, ts_buf, knn, k, epsilon);
        } else {
            // load entire leaves_sims.idx into sax_cache
            std::string sims_path = std::string(root_dir) + "/leaves_sims.idx";
            FILE *sims_file = fopen(sims_path.c_str(), "rb");
            if (sims_file == nullptr){
                // sims file not available: fall back to sequential scan
                hercules_skip_sequential_scan(lclist, query, dim, raw_file, ts_buf, knn, k, epsilon);
            } else {
                fseek(sims_file, 0, SEEK_END);
                long sims_size = ftell(sims_file);
                rewind(sims_file);
                size_t n_sax_words = (size_t)sims_size / ((size_t)paa_segments * sizeof(sax_type));
                std::vector<sax_type> sax_cache(n_sax_words * paa_segments);
                fread(sax_cache.data(), sizeof(sax_type), n_sax_words * paa_segments, sims_file);
                fclose(sims_file);

                // compute query PAA
                std::vector<float> query_paa(paa_segments);
                paa_from_ts(query, query_paa.data(), paa_segments, dim / paa_segments);

                // max_sax_cardinalities: all segments at the same bit cardinality
                std::vector<sax_type> max_sax_cardinalities(paa_segments, sax_bit_cardinality);

                // mindist_sqrt = ts_length / paa_segments 
                float mindist_sqrt = (float)(dim / paa_segments);

                // SAX filtering -> SCList
                std::atomic<unsigned int> cs_idx(0u);
                std::vector<HerculesCSWorkerData> cs_data((size_t)num_query_threads);
                for (int ti = 0; ti < num_query_threads; ti++){
                    cs_data[ti].lclist = &lclist;
                    cs_data[ti].cs_idx = &cs_idx;
                    cs_data[ti].sax_cache = &sax_cache;
                    cs_data[ti].query_paa = query_paa.data();
                    cs_data[ti].max_sax_cardinalities = max_sax_cardinalities.data();
                    cs_data[ti].sax_bit_cardinality = sax_bit_cardinality;
                    cs_data[ti].sax_cardinality = sax_cardinality;
                    cs_data[ti].paa_segments = paa_segments;
                    cs_data[ti].mindist_sqrt = mindist_sqrt;
                    cs_data[ti].knn = &knn;
                    cs_data[ti].k = k;
                }

                std::vector<pthread_t> cs_threads((size_t)num_query_threads);
                for (int ti = 0; ti < num_query_threads; ti++)
                    pthread_create(&cs_threads[ti], nullptr, hercules_cs_worker, (void *)&cs_data[ti]);
                for (int ti = 0; ti < num_query_threads; ti++)
                    pthread_join(cs_threads[ti], nullptr);

                // merge thread-local SCLists
                std::vector<SeriesCandidate> sclist;
                for (int ti = 0; ti < num_query_threads; ti++){
                    sclist.insert(sclist.end(),
                                  cs_data[ti].local_sclist.begin(),
                                  cs_data[ti].local_sclist.end());
                }

                // if SAX pruned too few series, then fall back to sequential scan
                float second_level_pruning = 1.0f - (float)sclist.size() / (float)n_series;

                if (second_level_pruning < sax_th){
                    hercules_skip_sequential_scan(lclist, query, dim, raw_file, ts_buf, knn, k, epsilon);
                } else {
                    // exact distance computation
                    std::string raw_path_str = std::string(root_dir) + "/leaves_raw.idx";
                    pthread_rwlock_t lock_bsf = PTHREAD_RWLOCK_INITIALIZER;
                    std::atomic<unsigned int> cr_idx(0u);

                    std::vector<HerculesCRWorkerData> cr_data((size_t)num_query_threads);
                    for (int ti = 0; ti < num_query_threads; ti++){
                        cr_data[ti].sclist = &sclist;
                        cr_data[ti].cr_idx = &cr_idx;
                        cr_data[ti].query = query;
                        cr_data[ti].dim = dim;
                        cr_data[ti].k = k;
                        cr_data[ti].raw_path = raw_path_str.c_str();
                        cr_data[ti].knn = &knn;
                        cr_data[ti].lock_bsf = &lock_bsf;
                    }

                    std::vector<pthread_t> cr_threads((size_t)num_query_threads);
                    for (int ti = 0; ti < num_query_threads; ti++)
                        pthread_create(&cr_threads[ti], nullptr, hercules_cr_worker, (void *)&cr_data[ti]);
                    for (int ti = 0; ti < num_query_threads; ti++)
                        pthread_join(cr_threads[ti], nullptr);

                    pthread_rwlock_destroy(&lock_bsf);
                }
            }
        }

        fclose(raw_file);

        for (idx_t i = 0; i < k; i++){
            I[i] = knn[i].series_idx;
            D[i] = knn[i].distance;
        }
    }

} 
