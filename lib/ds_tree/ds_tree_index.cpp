#include "ds_tree_index.hpp"

#include <cmath>
#include <cfloat>
#include <climits>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include <atomic>
#include <pthread.h>

namespace daisy
{

    void calc_mean_stdev(float *series, int start, int end, float *mean, float *stdev)
    {
        float sum_x_squares = 0, sum_x = 0;
        int count_x;

        *stdev = 0;
        *mean  = 0;

        if (start >= end) {
            fprintf(stderr, "error in stdev start >= end\n");
        } else {
            count_x = end - start;
            for (int i = start; i < end; i++) {
                sum_x += series[i];
                sum_x_squares += series[i] * series[i];
            }
            *mean = sum_x / count_x;
            sum_x_squares -= ((sum_x * sum_x) / count_x);
            *stdev = sqrtf(sum_x_squares / count_x);
        }
    }

    float calc_mean(float *series, int start, int end)
    {
        float mean = 0;
        if (start >= end) {
            fprintf(stderr, "error start > end \n");
        } else {
            for (int i = start; i < end; i++)
                mean += series[i];
            mean /= (end - start);
        }
        return mean;
    }

    int get_segment_start(const std::vector<int> &points, int idx)
    {
        return (idx == 0) ? 0 : points[idx - 1];
    }

    int get_segment_end(const std::vector<int> &points, int idx)
    {
        return points[idx];
    }

    int get_segment_length(const std::vector<int> &points, int i)
    {
        return (i == 0) ? points[i] : points[i] - points[i - 1];
    }

    bool series_segment_sketch_do_sketch(SegmentSketch *series_segment_sketch,
                                         float *series, int fromIdx, int toIdx)
    {
        calc_mean_stdev(series, fromIdx, toIdx,
                        &series_segment_sketch->indicators[0],
                        &series_segment_sketch->indicators[1]);
        return true;
    }

    bool node_segment_sketch_update_sketch(SegmentSketch *node_segment_sketch,
                                           float *series, int fromIdx, int toIdx)
    {
        SegmentSketch series_segment_sketch;
        if (!series_segment_sketch_do_sketch(&series_segment_sketch, series, fromIdx, toIdx)) {
            fprintf(stderr, "Error in node_segment_sketch_update_sketch: could not calculate series sketch.\n");
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
        for (int i = 0; i < (int)node->node_points.size(); i++) {
            int from = get_segment_start(node->node_points, i);
            int to   = get_segment_end(node->node_points, i);
            if (!node_segment_sketch_update_sketch(&node->node_segment_sketches[i], timeseries, from, to)) {
                fprintf(stderr, "Error: could not update vertical sketch.\n");
                return false;
            }
        }
        for (int i = 0; i < (int)node->hs_node_points.size(); i++) {
            if (!node_segment_sketch_update_sketch(&node->hs_node_segment_sketches[i], timeseries,
                                                   get_segment_start(node->hs_node_points, i),
                                                   get_segment_end(node->hs_node_points, i))) {
                fprintf(stderr, "Error: could not update horizontal sketch.\n");
                return false;
            }
        }
        ++node->node_size;
        return true;
    }

    bool calc_hs_split_points(std::vector<int> &hs_split_points,
                              const std::vector<int> &split_points, int min_length)
    {
        int c = 0;
        int segment_size = (int)split_points.size();
        hs_split_points.resize(segment_size * 2);

        for (int i = 0; i < segment_size; i++) {
            int length = (i == 0) ? split_points[i] : split_points[i] - split_points[i - 1];
            if (length >= min_length * 2) {
                int start = (i > 0) ? split_points[i - 1] : 0;
                hs_split_points[c++] = start + (length / 2);
            }
            hs_split_points[c++] = split_points[i];
        }
        hs_split_points.resize(c);
        return true;
    }

    bool calc_split_points(std::vector<int> &points, int ts_length, int segment_size)
    {
        int avg_length = ts_length / segment_size;
        points.resize(segment_size);
        for (int i = 0; i < segment_size; ++i)
            points[i] = (i + 1) * avg_length;
        points[segment_size - 1] = ts_length;
        return true;
    }

    bool node_init_segments(HerculesNode *node, const std::vector<int> &split_points)
    {
        node->node_points = split_points;
        calc_hs_split_points(node->hs_node_points, node->node_points, 1);
        node->node_segment_sketches.resize(node->node_points.size());
        node->hs_node_segment_sketches.resize(node->hs_node_points.size());
        return true;
    }

    SegmentSketch *init_segment_sketches(int num_child_segments)
    {
        return new SegmentSketch[num_child_segments];
    }

    HerculesNode *create_child_node(HerculesNode *parent)
    {
        HerculesNode *child = new HerculesNode();
        if (child == nullptr) {
            fprintf(stderr, "Error: could not initialize child node.\n");
            return nullptr;
        }
        child->parent  = parent;
        child->level   = parent->level + 1;
        child->is_leaf = true;
        return child;
    }

    static bool mean_node_segment_split_policy_split(node_split_policy *policy,
                                                      SegmentSketch sketch,
                                                      SegmentSketch *ret)
    {
        const int num_splits = 2;
        policy->indicator_split_idx = 0;
        policy->indicator_split_val = (sketch.indicators[0] + sketch.indicators[1]) / 2;

        for (int i = 0; i < num_splits; i++) {
            ret[i].num_indicators = sketch.num_indicators;
            for (int j = 0; j < ret[i].num_indicators; ++j)
                ret[i].indicators[j] = sketch.indicators[j];
        }
        ret[0].indicators[1] = policy->indicator_split_val;
        ret[1].indicators[0] = policy->indicator_split_val;
        return true;
    }

    static bool stdev_node_segment_split_policy_split(node_split_policy *policy,
                                                       SegmentSketch sketch,
                                                       SegmentSketch *ret)
    {
        const int num_splits = 2;
        policy->indicator_split_idx = 1;
        policy->indicator_split_val = (sketch.indicators[2] + sketch.indicators[3]) / 2;

        for (int i = 0; i < num_splits; i++) {
            ret[i].num_indicators = sketch.num_indicators;
            for (int j = 0; j < ret[i].num_indicators; ++j)
                ret[i].indicators[j] = sketch.indicators[j];
        }
        ret[0].indicators[2] = policy->indicator_split_val;
        ret[1].indicators[3] = policy->indicator_split_val;
        return true;
    }

    static int get_hs_split_point(const std::vector<int> &points, int from, int to)
    {
        return std::binary_search(points.begin(), points.end(), to) ? from : to;
    }

    float range_calc(SegmentSketch sketch, int len)
    {
        float mean_width  = sketch.indicators[0] - sketch.indicators[1];
        float stdev_upper = sketch.indicators[2];
        return (float)len * (mean_width * mean_width + stdev_upper * stdev_upper);
    }

    bool node_split_policy_route_to_left(HerculesNode *node, float *series,
                                         SegmentSketch *series_segment_sketch)
    {
        node_split_policy *policy = node->split_policy;
        if (!series_segment_sketch_do_sketch(series_segment_sketch, series,
                                             policy->split_from, policy->split_to)) {
            fprintf(stderr, "Error: could not calculate series segment sketch.\n");
        }
        return series_segment_sketch->indicators[policy->indicator_split_idx] < policy->indicator_split_val;
    }

    void split_node_evaluate_policies(HerculesNode *node, int *hs_split_point)
    {
        node_split_policy best;
        float max_diff = -FLT_MAX;
        *hs_split_point = -1;
        const int num_child_segments = 2;

        for (int i = 0; i < (int)node->node_points.size(); ++i) {
            SegmentSketch curr = node->node_segment_sketches[i];
            float node_range   = range_calc(curr, get_segment_length(node->node_points, i));

            for (int j = 0; j < 2; ++j) {
                node_split_policy tmp;
                SegmentSketch *children = init_segment_sketches(num_child_segments);

                if (j == 0) mean_node_segment_split_policy_split(&tmp, curr, children);
                else        stdev_node_segment_split_policy_split(&tmp, curr, children);

                float avg_child_range = (range_calc(children[0], get_segment_length(node->node_points, i)) +
                                         range_calc(children[1], get_segment_length(node->node_points, i))) / 2.0f;
                float diff = node_range - avg_child_range;

                if (diff > max_diff) {
                    max_diff = diff;
                    best.split_from           = get_segment_start(node->node_points, i);
                    best.split_to             = get_segment_end(node->node_points, i);
                    best.indicator_split_idx  = tmp.indicator_split_idx;
                    best.indicator_split_val  = tmp.indicator_split_val;
                }
                delete[] children;
            }
        }

        max_diff *= 2;

        for (int i = 0; i < (int)node->hs_node_points.size(); ++i) {
            SegmentSketch curr = node->hs_node_segment_sketches[i];
            float node_range   = range_calc(curr, get_segment_length(node->hs_node_points, i));

            for (int j = 0; j < 2; ++j) {
                node_split_policy tmp;
                SegmentSketch *children = init_segment_sketches(num_child_segments);

                if (j == 0) mean_node_segment_split_policy_split(&tmp, curr, children);
                else        stdev_node_segment_split_policy_split(&tmp, curr, children);

                float avg_child_range = (range_calc(children[0], get_segment_length(node->hs_node_points, i)) +
                                         range_calc(children[1], get_segment_length(node->hs_node_points, i))) / 2.0f;
                float diff = node_range - avg_child_range;

                if (diff > max_diff) {
                    max_diff = diff;
                    best.split_from           = get_segment_start(node->hs_node_points, i);
                    best.split_to             = get_segment_end(node->hs_node_points, i);
                    best.indicator_split_idx  = tmp.indicator_split_idx;
                    best.indicator_split_val  = tmp.indicator_split_val;
                    *hs_split_point = get_hs_split_point(node->node_points,
                                                         best.split_from, best.split_to);
                }
                delete[] children;
            }
        }

        delete node->split_policy;
        node->split_policy = new node_split_policy(best);
    }

    bool split_node_create_children(HerculesNode *node, std::vector<int> &child_node_points)
    {
        if (!node->is_leaf) {
            fprintf(stderr, "Error: trying to split a non-leaf node.\n");
            return false;
        }
        node->left_child = create_child_node(node);
        if (!node->left_child || !node_init_segments(node->left_child, child_node_points))
            return false;
        node->left_child->is_leaf = true;
        node->left_child->is_left = true;

        node->right_child = create_child_node(node);
        if (!node->right_child || !node_init_segments(node->right_child, child_node_points))
            return false;
        node->right_child->is_leaf = true;

        return true;
    }

    void split_node(HerculesNode *node, SegmentSketch *sketch,
                    const float *database, int dim)
    {
        int hs_split_point = -1;
        std::vector<int> child_node_points;

        split_node_evaluate_policies(node, &hs_split_point);

        if (hs_split_point < 0) {
            child_node_points = node->node_points;
        } else {
            child_node_points = node->node_points;
            child_node_points.push_back(hs_split_point);
            std::sort(child_node_points.begin(), child_node_points.end());
        }

        if (!split_node_create_children(node, child_node_points)) {
            fprintf(stderr, "Error: could not split node.\n");
            return;
        }

        for (int idx = 0; idx < (int)node->series_indices.size(); ++idx) {
            float *ts = const_cast<float *>(database + (long)node->series_indices[idx] * dim);
            SegmentSketch series_sketch;
            if (node_split_policy_route_to_left(node, ts, &series_sketch)) {
                update_node_statistics(node->left_child, ts);
                node->left_child->series_indices.push_back(node->series_indices[idx]);
            } else {
                update_node_statistics(node->right_child, ts);
                node->right_child->series_indices.push_back(node->series_indices[idx]);
            }
        }

        node->is_leaf = false;
        node->series_indices.clear();
    }

    // ---- parallel insert workers (Linux only) ----
#ifndef __APPLE__

    static void hercules_parallel_insert(HerculesNode *root, float *timeseries,
                                          int series_global_idx, const float *database_orig,
                                          int leaf_size, int dim)
    {
        HerculesNode *node = root;
        SegmentSketch sketch;

        while (!node->is_leaf) {
            if (node_split_policy_route_to_left(node, timeseries, &sketch))
                node = node->left_child;
            else
                node = node->right_child;
        }

        pthread_mutex_lock(&node->lock_data);
        while (!node->is_leaf) {
            pthread_mutex_unlock(&node->lock_data);
            while (!node->is_leaf) {
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

        if (__sync_fetch_and_add(d->flush_order, 0) != 0) {
            int buffer_counter = d->buffer_counter;
            pthread_barrier_wait(d->flush_barrier);
            __sync_fetch_and_sub(&d->buffer_counter, buffer_counter);
        }
    }

    static void hercules_flush_coordinator(void *transferdata)
    {
        HerculesInsertWorkerData *d = (HerculesInsertWorkerData *)transferdata;
        int num_threads = d->num_insert_workers;
        int worker_id   = d->thread_id;

        __sync_fetch_and_add(&d->threads_data[worker_id].continue_handshake, 1);

        for (int j = 0; j < num_threads; j++)
            while (__sync_fetch_and_add(&d->threads_data[j].continue_handshake, 0) != 1);

        if (__sync_fetch_and_add(d->flush_counter, 0) >= INT_MAX ||
            d->buffer_counter >= (int)d->buffer_max)
            __sync_fetch_and_add(d->flush_order, 1);

        int num_full = *d->flush_counter;
        __sync_fetch_and_sub(d->flush_counter, num_full);

        pthread_barrier_wait(d->continue_barrier);
        __sync_fetch_and_sub(&d->threads_data[worker_id].continue_handshake, 1);

        if (__sync_fetch_and_add(d->flush_order, 0) != 0) {
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

        while (!d->finished[toggle]) {
            if (d->buffer_counter < (int)d->buffer_max) {
                unsigned int ts_id = __sync_fetch_and_add(&d->db_counter[toggle], 1);
                while (ts_id < d->db_size[toggle]) {
                    unsigned int offset = ts_id * (unsigned int)d->dim;
                    float *timeseries = d->double_buffer[toggle] + offset;
                    int global_idx    = (int)(d->db_global_start[toggle] + ts_id);

                    hercules_parallel_insert(d->root, timeseries, global_idx,
                                              d->database_orig, d->leaf_size, d->dim);

                    ts_id = __sync_fetch_and_add(&d->db_counter[toggle], 1);
                }
            }

            pthread_barrier_wait(d->db_barrier);

            if (d->is_flusher) hercules_flush_coordinator(transferdata);
            else               hercules_flush_worker(transferdata);

            toggle = 1 - toggle;
        }
        return nullptr;
    }

#endif // __APPLE__

    // ---- index build ----
    HerculesNode *hercules_index_build(const float *database, int n, int dim,
                                       int leaf_size, int init_segments,
                                       int num_build_threads)
    {
        std::vector<int> split_points;
        if (!calc_split_points(split_points, dim, init_segments)) {
            fprintf(stderr, "Error in hercules_index_build: could not calculate split points.\n");
            return nullptr;
        }

        HerculesNode *root = new HerculesNode();
        if (!node_init_segments(root, split_points)) {
            fprintf(stderr, "Error in hercules_index_build: could not initialize root segments.\n");
            delete root;
            return nullptr;
        }

        if (num_build_threads <= 1) {
            for (int i = 0; i < n; ++i) {
                float *ts = const_cast<float *>(database + (long)i * dim);
                HerculesNode *node = root;
                while (!node->is_leaf) {
                    SegmentSketch sketch;
                    if (node_split_policy_route_to_left(node, ts, &sketch))
                        node = node->left_child;
                    else
                        node = node->right_child;
                }
                update_node_statistics(node, ts);
                node->series_indices.push_back(i);
                if (node->node_size >= (unsigned int)leaf_size) {
                    SegmentSketch sketch;
                    split_node(node, &sketch, database, dim);
                }
            }
            return root;
        }

#ifndef __APPLE__
        int num_workers = num_build_threads - 1;
        unsigned int initial_db_size = (unsigned int)n;

        pthread_barrier_t db_barrier, continue_barrier, flush_barrier;
        pthread_barrier_init(&db_barrier,       NULL, (unsigned int)num_build_threads);
        pthread_barrier_init(&continue_barrier, NULL, (unsigned int)num_workers);
        pthread_barrier_init(&flush_barrier,    NULL, (unsigned int)num_workers);

        unsigned int db_size[2]        = {0, 0};
        unsigned int db_counter[2]     = {0, 0};
        unsigned int db_global_start[2]= {0, 0};
        bool finished[2]               = {false, false};
        int  flush_counter             = 0;
        int  flush_order               = 0;
        int  toggle                    = 0;

        db_size[toggle] = (unsigned int)std::min((unsigned long long)initial_db_size,
                                                  (unsigned long long)n);

        float **double_buffer = (float **)calloc(2, sizeof(float *));
        double_buffer[0] = (float *)malloc(sizeof(float) * (size_t)dim * initial_db_size);
        double_buffer[1] = (float *)malloc(sizeof(float) * (size_t)dim * initial_db_size);

        memcpy(double_buffer[toggle], database,
               sizeof(float) * (size_t)dim * db_size[toggle]);
        db_global_start[toggle] = 0;
        toggle = 1 - toggle;

        HerculesInsertWorkerData *input_data =
            (HerculesInsertWorkerData *)malloc(sizeof(HerculesInsertWorkerData) * num_workers);

        for (int i = 0; i < num_workers; i++) {
            input_data[i].threads_data      = input_data;
            input_data[i].root              = root;
            input_data[i].database_orig     = database;
            input_data[i].db_barrier        = &db_barrier;
            input_data[i].continue_barrier  = &continue_barrier;
            input_data[i].flush_barrier     = &flush_barrier;
            input_data[i].buffer_max        = (unsigned int)INT_MAX;
            input_data[i].buffer_counter    = 0;
            input_data[i].continue_handshake= 0;
            input_data[i].is_flusher        = (i == 0);
            input_data[i].num_insert_workers= num_workers;
            input_data[i].thread_id         = i;
            input_data[i].double_buffer     = double_buffer;
            input_data[i].db_size           = db_size;
            input_data[i].db_counter        = db_counter;
            input_data[i].finished          = finished;
            input_data[i].flush_counter     = &flush_counter;
            input_data[i].flush_order       = &flush_order;
            input_data[i].initial_db_size   = initial_db_size;
            input_data[i].db_global_start   = db_global_start;
            input_data[i].dim               = dim;
            input_data[i].leaf_size         = leaf_size;
        }

        pthread_t *threadid = (pthread_t *)malloc(sizeof(pthread_t) * num_workers);
        for (int j = 0; j < num_workers; j++)
            pthread_create(&threadid[j], NULL, hercules_insert_worker, (void *)&input_data[j]);

        for (unsigned long long i = (unsigned long long)db_size[1 - toggle];
             i < (unsigned long long)n;
             i += (unsigned long long)db_size[toggle]) {
            db_size[toggle] = (unsigned int)std::min((unsigned long long)initial_db_size,
                                                      (unsigned long long)n - i);
            memcpy(double_buffer[toggle], database + (long)i * dim,
                   sizeof(float) * (size_t)dim * db_size[toggle]);
            db_global_start[toggle] = (unsigned int)i;
            db_counter[toggle]      = 0;
            toggle = 1 - toggle;
            pthread_barrier_wait(&db_barrier);
        }

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
        // macOS fallback: sequential build (pthread_barrier_t unavailable)
        for (int i = 0; i < n; ++i) {
            float *ts = const_cast<float *>(database + (long)i * dim);
            HerculesNode *node = root;
            while (!node->is_leaf) {
                SegmentSketch sketch;
                if (node_split_policy_route_to_left(node, ts, &sketch))
                    node = node->left_child;
                else
                    node = node->right_child;
            }
            update_node_statistics(node, ts);
            node->series_indices.push_back(i);
            if (node->node_size >= (unsigned int)leaf_size) {
                SegmentSketch sketch;
                split_node(node, &sketch, database, dim);
            }
        }
        return root;
#endif
    }

    HerculesNode *hercules_index_build(DataSource *ds, int leaf_size,
                                       int init_segments, int num_build_threads)
    {
        int n   = (int)ds->getTotalRecords();
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
        if (node == nullptr) return;
        destroy_tree(node->left_child);
        destroy_tree(node->right_child);
        delete node;
    }

} // namespace daisy
