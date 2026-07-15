#include "Hercules.hpp"
#include "../isax/SAX.hpp"

#include <cmath>
#include <cfloat>
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

// ---- disk I/O helpers ----

static uint64_t assign_file_pos(HerculesNode *node, uint64_t pos, unsigned long &leaf_count)
{
    if (!node->is_leaf) {
        pos = assign_file_pos(node->left_child,  pos, leaf_count);
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
    if (!node->is_leaf) {
        return write_leaf_data(node->left_child,  database, dim, paa_segments, sax_cardinality, sax_bit_cardinality, raw_file, sims_file) &&
               write_leaf_data(node->right_child, database, dim, paa_segments, sax_cardinality, sax_bit_cardinality, raw_file, sims_file);
    }

    int ts_values_per_segment = dim / paa_segments;
    std::vector<sax_type> sax(paa_segments);

    for (int idx = 0; idx < (int)node->series_indices.size(); idx++) {
        float *ts = const_cast<float *>(database + (long)node->series_indices[idx] * dim);
        fwrite(ts, sizeof(float), dim, raw_file);
        sax_from_ts(ts, sax.data(), ts_values_per_segment, paa_segments,
                    sax_cardinality, sax_bit_cardinality);
        fwrite(sax.data(), sizeof(sax_type), paa_segments, sims_file);
    }
    return true;
}

static bool write_tree_node(HerculesNode *node, FILE *file)
{
    unsigned char is_leaf = (unsigned char)node->is_leaf;
    fwrite(&is_leaf,       sizeof(unsigned char), 1, file);
    fwrite(&node->level,   sizeof(unsigned int),  1, file);

    if (!node->is_leaf) {
        if (!write_tree_node(node->left_child,  file)) return false;
        if (!write_tree_node(node->right_child, file)) return false;

        fwrite(&node->node_size, sizeof(unsigned int), 1, file);
        if (node->split_policy == nullptr) {
            fprintf(stderr, "Error in write_tree_node: internal node has null split_policy.\n");
            return false;
        }
        fwrite(node->split_policy, sizeof(node_split_policy), 1, file);

        int num_pts = (int)node->node_points.size();
        fwrite(&num_pts, sizeof(int), 1, file);
        fwrite(node->node_points.data(), sizeof(int), num_pts, file);
        for (int i = 0; i < num_pts; i++) {
            fwrite(&node->node_segment_sketches[i].num_indicators, sizeof(int), 1, file);
            fwrite(node->node_segment_sketches[i].indicators, sizeof(float),
                   node->node_segment_sketches[i].num_indicators, file);
        }
    } else {
        fwrite(&node->node_size, sizeof(unsigned int), 1, file);

        int num_pts = (int)node->node_points.size();
        fwrite(&num_pts, sizeof(int), 1, file);
        fwrite(node->node_points.data(), sizeof(int), num_pts, file);
        for (int i = 0; i < num_pts; i++) {
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
    mkdir(root_dir, 0755);

    std::string raw_path = std::string(root_dir) + "/leaves_raw.idx";
    std::string sims_path = std::string(root_dir) + "/leaves_sims.idx";
    std::string tree_path = std::string(root_dir) + "/root.idx";

    FILE *raw_file = fopen(raw_path.c_str(), "wb");
    FILE *sims_file = fopen(sims_path.c_str(), "wb");
    FILE *tree_file = fopen(tree_path.c_str(), "wb");

    if (!raw_file || !sims_file || !tree_file) {
        fprintf(stderr, "Error in hercules_index_write: could not open output files in %s\n", root_dir);
        if (raw_file) fclose(raw_file);
        if (sims_file) fclose(sims_file);
        if (tree_file) fclose(tree_file);
        return false;
    }

    unsigned long leaf_count = 0;
    unsigned int ts_size = (unsigned int)dim;
    unsigned int init_seg = (unsigned int)init_segments;
    unsigned int max_leaf = (unsigned int)leaf_size;

    fwrite(&leaf_count, sizeof(unsigned long), 1, tree_file);
    fwrite(&ts_size, sizeof(unsigned int), 1, tree_file);
    fwrite(&init_seg, sizeof(unsigned int), 1, tree_file);
    fwrite(&paa_segments, sizeof(int), 1, tree_file);
    fwrite(&sax_bit_cardinality, sizeof(int), 1, tree_file);
    fwrite(&max_leaf, sizeof(unsigned int), 1, tree_file);

    assign_file_pos(root, 0, leaf_count);

    if (!write_leaf_data(root, database, dim, paa_segments, sax_cardinality,
                         sax_bit_cardinality, raw_file, sims_file)) {
        fprintf(stderr, "Error in hercules_index_write: could not write leaf data.\n");
        fclose(raw_file); fclose(sims_file); fclose(tree_file);
        return false;
    }

    if (!write_tree_node(root, tree_file)) {
        fprintf(stderr, "Error in hercules_index_write: could not write tree node.\n");
        fclose(raw_file); fclose(sims_file); fclose(tree_file);
        return false;
    }

    fseek(tree_file, 0L, SEEK_SET);
    fwrite(&leaf_count, sizeof(unsigned long), 1, tree_file);

    fclose(raw_file);
    fclose(sims_file);
    fclose(tree_file);
    return true;
}

// ---- Hercules-specific search ----

static int count_leaves(HerculesNode *node)
{
    if (node->is_leaf) return 1;
    return count_leaves(node->left_child) + count_leaves(node->right_child);
}

static void hercules_skip_sequential_scan(const std::vector<LeafCandidate> &lclist,
                                          const float *query, int dim,
                                          FILE *raw_file,
                                          std::vector<float> &ts_buf,
                                          std::vector<HerculesKnnResult> &knn,
                                          idx_t k, float epsilon)
{
    for (size_t i = 0; i < lclist.size(); i++) {
        HerculesNode *leaf = lclist[i].node;
        float kth_bsf = knn[k - 1].distance;

        if (lclist[i].lb <= kth_bsf / (1.0f + epsilon)) {
            ts_buf.resize((size_t)leaf->node_size * dim);
            fseek(raw_file, (long)leaf->file_pos * dim * (long)sizeof(float), SEEK_SET);
            fread(ts_buf.data(), sizeof(float), (size_t)leaf->node_size * dim, raw_file);

            for (unsigned int idx = 0; idx < leaf->node_size; idx++) {
                kth_bsf = knn[k - 1].distance;
                float dist = l2sq(query, ts_buf.data() + (size_t)idx * dim, dim, kth_bsf);
                if (dist < kth_bsf)
                    knn_bounded_insert(knn, k, (idx_t)leaf->series_indices[idx], dist);
            }
        }
    }
}

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
    for (size_t ci = 0; ci < lclist.size(); ci++) {
        HerculesNode *leaf = lclist[ci].node;
        if (lclist[ci].lb <= kth_bsf) {
            for (unsigned int i = 0; i < leaf->node_size; i++) {
                uint64_t idx = leaf->file_pos + i;
                sax_type *sax = const_cast<sax_type *>(&sax_cache[idx * paa_segments]);
                float mindist = minidist_paa_to_isax(
                    const_cast<float *>(query_paa.data()),
                    sax,
                    const_cast<sax_type *>(max_sax_cardinalities.data()),
                    sax_bit_cardinality, sax_cardinality, paa_segments,
                    MINVAL, MAXVAL, mindist_sqrt);
                if (mindist <= kth_bsf) {
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

    while (true) {
        unsigned int ci = d->cs_idx->fetch_add(1u, std::memory_order_relaxed);
        if (ci >= (unsigned int)d->lclist->size()) break;

        const LeafCandidate &lc = (*d->lclist)[ci];
        float kth_bsf = (*d->knn)[d->k - 1].distance;

        if (lc.lb <= kth_bsf) {
            HerculesNode *leaf = lc.node;
            for (unsigned int i = 0; i < leaf->node_size; i++) {
                uint64_t idx = leaf->file_pos + i;
                sax_type *sax = const_cast<sax_type *>(&(*d->sax_cache)[idx * d->paa_segments]);
                float mindist = minidist_paa_to_isax(
                    const_cast<float *>(d->query_paa),
                    sax,
                    const_cast<sax_type *>(d->max_sax_cardinalities),
                    d->sax_bit_cardinality, d->sax_cardinality, d->paa_segments,
                    MINVAL, MAXVAL, d->mindist_sqrt);
                if (mindist <= kth_bsf) {
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

    while (true) {
        unsigned int t = d->cr_idx->fetch_add(1u, std::memory_order_relaxed);
        if (t >= (unsigned int)d->sclist->size()) break;

        const SeriesCandidate &sc = (*d->sclist)[t];
        float kth_bsf = (*d->knn)[d->k - 1].distance;

        if (sc.lb_sax <= kth_bsf) {
            fseek(raw_file, (long)sc.record_idx * d->dim * (long)sizeof(float), SEEK_SET);
            fread(ts_buf.data(), sizeof(float), d->dim, raw_file);
            float dist = l2sq(d->query, ts_buf.data(), d->dim, kth_bsf);
            if (dist < kth_bsf) {
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
    if (raw_file == nullptr) {
        fprintf(stderr, "hercules_knn_search: cannot open %s\n", raw_path.c_str());
        return;
    }

    std::vector<HerculesKnnResult> knn(k);
    std::vector<float> ts_buf;

    HerculesPQ pq;
    pq.push(std::make_pair(calculate_node_min_distance(root, query), root));

    approximate_knn_search(query, dim, pq, raw_file, ts_buf, knn, k, approx_leaves);

    std::vector<LeafCandidate> lclist;

    while (!pq.empty()) {
        std::pair<float, HerculesNode *> top = pq.top();
        pq.pop();
        float lb = top.first;
        HerculesNode *node = top.second;

        float kth_bsf = knn[k - 1].distance;
        if (lb > kth_bsf / (1.0f + epsilon)) break;

        if (node->is_leaf) {
            LeafCandidate c;
            c.node = node;
            c.lb = lb;
            lclist.push_back(c);
        } else {
            kth_bsf = knn[k - 1].distance;

            float child_lb = calculate_node_min_distance(node->left_child, query);
            if (child_lb < kth_bsf / (1.0f + epsilon))
                pq.push(std::make_pair(child_lb, node->left_child));

            child_lb = calculate_node_min_distance(node->right_child, query);
            if (child_lb < kth_bsf / (1.0f + epsilon))
                pq.push(std::make_pair(child_lb, node->right_child));
        }
    }

    std::sort(lclist.begin(), lclist.end(),
              [](const LeafCandidate &a, const LeafCandidate &b) {
                  return a.node->file_pos < b.node->file_pos;
              });

    int total_leaves = count_leaves(root);
    float first_level_pruning = 1.0f - (float)lclist.size() / (float)total_leaves;

    if (first_level_pruning < eapca_th) {
        hercules_skip_sequential_scan(lclist, query, dim, raw_file, ts_buf, knn, k, epsilon);
    } else {
        std::string sims_path = std::string(root_dir) + "/leaves_sims.idx";
        FILE *sims_file = fopen(sims_path.c_str(), "rb");

        if (sims_file == nullptr) {
            hercules_skip_sequential_scan(lclist, query, dim, raw_file, ts_buf, knn, k, epsilon);
        } else {
            fseek(sims_file, 0, SEEK_END);
            long sims_size = ftell(sims_file);
            rewind(sims_file);
            size_t n_sax_words = (size_t)sims_size / ((size_t)paa_segments * sizeof(sax_type));
            std::vector<sax_type> sax_cache(n_sax_words * paa_segments);
            fread(sax_cache.data(), sizeof(sax_type), n_sax_words * paa_segments, sims_file);
            fclose(sims_file);

            std::vector<float>    query_paa(paa_segments);
            paa_from_ts(query, query_paa.data(), paa_segments, dim / paa_segments);

            std::vector<sax_type> max_sax_cardinalities(paa_segments, sax_bit_cardinality);
            float mindist_sqrt = (float)(dim / paa_segments);

            std::atomic<unsigned int> cs_idx(0u);
            std::vector<HerculesCSWorkerData> cs_data((size_t)num_query_threads);
            for (int ti = 0; ti < num_query_threads; ti++) {
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

            std::vector<SeriesCandidate> sclist;
            for (int ti = 0; ti < num_query_threads; ti++)
                sclist.insert(sclist.end(),
                              cs_data[ti].local_sclist.begin(),
                              cs_data[ti].local_sclist.end());

            float second_level_pruning = 1.0f - (float)sclist.size() / (float)n_series;

            if (second_level_pruning < sax_th) {
                hercules_skip_sequential_scan(lclist, query, dim, raw_file, ts_buf, knn, k, epsilon);
            } else {
                std::string raw_path_str = std::string(root_dir) + "/leaves_raw.idx";
                pthread_rwlock_t lock_bsf = PTHREAD_RWLOCK_INITIALIZER;
                std::atomic<unsigned int> cr_idx(0u);

                std::vector<HerculesCRWorkerData> cr_data((size_t)num_query_threads);
                for (int ti = 0; ti < num_query_threads; ti++) {
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

    for (idx_t i = 0; i < k; i++) {
        I[i] = knn[i].series_idx;
        D[i] = knn[i].distance;
    }
}

static void hercules_range_scan(const std::vector<LeafCandidate> &lclist,
                                const float *query, int dim,
                                FILE *raw_file, std::vector<float> &ts_buf,
                                float r,
                                std::vector<std::pair<float, idx_t>> &results)
{
    for (size_t i = 0; i < lclist.size(); i++) {
        HerculesNode *leaf = lclist[i].node;
        ts_buf.resize((size_t)leaf->node_size * dim);
        fseek(raw_file, (long)leaf->file_pos * dim * (long)sizeof(float), SEEK_SET);
        fread(ts_buf.data(), sizeof(float), (size_t)leaf->node_size * dim, raw_file);
        for (unsigned int idx = 0; idx < leaf->node_size; idx++) {
            float dist = l2sq(query, ts_buf.data() + (size_t)idx * dim, dim, r);
            if (dist <= r)
                results.push_back({dist, (idx_t)leaf->series_indices[idx]});
        }
    }
}

static void *hercules_cs_range_worker(void *arg)
{
    HerculesCSRangeWorkerData *d = (HerculesCSRangeWorkerData *)arg;
    while (true) {
        unsigned int ci = d->cs_idx->fetch_add(1u, std::memory_order_relaxed);
        if (ci >= (unsigned int)d->lclist->size()) break;
        const LeafCandidate &lc = (*d->lclist)[ci];
        HerculesNode *leaf = lc.node;
        for (unsigned int i = 0; i < leaf->node_size; i++) {
            uint64_t idx = leaf->file_pos + i;
            sax_type *sax = const_cast<sax_type *>(&(*d->sax_cache)[idx * d->paa_segments]);
            float mindist = minidist_paa_to_isax(
                const_cast<float *>(d->query_paa), sax,
                const_cast<sax_type *>(d->max_sax_cardinalities),
                d->sax_bit_cardinality, d->sax_cardinality, d->paa_segments,
                MINVAL, MAXVAL, d->mindist_sqrt);
            if (mindist <= d->r) {
                SeriesCandidate sc;
                sc.record_idx = idx;
                sc.series_idx = (idx_t)leaf->series_indices[i];
                sc.lb_sax = mindist;
                d->local_sclist.push_back(sc);
            }
        }
    }
    return nullptr;
}

static void *hercules_cr_range_worker(void *arg)
{
    HerculesCRRangeWorkerData *d = (HerculesCRRangeWorkerData *)arg;
    FILE *raw_file = fopen(d->raw_path, "rb");
    if (raw_file == nullptr) return nullptr;

    std::vector<float> ts_buf(d->dim);
    while (true) {
        unsigned int t = d->cr_idx->fetch_add(1u, std::memory_order_relaxed);
        if (t >= (unsigned int)d->sclist->size()) break;

        const SeriesCandidate &sc = (*d->sclist)[t];
        if (sc.lb_sax <= d->r) {
            fseek(raw_file, (long)sc.record_idx * d->dim * (long)sizeof(float), SEEK_SET);
            fread(ts_buf.data(), sizeof(float), d->dim, raw_file);
            float dist = l2sq(d->query, ts_buf.data(), d->dim, d->r);
            if (dist <= d->r) {
                pthread_rwlock_wrlock(d->lock_range_results);
                d->range_results->push_back({dist, sc.series_idx});
                pthread_rwlock_unlock(d->lock_range_results);
            }
        }
    }
    fclose(raw_file);
    return nullptr;
}

std::vector<std::pair<float, idx_t>> hercules_range_search(
    HerculesNode *root, const float *query, int dim,
    float r, const char *root_dir,
    int paa_segments, sax_type sax_bit_cardinality, int sax_cardinality,
    float eapca_th, float sax_th, int n_series,
    int num_query_threads)
{
    std::string raw_path = std::string(root_dir) + "/leaves_raw.idx";
    FILE *raw_file = fopen(raw_path.c_str(), "rb");
    if (raw_file == nullptr) {
        fprintf(stderr, "hercules_range_search: cannot open %s\n", raw_path.c_str());
        return {};
    }

    std::vector<std::pair<float, idx_t>> results;
    std::vector<float> ts_buf;

    HerculesPQ pq;
    pq.push(std::make_pair(calculate_node_min_distance(root, query), root));

    std::vector<LeafCandidate> lclist;
    while (!pq.empty()) {
        std::pair<float, HerculesNode *> top = pq.top();
        pq.pop();
        float lb = top.first;
        HerculesNode *node = top.second;

        if (lb > r) break;

        if (node->is_leaf) {
            LeafCandidate c;
            c.node = node;
            c.lb = lb;
            lclist.push_back(c);
        } else {
            float child_lb = calculate_node_min_distance(node->left_child, query);
            if (child_lb <= r)
                pq.push(std::make_pair(child_lb, node->left_child));
            child_lb = calculate_node_min_distance(node->right_child, query);
            if (child_lb <= r)
                pq.push(std::make_pair(child_lb, node->right_child));
        }
    }

    std::sort(lclist.begin(), lclist.end(),
              [](const LeafCandidate &a, const LeafCandidate &b) {
                  return a.node->file_pos < b.node->file_pos;
              });

    int total_leaves = count_leaves(root);
    float first_level_pruning = 1.0f - (float)lclist.size() / (float)total_leaves;

    if (first_level_pruning < eapca_th) {
        hercules_range_scan(lclist, query, dim, raw_file, ts_buf, r, results);
    } else {
        std::string sims_path = std::string(root_dir) + "/leaves_sims.idx";
        FILE *sims_file = fopen(sims_path.c_str(), "rb");
        if (sims_file == nullptr) {
            hercules_range_scan(lclist, query, dim, raw_file, ts_buf, r, results);
        } else {
            fseek(sims_file, 0, SEEK_END);
            long sims_size = ftell(sims_file);
            rewind(sims_file);
            size_t n_sax_words = (size_t)sims_size / ((size_t)paa_segments * sizeof(sax_type));
            std::vector<sax_type> sax_cache(n_sax_words * paa_segments);
            fread(sax_cache.data(), sizeof(sax_type), n_sax_words * paa_segments, sims_file);
            fclose(sims_file);

            std::vector<float> query_paa(paa_segments);
            paa_from_ts(query, query_paa.data(), paa_segments, dim / paa_segments);

            std::vector<sax_type> max_sax_cardinalities(paa_segments, sax_bit_cardinality);
            float mindist_sqrt = (float)(dim / paa_segments);

            std::atomic<unsigned int> cs_idx(0u);
            std::vector<HerculesCSRangeWorkerData> cs_data((size_t)num_query_threads);
            for (int ti = 0; ti < num_query_threads; ti++) {
                cs_data[ti].lclist = &lclist;
                cs_data[ti].cs_idx = &cs_idx;
                cs_data[ti].sax_cache = &sax_cache;
                cs_data[ti].query_paa = query_paa.data();
                cs_data[ti].max_sax_cardinalities = max_sax_cardinalities.data();
                cs_data[ti].sax_bit_cardinality = sax_bit_cardinality;
                cs_data[ti].sax_cardinality = sax_cardinality;
                cs_data[ti].paa_segments = paa_segments;
                cs_data[ti].mindist_sqrt = mindist_sqrt;
                cs_data[ti].r = r;
            }

            std::vector<pthread_t> cs_threads((size_t)num_query_threads);
            for (int ti = 0; ti < num_query_threads; ti++)
                pthread_create(&cs_threads[ti], nullptr, hercules_cs_range_worker, &cs_data[ti]);
            for (int ti = 0; ti < num_query_threads; ti++)
                pthread_join(cs_threads[ti], nullptr);

            std::vector<SeriesCandidate> sclist;
            for (int ti = 0; ti < num_query_threads; ti++)
                sclist.insert(sclist.end(),
                              cs_data[ti].local_sclist.begin(),
                              cs_data[ti].local_sclist.end());

            float second_level_pruning = 1.0f - (float)sclist.size() / (float)n_series;

            if (second_level_pruning < sax_th) {
                hercules_range_scan(lclist, query, dim, raw_file, ts_buf, r, results);
            } else {
                pthread_rwlock_t lock_range_results = PTHREAD_RWLOCK_INITIALIZER;
                std::atomic<unsigned int> cr_idx(0u);
                std::string raw_path_str = raw_path;

                std::vector<HerculesCRRangeWorkerData> cr_data((size_t)num_query_threads);
                for (int ti = 0; ti < num_query_threads; ti++) {
                    cr_data[ti].sclist = &sclist;
                    cr_data[ti].cr_idx = &cr_idx;
                    cr_data[ti].query = query;
                    cr_data[ti].dim = dim;
                    cr_data[ti].r = r;
                    cr_data[ti].raw_path = raw_path_str.c_str();
                    cr_data[ti].range_results = &results;
                    cr_data[ti].lock_range_results = &lock_range_results;
                }

                std::vector<pthread_t> cr_threads((size_t)num_query_threads);
                for (int ti = 0; ti < num_query_threads; ti++)
                    pthread_create(&cr_threads[ti], nullptr, hercules_cr_range_worker, &cr_data[ti]);
                for (int ti = 0; ti < num_query_threads; ti++)
                    pthread_join(cr_threads[ti], nullptr);

                pthread_rwlock_destroy(&lock_range_results);
            }
        }
    }

    fclose(raw_file);
    return results;
}

// ---- Hercules class implementation ----

Hercules::Hercules(DistanceType distance_type)
    : Hercules(distance_type, HerculesConfig{})
{
}

Hercules::Hercules(DistanceType distance_type, const HerculesConfig &config)
    : SimilaritySearchAlgorithm(distance_type), config_(config)
{
    this->leaf_size = config.leaf_size;
}

Hercules::~Hercules()
{
    destroy_tree(root_);
}

void Hercules::buildIndex(DataSource *data_source)
{
    int n = (int)data_source->getTotalRecords();
    int dim = (int)data_source->getDim();

    if (n == 0 || dim == 0)
        throw std::runtime_error("Hercules::buildIndex: empty data source");

    const float *raw = data_source->rawPointer();
    std::vector<float> tmp;
    if (!raw) {
        tmp.resize((size_t)n * dim);
        data_source->reset();
        for (int i = 0; i < n; i++)
            data_source->nextRecord(tmp.data() + (size_t)i * dim);
        raw = tmp.data();
    }

    destroy_tree(root_);
    root_ = nullptr;

    root_ = hercules_index_build(raw, n, dim, config_.leaf_size, 1, config_.num_build_threads);
    if (root_ == nullptr)
        throw std::runtime_error("Hercules::buildIndex: index build failed");

    if (!config_.index_dir.empty()) {
        if (!hercules_index_write(root_, raw, dim, config_.leaf_size, 1,
                                   config_.paa_segments, config_.sax_cardinality,
                                   config_.sax_bit_cardinality,
                                   config_.index_dir.c_str()))
            throw std::runtime_error("Hercules::buildIndex: index write failed");
    }

    this->n_database = (idx_t)n;
    this->dim = (idx_t)dim;
}

void Hercules::searchIndex(const float *query, idx_t n_query, idx_t k, idx_t *I, float *D)
{
    if (root_ == nullptr)
        throw std::runtime_error("Hercules::searchIndex: index not built");
    if (config_.index_dir.empty())
        throw std::runtime_error("Hercules::searchIndex: index_dir not set");

    for (idx_t q = 0; q < n_query; q++)
        hercules_knn_search(root_, query + q * this->dim, (int)this->dim, k,
                            I + q * k, D + q * k,
                            config_.index_dir.c_str(),
                            /*epsilon=*/0.0f,
                            config_.lmax,
                            config_.paa_segments,
                            (sax_type)config_.sax_bit_cardinality,
                            config_.sax_cardinality,
                            config_.eapca_th,
                            config_.sax_th,
                            (int)this->n_database,
                            config_.num_query_threads);
}

void Hercules::searchIndex(const float *query, idx_t n_query, const SearchConfig &config,
                            std::vector<std::vector<idx_t>> &I,
                            std::vector<std::vector<float>> &D)
{
    if (config.type == QueryType::TOP_K) {
        SimilaritySearchAlgorithm::searchIndex(query, n_query, config, I, D);
        return;
    }
    if (root_ == nullptr)
        throw std::runtime_error("Hercules::searchIndex: index not built");
    if (config_.index_dir.empty())
        throw std::runtime_error("Hercules::searchIndex: index_dir not set");

    I.resize(n_query);
    D.resize(n_query);

    for (idx_t q = 0; q < n_query; q++) {
        auto hits = hercules_range_search(root_, query + q * this->dim, (int)this->dim,
                                          config.r,
                                          config_.index_dir.c_str(),
                                          config_.paa_segments,
                                          (sax_type)config_.sax_bit_cardinality,
                                          config_.sax_cardinality,
                                          config_.eapca_th,
                                          config_.sax_th,
                                          (int)this->n_database,
                                          config_.num_query_threads);
        std::sort(hits.begin(), hits.end());
        I[q].resize(hits.size());
        D[q].resize(hits.size());
        for (size_t j = 0; j < hits.size(); j++) {
            D[q][j] = hits[j].first;
            I[q][j] = hits[j].second;
        }
    }
}

} // namespace daisy
