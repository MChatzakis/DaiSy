#pragma once

#include "../ds_tree/ds_tree_index.hpp"
#include "../ds_tree/ds_tree_search.hpp"
#include "SimilaritySearchAlgorithm.hpp"
#include "../isax/iSAXTypes.hpp"

#include <atomic>
#include <string>
#include <stdexcept>
#include <pthread.h>

namespace daisy
{

struct HerculesConfig
{
    int leaf_size = 2000;
    int lmax = 80;
    float eapca_th = 0.25f;
    float sax_th = 0.50f;
    int paa_segments = 16;
    int sax_bit_cardinality = 3;
    int sax_cardinality = 8;
    int num_build_threads = 1;
    int num_query_threads = 1;
    int flush_threshold = 200000;
    int insert_buffer_size = 1000;
    int flush_buffer_size = 10000;
    std::string index_dir = "";
};

// ---- Hercules-specific search structs (moved from .cpp) ----

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

// ---- Hercules-specific index write / search ----
bool hercules_index_write(HerculesNode *root, const float *database,
                          int dim, int leaf_size, int init_segments,
                          int paa_segments, int sax_cardinality, int sax_bit_cardinality,
                          const char *root_dir);

void hercules_knn_search(HerculesNode *root, const float *query, int dim,
                          idx_t k, idx_t *I, float *D,
                          const char *root_dir, float epsilon, int approx_leaves,
                          int paa_segments, sax_type sax_bit_cardinality, int sax_cardinality,
                          float eapca_th, float sax_th, int n_series,
                          int num_query_threads = 1);

// ---- Hercules class ----
class Hercules : public SimilaritySearchAlgorithm
{
public:
    Hercules(DistanceType distance_type);
    Hercules(DistanceType distance_type, const HerculesConfig &config);

    using SimilaritySearchAlgorithm::buildIndex;

    void buildIndex(DataSource *data_source) override;

    void searchIndex(const float *query, idx_t n_query, idx_t k,
                     idx_t *I, float *D) override;

    void setNumThreads(int num_threads) override
    {
        config_.num_query_threads = num_threads;
    }

    ~Hercules() override;

private:
    HerculesConfig config_;
    HerculesNode *root_ = nullptr;
};

} // namespace daisy
