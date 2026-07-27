#pragma once

#include "../ds_tree/ds_tree_index.hpp"
#include "../ds_tree/ds_tree_search.hpp"
#include "SimilaritySearchAlgorithm.hpp"
#include "../isax/iSAXTypes.hpp"

#include <atomic>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>
#include <pthread.h>

namespace daisy
{

// Optional constructor parameters. Defaults follow the Hercules paper.
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

// A leaf that survived tree pruning, tagged with its EAPCA lower bound.
struct LeafCandidate {
    HerculesNode *node;
    float lb;
};

// A single series inside a leaf, tagged with its SAX lower bound.
struct SeriesCandidate {
    uint64_t record_idx;
    idx_t series_idx;
    float lb_sax;
};

// Candidate selection worker data. Scans leaves, refines with SAX and produces a per-thread series list.
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

// Candidate refinement worker data. Reads the real series from disk and updates the top-k queue.
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

// Range variant of the candidate selection worker.
struct HerculesCSRangeWorkerData {
    const std::vector<LeafCandidate> *lclist;
    std::atomic<unsigned int> *cs_idx;
    const std::vector<sax_type> *sax_cache;
    const float *query_paa;
    const sax_type *max_sax_cardinalities;
    sax_type sax_bit_cardinality;
    int sax_cardinality;
    int paa_segments;
    float mindist_sqrt;
    float r;
    std::vector<SeriesCandidate> local_sclist;
};

// Range variant of the candidate refinement worker.
struct HerculesCRRangeWorkerData {
    const std::vector<SeriesCandidate> *sclist;
    std::atomic<unsigned int> *cr_idx;
    const float *query;
    int dim;
    float r;
    const char *raw_path;
    std::vector<std::pair<float, idx_t>> *range_results;
    pthread_rwlock_t *lock_range_results;
};

// ---- Hercules-specific index write / search ----

// Flushes an in-memory Hercules tree and its raw records to root_dir.
bool hercules_index_write(HerculesNode *root, const float *database,
                          int dim, int leaf_size, int init_segments,
                          int paa_segments, int sax_cardinality, int sax_bit_cardinality,
                          const char *root_dir);

// Top-k search over a persisted Hercules index. Reads raw series from disk when needed.
void hercules_knn_search(HerculesNode *root, const float *query, int dim,
                          idx_t k, idx_t *I, float *D,
                          const char *root_dir, float epsilon, int approx_leaves,
                          int paa_segments, sax_type sax_bit_cardinality, int sax_cardinality,
                          float eapca_th, float sax_th, int n_series,
                          int num_query_threads = 1);

// Range search over a persisted Hercules index. Returns all (distance, position) within radius r.
std::vector<std::pair<float, idx_t>> hercules_range_search(
    HerculesNode *root, const float *query, int dim,
    float r, const char *root_dir,
    int paa_segments, sax_type sax_bit_cardinality, int sax_cardinality,
    float eapca_th, float sax_th, int n_series,
    int num_query_threads = 1);

// ---- Hercules class ----

// Non-iSAX exact similarity search based on the Hercules tree (EAPCA + SAX).
class Hercules : public SimilaritySearchAlgorithm
{
public:
    // Default constructor. Uses HerculesConfig defaults.
    Hercules(DistanceType distance_type);
    // Explicit constructor.
    Hercules(DistanceType distance_type, const HerculesConfig &config);

    using SimilaritySearchAlgorithm::buildIndex;

    // Build the Hercules tree from an in-memory data source and persist it to disk.
    void buildIndex(DataSource *data_source) override;

    // Top-k search using L2 Squared distance.
    void searchIndex(const float *query, idx_t n_query, idx_t k,
                     idx_t *I, float *D) override;

    // Range and top-k combined entry point. Reads mode and radius from config.
    void searchIndex(const float *query, idx_t n_query, const SearchConfig &config,
                     std::vector<std::vector<idx_t>> &I,
                     std::vector<std::vector<float>> &D) override;

    void setNumThreads(int num_threads) override
    {
        config_.num_query_threads = num_threads;
    }

    // Hercules only supports z-normalized data
    void setNormalized(bool normalized) override
    {
        if (!normalized)
            throw std::runtime_error(
                "Hercules currently supports only z-normalized data (equi-depth breakpoints not implemented).");
    }

    ~Hercules() override;

private:
    HerculesConfig config_;
    HerculesNode *root_ = nullptr;
};

} // namespace daisy
