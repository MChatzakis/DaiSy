#ifndef SOFA_HPP
#define SOFA_HPP

#ifdef SOFA_FFTW_ENABLED

#include "SimilaritySearchAlgorithm.hpp"
#include <queue>
#include <cfloat>
#include <omp.h>
#include <utility>
#include <vector>
#include <fftw3.h>
#include <pthread.h>
#include "../isax/iSAXIndex.hpp"
#include "../isax/iSAXPqueue.hpp"

namespace daisy
{

// ---- SOFA configuration ----
struct SofaConfig
{
    int search_workers = 4;
    int index_workers = 2;
    int leaf_size = 20000;
    int min_leaf_size = 20000;
    int word_length = 16;
    int alphabet_size = 8;
    int sample_size = 100000;
    int histogram_type = 2;
    int coeff_number = 0;
    bool is_norm = false;
};

// ---- worker data structs (moved from .cpp) ----

typedef struct SOFA_workerdata
{
    isax_node        **nodelist;
    int amountnode;
    ts_type           *fft;
    ts_type           *ts;
    pqueue_t          *pq;
    isax_index        *index;
    float minimum_distance;
    pthread_mutex_t   *lock_current_root_node;
    pthread_mutex_t   *lock_queue;
    pthread_barrier_t *lock_barrier;
    pthread_rwlock_t  *lock_bsf;
    int               *node_counter;
    pthread_mutex_t   *alllock;
    int               *allqueuelabel;
    pqueue_t         **allpq;
    int startqueuenumber;
    pqueue_bsf        *pq_bsf;
    int n_pqueue;
    float             *rawfile;
    float            **bins;
    bool is_norm;

    float r;
    std::vector<std::pair<float, idx_t>> *range_results;
    pthread_rwlock_t *lock_range_results;
} SOFA_workerdata;

struct SOFA_index_worker
{
    isax_index        *index;
    float             *raw_database;
    unsigned long     *shared_start_number;
    unsigned long stop_number;
    int workernumber;
    int total_workernumber;
    pthread_barrier_t *lock_barrier1;
    pthread_barrier_t *lock_barrier2;
    pthread_mutex_t   *lock_firstnode;
    int               *node_counter;
    int               *nodeid;
    int read_block_length;
    float            **bins;
    float norm_factor;
    bool is_norm;
    float             *ts_buf;
    fftwf_complex     *ts_out;
    fftwf_plan         plan_forward;
    float             *fft_transform;
};

struct SOFA_bins_worker
{
    isax_index    *index;
    float        **dft_mem_array;
    float         *raw_database;
    unsigned long start_number;
    unsigned long stop_number;
    long records;
    long records_offset;
    int workernumber;
    float norm_factor;
    bool is_norm;
    int coeff_number;
    float         *ts_buf;
    fftwf_complex *ts_out;
    fftwf_plan     plan_forward;
    float         *fft_transform;
};

struct SOFA_divide_worker_data
{
    float        **dft_mem_array;
    float        **bins;
    unsigned long start_number;
    unsigned long stop_number;
    unsigned int sample_size;
    int num_symbols;
    int histogram_type;
};

// ---- SOFA helper function declarations ----
void *SOFA_topk_search_worker(void *rfdata);
void *SOFA_range_search_worker(void *rfdata);

void calculate_node_range_sofa(isax_index *index, isax_node *node,
                                ts_type *query, float *fft,
                                float **bins, bool is_norm, float r,
                                std::vector<std::pair<float, idx_t>> *results,
                                pthread_rwlock_t *lock_results, float *rawfile);

void insert_tree_node_sofa(float **bins, bool is_norm, float *fft,
                           isax_node *node, isax_index *index, float bsf,
                           pqueue_t **pq, pthread_mutex_t *lock_queue,
                           int *tnumber, int n_pqueue);

void calculate_node_topk_sofa(isax_index *index, isax_node *node,
                              ts_type *query, float *fft,
                              float **bins, bool is_norm,
                              pqueue_bsf *pq_bsf,
                              pthread_rwlock_t *lock_bsf, float *rawfile);

void sofa_fft_from_ts(int ts_length, float norm_factor, bool is_norm,
                      fftwf_complex *ts_out, float *transform, fftwf_plan plan_forward,
                      int coeff_number);

void sofa_sfa_from_fft(float *transform, sax_type *sax_out, float **bins,
                       int paa_segments, int cardinality);

void *sofa_index_creation_worker(void *transferdata);

// ---- Sofa class ----
class Sofa : public SimilaritySearchAlgorithm
{
private:
    int read_block_length = 100000;
    int search_workers = 4;
    int index_workers = 2;
    int n_pqueue = 42;
    bool owns_database = false;

    float **bins = nullptr;
    float  *binsv = nullptr;
    int    *coefficients = nullptr;

    int sample_size = 100000;
    int histogram_type = 2;
    int coeff_number = 0;
    bool is_norm = false;

    void sfaBinsInit();
    void sfaSetBins();
    void sfaFreeBins();

    pqueue_bsf sofaSearchTopkL2Squared(float *ts, float *fft, node_list *nodelist, idx_t k);
    std::vector<std::pair<float, idx_t>> sofaSearchRangeL2Squared(float *ts, float *fft, node_list *nodelist, float r);
    void searchIndexL2Squared(const float *query, idx_t n_query, idx_t k, idx_t *I, float *D);
    void searchIndexRangeL2Squared(const float *query, idx_t n_query, float r,
                                   std::vector<std::vector<idx_t>> &I,
                                   std::vector<std::vector<float>> &D);
    void searchIndexDTW(const float *query, idx_t n_query, idx_t k, idx_t *I, float *D);

public:
    Sofa(DistanceType distance_type);
    Sofa(DistanceType distance_type, const SofaConfig &config);

    using SimilaritySearchAlgorithm::buildIndex;

    void buildIndex(DataSource *data_source) override;

    void buildIndex(const std::string &filename, idx_t dim, idx_t n_database = 0) override
    {
        throw std::runtime_error(
            "Sofa requires in-memory data. Use buildIndex(database, n_database, dim) instead.");
    }

    void searchIndex(const float *query, const idx_t n_query, const idx_t k,
                     idx_t *I, float *D) override;

    void searchIndex(const float *query, idx_t n_query, const SearchConfig &config,
                     std::vector<std::vector<idx_t>> &I,
                     std::vector<std::vector<float>> &D) override;

    int getNumThreads()        const { return SimilaritySearchAlgorithm::num_threads; }
    int getReadBlockLength()   const { return read_block_length; }
    int getSearchWorkers()     const { return search_workers; }
    int getIndexWorkers()      const { return index_workers; }
    int getWordLength()        const { return paa_segments; }
    int getAlphabetSize()      const { return sax_cardinality; }
    int getLeafSize()          const { return leaf_size; }
    int getMinLeafSize()       const { return min_leaf_size; }
    int getInitialLblSize()    const { return initial_lbl_size; }
    int getFlushLimit()        const { return flush_limit; }
    int getInitialFblSize()    const { return initial_fbl_size; }
    int getTotalLoadedLeaves() const { return total_loaded_leaves; }
    int getTightBound()        const { return tight_bound; }
    int getSampleSize()        const { return sample_size; }
    int getHistogramType()     const { return histogram_type; }
    int getCoeffNumber()       const { return coeff_number; }
    bool getIsNorm()            const { return is_norm; }

    void setNumThreads(int n)        { SimilaritySearchAlgorithm::num_threads = n; search_workers = n; }
    void setReadBlockLength(int n)   { read_block_length = n; }
    void setSearchWorkers(int n)     { search_workers = n; }
    void setIndexWorkers(int n)      { index_workers = n; }
    void setWordLength(int n)        { paa_segments = n; }
    void setAlphabetSize(int n)      { sax_cardinality = n; }
    void setLeafSize(int n)          { leaf_size = n; }
    void setMinLeafSize(int n)       { min_leaf_size = n; }
    void setInitialLblSize(int n)    { initial_lbl_size = n; }
    void setFlushLimit(int n)        { flush_limit = n; }
    void setInitialFblSize(int n)    { initial_fbl_size = n; }
    void setTotalLoadedLeaves(int n) { total_loaded_leaves = n; }
    void setTightBound(int n)        { tight_bound = n; }
    void setSampleSize(int n)        { sample_size = n; }
    void setHistogramType(int n)     { histogram_type = n; }
    void setCoeffNumber(int n)       { coeff_number = n; }
    void setIsNorm(bool v)           { is_norm = v; }

    ~Sofa();
};

} // namespace daisy

#endif // SOFA_FFTW_ENABLED
#endif // SOFA_HPP
