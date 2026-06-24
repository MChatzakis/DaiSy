#ifndef PARIS_HPP
#define PARIS_HPP

#include "SimilaritySearchAlgorithm.hpp"

#include "isax/iSAXIndex.hpp"
#include "isax/iSAXSearch.hpp"

#include <queue>
#include <cfloat>
#include <omp.h>
#include <pthread.h>

namespace daisy
{

#define MAXREADTHREAD 4

    struct ParISConfig
    {
        int search_workers = 64; 
        int index_workers = 32;  
        int warping_window = 10; 
        int leaf_size = 2000;
        int paa_segments = 16;
    };

    typedef struct ParIS_LDCW_data
    {
        isax_index *index;
        pthread_rwlock_t *lock_bsf;
        unsigned long start_number;
        unsigned long stop_number;
        ts_type *paa;
        ts_type *paaU; 
        ts_type *paaL; 
        ts_type *ts;
        float bsfdistance;
        unsigned long sum_of_lab;
        unsigned long *label_number;
        float *minidisvector;
    } ParIS_LDCW_data;

    typedef struct ParIS_read_worker_data
    {
        isax_index *index;
        ts_type *ts;
        ts_type *tsU; 
        ts_type *tsL; 
        unsigned long *counter;
        unsigned long *load_point;
        pthread_rwlock_t *lock_bsf;
        float *minidisvector;
        unsigned long sum_of_lab;
        pqueue_bsf *pq_bsf;
        int warpWind; 
    } ParIS_read_worker_data;

    void *mindistance_worker(void *essdata);
    void *topk_read_worker(void *read_pointer);
    void *mindistance_worker_dtw(void *essdata); 
    void *dtwknnreadworker(void *read_pointer);  
    pqueue_bsf exact_topk_serial_ParIS(ts_type *ts, ts_type *paa, isax_index *index, float minimum_distance, int min_checked_leaves, int k, int maxquerythread);
    void approximate_topk(ts_type *ts, ts_type *paa, isax_index *index, pqueue_bsf *pq_bsf);
    void approximate_topk_dtw(ts_type *ts, ts_type *paa, isax_index *index, pqueue_bsf *pq_bsf, int warpWind);
    void refine_topk_answer(ts_type *ts, ts_type *paa, isax_index *index, pqueue_bsf *pq_bsf, float minimum_distance, int limit);
    void calculate_node_topk(isax_index *index, isax_node *node, ts_type *query, pqueue_bsf *pq_bsf);
    void calculate_node_topk_dtw(isax_index *index, isax_node *node, ts_type *query, pqueue_bsf *pq_bsf, int warpWind);
    float calculate_minimum_distance(isax_index *index, isax_node *node, ts_type *raw_query, ts_type *query);

    class ParIS : public SimilaritySearchAlgorithm
    {
    private:
        int search_workers = 64;
        int index_workers = 32;
        int read_block_length = 100000;
        int n_pqueue = 42;

        void searchIndexL2Squared(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D);
        void searchIndexDTW(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D);
        pqueue_bsf exact_DTWknn_serial_ParIS(ts_type *ts, isax_index *index, int warpWind, int k);

    public:
        ParIS(DistanceType distance_type);
        ParIS(DistanceType distance_type, const ParISConfig &config);

        void setWarpingWindow(int w) { warping_window = w; }
        int getWarpingWindow() const { return warping_window; }

        int getNumThreads() const { return SimilaritySearchAlgorithm::num_threads; }
        void setNumThreads(int n) { SimilaritySearchAlgorithm::num_threads = n; }

        using SimilaritySearchAlgorithm::buildIndex;

        void buildIndex(DataSource *data_source) override;

        void buildIndex(float *database, idx_t n_database, idx_t dim) override
        {
            throw std::runtime_error("ParIS requires file-based data. Use buildIndex(filename, dim, n_database) instead.");
        }

        void searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D) override;

        ~ParIS();
    };

}

#endif