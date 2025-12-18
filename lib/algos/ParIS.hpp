#ifndef PARIS_HPP
#define PARIS_HPP

#include "SimilaritySearchAlgorithm.hpp"

#include "isax/iSAXIndex.hpp"
#include "isax/iSAXSearch.hpp"

#include <queue>
#include <cfloat>
#include <omp.h> 
#include <pthread.h>

namespace diNoLib
{
    // Constants for ParIS search
    #define MAXREADTHREAD 4

    typedef struct ParIS_LDCW_data
    {
        isax_index *index;
        pthread_rwlock_t *lock_bsf;
        unsigned long start_number;
        unsigned long stop_number;
        ts_type *paa;
        ts_type *paaU;  // Upper PAA bounds for DTW
        ts_type *paaL;  // Lower PAA bounds for DTW
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
        ts_type *tsU;  // Upper Lemire envelope for DTW
        ts_type *tsL;  // Lower Lemire envelope for DTW
        unsigned long *counter;
        unsigned long *load_point;
        pthread_rwlock_t *lock_bsf;
        float *minidisvector;
        unsigned long sum_of_lab;
        pqueue_bsf *pq_bsf;
        int warpWind;  // Warping window for DTW
    } ParIS_read_worker_data;

    void *mindistance_worker(void *essdata);
    void *topk_read_worker(void *read_pointer);
    void *mindistance_worker_dtw(void *essdata);  // DTW version of mindistance_worker
    void *dtwknnreadworker(void *read_pointer);  // DTW version of read worker
    pqueue_bsf exact_topk_serial_ParIS(ts_type *ts, ts_type *paa, isax_index *index, float minimum_distance, int min_checked_leaves, int k, int maxquerythread);
    void approximate_topk(ts_type *ts, ts_type *paa, isax_index *index, pqueue_bsf *pq_bsf);
    void refine_topk_answer(ts_type *ts, ts_type *paa, isax_index *index, pqueue_bsf *pq_bsf, float minimum_distance, int limit);
    void calculate_node_topk(isax_index *index, isax_node *node, ts_type *query, pqueue_bsf *pq_bsf);
    float calculate_minimum_distance(isax_index *index, isax_node *node, ts_type *raw_query, ts_type *query);

    class ParIS : public SimilaritySearchAlgorithm
    {
    private:
        int num_threads = 1;
        int paa_segments = 16;
        int sax_cardinality = 8;
        int leaf_size = 2000;
        int min_leaf_size = 10;
        int initial_lbl_size = 2000;
        int flush_limit = 200000;
        int initial_fbl_size = 100;
        int total_loaded_leaves = 1;
        int tight_bound = 0;
        int search_workers = 64;
        int index_workers = 32;
        int read_block_length = 100000;
        float minimum_distance = FLT_MAX;
        int min_checked_leaves = -1;
        int n_pqueue = 42;
        int warping_window = 10;  // warping window size (typically 10% of time series length)

        isax_index_settings *index_settings = nullptr;
        isax_index *index = nullptr;

        void searchIndexL2Squared(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D);
        pqueue_bsf exact_DTWknn_serial_ParIS(ts_type *ts, isax_index *index, int warpWind, int k);
                
    public:
        ParIS(DistanceType distance_type);
        void setNumThreads(int num_threads);
        int getNumThreads() const;
        void setWarpingWindow(int warping_window) { this->warping_window = warping_window; }
        int getWarpingWindow() const { return this->warping_window; }
        // Bring base class buildIndex overloads into scope
        using SimilaritySearchAlgorithm::buildIndex;
        
        void buildIndex(DataSource *data_source) override;
        
        // ParIS only supports file-based data - override to give clear error
        void buildIndex(float *database, idx_t n_database, idx_t dim) override {
            throw std::runtime_error("ParIS requires file-based data. Use buildIndex(filename, dim, n_database) instead.");
        }
        
        void searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D) override;
           

        ~ParIS();

    };

} // namespace diNoLib

#endif // PARIS_HPP