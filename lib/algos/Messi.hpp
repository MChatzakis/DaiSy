#ifndef MESSI_HPP
#define MESSI_HPP

#include "SimilaritySearchAlgorithm.hpp"

#include <queue>
#include <cfloat>
#include <omp.h>

#include "../isax/iSAXIndex.hpp"
#include "../isax/iSAXPqueue.hpp"

namespace diNoLib
{
    typedef struct localStack
    {
        isax_node **val;
        int top;
        int bottom;
    } localStack;

    typedef struct MESSI_workerdata
    {
        isax_node *current_root_node;
        ts_type *paa, *paaU, *paaL, *ts, *uo, *lo;
        pqueue_t *pq;
        isax_index *index;
        float minimum_distance;
        int limit;
        pthread_mutex_t *lock_current_root_node;
        pthread_mutex_t *lock_queue;
        pthread_barrier_t *lock_barrier;
        pthread_rwlock_t *lock_bsf;
        query_result *bsf_result;
        int *node_counter;
        isax_node **nodelist;
        int amountnode;
        localStack *localstk;
        localStack *allstk;
        pthread_mutex_t *locallock, *alllock;
        int *queuelabel, *allqueuelabel;
        pqueue_t **allpq;
        int startqueuenumber;
        int warpWind;
        pqueue_bsf *pq_bsf;

        int n_pqueue;
        float *rawfile;
    } MESSI_workerdata;

    void *MESSI_topk_search_worker_L2Squared(void *rfdata);
    void *MESSI_topk_search_worker_DTW(void *rfdata);

    void insert_tree_node_m_hybridpqueue(float *paa, isax_node *node, isax_index *index, float bsf, pqueue_t **pq, pthread_mutex_t *lock_queue, int *tnumber, int n_pqueue);
    void insert_tree_node_m_hybridpqueue_DTW(float *paaU, float *paaL, isax_node *node, isax_index *index, float bsf, pqueue_t **pq, pthread_mutex_t *lock_queue, int *tnumber, int n_pqueue);

    void calculate_node2_topk_inmemory(isax_index *index, isax_node *node, ts_type *query, ts_type *paa, pqueue_bsf *pq_bsf, pthread_rwlock_t *lock_queue, float *rawfile);
    void calculate_node_DTW2knn_inmemory(isax_index *index, isax_node *node, ts_type *query, float *uo, float *lo, ts_type *paa, ts_type *paaU, ts_type *paaL, float bsf, int warpWind, pqueue_bsf *pq_bsf, pthread_rwlock_t *lock_queue, float *rawfile);

    class Messi : public SimilaritySearchAlgorithm
    {
    private:
        int paa_segments = 16;
        int sax_cardinality = 8;
        int leaf_size = 2000;
        int min_leaf_size = 10;
        int initial_lbl_size = 2000;
        int flush_limit = 200000;
        int initial_fbl_size = 100;
        int total_loaded_leaves = 1;
        int tight_bound = 0;
        float minimum_distance = FLT_MAX;
        int min_checked_leaves = -1;
        
        int read_block_length = 100000;
        int search_workers = 4; //64;
        int index_workers = 2; //32;
        int n_pqueue = 42;
        int warping_window = 10;  // warping window size (typically 10% of time series length)

        isax_index_settings *index_settings = nullptr;
        isax_index *index = nullptr;

        pqueue_bsf MESSI_search_topk_L2Squared(ts_type *ts, ts_type *paa, node_list *nodelist, idx_t k);
        pqueue_bsf MESSI_search_topk_DTW(ts_type *ts, ts_type *paa, ts_type *paaU, ts_type *paaL, node_list *nodelist, idx_t k);

        void searchIndexL2Squared(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D);
        void searchIndexDTW(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D);

    public:
        Messi(DistanceType distance_type);
        void setNumThreads(int num_threads)
        {
            this->search_workers = num_threads;
            this->index_workers = num_threads;
        }
        int getNumThreads() const { return this->search_workers; }

        int getPaaSegments() const { return paa_segments; }
        int getSaxCardinality() const { return sax_cardinality; }
        int getLeafSize() const { return leaf_size; }
        int getMinLeafSize() const { return min_leaf_size; }
        int getInitialLblSize() const { return initial_lbl_size; }
        int getFlushLimit() const { return flush_limit; }
        int getInitialFblSize() const { return initial_fbl_size; }
        int getTotalLoadedLeaves() const { return total_loaded_leaves; }
        int getTightBound() const { return tight_bound; }
        
        int getSearchWorkers() const { return search_workers; }
        int getIndexWorkers() const { return index_workers; }
        int getReadBlockLength() const { return read_block_length; }
        int getWarpingWindow() const { return warping_window; }

        void setPaaSegments(int paa_segments) { this->paa_segments = paa_segments; }
        void setSaxCardinality(int sax_cardinality) { this->sax_cardinality = sax_cardinality; }
        void setLeafSize(int leaf_size) { this->leaf_size = leaf_size; }
        void setMinLeafSize(int min_leaf_size) { this->min_leaf_size = min_leaf_size; }
        void setInitialLblSize(int initial_lbl_size) { this->initial_lbl_size = initial_lbl_size; }
        void setFlushLimit(int flush_limit) { this->flush_limit = flush_limit; }
        void setInitialFblSize(int initial_fbl_size) { this->initial_fbl_size = initial_fbl_size; }
        void setTotalLoadedLeaves(int total_loaded_leaves) { this->total_loaded_leaves = total_loaded_leaves; }
        void setTightBound(int tight_bound) { this->tight_bound = tight_bound; }
        
        void setSearchWorkers(int search_workers) { this->search_workers = search_workers; }
        void setIndexWorkers(int index_workers) { this->index_workers = index_workers; }
        void setReadBlockLength(int read_block_length) { this->read_block_length = read_block_length; }
        void setWarpingWindow(int warping_window) { this->warping_window = warping_window; }

        void buildIndex(const float *database, const idx_t n_database, const idx_t dim) override;
        void searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D) override;

        ~Messi();
    };

} // namespace diNoLib

#endif // MESSI_HPP