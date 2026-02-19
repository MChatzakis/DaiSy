#ifndef MESSI_HPP
#define MESSI_HPP

#include "SimilaritySearchAlgorithm.hpp"

#include <queue>
#include <cfloat>
#include <omp.h>

#include "../isax/iSAXIndex.hpp"
#include "../isax/iSAXPqueue.hpp"

namespace daisy
{

    struct MessiConfig
    {
        int search_workers = 4;
        int index_workers = 2;
        int warping_window = 10;
        int leaf_size = 2000;
        int paa_segments = 16;
    };

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
        int read_block_length = 100000;
        int search_workers = 4;
        int index_workers = 2;
        int n_pqueue = 42;
        bool owns_database = false;  // track ownership of database buffer

        pqueue_bsf MESSI_search_topk_L2Squared(ts_type *ts, ts_type *paa, node_list *nodelist, idx_t k);
        pqueue_bsf MESSI_search_topk_DTW(ts_type *ts, node_list *nodelist, idx_t k);

        void searchIndexL2Squared(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D);
        void searchIndexDTW(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D);

    public:
        Messi(DistanceType distance_type);
        Messi(DistanceType distance_type, const MessiConfig &config);

        void setWarpingWindow(int w) { warping_window = w; }
        void setWarpWindow(int w) { warping_window = w; }

        using SimilaritySearchAlgorithm::buildIndex;

        void buildIndex(DataSource *data_source) override;

        void buildIndex(const std::string &filename, idx_t dim, idx_t n_database = 0) override
        {
            throw std::runtime_error("Messi requires in-memory data. Use buildIndex(database, n_database, dim) instead.");
        }

        void searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D) override;

        int getNumThreads() const { return SimilaritySearchAlgorithm::num_threads; }
        void setNumThreads(int n) {
            SimilaritySearchAlgorithm::num_threads = n;
            search_workers = n;
        }
        int getPaaSegments() const { return paa_segments; }
        void setPaaSegments(int n) { paa_segments = n; }
        int getSaxCardinality() const { return sax_cardinality; }
        void setSaxCardinality(int n) { sax_cardinality = n; }
        int getLeafSize() const { return leaf_size; }
        void setLeafSize(int n) { leaf_size = n; }
        int getMinLeafSize() const { return min_leaf_size; }
        void setMinLeafSize(int n) { min_leaf_size = n; }
        int getInitialLblSize() const { return initial_lbl_size; }
        void setInitialLblSize(int n) { initial_lbl_size = n; }
        int getFlushLimit() const { return flush_limit; }
        void setFlushLimit(int n) { flush_limit = n; }
        int getInitialFblSize() const { return initial_fbl_size; }
        void setInitialFblSize(int n) { initial_fbl_size = n; }
        int getTotalLoadedLeaves() const { return total_loaded_leaves; }
        void setTotalLoadedLeaves(int n) { total_loaded_leaves = n; }
        int getTightBound() const { return tight_bound; }
        void setTightBound(int n) { tight_bound = n; }
        int getSearchWorkers() const { return search_workers; }
        void setSearchWorkers(int n) { search_workers = n; }
        int getIndexWorkers() const { return index_workers; }
        void setIndexWorkers(int n) { index_workers = n; }
        int getReadBlockLength() const { return read_block_length; }
        void setReadBlockLength(int n) { read_block_length = n; }
        int getWarpingWindow() const { return warping_window; }

        ~Messi();
    };

}

#endif
