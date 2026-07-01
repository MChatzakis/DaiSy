#ifndef FRESH_HPP
#define FRESH_HPP

#include "SimilaritySearchAlgorithm.hpp"
#include <cfloat>
#include <omp.h>
#include "../isax/iSAXIndex.hpp"
#include "../isax/iSAXPqueue.hpp"

#define FRESH_CASPTR(A,B,C) __sync_bool_compare_and_swap((long *)(A), (long)(B), (long)(C))
#define FRESH_CASULONG(A,B,C) __sync_bool_compare_and_swap((unsigned long *)(A), (unsigned long)(B), (unsigned long)(C))

namespace daisy
{

    typedef struct fresh_array_element
    {
        float distance;
        isax_node *node;
    } fresh_array_element_t;

    typedef struct fresh_array_list_node
    {
        fresh_array_element_t *data;
        int num_node;
        struct fresh_array_list_node *next;
    } fresh_array_list_node_t;

    typedef struct fresh_array_list
    {
        fresh_array_list_node_t *Top;
        int element_size;
    } fresh_array_list_t;

    typedef struct fresh_sorted_array
    {
        fresh_array_element_t *data;
        int num_elements;
    } fresh_sorted_array_t;

    // Must stay layout-compatible with parallel_fbl_soft_buffer (same field offsets, same 56-byte size)
    // because iSAXSearch.cpp casts index->fbl through parallel_first_buffer_layer * and accesses
    // .initialized (offset 24) and .node (offset 0) via the MESSI slot type.
    typedef struct parallel_fbl_soft_buffer_lf
    {
        isax_node *volatile node;
        sax_type **sax_records;
        file_position_type **pos_records;
        int initialized;
        int _pad;
        int *max_buffer_size;
        int *buffer_size;
        root_mask_type mask;
    } parallel_fbl_soft_buffer_lf;

    // Must stay layout-compatible with parallel_first_buffer_layer so that
    // iSAXSearch.cpp's (parallel_first_buffer_layer *)(index->fbl)->soft_buffers lands at offset 32.
    typedef struct parallel_first_buffer_layer_lf
    {
        int number_of_buffers;
        int initial_buffer_size;
        int max_total_size;
        int current_record_index;
        char *current_record;
        char *hard_buffer;
        parallel_fbl_soft_buffer_lf *soft_buffers;
    } parallel_first_buffer_layer_lf;

    struct FreshConfig
    {
        int search_workers = 4;
        int index_workers = 2;
        int warping_window = 10;
        int leaf_size = 2000;
        int paa_segments = 16;
    };

    // FreSH's lock-free workerdata. bsf_result_p replaced by pq_bsf + lock_bsf
    // because DaiSy requires top-k; FreSH's CAS BSF update is 1-NN only.
    typedef struct FRESH_workerdata
    {
        ts_type *paa, *paaU, *paaL, *ts, *uo, *lo;
        fresh_array_list_t *array_lists;
        fresh_sorted_array_t *volatile *sorted_arrays;
        volatile unsigned char *queue_finished;
        volatile unsigned char *helper_queue_exist;
        volatile int *fai_queue_counters;
        volatile float **queue_bsf_distance;
        isax_index *index;
        float minimum_distance;
        pqueue_bsf *pq_bsf;
        pthread_rwlock_t *lock_bsf;
        volatile int *node_counter;
        volatile unsigned long *sorted_array_counter;
        volatile unsigned long *sorted_array_FAI_counter;
        isax_node **nodelist;
        int amountnode;
        int workernumber;
        int n_queues;
        int warpWind;
        volatile unsigned long *next_queue_data_pos;
        int query_id;
        pthread_barrier_t *wait_tree_pruning_phase_to_finish;
        pthread_barrier_t *wait_process_queue;
        float *rawfile;
    } FRESH_workerdata;

    void *FRESH_topk_search_worker_L2Squared(void *rfdata);
    void *FRESH_topk_search_worker_DTW(void *rfdata);

    class Fresh : public SimilaritySearchAlgorithm
    {
    private:
        int read_block_length = 100000;
        int search_workers = 4;
        int index_workers = 2;
        bool owns_database = false;

        pqueue_bsf FRESH_search_topk_L2Squared(ts_type *ts, ts_type *paa, node_list *nodelist, idx_t k);
        pqueue_bsf FRESH_search_topk_DTW(ts_type *ts, node_list *nodelist, idx_t k);

        void searchIndexL2Squared(const float *query, idx_t n_query, idx_t k, idx_t *I, float *D);
        void searchIndexDTW(const float *query, idx_t n_query, idx_t k, idx_t *I, float *D);

    public:
        Fresh(DistanceType distance_type);
        Fresh(DistanceType distance_type, const FreshConfig &config);

        void setWarpingWindow(int w) { warping_window = w; }
        void setWarpWindow(int w) { warping_window = w; }

        using SimilaritySearchAlgorithm::buildIndex;

        void buildIndex(DataSource *data_source) override;

        void buildIndex(const std::string &filename, idx_t dim, idx_t n_database = 0) override
        {
            throw std::runtime_error("Fresh requires in-memory data. Use buildIndex(database, n_database, dim) instead.");
        }

        void searchIndex(const float *query, idx_t n_query, idx_t k, idx_t *I, float *D) override;

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

        ~Fresh();
    };

}

#endif
