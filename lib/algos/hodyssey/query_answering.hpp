#ifndef QUERY_ANSWERING_HPP
#define QUERY_ANSWERING_HPP

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <pthread.h>
#include <cstdbool>
#include <sched.h>

#if ODYSSEY_MPI
#include <mpi.h>
#endif

#include "../../isax/iSAXTypes.hpp"
#include "../../isax/iSAXIndex.hpp"
#include "../../isax/iSAXPqueue.hpp"
#include "../../utils/TimerManager.hpp"
#include "workstealing.hpp"
#include "replication.hpp"
#include "bsf_sharing.hpp"
#include "indexing.hpp"

namespace diNoLib
{
    // Forward declaration
    class Odyssey;

    // Priority queue helper functions (inline for header-only usage)
    inline int cmp_pri(double next, double curr)
    {
        return (next > curr);
    }

    inline double get_pri(void *a)
    {
        return (double)((query_result *)a)->distance;
    }

    inline void set_pri(void *a, double pri)
    {
        ((query_result *)a)->distance = (float)pri;
    }

    inline size_t get_pos(void *a)
    {
        return ((query_result *)a)->pqueue_position;
    }

    inline void set_pos(void *a, size_t pos)
    {
        ((query_result *)a)->pqueue_position = pos;
    }

    struct OdysseyQuery
    {
        int id;
        ts_type *query;
        ts_type *paa;
        double initial_estimation;
        pqueue_bsf *initial_pq_bsfs;
    };

    struct SearchFunctionParams
    {
        int query_id;
        ts_type *ts;
        ts_type *paa;
        isax_index *index;
        NodeList *nodelist;  // Changed from node_list to NodeList
        float minimum_distance;

        double (*estimation_func)(double);
        CommunicationModuleData *comm_data;  // Changed from communication_module_data to CommunicationModuleData

        std::vector<BsfMessage> *shared_bsf_results;  // Changed from bsf_msg* to std::vector<BsfMessage>*
        pqueue_bsf *precomputed_bsfs;

        int k;
        int my_rank;
        int comm_sz;
        int query_threads;
        bool verbose;
        float *rawfile;
        int merge_offset;
        int query_counter;
        int pq_th_div_factor;
        float corr_threshold;

        // TimerManager *timer_manager;  // Commented out - only for profiling
        BsfSharingData *bsf_sharing_data;  // Changed from bsf_sharing_data_t to BsfSharingData
        WorkstealingData *workstealing_data;  // Changed from workstealing_data_t to WorkstealingData
        ReplicationData *replication_data;  // Changed from rep_data_t to ReplicationData

        std::string output_file;  // Changed from char* to std::string
    };

    struct WsSearchFunctionParams
    {
        int query_id;
        ts_type *ts;
        ts_type *paa;
        isax_index *index;
        isax_node *lca_node;
        NodeList *nodelist;  // Changed from node_list to NodeList
        float minimum_distance;
        float bsf;
        file_position_type bsf_pos;
        int *batch_ids;
        double (*estimation_func)(double);

        CommunicationModuleData *comm_data;  // Changed from communication_module_data to CommunicationModuleData

        std::vector<BsfMessage> *shared_bsf_results;  // Changed from bsf_msg* to std::vector<BsfMessage>*
        pqueue_bsf *precomputed_bsfs;

        int k;
        int my_rank;
        int comm_sz;
        int query_threads;
        bool verbose;
        float *rawfile;
        int merge_offset;
        int query_counter;
        int pq_th_div_factor;
        float corr_threshold;

        // TimerManager *timer_manager;  // Commented out - only for profiling
        BsfSharingData *bsf_sharing_data;  // Changed from bsf_sharing_data_t to BsfSharingData
        WorkstealingData *workstealing_data;  // Changed from workstealing_data_t to WorkstealingData
        ReplicationData *replication_data;  // Changed from rep_data_t to ReplicationData

        std::string output_file;  // Changed from char* to std::string
    };

    struct QaWorkerData
    {
        int workernumber;
        float minimum_distance;

        ts_type *paa, *ts;
        isax_index *index;

        query_result *bsf_result;

        SubtreeBatch *batches;  // Changed from subtree_batch to SubtreeBatch
        int total_batches;

        volatile int *batch_counter;
        volatile int *pq_counter;

        pthread_barrier_t *sync_barrier;
        pthread_mutex_t *bsf_lock;

        volatile char *receiving_workstealing;
        volatile char *priority_queues_filled;

        pqueue_t ***final_pq_list;
        int *final_pq_list_size;

        pthread_mutex_t *distances_lock;

        int *pqs_stolen;
        int *processed_pqs;

        CommunicationModuleData *comm_data;  // Changed from communication_module_data to CommunicationModuleData

        std::vector<BsfMessage> *shared_bsf_results;  // Changed from bsf_msg* to std::vector<BsfMessage>*

        int my_rank;
        int comm_sz;
        float *rawfile;
        int query_threads;
        int merge_offset;
        int query_counter;
        int pq_th_div_factor;

        float corr_threshold;

        bool verbose;

        ReplicationData *replication_data;  // Changed from rep_data_t to ReplicationData
        WorkstealingData *workstealing_data;  // Changed from workstealing_data_t to WorkstealingData
        BsfSharingData *bsf_sharing_data;  // Changed from bsf_sharing_data_t to BsfSharingData
        // TimerManager *timer_manager;  // Commented out - only for profiling

        std::string output_file;  // Changed from char* to std::string
    };

    constexpr double ERROR_THRESHOLD = 0.00001;

    struct CoordinatorData
    {
        CommunicationModuleData *comm_data;  // Changed from communication_module_data to CommunicationModuleData
        volatile char *threads_finished;
    };

    struct WorkstealingThreadData
    {
        volatile char *query_workers_finished;
        volatile char *priority_queues_filled;
        volatile char *receiving_workstealing;

        pthread_mutex_t *bsf_lock;
        query_result *bsf_result;

        BatchList *batchlist;  // Changed from batch_list to BatchList

        int *pqs_stolen;
        int *workstealing_times;

        pqueue_t ***final_pq_list;
        int *final_pq_list_size;

        isax_index *index;

        int my_rank;
        int comm_sz;
        float *rawfile;
        int query_threads;
        int merge_offset;
        int query_counter;
        int pq_th_div_factor;

        float corr_threshold;

        bool verbose;

        ReplicationData *replication_data;  // Changed from rep_data_t to ReplicationData
        WorkstealingData *workstealing_data;  // Changed from workstealing_data_t to WorkstealingData
        BsfSharingData *bsf_sharing_data;  // Changed from bsf_sharing_data_t to BsfSharingData
        // TimerManager *timer_manager;  // Commented out - only for profiling

        std::string output_file;  // Changed from char* to std::string
    };

    // Search functions (moved from indexing.h)
    float calculate_minimum_distance_inmemory(isax_index *index, isax_node *node, ts_type *raw_query, ts_type *query);
    void approximate_topk_inmemory(ts_type *ts, ts_type *paa, isax_index *index, pqueue_bsf *pq_bsf, float *rawfile, int merge_offset);
    void refine_topk_answer_inmemory(ts_type *ts, ts_type *paa, isax_index *index, pqueue_bsf *pq_bsf, float minimum_distance, int limit, float *rawfile, int merge_offset);
    void calculate_node_topk_inmemory(isax_index *index, isax_node *node, ts_type *query, pqueue_bsf *pq_bsf, float *rawfile, int merge_offset);

    // Management Functions
    int process_pq_of_batch_chatzakis(int current_pq_index, QaWorkerData *input_data);
    void gather_sort_pqueues(QaWorkerData *in_data);
    int estimate_th(double x, double (*estimation_func)(double));
    // COMMENTED: Threshold search not supported - only KNN queries supported
    // int process_pq_of_batch_threshold(int current_pq_index, QaWorkerData *input_data);
    void process_rs_batch(int batch_index, SubtreeBatch *batches, float bsf_distance, isax_index *index, ts_type *paa);  // Changed subtree_batch to SubtreeBatch
    void generate_pqs_of_rs_batch(isax_node *subtree_node, SubtreeBatch *batch, float bsf_distance, ts_type *paa, isax_index *index);  // Changed subtree_batch to SubtreeBatch

    // Thread workers: Prototypes: void * func(void * args)
    void *workstealing_manager(void *rfdata);
    void *dynamic_query_scheduler(void *rfdata);
    void *qa_exact_search_messi_worker(void *rfdata);
    void *exact_search_worker_inmemory_hybridpqueue_parallel_chatzakis(void *rfdata);
    void *qa_exact_search_messi_dynamic_worker(void *rfdata);
    void *qa_exact_search_odyssey_worker(void *rfdata);
    // COMMENTED: Threshold search not supported - only KNN queries supported
    // void *qa_exact_search_odyssey_th_worker(void *rfdata);

    // Query Answering functions: Prototypes: query_result = func(SearchFunctionParams)
    query_result qa_exact_search_odyssey_knn(SearchFunctionParams args);  // Changed search_function_params to SearchFunctionParams
    // query_result qa_exact_search_odyssey_th(SearchFunctionParams args);  // COMMENTED: Threshold search not supported

    // Workstealing helper function
    query_result qa_exact_search_odyssey_knn_workstealing(WsSearchFunctionParams ws_args);  // Changed ws_search_function_params to WsSearchFunctionParams
    // query_result qa_exact_search_odyssey_th_workstealing(WsSearchFunctionParams ws_args);  // COMMENTED: Threshold search not supported

} // namespace diNoLib

#endif // QUERY_ANSWERING_HPP
