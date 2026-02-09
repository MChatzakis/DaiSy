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

namespace daisy
{
    
    using ::daisy::TimerManager;
    
    class Odyssey;

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
        
        ts_type *paaU;
        ts_type *paaL;
    };

    struct SearchFunctionParams
    {
        int query_id;
        ts_type *ts;
        ts_type *paa;
        isax_index *index;
        NodeList *nodelist;  
        float minimum_distance;

        double (*estimation_func)(double);
        CommunicationModuleData *comm_data;  

        std::vector<BsfMessage> *shared_bsf_results;  
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

        BsfSharingData *bsf_sharing_data;  
        WorkstealingData *workstealing_data;  
        ReplicationData *replication_data;  

        std::string output_file;  

        int warp_window;
        ts_type *paaU;
        ts_type *paaL;
    };

    struct WsSearchFunctionParams
    {
        int query_id;
        ts_type *ts;
        ts_type *paa;
        isax_index *index;
        isax_node *lca_node;
        NodeList *nodelist;  
        float minimum_distance;
        float bsf;
        file_position_type bsf_pos;
        int *batch_ids;
        double (*estimation_func)(double);

        CommunicationModuleData *comm_data;  

        std::vector<BsfMessage> *shared_bsf_results;  
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

        BsfSharingData *bsf_sharing_data;  
        WorkstealingData *workstealing_data;  
        ReplicationData *replication_data;  

        std::string output_file;  

        int warp_window;
        ts_type *paaU;
        ts_type *paaL;
    };

    struct QaWorkerData
    {
        int workernumber;
        float minimum_distance;

        ts_type *paa, *ts;
        isax_index *index;

        query_result *bsf_result;

        SubtreeBatch *batches;  
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

        CommunicationModuleData *comm_data;  

        std::vector<BsfMessage> *shared_bsf_results;  

        int my_rank;
        int comm_sz;
        float *rawfile;
        int query_threads;
        int merge_offset;
        int query_counter;
        int pq_th_div_factor;

        float corr_threshold;

        bool verbose;

        ReplicationData *replication_data;  
        WorkstealingData *workstealing_data;  
        BsfSharingData *bsf_sharing_data;  

        std::string output_file;  

        int warp_window;
        ts_type *paaU;
        ts_type *paaL;
    };

    struct CoordinatorData
    {
        CommunicationModuleData *comm_data;  
        volatile char *threads_finished;
    };

    struct WorkstealingThreadData
    {
        volatile char *query_workers_finished;
        volatile char *priority_queues_filled;
        volatile char *receiving_workstealing;

        pthread_mutex_t *bsf_lock;
        query_result *bsf_result;

        BatchList *batchlist;  

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

        ReplicationData *replication_data;  
        WorkstealingData *workstealing_data;  
        BsfSharingData *bsf_sharing_data;  

        std::string output_file;  
    };

    float calculate_minimum_distance_inmemory(isax_index *index, isax_node *node, ts_type *raw_query, ts_type *query);
    void refine_topk_answer_inmemory_dtw(ts_type *ts, ts_type *paa, ts_type *paaU, ts_type *paaL, isax_index *index, int warp_window, pqueue_bsf *pq_bsf, float minimum_distance, int limit, float *rawfile, int merge_offset);
    void odyssey_compute_query_envelopes_dtw(OdysseyQuery *query, isax_index *index, int warping_window);

    int process_pq_of_batch_chatzakis(int current_pq_index, QaWorkerData *input_data);
    void gather_sort_pqueues(QaWorkerData *in_data);
    int estimate_th(double x, double (*estimation_func)(double));

    void process_rs_batch(int batch_index, SubtreeBatch *batches, float bsf_distance, isax_index *index, ts_type *paa, int warp_window = 0, ts_type *paaU = nullptr, ts_type *paaL = nullptr);
    void generate_pqs_of_rs_batch(isax_node *subtree_node, SubtreeBatch *batch, float bsf_distance, ts_type *paa, isax_index *index, int warp_window = 0, ts_type *paaU = nullptr, ts_type *paaL = nullptr);

    void *workstealing_manager(void *rfdata);
    void *dynamic_query_scheduler(void *rfdata);
    void *qa_exact_search_messi_worker(void *rfdata);
    void *exact_search_worker_inmemory_hybridpqueue_parallel_chatzakis(void *rfdata);
    void *qa_exact_search_messi_dynamic_worker(void *rfdata);
    void *qa_exact_search_odyssey_worker(void *rfdata);

    query_result qa_exact_search_odyssey_knn(SearchFunctionParams args);  

    query_result qa_exact_search_odyssey_knn_workstealing(WsSearchFunctionParams ws_args);  

} 

#endif 
