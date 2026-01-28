#ifndef INDEXING_HPP
#define INDEXING_HPP

#include <cstdlib>
#include <vector>
#include <mpi.h>
#include <pthread.h>

#include "../isax/iSAXIndex.hpp"
#include "../isax/iSAXPqueue.hpp"
#include "replication.hpp"

namespace diNoLib
{
    // TimerManager commented out - only used for profiling, not needed for functionality
    // namespace dinoLib {
    //     class TimerManager;
    // }
    // using TimerManager = ::dinoLib::TimerManager;

    constexpr int MAX_PQs_WORKSTEALING = 2000;

    enum class DynamicSchedulingMode
    {
        PERIODIC_CHECK = 0,
        COORDINATOR_IDLE,
        STANDALONE_THREAD
    };

    struct NodeList
    {
        isax_node **nlist;  // Array of node pointers
        int node_amount;

        int data_amount;
        ts_type *rawfile;
    };

    struct SubtreeBatch
    {
        int id;
        int from;
        int to;
        int size;

        volatile int current_subtree_to_process;
        volatile int current_pq_to_process;

        char processed_phase_1;
        char processed_phase_2;

        char is_getting_help_phase1;
        char is_getting_help_phase2;

        int pq_th;
        int pq_amount;

        pthread_mutex_t pq_insert_lock;
        pqueue_t *pq[MAX_PQs_WORKSTEALING];  // Static array for compatibility

        NodeList *nodelist;

        int max_pq_index;
        int min_pq_index;

        int is_stolen;
    };

    struct BatchList
    {
        SubtreeBatch *batches;  // Array of batches
        int batch_amount;
    };

    // Function pointer type for communication module
    using CommunicationModuleFunc = int (*)(int *q_loaded, int q_num, int *process_buffer, 
                                             MPI_Request *request, int *rec_message,
                                             MPI_Request *send_request, int *termination_message_id,
                                             ReplicationData *replication_data, int my_rank, int comm_sz,
                                             /* TimerManager *timer_manager, */ bool verbose);

    struct CommunicationModuleData
    {
        int q_num;
        int *q_loaded;      // current query loaded! (array)
        int *process_buffer; // array
        int *rec_message;    // array
        int *termination_message_id; // array

        ReplicationData *replication_data;
        int my_rank;
        int comm_sz;
        // TimerManager *timer_manager;  // Commented out - only for profiling
        bool verbose;

        DynamicSchedulingMode mode;

        MPI_Request *request;      // array
        MPI_Request *send_request; // array

        CommunicationModuleFunc module_func;
    };

    // Macro-like inline function for calling the module
    inline int call_module(CommunicationModuleData *comm_data)
    {
        return (comm_data->module_func)(comm_data->q_loaded, comm_data->q_num, 
                                        comm_data->process_buffer, comm_data->request,
                                        comm_data->rec_message, comm_data->send_request,
                                        comm_data->termination_message_id, 
                                        comm_data->replication_data, comm_data->my_rank, comm_data->comm_sz,
                                        /* comm_data->timer_manager, */ comm_data->verbose);
    }

    // Batch creation and tree analysis functions
    BatchList* create_subtree_batches(NodeList *nodelist, int number_of_batches_to_create, int pq_th);

    long int find_total_nodes(isax_node *root_node);
    long int find_total_leafs_nodes(isax_node *root_node);
    long int find_tree_height(isax_node *root_node);
    long int find_total_tree_leafs_depths(isax_node *root_node, long int depth);
    long int count_ts_in_nodes(isax_node *root_node, const char parallelism_in_subtree, 
                               const char recBuf_helpers_exist);
    long int find_total_nodes_tmp(isax_node *root_node);
    long int find_tree_height_tmp(isax_node *root_node);

    void print_index_stats(isax_index *index, int my_rank);

    // ========================================================================
    // EKOSMAS-specific structures and functions for Odyssey
    // ========================================================================

    // No parallelism within subtrees (used by EKOSMAS workers)
    constexpr char NO_PARALLELISM_IN_SUBTREE = 0;

    // Worker input data for EKOSMAS in-memory indexing
    struct buffer_data_inmemory_ekosmas
    {
        isax_index *index;
        pthread_mutex_t *lock_firstnode;
        int workernumber;

        unsigned long *shared_start_number;          // shared block counter
        idx_t ts_num;                                // total time series for this node
        pthread_barrier_t *wait_summaries_to_compute;
        int *node_counter;                           // tree node counter (for FAI)

        char parallelism_in_subtree;                 // currently NO_PARALLELISM_IN_SUBTREE
        volatile unsigned long *next_iSAX_group;     // bookkeeping array for iSAX groups

        float *rawfile;                              // pointer to local raw time series
        bool deterministic_index;                    // deterministic index construction flag

        int index_threads;                           // total index threads on this node
        int readblock;                               // read block length
        int my_rank;                                 // MPI rank
        int comm_sz;                                 // MPI world size

        ReplicationData *replication_data;           // replication information
        // ::dinoLib::TimerManager *timer_manager;     // Commented out - only for profiling
    };

    // Index creation functions (only sequence similarity, no subsequence)
    root_mask_type isax_pRecBuf_index_insert_inmemory_ekosmas(isax_index *index,
                                                              sax_type *sax,
                                                              file_position_type *pos, 
                                                              pthread_mutex_t *lock_firstnode,
                                                              int workernumber, 
                                                              int total_workernumber);

    void tree_index_creation_from_pRecBuf_fai_blocking(void *transferdata);

    void* index_creation_sequence_worker(void *transferdata);
    // Note: index_create_subsequence_worker removed - subsequence querying not supported

} // namespace diNoLib

#endif // INDEXING_HPP
