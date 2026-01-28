#include "../hodyssey/bsf_sharing.hpp"
#include "../../utils/TimerManager.hpp"
#include "../../isax/iSAXPqueue.hpp"
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cstddef>  // for offsetof

namespace diNoLib
{
    // MPI Datatype for BSF message (initialized in create_bsf_msg_mpi_type)
    MPI_Datatype bsf_msg_type;

    // Helper macro for allocation checking (if not defined elsewhere)
    #ifndef CHECK_ALLOC
    #define CHECK_ALLOC(ptr, rank) \
        if ((ptr) == nullptr) { \
            fprintf(stderr, "[Node %d] Error: Memory allocation failed in %s:%d\n", (rank), __FILE__, __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); \
        }
    #endif

    // Debug flag for BSF sharing prints (can be set via environment variable or config)
    #ifndef ENABLE_PRINTS_BSF_SHARING
    #define ENABLE_PRINTS_BSF_SHARING 0  // Set to 1 to enable debug prints
    #endif

    void create_bsf_msg_mpi_type()
    {
        int lengths[3] = {1, 1, 1};
        const MPI_Aint displacements[3] = {
            offsetof(BsfMessage, q_num),
            offsetof(BsfMessage, bsf),
            offsetof(BsfMessage, position)
        };
        MPI_Datatype types[3] = {MPI_INT, MPI_FLOAT, MPI_UNSIGNED_LONG_LONG}; // query_id, bsf, pos of bsf
        MPI_Type_create_struct(3, lengths, displacements, types, &bsf_msg_type);
        MPI_Type_commit(&bsf_msg_type);
    }

    void bsf_sharing_init(BsfSharingData &bsf_sharing_data, int my_rank, int comm_sz)
    {
        if (comm_sz == 1)
        {
            return;
        }

        create_bsf_msg_mpi_type();

        // Initialize/reset the data structure
        bsf_sharing_data.communicators.clear();
        bsf_sharing_data.requests.clear();
        bsf_sharing_data.shared_bsfs.clear();
        bsf_sharing_data.bcasts_per_query.clear();
        
        bsf_sharing_data.bsf_broadcasts_counter = 0;
        bsf_sharing_data.bsf_receives_counter = 0;
        bsf_sharing_data.bsf_correct_receives_counter = 0;

        // Resize vectors to comm_sz
        bsf_sharing_data.communicators.resize(comm_sz);
        bsf_sharing_data.requests.resize(comm_sz);
        bsf_sharing_data.shared_bsfs.resize(comm_sz);
        bsf_sharing_data.bcasts_per_query.resize(comm_sz);

        // Each node broadcasts in one communicator, and listens to all communicators.
        for (int i = 0; i < comm_sz; ++i)
        {
            MPI_Comm_dup(MPI_COMM_WORLD, &(bsf_sharing_data.communicators[i]));

            bsf_sharing_data.requests[i] = MPI_REQUEST_NULL;
            bsf_sharing_data.shared_bsfs[i].q_num = -1;
            bsf_sharing_data.shared_bsfs[i].bsf = -1.0f;
            bsf_sharing_data.shared_bsfs[i].position = 0;
            bsf_sharing_data.bcasts_per_query[i] = 0;
        }
    }

    void bsf_sharing_destroy(BsfSharingData &bsf_sharing_data, int comm_sz)
    {
        if (comm_sz == 1)
        {
            return;
        }

        for (int i = 0; i < comm_sz; ++i)
        {
            MPI_Comm_free(&(bsf_sharing_data.communicators[i]));
        }

        // Vectors will be automatically cleaned up by destructor
        bsf_sharing_data.communicators.clear();
        bsf_sharing_data.requests.clear();
        bsf_sharing_data.shared_bsfs.clear();
        bsf_sharing_data.bcasts_per_query.clear();

        MPI_Type_free(&bsf_msg_type);
    }

    void bsf_sharing_bcast_bsf(BsfSharingData &bsf_sharing_data, pqueue_bsf *pq_bsf, 
                                int workernumber, int my_rank, int query_counter, 
                                ::dinoLib::timer_manager_t *timer_manager)
    {
        if (!bsf_sharing_data.bsf_sharing_enabled || workernumber != 0 || pq_bsf == nullptr)
        {
            return;
        }

        static bool first_time = true;

        float current_BSF = pq_bsf->knn[pq_bsf->k - 1];
        file_position_type position = pq_bsf->position[pq_bsf->k - 1];
        int query_id = query_counter;

        if (first_time)
        {
            bsf_sharing_data.shared_bsfs[my_rank].bsf = current_BSF;
            bsf_sharing_data.shared_bsfs[my_rank].position = position;
            bsf_sharing_data.shared_bsfs[my_rank].q_num = query_id;

            if (ENABLE_PRINTS_BSF_SHARING)
            {
                printf("[BSF-SENDER - Node %d]: First Time Initial ever bcasting BSF(%d, %llu, %f).\n", 
                       my_rank, bsf_sharing_data.shared_bsfs[my_rank].q_num, 
                       bsf_sharing_data.shared_bsfs[my_rank].position, 
                       bsf_sharing_data.shared_bsfs[my_rank].bsf);
            }

            bsf_sharing_data.bsf_broadcasts_counter++;

            ::dinoLib::timer_start(timer_manager, "COMM");
            MPI_Ibcast(&bsf_sharing_data.shared_bsfs[my_rank], 1, bsf_msg_type, my_rank, 
                      bsf_sharing_data.communicators[my_rank], &bsf_sharing_data.requests[my_rank]);
            ::dinoLib::timer_stop(timer_manager, "COMM");

            first_time = false;
            return;
        }

        int ready;
        ::::dinoLib::timer_start(timer_manager, "COMM");
        MPI_Test(&bsf_sharing_data.requests[my_rank], &ready, MPI_STATUS_IGNORE);
        ::::dinoLib::timer_stop(timer_manager, "COMM");

        if (!ready)
        {
            return;
        }

        bool bsf_changed = (current_BSF < bsf_sharing_data.shared_bsfs[my_rank].bsf);
        bool query_changed = (query_id != bsf_sharing_data.shared_bsfs[my_rank].q_num);
        if (bsf_changed || query_changed)
        {
            bsf_sharing_data.shared_bsfs[my_rank].bsf = current_BSF;
            bsf_sharing_data.shared_bsfs[my_rank].position = position;
            bsf_sharing_data.shared_bsfs[my_rank].q_num = query_id;

            bsf_sharing_data.bsf_broadcasts_counter++;

            if (ENABLE_PRINTS_BSF_SHARING)
            {
                printf("[BSF-SENDER - Node %d]: Sends an improvement BSF = (%d, %llu, %f)\n", 
                       my_rank, bsf_sharing_data.shared_bsfs[my_rank].q_num, 
                       bsf_sharing_data.shared_bsfs[my_rank].position, 
                       bsf_sharing_data.shared_bsfs[my_rank].bsf);
            }

            ::dinoLib::timer_start(timer_manager, "COMM");
            MPI_Ibcast(&bsf_sharing_data.shared_bsfs[my_rank], 1, bsf_msg_type, my_rank, 
                      bsf_sharing_data.communicators[my_rank], &bsf_sharing_data.requests[my_rank]);
            ::dinoLib::timer_stop(timer_manager, "COMM");
        }
    }

    void bsf_sharing_recv_bsf(BsfSharingData &bsf_sharing_data, pqueue_bsf *pq_bsf, 
                              int workernumber, std::vector<BsfMessage> &shared_bsf_results, 
                              pthread_mutex_t *lock_bsf, int my_rank, int comm_sz, int query_counter)
    {
        if (workernumber != 0 || !bsf_sharing_data.bsf_sharing_enabled || pq_bsf == nullptr)
        {
            return;
        }

        for (int process_rank = 0; process_rank < comm_sz; process_rank++)
        {
            int ready;
            MPI_Test(&bsf_sharing_data.requests[process_rank], &ready, MPI_STATUS_IGNORE);

            if (my_rank != process_rank && ready)
            {
                // Process process_rank has shared a new BSF value
                do
                {
                    bsf_sharing_data.bsf_receives_counter++;

                    float bsf_received = bsf_sharing_data.shared_bsfs[process_rank].bsf;
                    int query_id_received = bsf_sharing_data.shared_bsfs[process_rank].q_num;
                    file_position_type position_received = bsf_sharing_data.shared_bsfs[process_rank].position;

                    float current_bsf_shared = bsf_sharing_data.shared_bsfs[my_rank].bsf;
                    float current_bsf_local = pq_bsf->knn[pq_bsf->k - 1];

                    // Bookkeeping: update shared_bsf_results if received BSF is better
                    if (query_id_received >= 0 && query_id_received < (int)shared_bsf_results.size() &&
                        bsf_received < shared_bsf_results[query_id_received].bsf)
                    {
                        if (ENABLE_PRINTS_BSF_SHARING)
                        {
                            printf("[BSF SHARING] - Node %d received an improved bsf-msg (%d, %f, %llu) from node %d. QueryCounter = %d\n", 
                                   my_rank, query_id_received, bsf_received, position_received, process_rank, query_counter);
                        }

                        shared_bsf_results[query_id_received].bsf = bsf_received;
                        shared_bsf_results[query_id_received].position = position_received;
                        shared_bsf_results[query_id_received].q_num = query_id_received;
                    }

                    // Check if the received BSF is for my current query and if it is smaller than my current BSF
                    if (query_counter == query_id_received && bsf_received < current_bsf_shared && bsf_received < current_bsf_local)
                    {
                        bsf_sharing_data.shared_bsfs[my_rank].bsf = bsf_sharing_data.shared_bsfs[process_rank].bsf;
                        bsf_sharing_data.shared_bsfs[my_rank].position = bsf_sharing_data.shared_bsfs[process_rank].position;
                        bsf_sharing_data.shared_bsfs[my_rank].q_num = query_counter;

                        if (query_id_received >= 0 && query_id_received < (int)shared_bsf_results.size())
                        {
                            shared_bsf_results[query_id_received].bsf = bsf_received;
                            shared_bsf_results[query_id_received].position = position_received;
                            shared_bsf_results[query_id_received].q_num = query_id_received;
                        }
                    }

                    // Continue listening to the communicator of process_rank, and consume all the messages of this communicator
                    MPI_Ibcast(&bsf_sharing_data.shared_bsfs[process_rank], 1, bsf_msg_type, process_rank, 
                              bsf_sharing_data.communicators[process_rank], &bsf_sharing_data.requests[process_rank]);
                    MPI_Test(&bsf_sharing_data.requests[process_rank], &ready, MPI_STATUS_IGNORE);
                } while (ready);

                float new_bsf = bsf_sharing_data.shared_bsfs[my_rank].bsf;
                file_position_type new_position = bsf_sharing_data.shared_bsfs[my_rank].position;
                if (new_bsf < pq_bsf->knn[pq_bsf->k - 1])
                {
                    bsf_sharing_data.bsf_correct_receives_counter++;
                    pthread_mutex_lock(lock_bsf);
                    pqueue_bsf_insert_invalidate_worse_entries(pq_bsf, new_bsf, new_position, nullptr);
                    pthread_mutex_unlock(lock_bsf);
                }
            }
        }
    }

    void bsf_sharing_update_from_bookkeeping(BsfSharingData &bsf_sharing_data, 
                                              pqueue_bsf *pq_bsf, 
                                              std::vector<BsfMessage> &shared_bsf_results, 
                                              int query_counter)
    {
        if (!bsf_sharing_data.bsf_sharing_enabled || pq_bsf == nullptr)
        {
            return;
        }

        if (query_counter < 0 || query_counter >= (int)shared_bsf_results.size())
        {
            return;
        }

        float last_bsf = pq_bsf->knn[pq_bsf->k - 1];
        file_position_type last_position = pq_bsf->position[pq_bsf->k - 1];

        if (shared_bsf_results[query_counter].bsf < last_bsf)
        {
            pqueue_bsf_insert_invalidate_worse_entries(pq_bsf, shared_bsf_results[query_counter].bsf, 
                                                      shared_bsf_results[query_counter].position, nullptr);
        }
        else if (shared_bsf_results[query_counter].bsf > last_bsf)
        {
            shared_bsf_results[query_counter].bsf = last_bsf;
            shared_bsf_results[query_counter].position = last_position;
            shared_bsf_results[query_counter].q_num = query_counter;
        }
    }

} // namespace diNoLib
