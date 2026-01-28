#ifndef BSF_SHARING_HPP
#define BSF_SHARING_HPP

#include "../isax/iSAXTypes.hpp"
#include "../isax/iSAXIndex.hpp"
#include <vector>
#include <memory>
#include <mutex>
#include <mpi.h>
#include <pthread.h>

namespace diNoLib
{
    // Forward declaration for timer_manager_t if needed
    // For now, using void* or we can create a proper TimerManager class later
    struct TimerManager;

    // MPI Datatype for BSF message (needs to be initialized)
    extern MPI_Datatype bsf_msg_type;

    struct BsfMessage
    {
        int q_num;
        float bsf;
        file_position_type position;
    };

    struct BsfSharingData
    {
        bool bsf_sharing_enabled = false;

        std::vector<MPI_Comm> communicators;     // we need the communicators because each node sends the BSF value to the other node
        std::vector<MPI_Request> requests;       // one for each communicator, to check if there is a message arrived
        std::vector<BsfMessage> shared_bsfs;    // one for each node, represent the received value from each
        std::vector<int> bcasts_per_query;

        int bsf_broadcasts_counter = 0;
        int bsf_receives_counter = 0;
        int bsf_correct_receives_counter = 0;
    };

    // BSF sharing initialization and management functions
    void bsf_sharing_init(BsfSharingData &bsf_sharing_data, int my_rank, int comm_sz);
    
    void bsf_sharing_destroy(BsfSharingData &bsf_sharing_data, int comm_sz);
    
    void bsf_sharing_bcast_bsf(BsfSharingData &bsf_sharing_data, pqueue_bsf *pq_bsf, 
                                int workernumber, int my_rank, int query_counter, 
                                dinoLib::timer_manager_t *timer_manager);
    
    void bsf_sharing_recv_bsf(BsfSharingData &bsf_sharing_data, pqueue_bsf *pq_bsf, 
                              int workernumber, std::vector<BsfMessage> &shared_bsf_results, 
                              pthread_mutex_t *lock_bsf, int my_rank, int comm_sz, int query_counter);
    
    void bsf_sharing_update_from_bookkeeping(BsfSharingData &bsf_sharing_data, 
                                              pqueue_bsf *pq_bsf, 
                                              std::vector<BsfMessage> &shared_bsf_results, 
                                              int query_counter);

} // namespace diNoLib

#endif // BSF_SHARING_HPP
