#ifndef BSF_SHARING_HPP
#define BSF_SHARING_HPP

#include "../../isax/iSAXTypes.hpp"
#include "../../isax/iSAXIndex.hpp"
#include "../../isax/iSAXPqueue.hpp"
#include "../../utils/TimerManager.hpp"
#include <vector>
#include <memory>
#include <mutex>
#include <mpi.h>
#include <pthread.h>

namespace daisy
{

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

        std::vector<MPI_Comm> communicators;     
        std::vector<MPI_Request> requests;       
        std::vector<BsfMessage> shared_bsfs;    
        std::vector<int> bcasts_per_query;

        int bsf_broadcasts_counter = 0;
        int bsf_receives_counter = 0;
        int bsf_correct_receives_counter = 0;
    };

    void bsf_sharing_init(BsfSharingData &bsf_sharing_data, int my_rank, int comm_sz);
    
    void bsf_sharing_destroy(BsfSharingData &bsf_sharing_data, int comm_sz);
    
    struct ReplicationData;
    void bsf_sharing_bcast_bsf(BsfSharingData &bsf_sharing_data, pqueue_bsf *pq_bsf, 
                                int workernumber, int my_rank, int query_counter, 
                                const ReplicationData *replication_data,
                                ::daisy::timer_manager_t *timer_manager);
    
    void bsf_sharing_recv_bsf(BsfSharingData &bsf_sharing_data, pqueue_bsf *pq_bsf, 
                              int workernumber, std::vector<BsfMessage> &shared_bsf_results, 
                              pthread_mutex_t *lock_bsf, int my_rank, int comm_sz, int query_counter);
    
    void bsf_sharing_update_from_bookkeeping(BsfSharingData &bsf_sharing_data, 
                                              pqueue_bsf *pq_bsf, 
                                              std::vector<BsfMessage> &shared_bsf_results, 
                                              int query_counter);

} 

#endif 
