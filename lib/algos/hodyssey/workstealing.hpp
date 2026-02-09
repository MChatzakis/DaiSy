#ifndef WORKSTEALING_HPP
#define WORKSTEALING_HPP

#include <vector>
#include <mpi.h>
#include "../../isax/iSAXIndex.hpp"

namespace daisy
{
    
    constexpr int WORKSTEALING_INFORM_AVAILABILITY = 900;
    constexpr int WORKSTEALING_QUERY_ANSWERING_COMPLETION = 901;
    constexpr int WORKSTEALING_DATA_SEND = 902;

    enum class WorkstealingType
    {
        DISABLED = 0,
        S_WS = 1,
        P_WS = 2
    };

    struct WorkstealingData
    {
        int items_to_send;

        bool deterministic_index = false;             

        std::vector<MPI_Request> global_helper_requests; 
        WorkstealingType ws_type = WorkstealingType::S_WS; 
    };

    void ws_init(WorkstealingData &workstealing_data, int comm_sz);
    void ws_destroy(WorkstealingData &workstealing_data, int comm_sz);

    int ws_cmp_isax_nodes(isax_index *index, isax_node *node_1, isax_node *node_2);
    int ws_deep_cmp_isax_nodes(isax_index *index, isax_node *node1, isax_node *node2);

    isax_node* ws_locate_node(isax_index *index, isax_node *node_to_locate);
    isax_node* ws_compute_lca(isax_index *index, isax_node *node_1, isax_node *node_2);

} 

#endif 
