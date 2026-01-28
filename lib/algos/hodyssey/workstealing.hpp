#ifndef WORKSTEALING_HPP
#define WORKSTEALING_HPP

#include <vector>
#include <mpi.h>
#include "../isax/iSAXIndex.hpp"

namespace diNoLib
{
    // Workstealing message tags
    constexpr int WORKSTEALING_INFORM_AVAILABILITY = 900;
    constexpr int WORKSTEALING_QUERY_ANSWERING_COMPLETION = 901;
    constexpr int WORKSTEALING_DATA_SEND = 902;
    constexpr int WORKSTEALING_BSF_SHARE = 903;

    enum class WorkstealingType
    {
        DISABLED = 0,
        S_WS = 1,
        P_WS = 2
    };

    struct WorkstealingData
    {
        int items_to_send;
        
        bool first_time_flag = false;                 // First time bsf sharing requires some special handling
        bool deterministic_index = false;             // Create deterministic index (Index across nodes with same data will be exactly the same)
        
        float limit_factor;                  // If the current node has processed >=80% of PQs, do not accept workstealing
        
        std::vector<MPI_Request> global_helper_requests; // Helper node requests (should be global single object)
        WorkstealingType ws_type = WorkstealingType::S_WS; // 0 for no ws, 1 for first algorithm, 2 for second.
    };

    // Workstealing initialization and cleanup
    void ws_init(WorkstealingData &workstealing_data, int comm_sz);
    void ws_destroy(WorkstealingData &workstealing_data, int comm_sz);

    // Node comparison functions for workstealing
    int ws_cmp_isax_nodes(isax_index *index, isax_node *node_1, isax_node *node_2);
    int ws_deep_cmp_isax_nodes(isax_index *index, isax_node *node1, isax_node *node2);

    // Node location and LCA (Lowest Common Ancestor) computation
    isax_node* ws_locate_node(isax_index *index, isax_node *node_to_locate);
    isax_node* ws_compute_lca(isax_index *index, isax_node *node_1, isax_node *node_2);

} // namespace diNoLib

#endif // WORKSTEALING_HPP
