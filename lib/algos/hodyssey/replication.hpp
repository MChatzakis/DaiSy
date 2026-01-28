#ifndef REPLICATION_GROUP_HPP
#define REPLICATION_GROUP_HPP

#include <vector>
#include <string>
#include <memory>
#include <cstdint>

namespace diNoLib
{
    // Use idx_t from SimilaritySearchAlgorithm.hpp as data_size_type
    using idx_t = unsigned long long;

    struct SystemNode
    {
        int node_id;
        int index_threads;
        int query_threads;
    };

    struct ReplicationGroup
    {
        int group_id; // group id

        int total_nodes;      // how many nodes this group has
        int coordinator_node; // coordinator node id
        std::vector<SystemNode> nodes;

        idx_t total_time_series; // how many data series this group works on

        int index_threads_total;
        int query_threads_total;
    };

    struct ReplicationData
    {
        int total_groups = 1;
        std::string node_groups_config;
        int bsf_sharing_pdr = 1; // Share BSF among node groups

        std::vector<ReplicationGroup> node_groups;
        std::vector<int> node_group_mappings;
    };

    // Replication initialization and management functions
    void rep_init(ReplicationData &replication_data, idx_t dataset_size, int my_rank, int comm_sz, 
                  int &index_threads, int &query_threads);
    
    void rep_destroy(ReplicationData &replication_data);

    /* Replication group init and management */
    void rep_init_from_file(ReplicationData &replication_data, idx_t dataset_size, int my_rank, 
                            int comm_sz, int &index_threads, int &query_threads);
    
    void rep_init_from_params(ReplicationData &replication_data, idx_t dataset_size, int my_rank, 
                              int comm_sz, int &index_threads, int &query_threads);
    
    void rep_log_info(const ReplicationData &replication_data, int index_threads, int query_threads);
    
    long rep_allocate_data_series_default(ReplicationData &replication_data, int group_id, 
                                          idx_t dataset_size);

    /* Replication group navigation functions */
    int rep_find_group(const ReplicationData &replication_data, int rank);
    
    int rep_find_coordinator_node_rank(const ReplicationData &replication_data, int rank);
    
    int rep_get_repgroup_nodes(const ReplicationData &replication_data, int rank);
    
    bool rep_is_last_node_of_group(const ReplicationData &replication_data, int rank);
    
    idx_t rep_get_time_series_of_group(const ReplicationData &replication_data, int rank);
    
    idx_t rep_get_time_series_offset(const ReplicationData &replication_data, int rank);
    
    ReplicationGroup rep_get_group(const ReplicationData &replication_data, int rank);

} // namespace diNoLib

#endif // REPLICATION_GROUP_HPP
