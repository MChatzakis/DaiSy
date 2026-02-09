#ifndef REPLICATION_GROUP_HPP
#define REPLICATION_GROUP_HPP

#include <vector>
#include <string>
#include <memory>
#include <cstdint>

namespace daisy
{
    
    using idx_t = unsigned long long;

    struct SystemNode
    {
        int node_id;
        int index_threads;
        int query_threads;
    };

    struct ReplicationGroup
    {
        int group_id; 

        int total_nodes;      
        int coordinator_node; 
        std::vector<SystemNode> nodes;

        idx_t total_time_series; 

        int index_threads_total;
        int query_threads_total;
    };

    struct ReplicationData
    {
        int total_groups = 1;
        std::string node_groups_config;

        std::vector<ReplicationGroup> node_groups;
        std::vector<int> node_group_mappings;
    };

    void rep_init(ReplicationData &replication_data, idx_t dataset_size, int my_rank, int comm_sz, 
                  int &index_threads, int &query_threads);
    
    void rep_destroy(ReplicationData &replication_data);

    void rep_init_from_file(ReplicationData &replication_data, idx_t dataset_size, int my_rank, 
                            int comm_sz, int &index_threads, int &query_threads);
    
    void rep_init_from_params(ReplicationData &replication_data, idx_t dataset_size, int my_rank, 
                              int comm_sz, int &index_threads, int &query_threads);
    
    void rep_log_info(const ReplicationData &replication_data, int index_threads, int query_threads);
    
    long rep_allocate_data_series_default(ReplicationData &replication_data, int group_id, 
                                          idx_t dataset_size);

    int rep_find_group(const ReplicationData &replication_data, int rank);
    
    int rep_find_coordinator_node_rank(const ReplicationData &replication_data, int rank);
    
    int rep_get_repgroup_nodes(const ReplicationData &replication_data, int rank);
    
    idx_t rep_get_time_series_of_group(const ReplicationData &replication_data, int rank);
    
    idx_t rep_get_time_series_offset(const ReplicationData &replication_data, int rank);

} 

#endif 
