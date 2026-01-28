#include "../hodyssey/replication.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace diNoLib
{
    void rep_log_info(const ReplicationData &replication_data, int index_threads, int query_threads)
    {
        printf("=============== Replication Groups ===============\n");

        printf("Replication Conf File: [%s]\n", 
               replication_data.node_groups_config.empty() ? "Not Provided" : replication_data.node_groups_config.c_str());
        printf("Total Replication Groups: [%d]\n", replication_data.total_groups);
        printf("[Index Workers, Query Workers]: [%d, %d]\n", index_threads, query_threads);

        for (int i = 0; i < replication_data.total_groups; i++)
        {
            printf("Replication Group [%d]: \n", replication_data.node_groups[i].group_id);
            printf("    Time Series: %llu\n", replication_data.node_groups[i].total_time_series);
            printf("    Total Nodes: %d\n", replication_data.node_groups[i].total_nodes);
            printf("    Coordinator Node: %d\n", replication_data.node_groups[i].coordinator_node);
            printf("    Index Threads (Total): %d\n", replication_data.node_groups[i].index_threads_total);
            printf("    Query Threads (Total): %d\n", replication_data.node_groups[i].query_threads_total);

            printf("    Nodes: [ ");
            for (int n = 0; n < replication_data.node_groups[i].total_nodes; n++)
            {
                printf("%d ", replication_data.node_groups[i].nodes[n].node_id);
            }
            printf("]\n");

            printf("    Index Threads: [ ");
            for (int n = 0; n < replication_data.node_groups[i].total_nodes; n++)
            {
                printf("%d ", replication_data.node_groups[i].nodes[n].index_threads);
            }
            printf("]\n");

            printf("    Query Threads: [ ");
            for (int n = 0; n < replication_data.node_groups[i].total_nodes; n++)
            {
                printf("%d ", replication_data.node_groups[i].nodes[n].query_threads);
            }
            printf("]\n");
        }

        printf("==================================================\n");
    }

    void rep_init(ReplicationData &replication_data, idx_t dataset_size, int my_rank, int comm_sz, 
                  int &index_threads, int &query_threads)
    {
        if (replication_data.node_groups_config.empty())
        {
            rep_init_from_params(replication_data, dataset_size, my_rank, comm_sz, index_threads, query_threads);
        }
        else
        {
            rep_init_from_file(replication_data, dataset_size, my_rank, comm_sz, index_threads, query_threads);
        }
    }

    void rep_destroy(ReplicationData &replication_data)
    {
        // With std::vector, no manual free needed - vectors will be automatically cleaned up
        // But we can clear them explicitly for clarity
        replication_data.node_groups.clear();
        replication_data.node_group_mappings.clear();
    }

    void rep_init_from_params(ReplicationData &replication_data, idx_t dataset_size, int my_rank, 
                              int comm_sz, int &index_threads, int &query_threads)
    {
        // This function is called when only replication_data->total_groups is provided, 
        // and no conf file is present

        replication_data.node_groups.clear();
        replication_data.node_groups.resize(replication_data.total_groups);
        replication_data.node_group_mappings.clear();
        replication_data.node_group_mappings.resize(comm_sz);

        int nodes_of_each_group = comm_sz / replication_data.total_groups;
        
        for (int i = 0; i < replication_data.total_groups; i++)
        {
            int nodes_of_this_group;
            if (i == replication_data.total_groups - 1)
            {
                nodes_of_this_group = comm_sz - nodes_of_each_group * (replication_data.total_groups - 1);
            }
            else
            {
                nodes_of_this_group = nodes_of_each_group;
            }

            replication_data.node_groups[i].total_nodes = nodes_of_this_group;
            replication_data.node_groups[i].group_id = i;
            replication_data.node_groups[i].nodes.clear();
            replication_data.node_groups[i].nodes.resize(nodes_of_this_group);

            replication_data.node_groups[i].index_threads_total = 0;
            replication_data.node_groups[i].query_threads_total = 0;

            for (int node = 0; node < nodes_of_this_group; node++)
            {
                replication_data.node_groups[i].nodes[node].node_id = i * nodes_of_each_group + node;
                replication_data.node_groups[i].nodes[node].index_threads = index_threads;
                replication_data.node_groups[i].nodes[node].query_threads = query_threads;

                replication_data.node_groups[i].index_threads_total += index_threads;
                replication_data.node_groups[i].query_threads_total += query_threads;

                if (node == 0)
                {
                    replication_data.node_groups[i].coordinator_node = replication_data.node_groups[i].nodes[node].node_id;
                }

                replication_data.node_group_mappings[replication_data.node_groups[i].nodes[node].node_id] = i;
            }

            replication_data.node_groups[i].total_time_series = rep_allocate_data_series_default(replication_data, i, dataset_size);
        }

        // Checking if the configuration makes sense.
        // The checks are minimalistic and do not cover all cases where something could be wrong.
        long time_series_sum = 0;
        int nodes_sum = 0;
        for (int i = 0; i < replication_data.total_groups; i++)
        {
            nodes_sum += replication_data.node_groups[i].total_nodes;
            time_series_sum += replication_data.node_groups[i].total_time_series;
        }

        if (time_series_sum != (long)dataset_size)
        {
            if (my_rank == 0)
                printf("Error: The sum of the data series allocated to each group is not equal to the whole dataset size.\n");

            exit(EXIT_FAILURE);
        }

        if (comm_sz != nodes_sum)
        {
            if (my_rank == 0)
                printf("Error: The sum of the nodes is different from total nodes\n");

            exit(EXIT_FAILURE);
        }
    }

    void rep_init_from_file(ReplicationData &replication_data, idx_t dataset_size, int my_rank, 
                            int comm_sz, int &index_threads, int &query_threads)
    {
        FILE *conf = fopen(replication_data.node_groups_config.c_str(), "r");
        if (!conf)
        {
            if (my_rank == 0)
                perror("Could not open node group configuration file");
            exit(EXIT_FAILURE);
        }

        int total_groups_read;
        if (fscanf(conf, "%d", &total_groups_read) != 1) {
            fprintf(stderr, "Error reading total_groups from configuration file.\n");
            fclose(conf);
            exit(EXIT_FAILURE);
        }
        replication_data.total_groups = total_groups_read;

        replication_data.node_groups.clear();
        replication_data.node_groups.resize(replication_data.total_groups);
        replication_data.node_group_mappings.clear();
        replication_data.node_group_mappings.resize(comm_sz);

        for (int i = 0; i < replication_data.total_groups; i++)
        {
            int group_id;
            int group_size;
            if (fscanf(conf, "%d %d", &group_id, &group_size) != 2) {
                fprintf(stderr, "Error reading group_id and group_size from configuration file.\n");
                fclose(conf);
                exit(EXIT_FAILURE);
            }

            replication_data.node_groups[i].total_nodes = group_size;
            replication_data.node_groups[i].group_id = group_id;
            replication_data.node_groups[i].nodes.clear();
            replication_data.node_groups[i].nodes.resize(group_size);

            replication_data.node_groups[i].index_threads_total = 0;
            replication_data.node_groups[i].query_threads_total = 0;

            for (int j = 0; j < group_size; j++)
            {
                int node_id;
                int node_index_threads;
                int node_query_threads;
                if (fscanf(conf, "%d %d %d", &node_id, &node_index_threads, &node_query_threads) != 3) {
                    fprintf(stderr, "Error reading node configuration (node_id, node_index_threads, node_query_threads) from file.\n");
                    fclose(conf);
                    exit(EXIT_FAILURE);
                }

                replication_data.node_groups[i].nodes[j].node_id = node_id;
                replication_data.node_groups[i].nodes[j].index_threads = node_index_threads;
                replication_data.node_groups[i].nodes[j].query_threads = node_query_threads;

                if (node_id == my_rank)
                {
                    index_threads = node_index_threads;
                    query_threads = node_query_threads;
                }

                replication_data.node_groups[i].index_threads_total += node_index_threads;
                replication_data.node_groups[i].query_threads_total += node_query_threads;

                if (j == 0)
                {
                    replication_data.node_groups[i].coordinator_node = replication_data.node_groups[i].nodes[j].node_id;
                }

                replication_data.node_group_mappings[node_id] = group_id;
            }

            int total_time_series;
            if (fscanf(conf, "%d", &total_time_series) != 1) {
                fprintf(stderr, "Error reading total_time_series from configuration file.\n");
                fclose(conf);
                exit(EXIT_FAILURE);
            }

            // Configuration could give any number of time series to each group.
            // In case we have a negative number, we allocate the time series automatically.
            if (total_time_series >= 0)
                replication_data.node_groups[i].total_time_series = total_time_series;
            else
                replication_data.node_groups[i].total_time_series = rep_allocate_data_series_default(replication_data, group_id, dataset_size);
        }

        fclose(conf);

        // Checking if the configuration makes sense.
        // The checks are minimalistic and do not cover all cases where something could be wrong.
        long time_series_sum = 0;
        int nodes_sum = 0;
        for (int i = 0; i < replication_data.total_groups; i++)
        {
            nodes_sum += replication_data.node_groups[i].total_nodes;
            time_series_sum += replication_data.node_groups[i].total_time_series;
        }

        if (time_series_sum != (long)dataset_size)
        {
            if (my_rank == 0)
                printf("Error: The sum of the data series allocated to each group is not equal to the whole dataset size.\n");

            exit(EXIT_FAILURE);
        }

        if (comm_sz != nodes_sum)
        {
            if (my_rank == 0)
                printf("Error: The sum of the nodes is different from total nodes\n");

            exit(EXIT_FAILURE);
        }
    }

    long rep_allocate_data_series_default(ReplicationData &replication_data, int group_id, idx_t dataset_size)
    {
        int chunks = replication_data.total_groups;
        int chunk_size_aprox = (int)(dataset_size / replication_data.total_groups);

        if (group_id != chunks - 1)
        {
            return chunk_size_aprox;
        }

        return dataset_size - chunk_size_aprox * (chunks - 1);
    }

    int rep_find_group(const ReplicationData &replication_data, int rank)
    {
        if (rank < 0 || rank >= (int)replication_data.node_group_mappings.size())
        {
            return -1;  // Invalid rank
        }
        return replication_data.node_group_mappings[rank];
    }

    bool rep_is_last_node_of_group(const ReplicationData &replication_data, int rank)
    {
        int group_id = rep_find_group(replication_data, rank);
        if (group_id < 0 || group_id >= (int)replication_data.node_groups.size())
        {
            return false;
        }
        
        const ReplicationGroup &group = replication_data.node_groups[group_id];
        if (group.total_nodes == 0)
        {
            return false;
        }
        
        return rank == group.nodes[group.total_nodes - 1].node_id;
    }

    int rep_find_coordinator_node_rank(const ReplicationData &replication_data, int rank)
    {
        int group_id = rep_find_group(replication_data, rank);
        if (group_id < 0 || group_id >= (int)replication_data.node_groups.size())
        {
            return -1;  // Invalid group
        }
        return replication_data.node_groups[group_id].coordinator_node;
    }

    int rep_get_repgroup_nodes(const ReplicationData &replication_data, int rank)
    {
        int group_id = rep_find_group(replication_data, rank);
        if (group_id < 0 || group_id >= (int)replication_data.node_groups.size())
        {
            return 0;
        }
        return replication_data.node_groups[group_id].total_nodes;
    }

    idx_t rep_get_time_series_of_group(const ReplicationData &replication_data, int rank)
    {
        int group_id = rep_find_group(replication_data, rank);
        if (group_id < 0 || group_id >= (int)replication_data.node_groups.size())
        {
            return 0;
        }
        return replication_data.node_groups[group_id].total_time_series;
    }

    ReplicationGroup rep_get_group(const ReplicationData &replication_data, int rank)
    {
        int group_id = rep_find_group(replication_data, rank);
        if (group_id < 0 || group_id >= (int)replication_data.node_groups.size())
        {
            // Return empty group as error indicator
            ReplicationGroup empty_group;
            empty_group.group_id = -1;
            empty_group.total_nodes = 0;
            empty_group.coordinator_node = -1;
            empty_group.total_time_series = 0;
            empty_group.index_threads_total = 0;
            empty_group.query_threads_total = 0;
            return empty_group;
        }
        return replication_data.node_groups[group_id];
    }

    idx_t rep_get_time_series_offset(const ReplicationData &replication_data, int rank)
    {
        int group_id = rep_find_group(replication_data, rank);
        if (group_id < 0)
        {
            return 0;
        }
        
        idx_t offset = 0;
        for (int i = 0; i < group_id; i++)
        {
            if (i < (int)replication_data.node_groups.size())
            {
                offset += replication_data.node_groups[i].total_time_series;
            }
        }
        return offset;
    }

} // namespace diNoLib
