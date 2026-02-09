#include "../hodyssey/workstealing.hpp"
#include "../../isax/iSAXTypes.hpp"
#include <cstdio>
#include <cstdlib>

namespace daisy
{
    void ws_init(WorkstealingData &workstealing_data, int comm_sz)
    {
        if (workstealing_data.ws_type == WorkstealingType::DISABLED || comm_sz == 1)
        {
            return;
        }

        workstealing_data.ws_type = WorkstealingType::S_WS;

        workstealing_data.items_to_send = 4;
        workstealing_data.deterministic_index = false;

        workstealing_data.global_helper_requests.clear();
        workstealing_data.global_helper_requests.resize(comm_sz);
    }

    void ws_destroy(WorkstealingData &workstealing_data, int comm_sz)
    {
        if (workstealing_data.ws_type == WorkstealingType::DISABLED || comm_sz == 1)
        {
            return;
        }

        workstealing_data.global_helper_requests.clear();
    }

    int ws_cmp_isax_nodes(isax_index *index, isax_node *node_1, isax_node *node_2)
    {
        return node_1 == node_2 ? 1 : 0;  
    }

    isax_node *ws_compute_lca(isax_index *index, isax_node *node_1, isax_node *node_2)
    {
        if (node_1 == nullptr)
        {
            printf("LCA ISAX Error: Encountered null start node.\n");
            exit(EXIT_FAILURE);
        }

        if (node_2 == nullptr)
        {
            printf("LCA ISAX Error: Encountered null end node.\n");
            exit(EXIT_FAILURE);
        }

        isax_node *lca = nullptr;

        isax_node *node_1_tracker = node_1;
        isax_node *node_2_tracker = node_2;
        long int depth_1 = 0;
        long int depth_2 = 0;

        while (node_1_tracker->parent != nullptr)
        {
            node_1_tracker = node_1_tracker->parent;
            depth_1++;
        }

        while (node_2_tracker->parent != nullptr)
        {
            node_2_tracker = node_2_tracker->parent;
            depth_2++;
        }

        if (depth_1 == 0)
        {
            return node_1_tracker;
        }

        if (depth_2 == 0)
        {
            return node_2_tracker;
        }

        node_1_tracker = node_1;
        node_2_tracker = node_2;

        while (depth_1 > depth_2)
        {
            node_1_tracker = node_1_tracker->parent;
            depth_1--;
        }

        while (depth_2 > depth_1)
        {
            node_2_tracker = node_2_tracker->parent;
            depth_2--;
        }

        int depth = depth_1;
        while (ws_cmp_isax_nodes(index, node_1_tracker, node_2_tracker) != 1)
        {
            if (node_1_tracker->parent == nullptr)
            {
                lca = node_1_tracker;
                break;
            }

            node_1_tracker = node_1_tracker->parent;
            node_2_tracker = node_2_tracker->parent;
            depth--;
        }

        lca = node_1_tracker;

        return lca;
    }

    int ws_deep_cmp_isax_nodes(isax_index *index, isax_node *node1, isax_node *node2)
    {
        
        if (node1 == nullptr || node2 == nullptr)
        {
            return 0;
        }

        if (node1->isax_values == nullptr || node2->isax_values == nullptr ||
            node1->isax_cardinalities == nullptr || node2->isax_cardinalities == nullptr)
        {
            return 0;
        }

        for (int i = 0; i < index->settings->paa_segments; i++)
        {
            if (node1->isax_values[i] != node2->isax_values[i])
            {
                return 0;
            }

            if (node1->isax_cardinalities[i] != node2->isax_cardinalities[i])
            {
                return 0;
            }
        }

        return 1;
    }

    isax_node *ws_locate_node(isax_index *index, isax_node *node_to_locate)
    {
        if (node_to_locate == nullptr || node_to_locate->isax_values == nullptr)
        {
            printf("ws_locate_node: Invalid node_to_locate (null or missing isax_values)\n");
            exit(EXIT_FAILURE);
        }

        sax_type *sax = node_to_locate->isax_values;
        root_mask_type root_mask = 0;
        CREATE_MASK(root_mask, index, sax);

        parallel_first_buffer_layer_ekosmas *p_fbl = (parallel_first_buffer_layer_ekosmas *)(index->fbl);
        
        if (p_fbl == nullptr)
        {
            printf("ws_locate_node: index->fbl is null!\n");
            exit(EXIT_FAILURE);
        }

        if (root_mask < 0 || root_mask >= p_fbl->number_of_buffers)
        {
            printf("ws_locate_node: Invalid root_mask %llu (number_of_buffers = %d)\n", root_mask, p_fbl->number_of_buffers);
            exit(EXIT_FAILURE);
        }

        parallel_fbl_soft_buffer_ekosmas *soft_buffer = &(p_fbl->soft_buffers[(int)root_mask]);

        if (soft_buffer->initialized)
        {
            isax_node *node = soft_buffer->node;

            if (node == nullptr)
            {
                printf("ws_locate_node: Initial node could not be found!\n");
                exit(EXIT_FAILURE);
            }

            while (node != nullptr && ws_deep_cmp_isax_nodes(index, node, node_to_locate) != 1)
            {
                
                if (node->right_child == nullptr && node->left_child == nullptr)
                {
                    break;
                }

                if (node->split_data == nullptr || node->split_data->splitpoint < 0)
                {
                    printf("ws_locate_node: Invalid split_data or splitpoint\n");
                    exit(EXIT_FAILURE);
                }

                int location = index->settings->sax_bit_cardinality - 1 - node->split_data->split_mask[node->split_data->splitpoint];
                if (location < 0 || location >= (int)(sizeof(root_mask_type) * 8))
                {
                    printf("ws_locate_node: Invalid location %d\n", location);
                    exit(EXIT_FAILURE);
                }

                root_mask_type mask = index->settings->bit_masks[location];

                if (sax[node->split_data->splitpoint] & mask)
                {
                    node = node->right_child;
                }
                else
                {
                    node = node->left_child;
                }
            }

            if (node == nullptr)
            {
                printf("ws_locate_node: Could not locate the given iSAX node in the current index!, node NULL\n");
                exit(EXIT_FAILURE);
            }

            return node;
        }
        else
        {
            printf("ws_locate_node: Could not locate the given iSAX node in the current index! (soft_buffer not initialized)\n");
            exit(EXIT_FAILURE);
        }

        return nullptr;
    }

} 
