#include "iSAXSearch.hpp"
#include "iSAXIndex.hpp"
#include "iSAXPqueue.hpp"

#include <float.h>

namespace diNoLib
{
    void calculate_node_topk_inmemory(isax_index *index, isax_node *node, ts_type *query, pqueue_bsf *pq_bsf, float *rawfile)
    {
        //COUNT_CHECKED_NODE()
        // If node has buffered data
        if (node->buffer != NULL)
        {
            int i;
            for (i = 0; i < node->buffer->full_buffer_size; i++)
            {
                float dist = ts_euclidean_distance_SIMD(query, node->buffer->full_ts_buffer[i],
                                                        index->settings->timeseries_size, pq_bsf->knn[pq_bsf->k - 1]);
                //__sync_fetch_and_add(&RDcalculationnumber, 1);

                if (dist <= pq_bsf->knn[pq_bsf->k - 1])
                {
                    pqueue_bsf_insert(pq_bsf, dist, 0, node);
                }
            }

            for (i = 0; i < node->buffer->tmp_full_buffer_size; i++)
            {
                float dist = ts_euclidean_distance_SIMD(query, node->buffer->tmp_full_ts_buffer[i],
                                                        index->settings->timeseries_size, pq_bsf->knn[pq_bsf->k - 1]);
                // __sync_fetch_and_add(&RDcalculationnumber, 1);

                if (dist <= pq_bsf->knn[pq_bsf->k - 1])
                {
                    pqueue_bsf_insert(pq_bsf, dist, 0, node);
                }
            }
            for (i = 0; i < node->buffer->partial_buffer_size; i++)
            {

                float dist = ts_euclidean_distance_SIMD(query, &(rawfile[*node->buffer->partial_position_buffer[i]]),
                                                        index->settings->timeseries_size, pq_bsf->knn[pq_bsf->k - 1]);
                //__sync_fetch_and_add(&RDcalculationnumber, 1);

                if (dist <= pq_bsf->knn[pq_bsf->k - 1])
                {
                    pqueue_bsf_insert(pq_bsf, dist, *node->buffer->partial_position_buffer[i] / index->settings->timeseries_size, node);
                }
            }
        }
        //////////////////////////////////////
    }

    void approximate_topk_inmemory(ts_type *ts, ts_type *paa, isax_index *index, pqueue_bsf *pq_bsf, float *rawfile)
    {

        sax_type *sax = (sax_type *)malloc(sizeof(sax_type) * index->settings->paa_segments);
        sax_from_paa(paa, sax, index->settings->paa_segments, index->settings->sax_alphabet_cardinality, index->settings->sax_bit_cardinality);

        root_mask_type root_mask = 0;
        CREATE_MASK(root_mask, index, sax);

        if ((&((parallel_first_buffer_layer *)(index->fbl))->soft_buffers[(int)root_mask])->initialized)
        {
            isax_node *node = (&((parallel_first_buffer_layer *)(index->fbl))->soft_buffers[(int)root_mask])->node;
            // Traverse tree

            // Adaptive splitting

            while (!node->is_leaf)
            {
                int location = index->settings->sax_bit_cardinality - 1 -
                               node->split_data->split_mask[node->split_data->splitpoint];
                root_mask_type mask = index->settings->bit_masks[location];

                if (sax[node->split_data->splitpoint] & mask)
                {
                    node = node->right_child;
                }
                else
                {
                    node = node->left_child;
                }

                // Adaptive splitting
            }
            calculate_node_topk_inmemory(index, node, ts, pq_bsf, rawfile);
        }
        else
        {
        }
        for (int i = 0; i < pq_bsf->k - 1; ++i)
        {
            pq_bsf->knn[i] = pq_bsf->knn[pq_bsf->k - 1];
        }
        free(sax);
    }

    void refine_topk_answer_inmemory(ts_type *ts, ts_type *paa, isax_index *index, pqueue_bsf *pq_bsf, float minimum_distance, int limit, float *rawfile)
    {

        int tight_bound = index->settings->tight_bound;
        int aggressive_check = index->settings->aggressive_check;

        int j = 0;
        pqueue_t *pq = pqueue_init(index->settings->root_nodes_size, cmp_pri, get_pri, set_pri, get_pos, set_pos);

        // Insert all root nodes in heap.
        isax_node *current_root_node = index->first_node;

        while (current_root_node != NULL)
        {
            query_result *mindist_result = (query_result*) malloc(sizeof(query_result));

            mindist_result->distance = minidist_paa_to_isax(paa, current_root_node->isax_values,
                                                            current_root_node->isax_cardinalities,
                                                            index->settings->sax_bit_cardinality,
                                                            index->settings->sax_alphabet_cardinality,
                                                            index->settings->paa_segments,
                                                            MINVAL, MAXVAL,
                                                            index->settings->mindist_sqrt);
            mindist_result->node = current_root_node;
            pqueue_insert(pq, mindist_result);

            current_root_node = current_root_node->next;
        }
        query_result *n;
        int checks = 0;
        while ((n = (query_result *) pqueue_pop(pq)))
        {
            // The best node has a worse mindist, so search is finished!
            if (n->distance >= pq_bsf->knn[pq_bsf->k - 1] || n->distance > minimum_distance)
            {
                pqueue_insert(pq, n);
                break;
            }
            else
            {
                // If it is a leaf, check its real distance.
                if (n->node->is_leaf)
                {
                    // *** ADAPTIVE SPLITTING ***
                    if (!n->node->has_full_data_file &&
                        (n->node->leaf_size > index->settings->min_leaf_size))
                    {
                        // Split and push again in the queue
                        split_node(index, n->node);
                        pqueue_insert(pq, n);
                        continue;
                    }
                    // *** EXTRA BOUNDING ***
                    if (tight_bound)
                    {
                        j++;
                        float mindistance = calculate_minimum_distance_inmemory(index, n->node, ts, paa);

                        if (mindistance >= pq_bsf->knn[pq_bsf->k - 1])
                        {
                            free(n);
                            continue;
                        }
                    }
                    // *** REAL DISTANCE ***
                    checks++;
                    calculate_node_topk_inmemory(index, n->node, ts, pq_bsf, rawfile);

                    if (pq_bsf->knn[pq_bsf->k - 1] < FLT_MAX)
                    {
                        pqueue_insert(pq, n);
                        break;
                    }
                }
                else
                {
                    // If it is an intermediate node calculate mindist for children
                    // and push them in the queue
                    if (n->node->left_child->isax_cardinalities != NULL)
                    {
                        if (n->node->left_child->is_leaf && !n->node->left_child->has_partial_data_file && aggressive_check)
                        {
                            calculate_node_topk_inmemory(index, n->node->left_child, ts, pq_bsf, rawfile);
                        }
                        else
                        {
                            query_result *mindist_result = (query_result *) malloc(sizeof(query_result));
                            mindist_result->distance = minidist_paa_to_isax(paa, n->node->left_child->isax_values,
                                                                            n->node->left_child->isax_cardinalities,
                                                                            index->settings->sax_bit_cardinality,
                                                                            index->settings->sax_alphabet_cardinality,
                                                                            index->settings->paa_segments,
                                                                            MINVAL, MAXVAL,
                                                                            index->settings->mindist_sqrt);
                            mindist_result->node = n->node->left_child;
                            pqueue_insert(pq, mindist_result);
                        }
                    }
                    if (n->node->right_child->isax_cardinalities != NULL)
                    {
                        if (n->node->right_child->is_leaf && !n->node->left_child->has_partial_data_file && aggressive_check)
                        {
                            calculate_node_topk_inmemory(index, n->node->right_child, ts, pq_bsf, rawfile);
                        }
                        else
                        {
                            query_result *mindist_result = (query_result *) malloc(sizeof(query_result));
                            mindist_result->distance = minidist_paa_to_isax(paa, n->node->right_child->isax_values,
                                                                            n->node->right_child->isax_cardinalities,
                                                                            index->settings->sax_bit_cardinality,
                                                                            index->settings->sax_alphabet_cardinality,
                                                                            index->settings->paa_segments,
                                                                            MINVAL, MAXVAL,
                                                                            index->settings->mindist_sqrt);
                            mindist_result->node = n->node->right_child;
                            pqueue_insert(pq, mindist_result);
                        }
                    }
                }

                // Free the node currently popped.
                free(n);
            }
        }
        // Free the nodes that where not popped.
        while ((n = (query_result *) pqueue_pop(pq)))
        {
            free(n);
        }
        for (int i = 0; i < pq_bsf->k - 1; ++i)
        {
            pq_bsf->knn[i] = pq_bsf->knn[pq_bsf->k - 1];
        }
        // Free the priority queue.

        pqueue_free(pq);
    }

} // namespace diNoLib
