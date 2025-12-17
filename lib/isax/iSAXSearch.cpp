#include "iSAXSearch.hpp"
#include "iSAXIndex.hpp"
#include "iSAXPqueue.hpp"

#include <float.h>


namespace diNoLib
{
    void calculate_node_topk_inmemory(isax_index *index, isax_node *node, ts_type *query, pqueue_bsf *pq_bsf, float *rawfile)
    {
        // COUNT_CHECKED_NODE()
        //  If node has buffered data
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
        fprintf(stderr, "DEBUG refine_topk: entering function\n"); fflush(stderr);
        fprintf(stderr, "DEBUG refine_topk: index=%p, index->settings=%p\n", (void*)index, (void*)(index ? index->settings : nullptr)); fflush(stderr);

        int tight_bound = index->settings->tight_bound;
        int aggressive_check = index->settings->aggressive_check;
        fprintf(stderr, "DEBUG refine_topk: tight_bound=%d, aggressive_check=%d\n", tight_bound, aggressive_check); fflush(stderr);

        int j = 0;
        fprintf(stderr, "DEBUG refine_topk: root_nodes_size=%d\n", index->settings->root_nodes_size); fflush(stderr);
        pqueue_t *pq = pqueue_init(index->settings->root_nodes_size, cmp_pri, get_pri, set_pri, get_pos, set_pos);
        fprintf(stderr, "DEBUG refine_topk: pqueue initialized, pq=%p\n", (void*)pq); fflush(stderr);

        // Insert all root nodes in heap.
        isax_node *current_root_node = index->first_node;
        fprintf(stderr, "DEBUG refine_topk: first_node=%p\n", (void*)current_root_node); fflush(stderr);

        int node_count = 0;
        while (current_root_node != NULL)
        {
            query_result *mindist_result = (query_result *)malloc(sizeof(query_result));

            fprintf(stderr, "DEBUG refine_topk: node %d, isax_values=%p, isax_cardinalities=%p\n", 
                    node_count, (void*)current_root_node->isax_values, (void*)current_root_node->isax_cardinalities); fflush(stderr);

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
            node_count++;
        }
        fprintf(stderr, "DEBUG refine_topk: inserted %d root nodes into pqueue\n", node_count); fflush(stderr);
        query_result *n;
        int checks = 0;
        while ((n = (query_result *)pqueue_pop(pq)))
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
                            query_result *mindist_result = (query_result *)malloc(sizeof(query_result));
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
                            query_result *mindist_result = (query_result *)malloc(sizeof(query_result));
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
        while ((n = (query_result *)pqueue_pop(pq)))
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

    // new
    float dtw(float *A, float *B, float *cb, int m, int r, float bsf)
    {

        float *cost;
        float *cost_prev;
        float *cost_tmp;
        int i, j, k;
        float x, y, z, min_cost;

        /// Instead of using matrix of size O(m^2) or O(mr), we will reuse two array of size O(r).
        cost = (float *)malloc(sizeof(float) * (2 * r + 1));
        for (k = 0; k < 2 * r + 1; k++)
            cost[k] = FLT_MAX;

        cost_prev = (float *)malloc(sizeof(float) * (2 * r + 1));
        for (k = 0; k < 2 * r + 1; k++)
            cost_prev[k] = FLT_MAX;

        for (i = 0; i < m; i++)
        {
            k = max(0, r - i);
            min_cost = FLT_MAX;

            for (j = max(0, i - r); j <= min(m - 1, i + r); j++, k++)
            {
                /// Initialize all row and column
                if ((i == 0) && (j == 0))
                {
                    cost[k] = dist(A[0], B[0]);
                    min_cost = cost[k];
                    continue;
                }

                if ((j - 1 < 0) || (k - 1 < 0))
                    y = FLT_MAX;
                else
                    y = cost[k - 1];
                if ((i - 1 < 0) || (k + 1 > 2 * r))
                    x = FLT_MAX;
                else
                    x = cost_prev[k + 1];
                if ((i - 1 < 0) || (j - 1 < 0))
                    z = FLT_MAX;
                else
                    z = cost_prev[k];

                /// Classic DTW calculation
                cost[k] = min(min(x, y), z) + dist(A[i], B[j]);

                /// Find minimum cost in row for early abandoning (possibly to use column instead of row).
                if (cost[k] < min_cost)
                {
                    min_cost = cost[k];
                }
            }

            /// We can abandon early if the current cummulative distace with lower bound together are larger than bsf
            if (i + r < m - 1 && min_cost + cb[i + r + 1] >= bsf)
            {
                free(cost);
                free(cost_prev);
                return min_cost + cb[i + r + 1];
            }
            /// Move current array to previous array.
            cost_tmp = cost;
            cost = cost_prev;
            cost_prev = cost_tmp;
        }
        k--;

        /// the DTW distance is in the last cell in the matrix of size O(m^2) or at the middle of our array.
        float final_dtw = cost_prev[k];
        free(cost);
        free(cost_prev);
        return final_dtw;
    }

    void calculate_node_DTWknn_inmemory(isax_index *index, isax_node *node, ts_type *query, int warpWind, pqueue_bsf *pq_bsf, float *rawfile)
    {
        // COUNT_CHECKED_NODE()
        //  If node has buffered data

        if (node->buffer != NULL)
        {
            float *cb = (float *)calloc(index->settings->timeseries_size, sizeof(float));
            int i;
            // RDcalculationnumber=RDcalculationnumber+node->buffer->partial_buffer_size;
            for (i = 0; i < node->buffer->partial_buffer_size; i++)
            {

                float dist = dtw(query, &(rawfile[*node->buffer->partial_position_buffer[i]]), cb, index->settings->timeseries_size, warpWind, FLT_MAX);

                // ts_euclidean_distance_SIMD(query, &(rawfile[*node->buffer->partial_position_buffer[i]]),
                //                                   index->settings->timeseries_size, bsf);

                pqueue_bsf_insert(pq_bsf, dist, *node->buffer->partial_position_buffer[i] / index->settings->timeseries_size, node);
            }
            free(cb);
        }
    }

    void approximate_DTWtopk_inmemory(ts_type *ts, ts_type *paa, isax_index *index, int warpWind, pqueue_bsf *pq_bsf, float *rawfile)
    {

        sax_type *sax = (sax_type *)malloc(sizeof(sax_type) * index->settings->paa_segments);
        sax_from_paa(paa, sax, index->settings->paa_segments,
                     index->settings->sax_alphabet_cardinality,
                     index->settings->sax_bit_cardinality);

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

            calculate_node_DTWknn_inmemory(index, node, ts, warpWind, pq_bsf, rawfile);
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
} // namespace diNoLib
