#include "../hodyssey/Odyssey.hpp"
#include "../hodyssey/indexing.hpp"  
#include "../hodyssey/query_answering.hpp"  
#include "../hodyssey/replication.hpp"      
#include "../hodyssey/workstealing.hpp"     
#include "../../isax/iSAXSearch.hpp"  
#include "../../isax/iSAXIndex.hpp"   
#include "../../isax/iSAXPqueue.hpp"   
#include "../../isax/SAX.hpp"        
#include "../../isax/iSAXTypes.hpp"   
#include "../hodyssey/bsf_sharing.hpp" 
#include "common.hpp"
#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <unordered_set>
#include <thread>
#include <chrono>
#if ODYSSEY_MPI
#include <mpi.h>
#endif

#ifndef CHECK_ALLOC
#define CHECK_ALLOC(ptr, rank) \
    do { \
        if ((ptr) == nullptr) { \
            fprintf(stderr, "[Node %d] Error: Memory allocation failed in %s:%d\n", (rank), __FILE__, __LINE__); \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)
#endif

namespace daisy
{
#if ODYSSEY_MPI
    
    static void odyssey_merge_knn_results_mpi(int my_rank, int comm_sz, int q_num, int topk, idx_t *I, float *D)
    {
        if (comm_sz <= 1)
            return;
        const size_t per_rank = static_cast<size_t>(q_num) * static_cast<size_t>(topk);
        idx_t *all_I = nullptr;
        float *all_D = nullptr;
        if (my_rank == 0)
        {
            all_I = static_cast<idx_t *>(std::malloc(per_rank * static_cast<size_t>(comm_sz) * sizeof(idx_t)));
            all_D = static_cast<float *>(std::malloc(per_rank * static_cast<size_t>(comm_sz) * sizeof(float)));
            CHECK_ALLOC(all_I, my_rank);
            CHECK_ALLOC(all_D, my_rank);
        }
        MPI_Gather(I, static_cast<int>(per_rank), MPI_UNSIGNED_LONG_LONG,
                   all_I, static_cast<int>(per_rank), MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        MPI_Gather(D, static_cast<int>(per_rank), MPI_FLOAT,
                   all_D, static_cast<int>(per_rank), MPI_FLOAT, 0, MPI_COMM_WORLD);
        if (my_rank == 0)
        {
            std::vector<std::pair<float, idx_t>> merged;
            merged.reserve(static_cast<size_t>(comm_sz) * static_cast<size_t>(topk));
            for (int q = 0; q < q_num; q++)
            {
                merged.clear();
                for (int r = 0; r < comm_sz; r++)
                {
                    for (int j = 0; j < topk; j++)
                    {
                        size_t idx = static_cast<size_t>(r) * per_rank + static_cast<size_t>(q) * static_cast<size_t>(topk) + static_cast<size_t>(j);
                        idx_t pos = all_I[idx];
                        float dist = all_D[idx];
                        
                        if (dist >= FLT_MAX * 0.99f)
                            continue;
                        
                        if (pos == static_cast<idx_t>(static_cast<long>(-1)))
                            continue;
                        merged.push_back({dist, pos});
                    }
                }
                std::sort(merged.begin(), merged.end());
                std::unordered_set<idx_t> seen;
                int count = 0;
                idx_t *out_I = I + q * topk;
                float *out_D = D + q * topk;
                for (const auto &p : merged)
                {
                    if (seen.find(p.second) == seen.end())
                    {
                        seen.insert(p.second);
                        out_I[count] = p.second;
                        out_D[count] = p.first;
                        count++;
                        if (count >= topk)
                            break;
                    }
                }
                
                if (count > 0 && count < topk)
                {
                    for (int j = count; j < topk; j++)
                    {
                        out_I[j] = out_I[count - 1];
                        out_D[j] = out_D[count - 1];
                    }
                }
                
                if (count == 0)
                {
                    for (int j = 0; j < topk; j++)
                    {
                        out_I[j] = 0;
                        out_D[j] = FLT_MAX;
                    }
                }
            }
            std::free(all_I);
            std::free(all_D);
        }
        
        MPI_Bcast(I, static_cast<int>(per_rank), MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(D, static_cast<int>(per_rank), MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
#endif

    OdysseyQuery* load_queries_from_buffer(const float *query_buf, int q_num, isax_index *index, int my_rank)
    {
        const int ts_len = index->settings->timeseries_size;
        OdysseyQuery *queries = (OdysseyQuery *)malloc(sizeof(OdysseyQuery) * static_cast<size_t>(q_num));
        CHECK_ALLOC(queries, my_rank);

        for (int i = 0; i < q_num; i++)
        {
            queries[i].id = i;
            queries[i].initial_estimation = 0.0;
            queries[i].initial_pq_bsfs = nullptr;
            queries[i].paa = nullptr;
            queries[i].paaU = nullptr;
            queries[i].paaL = nullptr;

            queries[i].query = (ts_type *)malloc(sizeof(ts_type) * static_cast<size_t>(ts_len));
            CHECK_ALLOC(queries[i].query, my_rank);

            std::memcpy(queries[i].query, query_buf + static_cast<size_t>(i) * static_cast<size_t>(ts_len),
                        sizeof(ts_type) * static_cast<size_t>(ts_len));
        }
        return queries;
    }

    void free_queries(OdysseyQuery *queries, int q_num)
    {
        if (!queries) return;
        for (int i = 0; i < q_num; i++)
        {
            if (queries[i].paa)
            {
                free(queries[i].paa);
                queries[i].paa = nullptr;
            }
            if (queries[i].initial_pq_bsfs)
            {
                pqueue_bsf_destroy(queries[i].initial_pq_bsfs);
                queries[i].initial_pq_bsfs = nullptr;
            }
            free(queries[i].query);
            queries[i].query = nullptr;
            if (queries[i].paaU)
            {
                free(queries[i].paaU);
                queries[i].paaU = nullptr;
            }
            if (queries[i].paaL)
            {
                free(queries[i].paaL);
                queries[i].paaL = nullptr;
            }
        }
        free(queries);
    }

    double predict_exec_time(float bsf, const char *dataset_type)
    {
        if (strcmp(dataset_type, "default") == 0)
        {
            return static_cast<double>(bsf);
        }
        else if (strcmp(dataset_type, "seismic") == 0)
        {
            return static_cast<double>(bsf);
        }
        else if (strcmp(dataset_type, "astro") == 0)
        {
            return static_cast<double>(bsf);
        }
        else if (strcmp(dataset_type, "deep") == 0)
        {
            return static_cast<double>(bsf);
        }
        else if (strcmp(dataset_type, "sift") == 0)
        {
            return static_cast<double>(bsf);
        }
        else if (strcmp(dataset_type, "t2i") == 0)
        {
            return static_cast<double>(bsf);
        }
        else if (strcmp(dataset_type, "random") == 0)
        {
            return static_cast<double>(bsf);
        }
        else
        {
            printf("ERROR: Dataset type %s does not have an associated query time prediction function.\n", dataset_type);
            std::exit(EXIT_FAILURE);
        }

        return static_cast<double>(bsf);
    }

    static void rerank_pq_exact_l2(pqueue_bsf *pq, ts_type *query, float *rawfile, int dim)
    {
        if (!pq || !rawfile || !query)
            return;

        const int k = pq->k;
        std::vector<std::pair<float, file_position_type>> cand;
        cand.reserve(static_cast<size_t>(k));

        for (int i = 0; i < k; i++)
        {
            file_position_type pos = static_cast<file_position_type>(pq->position[i]);
            if (pos < 0)
                continue;

            const ts_type *base = rawfile + static_cast<size_t>(pos) * static_cast<size_t>(dim);
            float dist = 0.0f;
            for (int d = 0; d < dim; d++)
            {
                float diff = static_cast<float>(query[d] - base[d]);
                dist += diff * diff;
            }
            cand.emplace_back(dist, pos);
        }

        if (cand.empty())
            return;

        std::sort(cand.begin(), cand.end(), [](const auto &a, const auto &b) {
            if (a.first != b.first)
                return a.first < b.first;
            return a.second < b.second;
        });

        std::vector<std::pair<float, file_position_type>> uniq;
        uniq.reserve(cand.size());
        file_position_type last_pos = static_cast<file_position_type>(-1);
        for (const auto &p : cand)
        {
            if (p.second != last_pos)
            {
                uniq.push_back(p);
                last_pos = p.second;
            }
        }

        for (int i = 0; i < k; i++)
        {
            pq->knn[i] = FLT_MAX;
            pq->position[i] = -1;
        }

        int copy_count = std::min(k, static_cast<int>(uniq.size()));
        for (int i = 0; i < copy_count; i++)
        {
            pq->knn[i] = uniq[static_cast<size_t>(i)].first;
            pq->position[i] = static_cast<long>(uniq[static_cast<size_t>(i)].second);
        }

        if (copy_count > 0 && copy_count < k)
        {
            float pad_dist = pq->knn[copy_count - 1];
            long pad_pos = pq->position[copy_count - 1];
            for (int i = copy_count; i < k; i++)
            {
                pq->knn[i] = pad_dist;
                pq->position[i] = pad_pos;
            }
        }
    }

    int cmp_query(const void *a, const void *b)
    {
        const OdysseyQuery *query_a = static_cast<const OdysseyQuery *>(a);
        const OdysseyQuery *query_b = static_cast<const OdysseyQuery *>(b);

        if (query_a->initial_estimation < query_b->initial_estimation)
        {
            return -1;
        }
        else if (query_a->initial_estimation > query_b->initial_estimation)
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }

    void refine_topk_answer_inmemory_dtw(ts_type *ts, ts_type *paa, ts_type *paaU, ts_type *paaL, isax_index *index, int warp_window, pqueue_bsf *pq_bsf, float minimum_distance, int limit, float *rawfile, int merge_offset)
    {
        (void)paa;
        (void)limit;
        (void)merge_offset;

        int aggressive_check = index->settings->aggressive_check;
        pqueue_t *pq = pqueue_init(index->settings->root_nodes_size, cmp_pri, get_pri, set_pri, get_pos, set_pos);

        isax_node *current_root_node = index->first_node;
        while (current_root_node != nullptr)
        {
            if (current_root_node->isax_values == nullptr || current_root_node->isax_cardinalities == nullptr)
            {
                current_root_node = current_root_node->next;
                continue;
            }
            query_result *mindist_result = static_cast<query_result *>(std::malloc(sizeof(query_result)));
            if (!mindist_result) { pqueue_free(pq); return; }
            mindist_result->distance = minidist_paa_to_isax_DTW(paaU, paaL, current_root_node->isax_values,
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
        while ((n = static_cast<query_result *>(pqueue_pop(pq))) != nullptr)
        {
            if (!n->node) { std::free(n); continue; }
            if (n->node->isax_values == nullptr || n->node->isax_cardinalities == nullptr) { std::free(n); continue; }
            if (n->distance >= pq_bsf->knn[pq_bsf->k - 1] || n->distance > minimum_distance)
            {
                pqueue_insert(pq, n);
                break;
            }
            if (n->node->is_leaf)
            {
                if (!n->node->has_full_data_file &&
                    (n->node->leaf_size > index->settings->min_leaf_size) &&
                    n->node->buffer != nullptr)
                {
                    split_node(index, n->node);
                    if (!n->node->is_leaf)
                    {
                        pqueue_insert(pq, n);
                        continue;
                    }
                }
                calculate_node_DTWknn_inmemory(index, n->node, ts, warp_window, pq_bsf, rawfile);
                if (pq_bsf->knn[pq_bsf->k - 1] < FLT_MAX)
                {
                    pqueue_insert(pq, n);
                    break;
                }
            }
            else
            {
                if (n->node->left_child != nullptr && n->node->left_child->isax_values != nullptr && n->node->left_child->isax_cardinalities != nullptr)
                {
                    if (n->node->left_child->is_leaf && !n->node->left_child->has_partial_data_file && aggressive_check)
                        calculate_node_DTWknn_inmemory(index, n->node->left_child, ts, warp_window, pq_bsf, rawfile);
                    else
                    {
                        query_result *mindist_result = static_cast<query_result *>(std::malloc(sizeof(query_result)));
                        if (mindist_result)
                        {
                            mindist_result->distance = minidist_paa_to_isax_DTW(paaU, paaL, n->node->left_child->isax_values,
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
                }
                if (n->node->right_child != nullptr && n->node->right_child->isax_values != nullptr && n->node->right_child->isax_cardinalities != nullptr)
                {
                    if (n->node->right_child->is_leaf && !n->node->right_child->has_partial_data_file && aggressive_check)
                        calculate_node_DTWknn_inmemory(index, n->node->right_child, ts, warp_window, pq_bsf, rawfile);
                    else
                    {
                        query_result *mindist_result = static_cast<query_result *>(std::malloc(sizeof(query_result)));
                        if (mindist_result)
                        {
                            mindist_result->distance = minidist_paa_to_isax_DTW(paaU, paaL, n->node->right_child->isax_values,
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
            }
            std::free(n);
        }
        while ((n = static_cast<query_result *>(pqueue_pop(pq))) != nullptr)
            std::free(n);
        for (int i = 0; i < pq_bsf->k - 1; ++i)
            pq_bsf->knn[i] = pq_bsf->knn[pq_bsf->k - 1];
        pqueue_free(pq);
    }

    void odyssey_compute_query_envelopes_dtw(OdysseyQuery *query, isax_index *index, int warping_window)
    {
        const int ts_len = index->settings->timeseries_size;
        const int paa_segments = index->settings->paa_segments;
        const int ts_values_per_paa_segment = index->settings->ts_values_per_paa_segment;

        ts_type *upper_lemire = static_cast<ts_type *>(std::malloc(sizeof(ts_type) * static_cast<size_t>(ts_len)));
        ts_type *lower_lemire = static_cast<ts_type *>(std::malloc(sizeof(ts_type) * static_cast<size_t>(ts_len)));
        if (!upper_lemire || !lower_lemire)
        {
            if (upper_lemire) std::free(upper_lemire);
            if (lower_lemire) std::free(lower_lemire);
            return;
        }
        lower_upper_lemire(query->query, ts_len, warping_window, lower_lemire, upper_lemire);

        query->paaU = static_cast<ts_type *>(std::malloc(sizeof(ts_type) * static_cast<size_t>(paa_segments)));
        query->paaL = static_cast<ts_type *>(std::malloc(sizeof(ts_type) * static_cast<size_t>(paa_segments)));
        if (!query->paaU || !query->paaL)
        {
            std::free(upper_lemire);
            std::free(lower_lemire);
            if (query->paaU) { std::free(query->paaU); query->paaU = nullptr; }
            if (query->paaL) { std::free(query->paaL); query->paaL = nullptr; }
            return;
        }
        paa_from_ts(upper_lemire, query->paaU, paa_segments, ts_values_per_paa_segment);
        paa_from_ts(lower_lemire, query->paaL, paa_segments, ts_values_per_paa_segment);
        std::free(upper_lemire);
        std::free(lower_lemire);
    }

    void odyssey_preprocess_and_sort_queries(Odyssey *odyssey, OdysseyQuery *queries, int q_num, bool apply_sort)
    {
        const int MASTER = 0;

        isax_index *index = odyssey->index;
        const char *dataset_type = odyssey->dataset_type.c_str();
        int comm_sz = odyssey->comm_sz;
        int my_rank = odyssey->my_rank;
        int topk = odyssey->top_k;
        BsfSharingData &bsf_sharing_data = odyssey->bsf_sharing_data;
        bool verbose = odyssey->verbose;
        float *rawfile = odyssey->rawfile;

        for (int i = 0; i < q_num; i++)
        {
            queries[i].paa = (ts_type *)malloc(sizeof(ts_type) * static_cast<size_t>(index->settings->paa_segments));
            CHECK_ALLOC(queries[i].paa, my_rank);

            paa_from_ts(queries[i].query, queries[i].paa, index->settings->paa_segments, index->settings->ts_values_per_paa_segment);

            if (odyssey->distance_type == DistanceType::L2_SQUARED)
            {
                queries[i].initial_pq_bsfs = pqueue_bsf_init(topk);
                approximate_topk_inmemory(queries[i].query, queries[i].paa, index, queries[i].initial_pq_bsfs, rawfile);

                if (queries[i].initial_pq_bsfs->knn[topk - 1] == FLT_MAX)
                {
                    int min_checked_leaves = -1;
                    refine_topk_answer_inmemory(queries[i].query, queries[i].paa, index, queries[i].initial_pq_bsfs, FLT_MAX, min_checked_leaves, rawfile);
                }
            }
            else
            {
                
                odyssey_compute_query_envelopes_dtw(&queries[i], index, odyssey->warping_window);

                queries[i].initial_pq_bsfs = pqueue_bsf_init(topk);
                approximate_DTWtopk_inmemory(queries[i].query, queries[i].paa, index, odyssey->warping_window, queries[i].initial_pq_bsfs, rawfile);

                if (queries[i].initial_pq_bsfs->knn[topk - 1] == FLT_MAX)
                {
                    int min_checked_leaves = -1;
                    refine_topk_answer_inmemory_dtw(queries[i].query, queries[i].paa, queries[i].paaU, queries[i].paaL, index, odyssey->warping_window, queries[i].initial_pq_bsfs, FLT_MAX, min_checked_leaves, rawfile, odyssey->merge_offset);
                }
            }
        }

        if (!apply_sort)
        {
            return;
        }

        {
            float *all_bsfs = (float *)malloc(sizeof(float) * static_cast<size_t>(q_num) * static_cast<size_t>(topk));
            CHECK_ALLOC(all_bsfs, my_rank);

            file_position_type *all_pos = (file_position_type *)malloc(sizeof(file_position_type) * static_cast<size_t>(q_num) * static_cast<size_t>(topk));
            CHECK_ALLOC(all_pos, my_rank);

            for (int i = 0; i < q_num; i++)
            {
                for (int j = 0; j < topk; j++)
                {
                    all_bsfs[i * topk + j] = queries[i].initial_pq_bsfs->knn[j];
                    all_pos[i * topk + j] = queries[i].initial_pq_bsfs->position[j];
                }
            }

#if ODYSSEY_MPI
            if (comm_sz > 1 && static_cast<int>(bsf_sharing_data.communicators.size()) > my_rank)
            {
                MPI_Request local_send_requests[2];
                MPI_Ibcast(all_bsfs, q_num * topk, MPI_FLOAT, my_rank, bsf_sharing_data.communicators[my_rank], &local_send_requests[0]);
                MPI_Ibcast(all_pos, q_num * topk, MPI_UNSIGNED_LONG_LONG, my_rank, bsf_sharing_data.communicators[my_rank], &local_send_requests[1]);

                float *all_bsfs_recv = (float *)malloc(sizeof(float) * static_cast<size_t>(q_num) * static_cast<size_t>(topk));
                CHECK_ALLOC(all_bsfs_recv, my_rank);

                file_position_type *all_pos_recv = (file_position_type *)malloc(sizeof(file_position_type) * static_cast<size_t>(q_num) * static_cast<size_t>(topk));
                CHECK_ALLOC(all_pos_recv, my_rank);

                for (int i = 0; i < comm_sz; i++)
                {
                    if (i == my_rank)
                        continue;

                    MPI_Request recv_req;
                    MPI_Ibcast(all_bsfs_recv, q_num * topk, MPI_FLOAT, i, bsf_sharing_data.communicators[i], &recv_req);
                    MPI_Wait(&recv_req, MPI_STATUS_IGNORE);

                    MPI_Ibcast(all_pos_recv, q_num * topk, MPI_UNSIGNED_LONG_LONG, i, bsf_sharing_data.communicators[i], &recv_req);
                    MPI_Wait(&recv_req, MPI_STATUS_IGNORE);

                    for (int j = 0; j < q_num; j++)
                    {
                        for (int kk = 0; kk < topk; kk++)
                        {
                            float current_k_bsf = all_bsfs_recv[j * topk + kk];
                            file_position_type current_k_pos = all_pos_recv[j * topk + kk];

                            if (current_k_bsf < queries[j].initial_pq_bsfs->knn[topk - 1])
                            {
                                pqueue_bsf_insert(queries[j].initial_pq_bsfs, current_k_bsf, static_cast<long int>(current_k_pos), nullptr);
                            }
                        }
                    }
                }

                bsf_sharing_data.shared_bsfs[my_rank].bsf = queries[0].initial_pq_bsfs->knn[topk - 1];
                bsf_sharing_data.shared_bsfs[my_rank].position = queries[0].initial_pq_bsfs->position[topk - 1];
                bsf_sharing_data.shared_bsfs[my_rank].q_num = 0;

                MPI_Waitall(2, local_send_requests, MPI_STATUSES_IGNORE);

                free(all_bsfs_recv);
                free(all_pos_recv);

                for (int process_rank = 0; process_rank < comm_sz; process_rank++)
                {
                    if (process_rank == my_rank)
                        continue;
                    if (static_cast<int>(bsf_sharing_data.requests.size()) > process_rank)
                        MPI_Ibcast(&bsf_sharing_data.shared_bsfs[process_rank], 1, bsf_msg_type, process_rank, bsf_sharing_data.communicators[process_rank], &bsf_sharing_data.requests[process_rank]);
                }
            }
#endif
            free(all_bsfs);
            free(all_pos);
        }

        for (int i = 0; i < q_num; i++)
        {
            float k_th_bsf = queries[i].initial_pq_bsfs->knn[topk - 1];
            queries[i].initial_estimation = predict_exec_time(k_th_bsf, dataset_type);
        }

        qsort(queries, static_cast<size_t>(q_num), sizeof(OdysseyQuery), cmp_query);

#if ODYSSEY_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        if (my_rank == MASTER && verbose)
        {
            printf("[Node %d]: Initial BSFs exchanged. Proceeding to QA.\n", my_rank);
        }
    }

#if ODYSSEY_MPI
    constexpr int DISTRIBUTED_QUERIES_SEND_QUERY = 800;
    constexpr int DISTRIBUTED_QUERIES_REQUEST_QUERY = 801;
    constexpr int DYNAMIC_TERMINATION_MESSAGE = -1;
    constexpr bool ENABLE_PRINTS_WORKSTEALING = false;

    static void shuffle_int_array(int *arr, int n)
    {
        static std::mt19937 rng(static_cast<unsigned>(std::random_device{}()));
        for (int i = n - 1; i > 0; i--)
        {
            std::uniform_int_distribution<int> dist(0, i);
            int j = dist(rng);
            int t = arr[i];
            arr[i] = arr[j];
            arr[j] = t;
        }
    }

    void send_initial_queries_module_coordinator_async_chatzakis(int *q_loaded, int my_rank, int comm_sz,
                                                                 int distributed_queries_initial_burst,
                                                                 int **process_buffer_initial,
                                                                 int *rec_message, MPI_Request *request, MPI_Request *send_request,
                                                                 int q_num, int *termination_message_id,
                                                                 ReplicationData *replication_data)
    {
        (void)q_num;
        (void)termination_message_id;
        int coordinator_of_current_group_rank = rep_find_coordinator_node_rank(*replication_data, my_rank);
        int repgroup_nodes = rep_get_repgroup_nodes(*replication_data, my_rank);

        for (int i = 0; i < distributed_queries_initial_burst; i++)
        {
            for (int rank = coordinator_of_current_group_rank + 1; rank < (coordinator_of_current_group_rank + repgroup_nodes); rank++)
            {
                process_buffer_initial[rank][i] = (*q_loaded)++;
            }
        }

        for (int i = 0; i < distributed_queries_initial_burst; i++)
        {
            for (int rank = coordinator_of_current_group_rank + 1; rank < (coordinator_of_current_group_rank + repgroup_nodes); rank++)
            {
                MPI_Isend(&process_buffer_initial[rank][i], 1, MPI_INT, rank, DISTRIBUTED_QUERIES_SEND_QUERY, MPI_COMM_WORLD, &send_request[rank]);
            }
        }

        for (int rank = coordinator_of_current_group_rank + 1; rank < (coordinator_of_current_group_rank + repgroup_nodes); rank++)
        {
            MPI_Irecv(rec_message, 1, MPI_INT, rank, DISTRIBUTED_QUERIES_REQUEST_QUERY, MPI_COMM_WORLD, &request[rank]);
        }
    }

    int send_queries_module_coordinator_async_chatzakis(int *q_loaded, int q_num, int *process_buffer, MPI_Request *request, int *rec_message,
                                                        MPI_Request *send_request, int *termination_message_id,
                                                        ReplicationData *replication_data, int my_rank, int comm_sz,
                                                        bool verbose)
    {
        int ready;
        int coordinator_of_current_group_rank = rep_find_coordinator_node_rank(*replication_data, my_rank);
        int repgroup_nodes = rep_get_repgroup_nodes(*replication_data, my_rank);

        bool termination_message_sent = false;
        if (termination_message_sent)
        {
            return 0;
        }

        bool *node_requested = (bool *)malloc(sizeof(bool) * static_cast<size_t>(comm_sz));
        CHECK_ALLOC(node_requested, my_rank);

        for (int rank = coordinator_of_current_group_rank + 1; rank < (coordinator_of_current_group_rank + repgroup_nodes); rank++)
        {
            MPI_Test(&request[rank], &ready, MPI_STATUS_IGNORE);

            if (ready)
            {
                MPI_Wait(&send_request[rank], MPI_STATUS_IGNORE);

                process_buffer[rank] = (*q_loaded)++;
                node_requested[rank] = true;

                MPI_Irecv(rec_message, 1, MPI_INT, rank, DISTRIBUTED_QUERIES_REQUEST_QUERY, MPI_COMM_WORLD, &request[rank]);
            }
            else
            {
                node_requested[rank] = false;
            }
        }

        for (int rank = coordinator_of_current_group_rank + 1; rank < (coordinator_of_current_group_rank + repgroup_nodes); rank++)
        {
            if (node_requested[rank])
            {
                MPI_Isend(&process_buffer[rank], 1, MPI_INT, rank, DISTRIBUTED_QUERIES_SEND_QUERY, MPI_COMM_WORLD, &send_request[rank]);
            }
        }

        free(node_requested);

        if ((*q_loaded) >= q_num)
        {
            for (int rank = coordinator_of_current_group_rank + 1; rank < (coordinator_of_current_group_rank + repgroup_nodes); rank++)
            {
                if (verbose)
                    printf("[Node %d]: Sending to node %d a termination message\n", my_rank, rank);

                MPI_Isend(termination_message_id, 1, MPI_INT, rank, DISTRIBUTED_QUERIES_SEND_QUERY, MPI_COMM_WORLD, &send_request[rank]);
            }

            termination_message_sent = true;
            return 0;
        }

        return 1;
    }

    void odyssey_perform_workstealing(Odyssey *odyssey, OdysseyQuery *queries, NodeList nodelist,
                                      ws_func_type ws_func, double (*estimation_func)(double),
                                      query_result *results, std::vector<BsfMessage> *shared_bsf_results)
    {
        int my_rank = odyssey->my_rank;
        int comm_sz = odyssey->comm_sz;
        int query_threads = odyssey->query_threads;
        isax_index *index = odyssey->index;
        ReplicationData *replication_data = &odyssey->replication_data;
        WorkstealingData *workstealing_data = &odyssey->workstealing_data;
        bool verbose = odyssey->verbose;
        const float minimum_distance = FLT_MAX;
        int topk = odyssey->top_k;

        ts_type *paa = (ts_type *)malloc(sizeof(ts_type) * static_cast<size_t>(index->settings->paa_segments));
        CHECK_ALLOC(paa, my_rank);

        int term_message = DYNAMIC_TERMINATION_MESSAGE;
        int recv_message = 0;
        int ready = 0;
        std::vector<MPI_Request> send_request(static_cast<size_t>(comm_sz));
        std::vector<MPI_Request> recv_request(static_cast<size_t>(comm_sz));

        int repgroup_nodes = rep_get_repgroup_nodes(*replication_data, my_rank);
        int coordinator_of_current_group_rank = rep_find_coordinator_node_rank(*replication_data, my_rank);

        std::vector<int> nodes_of_group(static_cast<size_t>(repgroup_nodes));
        int in = 0;
        for (int rank = coordinator_of_current_group_rank; rank < (coordinator_of_current_group_rank + repgroup_nodes); rank++)
        {
            nodes_of_group[static_cast<size_t>(in++)] = rank;
        }

        shuffle_int_array(nodes_of_group.data(), repgroup_nodes);

        for (int i = 0; i < repgroup_nodes; i++)
        {
            int rank = nodes_of_group[static_cast<size_t>(i)];
            if (rank != my_rank)
            {
                if (ENABLE_PRINTS_WORKSTEALING && verbose)
                    printf("[WORKSTEALING - Node %d]: Sending to node %d that it has finished working.\n", my_rank, rank);

                MPI_Isend(&term_message, 1, MPI_INT, rank, WORKSTEALING_QUERY_ANSWERING_COMPLETION, MPI_COMM_WORLD, &send_request[static_cast<size_t>(rank)]);
            }
        }

        for (int i = 0; i < repgroup_nodes; i++)
        {
            int rank = nodes_of_group[static_cast<size_t>(i)];
            if (rank != my_rank)
            {
                MPI_Irecv(&recv_message, 1, MPI_INT, rank, WORKSTEALING_QUERY_ANSWERING_COMPLETION, MPI_COMM_WORLD, &recv_request[static_cast<size_t>(rank)]);
            }
        }

        std::vector<bool> working_nodes(static_cast<size_t>(repgroup_nodes), true);
        int finished_nodes = 1;

        while (finished_nodes < repgroup_nodes)
        {
            for (int i = 0; i < repgroup_nodes; i++)
            {
                int rank = nodes_of_group[static_cast<size_t>(i)];

                if (rank != my_rank && working_nodes[static_cast<size_t>(i)])
                {
                    MPI_Test(&recv_request[static_cast<size_t>(rank)], &ready, MPI_STATUS_IGNORE);

                    if (ready)
                    {
                        if (ENABLE_PRINTS_WORKSTEALING && verbose)
                            printf("[WORKSTEALING HELPER - Node %d]: Found that node %d has already finished his work.\n", my_rank, rank);
                        working_nodes[static_cast<size_t>(i)] = false;
                        finished_nodes++;
                    }
                    else
                    {
                        if (ENABLE_PRINTS_WORKSTEALING && verbose)
                            printf("[WORKSTEALING HELPER - Node %d] - Found that node %d is still working. Sending a steal request.\n", my_rank, rank);

                        MPI_Isend(&term_message, 1, MPI_INT, rank, WORKSTEALING_INFORM_AVAILABILITY, MPI_COMM_WORLD, &send_request[static_cast<size_t>(rank)]);

                        int data_size = 3 + workstealing_data->items_to_send;
                        std::vector<unsigned long long> datas(static_cast<size_t>(data_size));

                        MPI_Request data_req;
                        MPI_Irecv(datas.data(), data_size, MPI_UNSIGNED_LONG_LONG, rank, WORKSTEALING_DATA_SEND, MPI_COMM_WORLD, &data_req);

                        int req_ready = 0;
                        MPI_Test(&data_req, &req_ready, MPI_STATUS_IGNORE);
                        while (!req_ready)
                        {
                            MPI_Test(&data_req, &req_ready, MPI_STATUS_IGNORE);
                            MPI_Test(&recv_request[static_cast<size_t>(rank)], &ready, MPI_STATUS_IGNORE);
                            if (ready)
                            {
                                if (ENABLE_PRINTS_WORKSTEALING && verbose)
                                    printf("[WORKSTEALING - Node %d]: Found that node %d has already finished his work (While waiting to receive stolen work).\n", my_rank, rank);
                                working_nodes[static_cast<size_t>(i)] = false;
                                finished_nodes++;
                                req_ready = 1;
                            }
                        }

                        if (working_nodes[static_cast<size_t>(i)] == false)
                            break;

                        int query_num = static_cast<int>(datas[0]);
                        float bsf = static_cast<float>(datas[1]);
                        file_position_type bsf_position = static_cast<file_position_type>(datas[2]);

                        if (ENABLE_PRINTS_WORKSTEALING && verbose)
                        {
                            printf("[WORKSTEALING HELPER - Node %d]: Received message from node %d : [%d, %f, %llu, ", my_rank, rank, query_num, bsf, (unsigned long long)bsf_position);
                            for (int j = 0; j < workstealing_data->items_to_send; j++)
                                printf("%d ", static_cast<int>(datas[3 + static_cast<size_t>(j)]));
                            printf("]\n");
                        }

                        if (query_num >= 0)
                        {
                            std::vector<int> batches_to_create(static_cast<size_t>(workstealing_data->items_to_send));
                            for (int j = 0; j < workstealing_data->items_to_send; j++)
                                batches_to_create[static_cast<size_t>(j)] = static_cast<int>(datas[3 + static_cast<size_t>(j)]);

                            NodeList final_node_list = nodelist;
                            isax_node *original_lca_node = nullptr;

                            WsSearchFunctionParams ws_params;
                            ws_params.bsf_pos = bsf_position;
                            ws_params.ts = queries[query_num].query;
                            ws_params.paa = queries[query_num].paa;
                            ws_params.query_id = query_num;
                            ws_params.index = index;
                            ws_params.minimum_distance = minimum_distance;
                            ws_params.nodelist = &final_node_list;
                            ws_params.bsf = bsf;
                            ws_params.estimation_func = estimation_func;
                            ws_params.batch_ids = batches_to_create.data();
                            ws_params.shared_bsf_results = shared_bsf_results;
                            ws_params.comm_data = nullptr;
                            ws_params.lca_node = original_lca_node;
                            ws_params.k = topk;
                            ws_params.precomputed_bsfs = queries[query_num].initial_pq_bsfs;
                            ws_params.my_rank = my_rank;
                            ws_params.comm_sz = comm_sz;
                            ws_params.query_threads = query_threads;
                            ws_params.verbose = verbose;
                            ws_params.rawfile = odyssey->rawfile;
                            ws_params.replication_data = replication_data;
                            ws_params.output_file = odyssey->output_file;
                            ws_params.corr_threshold = odyssey->corr_threshold;
                            ws_params.bsf_sharing_data = &odyssey->bsf_sharing_data;
                            ws_params.workstealing_data = workstealing_data;
                            ws_params.pq_th_div_factor = odyssey->pq_th_div_factor;
                            ws_params.merge_offset = odyssey->merge_offset;
                            ws_params.query_counter = query_num;
                            ws_params.warp_window = (odyssey->distance_type == DistanceType::DTW) ? odyssey->warping_window : 0;
                            ws_params.paaU = (odyssey->distance_type == DistanceType::DTW) ? queries[query_num].paaU : nullptr;
                            ws_params.paaL = (odyssey->distance_type == DistanceType::DTW) ? queries[query_num].paaL : nullptr;

                            query_result result = ws_func(ws_params);

                            if (results[query_num].pq_bsf != nullptr)
                            {
                                for (int knni = 0; knni < result.pq_bsf->k; knni++)
                                {
                                    if (result.pq_bsf->knn[knni] < results[query_num].pq_bsf->knn[results[query_num].pq_bsf->k - 1])
                                    {
                                        pqueue_bsf_insert(results[query_num].pq_bsf, result.pq_bsf->knn[knni], static_cast<long int>(result.pq_bsf->position[knni]), nullptr);
                                    }
                                }
                                pqueue_bsf_destroy(result.pq_bsf);
                            }
                            else
                            {
                                results[query_num] = result;
                            }

                            if (ENABLE_PRINTS_WORKSTEALING && verbose)
                            {
                                printf("[Node: %d]: Stolen query: %d (id %d), result: %f\n", my_rank, query_num, queries[query_num].id,
                                       results[query_num].pq_bsf != nullptr ? results[query_num].pq_bsf->knn[0] : 0.0f);
                            }
                        }
                        else
                        {
                            working_nodes[static_cast<size_t>(i)] = false;
                            finished_nodes++;
                        }
                    }
                }
            }
        }

        free(paa);
    }
#else
    void send_initial_queries_module_coordinator_async_chatzakis(int *q_loaded, int my_rank, int comm_sz,
                                                                 int distributed_queries_initial_burst,
                                                                 int **process_buffer_initial,
                                                                 int *rec_message, MPI_Request *request, MPI_Request *send_request,
                                                                 int q_num, int *termination_message_id,
                                                                 ReplicationData *replication_data)
    {
        (void)q_loaded;
        (void)my_rank;
        (void)comm_sz;
        (void)distributed_queries_initial_burst;
        (void)process_buffer_initial;
        (void)rec_message;
        (void)request;
        (void)send_request;
        (void)q_num;
        (void)termination_message_id;
        (void)replication_data;
    }

    int send_queries_module_coordinator_async_chatzakis(int *q_loaded, int q_num, int *process_buffer, MPI_Request *request, int *rec_message,
                                                        MPI_Request *send_request, int *termination_message_id,
                                                        ReplicationData *replication_data, int my_rank, int comm_sz,
                                                        bool verbose)
    {
        (void)q_loaded;
        (void)q_num;
        (void)process_buffer;
        (void)request;
        (void)rec_message;
        (void)send_request;
        (void)termination_message_id;
        (void)replication_data;
        (void)my_rank;
        (void)comm_sz;
        (void)verbose;
        return 0;
    }

    void odyssey_perform_workstealing(Odyssey *odyssey, OdysseyQuery *queries, NodeList nodelist,
                                      ws_func_type ws_func, double (*estimation_func)(double),
                                      query_result *results, std::vector<BsfMessage> *shared_bsf_results)
    {
        (void)odyssey;
        (void)queries;
        (void)nodelist;
        (void)ws_func;
        (void)estimation_func;
        (void)results;
        (void)shared_bsf_results;
    }
#endif

    NodeList initialize_node_list(isax_index *index, int my_rank)
    {
        NodeList nodelist;
        const int capacity = (1 << index->settings->paa_segments);
        nodelist.nlist = (isax_node **)malloc(sizeof(isax_node *) * static_cast<size_t>(capacity));
        CHECK_ALLOC(nodelist.nlist, my_rank);
        if (!nodelist.nlist)
        {
            printf("Error: nodelist.nlist is NULL!\n");
            std::exit(EXIT_FAILURE);
        }
        nodelist.node_amount = 0;
        nodelist.data_amount = 0;
        nodelist.rawfile = nullptr;

        parallel_first_buffer_layer_ekosmas *fbl = (parallel_first_buffer_layer_ekosmas *)(index->fbl);
        if (!fbl)
        {
            return nodelist;
        }

        for (int j = 0; j < fbl->number_of_buffers; j++)
        {
            parallel_fbl_soft_buffer_ekosmas *current_fbl_node = &fbl->soft_buffers[j];
            if (!current_fbl_node->initialized)
            {
                continue;
            }

            nodelist.nlist[nodelist.node_amount] = current_fbl_node->node;
            if (!current_fbl_node->node)
            {
                printf("Error: node is NULL! (buffer %d)\n", j);
                std::fflush(stdout);
                std::exit(EXIT_FAILURE);
            }
            nodelist.node_amount++;
        }

        return nodelist;
    }

    int estimate_th(double x, double (*estimation_func)(double))
    {
        if (estimation_func == nullptr)
        {
            printf("estimate_th: PQ TH estimation function is NULL\n");
            std::exit(EXIT_FAILURE);
        }
        return static_cast<int>(estimation_func(x));
    }

    static int pq_comparator(const void *a, const void *b)
    {
        pqueue_t *p = *(pqueue_t **)a;
        pqueue_t *q = *(pqueue_t **)b;
        query_result *ra = static_cast<query_result *>(pqueue_peek(p));
        query_result *rb = static_cast<query_result *>(pqueue_peek(q));
        if (ra == nullptr)
            return 1;
        if (rb == nullptr)
            return -1;
        if (ra->distance < rb->distance)
            return -1;
        if (ra->distance > rb->distance)
            return 1;
        return 0;
    }

    void generate_pqs_of_rs_batch(isax_node *subtree_node, SubtreeBatch *batch, float bsf_distance, ts_type *paa, isax_index *index, int warp_window, ts_type *paaU, ts_type *paaL)
    {
        if (subtree_node == nullptr || subtree_node->isax_values == nullptr || subtree_node->isax_cardinalities == nullptr)
            return;

        float distance;
        if (warp_window > 0 && paaU != nullptr && paaL != nullptr)
            distance = minidist_paa_to_isax_DTW(paaU, paaL, subtree_node->isax_values, subtree_node->isax_cardinalities,
                                                 index->settings->sax_bit_cardinality, index->settings->sax_alphabet_cardinality,
                                                 index->settings->paa_segments, MINVAL, MAXVAL, index->settings->mindist_sqrt);
        else
            distance = minidist_paa_to_isax(paa, subtree_node->isax_values, subtree_node->isax_cardinalities,
                                            index->settings->sax_bit_cardinality, index->settings->sax_alphabet_cardinality,
                                            index->settings->paa_segments, MINVAL, MAXVAL, index->settings->mindist_sqrt);

        if (distance >= bsf_distance)
            return;

        if (subtree_node->is_leaf)
        {
            query_result *mindist_result = static_cast<query_result *>(std::malloc(sizeof(query_result)));
            if (mindist_result == nullptr)
            {
                printf("MEMORY ERROR: query_result allocation failed. Exiting...\n");
                std::exit(EXIT_FAILURE);
            }
            mindist_result->node = subtree_node;
            mindist_result->distance = distance;

            pthread_mutex_lock(&batch->pq_insert_lock);

            if (batch->pq_amount >= MAX_PQs_WORKSTEALING)
            {
                printf("MEMORY ERROR: Priority queues per batch exceeded the limit of %d. Increase the limit. Exiting...\n", MAX_PQs_WORKSTEALING);
                pthread_mutex_unlock(&batch->pq_insert_lock);
                std::free(mindist_result);
                std::exit(EXIT_FAILURE);
            }

            if (batch->pq[batch->pq_amount] == nullptr)
            {
                batch->pq[batch->pq_amount] = pqueue_init(static_cast<size_t>(batch->pq_th), cmp_pri, get_pri, set_pri, get_pos, set_pos);
                if (batch->pq[batch->pq_amount] == nullptr)
                {
                    printf("MEMORY ERROR: Priority queue allocation failed. Exiting...\n");
                    pthread_mutex_unlock(&batch->pq_insert_lock);
                    std::free(mindist_result);
                    std::exit(EXIT_FAILURE);
                }
                batch->pq[batch->pq_amount]->is_stolen = 0;
                batch->pq[batch->pq_amount]->is_processed = 0;
                batch->pq[batch->pq_amount]->batch_id = batch->id;
                batch->pq[batch->pq_amount]->starting_node = subtree_node;
                batch->pq[batch->pq_amount]->ending_node = nullptr;
                batch->pq[batch->pq_amount]->lca_node = nullptr;
            }

            if (pqueue_insert(batch->pq[batch->pq_amount], mindist_result) != 0)
            {
                printf("MEMORY ERROR: Priority queue insertion failed. Exiting...\n");
                pthread_mutex_unlock(&batch->pq_insert_lock);
                std::free(mindist_result);
                std::exit(EXIT_FAILURE);
            }

            batch->pq[batch->pq_amount]->ending_node = subtree_node;

            if (pqueue_size(batch->pq[batch->pq_amount]) >= static_cast<size_t>(batch->pq_th))
                batch->pq_amount++;

            pthread_mutex_unlock(&batch->pq_insert_lock);
        }
        else
        {
            generate_pqs_of_rs_batch(subtree_node->right_child, batch, bsf_distance, paa, index, warp_window, paaU, paaL);
            generate_pqs_of_rs_batch(subtree_node->left_child, batch, bsf_distance, paa, index, warp_window, paaU, paaL);
        }
    }

    void process_rs_batch(int batch_index, SubtreeBatch *batches, float bsf_distance, isax_index *index, ts_type *paa, int warp_window, ts_type *paaU, ts_type *paaL)
    {
        while (1)
        {
            int subtree_index = static_cast<int>(__atomic_fetch_add(reinterpret_cast<volatile int *>(&batches[batch_index].current_subtree_to_process), 1, __ATOMIC_SEQ_CST));

            if (subtree_index >= batches[batch_index].size)
                break;

            isax_node *subtree_node = (batches[batch_index].nodelist->nlist)[static_cast<size_t>(subtree_index + batches[batch_index].from)];
            generate_pqs_of_rs_batch(subtree_node, &batches[batch_index], bsf_distance, paa, index, warp_window, paaU, paaL);
        }
    }

    void gather_sort_pqueues(QaWorkerData *in_data)
    {
        int total_pqs = 0;

        for (int i = 0; i < in_data->total_batches; i++)
        {
            int pqs = in_data->batches[i].pq_amount + 1;
            for (int j = 0; j < pqs; j++)
            {
                if (in_data->batches[i].pq[j] != nullptr && pqueue_size(in_data->batches[i].pq[j]) > 1)
                    total_pqs++;
            }
        }

        *(in_data->final_pq_list_size) = total_pqs;
        *(in_data->final_pq_list) = static_cast<pqueue_t **>(std::malloc(sizeof(pqueue_t *) * static_cast<size_t>(total_pqs)));
        if (*(in_data->final_pq_list) == nullptr)
        {
            printf("MEMORY ERROR: final_pq_list allocation failed. Exiting...\n");
            std::exit(EXIT_FAILURE);
        }

        int curr_pq = 0;
        for (int i = 0; i < in_data->total_batches; i++)
        {
            int pqs = in_data->batches[i].pq_amount + 1;
            for (int j = 0; j < pqs; j++)
            {
                if (in_data->batches[i].pq[j] != nullptr)
                {
                    if (pqueue_size(in_data->batches[i].pq[j]) > 1)
                    {
                        in_data->batches[i].pq[j]->batch_id = i;
                        (*(in_data->final_pq_list))[curr_pq++] = in_data->batches[i].pq[j];
                    }
                    else
                    {
                        pqueue_free(in_data->batches[i].pq[j]);
                        in_data->batches[i].pq[j] = nullptr;
                    }
                }
            }
        }

        std::qsort(*(in_data->final_pq_list), static_cast<size_t>(total_pqs), sizeof(pqueue_t *), pq_comparator);

        for (int pq_num = 0; pq_num < total_pqs; pq_num++)
        {
            pqueue_t *p = (*(in_data->final_pq_list))[pq_num];
            SubtreeBatch *belonging_batch = &in_data->batches[p->batch_id];

            if (belonging_batch->min_pq_index > pq_num)
                belonging_batch->min_pq_index = pq_num;
            if (belonging_batch->max_pq_index < pq_num)
                belonging_batch->max_pq_index = pq_num;
        }

        *(in_data->priority_queues_filled) = 1;
    }

    int process_pq_of_batch_chatzakis(int current_pq_index, QaWorkerData *input_data)
    {
        pqueue_t **final_pq_list = *(input_data->final_pq_list);
        int final_pq_list_size = *(input_data->final_pq_list_size);
        if (current_pq_index >= final_pq_list_size)
            return 0;
        pqueue_t *pq = final_pq_list[current_pq_index];
        if (pq == nullptr || pq->is_stolen)
            return 0;

        pqueue_bsf *pq_bsf = input_data->bsf_result->pq_bsf;
        query_result *n = static_cast<query_result *>(pqueue_pop(pq));

        if (n == nullptr)
            return 0;

        float bsf = pq_bsf->knn[pq_bsf->k - 1];
        if (n->distance > bsf || n->distance > input_data->minimum_distance)
        {
            std::free(n);
            return 0;
        }

        if (n->node->is_leaf)
        {
            isax_index *index = input_data->index;
            ts_type *query = input_data->ts;
            ts_type *paa = input_data->paa;
            ts_type *paaU = input_data->paaU;
            ts_type *paaL = input_data->paaL;
            int warp_window = input_data->warp_window;
            float *rawfile = input_data->rawfile;
            isax_node *node = n->node;
            int my_rank = input_data->my_rank;
            ReplicationData *replication_data = input_data->replication_data;
            int merge_offset = input_data->merge_offset;
            const int ts_size = index->settings->timeseries_size;

            if (node->buffer != nullptr)
            {
                for (int i = 0; i < node->buffer->partial_buffer_size; i++)
                {
                    if (node->buffer->partial_position_buffer[i] == nullptr)
                        continue;

                    float distmin;
                    if (warp_window > 0 && paaU != nullptr && paaL != nullptr)
                        distmin = minidist_paa_to_isax_raw_DTW_SIMD(
                            paaU, paaL, node->buffer->partial_sax_buffer[i],
                            index->settings->max_sax_cardinalities,
                            index->settings->sax_bit_cardinality,
                            index->settings->sax_alphabet_cardinality,
                            index->settings->paa_segments, MINVAL, MAXVAL,
                            index->settings->mindist_sqrt);
                    else
                        distmin = minidist_paa_to_isax_rawa_SIMD(
                            paa, node->buffer->partial_sax_buffer[i],
                            index->settings->max_sax_cardinalities,
                            index->settings->sax_bit_cardinality,
                            index->settings->sax_alphabet_cardinality,
                            index->settings->paa_segments, MINVAL, MAXVAL,
                            index->settings->mindist_sqrt);

                    if (distmin <= bsf)
                    {
                        file_position_type pos = *node->buffer->partial_position_buffer[i];
                        float dist;
                        if (warp_window > 0)
                        {
                            float *cb = static_cast<float *>(std::calloc(static_cast<size_t>(ts_size), sizeof(float)));
                            if (cb)
                            {
                                dist = dtw(query, &rawfile[pos], cb, ts_size, warp_window, pq_bsf->knn[pq_bsf->k - 1]);
                                std::free(cb);
                            }
                            else
                                dist = FLT_MAX;
                        }
                        else
                            dist = ts_euclidean_distance_SIMD(
                                query, &rawfile[pos],
                                ts_size,
                                pq_bsf->knn[pq_bsf->k - 1]);

                        if (dist < pq_bsf->knn[pq_bsf->k - 1])
                        {
                            pthread_mutex_lock(input_data->bsf_lock);
                            file_position_type local_ts_index = pos / static_cast<file_position_type>(ts_size);
                            
                            pqueue_bsf_insert(pq_bsf, dist, static_cast<long int>(local_ts_index), node);
                            pthread_mutex_unlock(input_data->bsf_lock);

                            bsf_sharing_recv_bsf(*input_data->bsf_sharing_data, pq_bsf, input_data->workernumber, *input_data->shared_bsf_results, input_data->bsf_lock, my_rank, input_data->comm_sz, input_data->query_counter);
                            bsf_sharing_bcast_bsf(*input_data->bsf_sharing_data, pq_bsf, input_data->workernumber, my_rank, input_data->query_counter, input_data->replication_data, nullptr);
                        }
                    }
                }
            }
        }

        std::free(n);
        return 1;
    }

    void *workstealing_manager(void *rfdata)
    {
        WorkstealingThreadData *data = static_cast<WorkstealingThreadData *>(rfdata);
        volatile char *threads_finished = data->query_workers_finished;
        static bool first_time = true;

        ReplicationData *replication_data = data->replication_data;
        WorkstealingData *workstealing_data = data->workstealing_data;
        int my_rank = data->my_rank;
        int query_counter = data->query_counter;

        int recv_message = 0;
        int ready = 0;

        pqueue_t **final_pq_list = *(data->final_pq_list);
        int final_pq_list_size = *(data->final_pq_list_size);
        int local_pqs_stolen = 0;
        int local_ws_times = 0;

        int coordinator_of_current_group_rank = rep_find_coordinator_node_rank(*replication_data, my_rank);
        int repgroup_nodes = rep_get_repgroup_nodes(*replication_data, my_rank);

        if (first_time)
        {
            for (int rank = coordinator_of_current_group_rank; rank < (coordinator_of_current_group_rank + repgroup_nodes); rank++)
            {
                if (rank == my_rank)
                    continue;
                if (rank >= 0 && rank < static_cast<int>(workstealing_data->global_helper_requests.size()))
                {
                    MPI_Irecv(&recv_message, 1, MPI_INT, rank, WORKSTEALING_INFORM_AVAILABILITY, MPI_COMM_WORLD, &workstealing_data->global_helper_requests[static_cast<size_t>(rank)]);
                }
            }
            first_time = false;
        }

        while (*threads_finished == 0)
        {
            for (int rank = coordinator_of_current_group_rank; rank < (coordinator_of_current_group_rank + repgroup_nodes); rank++)
            {
                if (rank == my_rank)
                    continue;

                if (rank < 0 || rank >= static_cast<int>(workstealing_data->global_helper_requests.size()))
                    continue;

                MPI_Test(&workstealing_data->global_helper_requests[static_cast<size_t>(rank)], &ready, MPI_STATUS_IGNORE);

                if (ready)
                {
                    if (ENABLE_PRINTS_WORKSTEALING)
                    {
                        printf("[WORKSTEALING MAIN - Node %d] - Workstealing thread found that node %d can help!\n", my_rank, rank);
                    }

                    *(data->receiving_workstealing) = 1;

                    while (!(*data->priority_queues_filled))
                    {
                        
                    }

                    if (workstealing_data->ws_type == WorkstealingType::S_WS)
                    {
                        std::vector<int> batch_ids_to_send(static_cast<size_t>(workstealing_data->items_to_send));
                        bool are_batches_available = true;

                        for (int i = 0; i < workstealing_data->items_to_send; i++)
                        {
                            int selected_batch_id = -1;
                            int farest_min = 0;
                            for (int b = 0; b < data->batchlist->batch_amount; b++)
                            {
                                if (data->batchlist->batches[b].is_stolen || data->batchlist->batches[b].pq[0] == nullptr)
                                    continue;

                                int current_min = data->batchlist->batches[b].min_pq_index;
                                if (current_min >= farest_min)
                                {
                                    farest_min = current_min;
                                    selected_batch_id = b;
                                }
                            }

                            if (selected_batch_id == -1)
                            {
                                are_batches_available = false;
                                break;
                            }
                            batch_ids_to_send[static_cast<size_t>(i)] = selected_batch_id;
                            data->batchlist->batches[selected_batch_id].is_stolen = 1;
                        }

                        if (are_batches_available)
                        {
                            for (int id = 0; id < workstealing_data->items_to_send; id++)
                            {
                                int bid = batch_ids_to_send[static_cast<size_t>(id)];
                                for (int i = 0; i < data->batchlist->batches[bid].pq_amount + 1; i++)
                                {
                                    if (data->batchlist->batches[bid].pq[i] != nullptr && data->batchlist->batches[bid].pq[i]->size > 1)
                                    {
                                        data->batchlist->batches[bid].pq[i]->is_stolen = 1;
                                        local_pqs_stolen++;
                                        data->batchlist->batches[bid].pq[i]->is_processed = 1;
                                    }
                                }
                            }

                            local_ws_times++;

                            float bsf_val = 0.0f;
                            file_position_type pos_val = 0;
                            if (data->bsf_result->pq_bsf != nullptr)
                            {
                                bsf_val = data->bsf_result->pq_bsf->knn[data->bsf_result->pq_bsf->k - 1];
                                pos_val = data->bsf_result->pq_bsf->position[data->bsf_result->pq_bsf->k - 1];
                            }

                            std::vector<unsigned long long> datas(static_cast<size_t>(3 + workstealing_data->items_to_send));
                            datas[0] = static_cast<unsigned long long>(query_counter);
                            union { float f; unsigned long long u; } u;
                            u.f = bsf_val;
                            datas[1] = u.u;
                            datas[2] = static_cast<unsigned long long>(pos_val);
                            for (int i = 0; i < workstealing_data->items_to_send; i++)
                                datas[static_cast<size_t>(3 + i)] = static_cast<unsigned long long>(batch_ids_to_send[static_cast<size_t>(i)]);

                            if (ENABLE_PRINTS_WORKSTEALING)
                            {
                                printf("[WORKSTEALING MAIN NODE - Node %d]: Sending message to node %d :  [ %d, %f, %llu, ", my_rank, rank, (int)datas[0], bsf_val, (unsigned long long)datas[2]);
                                for (int t = 0; t < workstealing_data->items_to_send; t++)
                                    printf("%d ", (int)datas[static_cast<size_t>(t + 3)]);
                                printf("]\n");
                            }

                            MPI_Send(datas.data(), 3 + workstealing_data->items_to_send, MPI_UNSIGNED_LONG_LONG, rank, WORKSTEALING_DATA_SEND, MPI_COMM_WORLD);
                        }
                        else
                        {
                            std::vector<unsigned long long> datas(static_cast<size_t>(3 + workstealing_data->items_to_send));
                            datas[0] = static_cast<unsigned long long>(-1);
                            datas[1] = 0;
                            datas[2] = 0;
                            MPI_Send(datas.data(), 3 + workstealing_data->items_to_send, MPI_UNSIGNED_LONG_LONG, rank, WORKSTEALING_DATA_SEND, MPI_COMM_WORLD);
                        }
                    }
                    else if (workstealing_data->ws_type == WorkstealingType::P_WS)
                    {
                        final_pq_list_size = *(data->final_pq_list_size);
                        final_pq_list = *(data->final_pq_list);

                        std::vector<pqueue_t *> pq_candidates(static_cast<size_t>(workstealing_data->items_to_send));
                        int pq_candidates_size = 0;

                        if (ENABLE_PRINTS_WORKSTEALING)
                        {
                            printf("[WORKSTEALING MAIN NODE %d]: Total priority queue size: %d\n", my_rank, final_pq_list_size);
                        }

                        int segments = data->index->settings->paa_segments;
                        for (int pqindex = final_pq_list_size - local_pqs_stolen - 1; pqindex >= 0; pqindex--)
                        {
                            if (final_pq_list[pqindex]->is_stolen || final_pq_list[pqindex]->is_processed)
                                continue;

                            if (pq_candidates_size == workstealing_data->items_to_send)
                                break;

                            local_pqs_stolen++;
                            local_ws_times++;

                            final_pq_list[pqindex]->is_stolen = 1;
                            final_pq_list[pqindex]->is_processed = 1;

                            pq_candidates[static_cast<size_t>(pq_candidates_size)] = final_pq_list[pqindex];
                            pq_candidates_size++;
                        }

                        if (pq_candidates_size == 0)
                        {
                            int data_size = 2 + pq_candidates_size * (segments * 2);
                            std::vector<float> send_datas(static_cast<size_t>(data_size > 0 ? data_size : 1));
                            send_datas[0] = -1.0f;

                            if (ENABLE_PRINTS_WORKSTEALING)
                            {
                                printf("[WORKSTEALING MAIN NODE %d]: Nothing to send to node %d\n", my_rank, rank);
                            }
                            MPI_Send(send_datas.data(), data_size, MPI_FLOAT, rank, WORKSTEALING_DATA_SEND, MPI_COMM_WORLD);
                        }
                        else
                        {
                            int data_size = 2 + 1 + workstealing_data->items_to_send * (segments * 2);
                            std::vector<float> send_datas(static_cast<size_t>(data_size));

                            send_datas[0] = static_cast<float>(query_counter);
                            float bsf_for_pws = 0.0f;
                            if (data->bsf_result->pq_bsf != nullptr)
                                bsf_for_pws = data->bsf_result->pq_bsf->knn[data->bsf_result->pq_bsf->k - 1];
                            send_datas[1] = bsf_for_pws;
                            send_datas[2] = static_cast<float>(pq_candidates_size);

                            isax_index *data_index = data->index;

                            for (int pqindex = 0; pqindex < pq_candidates_size; pqindex++)
                            {
                                pqueue_t *pq = pq_candidates[static_cast<size_t>(pqindex)];

                                isax_node *starting_node = pq->starting_node;
                                isax_node *ending_node = pq->ending_node;
                                if (starting_node == nullptr || ending_node == nullptr)
                                {
                                    printf("ERROR: Starting or ending node is NULL\n");
                                    std::exit(EXIT_FAILURE);
                                }

                                isax_node *lca_node = ws_compute_lca(data_index, starting_node, ending_node);
                                if (lca_node == nullptr)
                                {
                                    printf("ERROR: LCA node is NULL\n");
                                    std::exit(EXIT_FAILURE);
                                }

                                for (int seg = 0; seg < segments; seg++)
                                {
                                    send_datas[static_cast<size_t>(3 + (pqindex * (segments * 2)) + seg)] = static_cast<float>(lca_node->isax_values[seg]);
                                    send_datas[static_cast<size_t>(3 + (pqindex * (segments * 2)) + (segments + seg))] = static_cast<float>(lca_node->isax_cardinalities[seg]);
                                }
                            }

                            if (ENABLE_PRINTS_WORKSTEALING)
                            {
                                printf("[WORKSTEALING MAIN NODE %d]: Sendingg %d priority queues to node %d\n", my_rank, pq_candidates_size, rank);
                            }

                            MPI_Send(send_datas.data(), data_size, MPI_FLOAT, rank, WORKSTEALING_DATA_SEND, MPI_COMM_WORLD);
                        }
                    }

                    MPI_Irecv(&recv_message, 1, MPI_INT, rank, WORKSTEALING_INFORM_AVAILABILITY, MPI_COMM_WORLD, &workstealing_data->global_helper_requests[static_cast<size_t>(rank)]);
                }
            }
        }

        *(data->pqs_stolen) += local_pqs_stolen;
        *(data->workstealing_times) += local_ws_times;

        pthread_exit(nullptr);
        return nullptr;
    }

    void *dynamic_query_scheduler(void *rfdata)
    {
        CoordinatorData *data = static_cast<CoordinatorData *>(rfdata);

        CommunicationModuleData *comm_data = data->comm_data;
        volatile char *threads_finished = data->threads_finished;

        while (!(*threads_finished))
        {
            if (comm_data != nullptr)
            {
                call_module(comm_data);
            }
        }

        pthread_exit(nullptr);
        return nullptr;
    }

    void *qa_exact_search_odyssey_worker(void *rfdata)
    {
        QaWorkerData *in_data = static_cast<QaWorkerData *>(rfdata);
        CommunicationModuleData *comm_data = in_data->comm_data;
        int my_rank = in_data->my_rank;
        int comm_sz = in_data->comm_sz;
        int query_counter = in_data->query_counter;
        BsfSharingData *bsf_sharing_data = in_data->bsf_sharing_data;

        int k = in_data->bsf_result->pq_bsf->k;

        for (;;)
        {
            int current_batch_index = __sync_fetch_and_add(in_data->batch_counter, 1);
            if (current_batch_index >= in_data->total_batches)
                break;

            if (comm_data != nullptr && in_data->workernumber == 0 && comm_data->mode == DynamicSchedulingMode::PERIODIC_CHECK)
            {
                call_module(comm_data);
            }

            float bsf = in_data->bsf_result->pq_bsf->knn[k - 1];
            process_rs_batch(current_batch_index, in_data->batches, bsf, in_data->index, in_data->paa, in_data->warp_window, in_data->paaU, in_data->paaL);
            in_data->batches[current_batch_index].processed_phase_1 = 1;
            bsf_sharing_recv_bsf(*bsf_sharing_data, in_data->bsf_result->pq_bsf, in_data->workernumber, *in_data->shared_bsf_results, in_data->bsf_lock, my_rank, comm_sz, query_counter);
        }

        for (int i = 0; i < in_data->total_batches; i++)
        {
            if (!in_data->batches[i].processed_phase_1 && !in_data->batches[i].is_getting_help_phase1)
            {
                in_data->batches[i].is_getting_help_phase1 = 1;
                int k_help = in_data->bsf_result->pq_bsf->k;
                process_rs_batch(i, in_data->batches, in_data->bsf_result->pq_bsf->knn[k_help - 1], in_data->index, in_data->paa, in_data->warp_window, in_data->paaU, in_data->paaL);
                bsf_sharing_recv_bsf(*bsf_sharing_data, in_data->bsf_result->pq_bsf, in_data->workernumber, *in_data->shared_bsf_results, in_data->bsf_lock, my_rank, comm_sz, query_counter);
            }
        }

        pthread_barrier_wait(in_data->sync_barrier);

        if (in_data->workernumber == 0)
        {
            gather_sort_pqueues(in_data);
        }

        pthread_barrier_wait(in_data->sync_barrier);

        while (1)
        {
            int current_pq_index = __sync_fetch_and_add(in_data->pq_counter, 1);
            if (current_pq_index >= *(in_data->final_pq_list_size))
                break;

            pqueue_t *pq = (*(in_data->final_pq_list))[current_pq_index];
            if (pq != nullptr && pq->is_stolen)
                continue;

            while (process_pq_of_batch_chatzakis(current_pq_index, in_data))
            {
                bsf_sharing_recv_bsf(*bsf_sharing_data, in_data->bsf_result->pq_bsf, in_data->workernumber, *in_data->shared_bsf_results, in_data->bsf_lock, my_rank, comm_sz, query_counter);
                bsf_sharing_bcast_bsf(*bsf_sharing_data, in_data->bsf_result->pq_bsf, in_data->workernumber, my_rank, query_counter, in_data->replication_data, nullptr);

                if (comm_data != nullptr && in_data->workernumber == 0 && comm_data->mode == DynamicSchedulingMode::PERIODIC_CHECK)
                {
                    call_module(comm_data);
                }
            }

            if (pq != nullptr)
                pq->is_processed = 1;
        }

        pthread_barrier_wait(in_data->sync_barrier);
        pthread_exit(nullptr);
    }

    query_result qa_exact_search_odyssey_knn(SearchFunctionParams args)
    {
        int query_id = args.query_id;
        ts_type *ts = args.ts;
        ts_type *paa = args.paa;
        isax_index *index = args.index;
        NodeList *nodelist = args.nodelist;
        float minimum_distance = args.minimum_distance;
        double (*estimation_func)(double) = args.estimation_func;
        CommunicationModuleData *comm_data = args.comm_data;
        int k = args.k;
        pqueue_bsf *precomputed_bsfs = args.precomputed_bsfs;

        float *rawfile = args.rawfile;
        int merge_offset = args.merge_offset;
        int my_rank = args.my_rank;
        int query_threads = args.query_threads;
        int pq_th_div_factor = args.pq_th_div_factor;
        int comm_sz = args.comm_sz;
        BsfSharingData *bsf_sharing_data = args.bsf_sharing_data;
        WorkstealingData *workstealing_data = args.workstealing_data;
        ReplicationData *replication_data = args.replication_data;

        args.query_counter = query_id;
        int query_counter = args.query_counter;

        std::vector<BsfMessage> *shared_bsf_results = args.shared_bsf_results;

        query_result bsf_result;
        bsf_result.distance = FLT_MAX;
        bsf_result.node = nullptr;
        bsf_result.pqueue_position = 0;
        bsf_result.pq_bsf = nullptr;
        bsf_result.total_time = 0.0f;

        if (!precomputed_bsfs)
        {
            bsf_result.pq_bsf = pqueue_bsf_init(k);
            if (args.warp_window > 0 && args.paaU != nullptr && args.paaL != nullptr)
            {
                approximate_DTWtopk_inmemory(ts, paa, index, args.warp_window, bsf_result.pq_bsf, rawfile);
                if (bsf_result.pq_bsf->knn[k - 1] == FLT_MAX)
                {
                    int min_checked_leaves = -1;
                    refine_topk_answer_inmemory_dtw(ts, paa, args.paaU, args.paaL, index, args.warp_window, bsf_result.pq_bsf, minimum_distance, min_checked_leaves, rawfile, args.merge_offset);
                }
            }
            else
            {
                approximate_topk_inmemory(ts, paa, index, bsf_result.pq_bsf, rawfile);
                int min_checked_leaves = -1;

                refine_topk_answer_inmemory(ts, paa, index, bsf_result.pq_bsf,
                                            minimum_distance, min_checked_leaves, rawfile);
            }
        }
        else
        {
            bsf_result.pq_bsf = pqueue_bsf_init_from_src(k, precomputed_bsfs);
        }

        if (rawfile != nullptr && args.warp_window == 0)
        {
            rerank_pq_exact_l2(bsf_result.pq_bsf, ts, rawfile, index->settings->timeseries_size);
        }   

        if (bsf_result.pq_bsf->knn[k - 1] == 0.0f)
        {
            return bsf_result;
        }

        int median = estimate_th(bsf_result.pq_bsf->knn[k - 1], estimation_func);
        int query_th = median / pq_th_div_factor;
        if (query_th <= 0)
            query_th = 1;

        BatchList *batchlist = create_subtree_batches(nodelist, query_threads, query_th);

        std::vector<pthread_t> threadid(static_cast<size_t>(query_threads));
        pthread_t workstealing_thread = 0;
        pthread_t coordinator_thread_id = 0;

        pthread_barrier_t sync_barrier;
        pthread_barrier_init(&sync_barrier, nullptr, query_threads);

        pthread_mutex_t bsf_lock;
        pthread_mutex_t distances_lock;
        pthread_mutex_init(&bsf_lock, nullptr);
        pthread_mutex_init(&distances_lock, nullptr);

        volatile int batch_counter = 0;
        volatile int pq_counter = 0;
        volatile char query_workers_finished = 0;
        volatile char receiving_workstealing = 0;
        volatile char priority_queues_filled = 0;

        SubtreeBatch *batches = batchlist->batches;
        int total_batches = batchlist->batch_amount;

        pqueue_t **all_pqs = nullptr;
        int all_pqs_size = 0;
        int stolen_pqs = 0;
        int processed_pqs = 0;

        bsf_sharing_update_from_bookkeeping(*bsf_sharing_data, bsf_result.pq_bsf, *shared_bsf_results, query_counter);

        std::vector<QaWorkerData> workerdata(static_cast<size_t>(query_threads));
        for (int i = 0; i < query_threads; i++)
        {
            workerdata[static_cast<size_t>(i)].workernumber = i;
            workerdata[static_cast<size_t>(i)].ts = ts;
            workerdata[static_cast<size_t>(i)].paa = paa;
            workerdata[static_cast<size_t>(i)].minimum_distance = minimum_distance;
            workerdata[static_cast<size_t>(i)].bsf_result = &bsf_result;
            workerdata[static_cast<size_t>(i)].batches = batches;
            workerdata[static_cast<size_t>(i)].total_batches = total_batches;
            workerdata[static_cast<size_t>(i)].batch_counter = &batch_counter;
            workerdata[static_cast<size_t>(i)].sync_barrier = &sync_barrier;
            workerdata[static_cast<size_t>(i)].bsf_lock = &bsf_lock;
            workerdata[static_cast<size_t>(i)].receiving_workstealing = &receiving_workstealing;
            workerdata[static_cast<size_t>(i)].index = index;
            workerdata[static_cast<size_t>(i)].pq_counter = &pq_counter;
            workerdata[static_cast<size_t>(i)].final_pq_list = &all_pqs;
            workerdata[static_cast<size_t>(i)].final_pq_list_size = &all_pqs_size;
            workerdata[static_cast<size_t>(i)].distances_lock = &distances_lock;
            workerdata[static_cast<size_t>(i)].priority_queues_filled = &priority_queues_filled;
            workerdata[static_cast<size_t>(i)].pqs_stolen = &stolen_pqs;
            workerdata[static_cast<size_t>(i)].processed_pqs = &processed_pqs;
            workerdata[static_cast<size_t>(i)].comm_data = comm_data;
            workerdata[static_cast<size_t>(i)].shared_bsf_results = shared_bsf_results;
            workerdata[static_cast<size_t>(i)].query_counter = query_counter;
            workerdata[static_cast<size_t>(i)].replication_data = replication_data;
            workerdata[static_cast<size_t>(i)].workstealing_data = workstealing_data;
            workerdata[static_cast<size_t>(i)].query_threads = query_threads;
            workerdata[static_cast<size_t>(i)].comm_sz = comm_sz;
            workerdata[static_cast<size_t>(i)].my_rank = my_rank;
            workerdata[static_cast<size_t>(i)].merge_offset = merge_offset;
            workerdata[static_cast<size_t>(i)].bsf_sharing_data = bsf_sharing_data;
            workerdata[static_cast<size_t>(i)].rawfile = rawfile;
            workerdata[static_cast<size_t>(i)].pq_th_div_factor = pq_th_div_factor;
            workerdata[static_cast<size_t>(i)].corr_threshold = args.corr_threshold;
            workerdata[static_cast<size_t>(i)].verbose = args.verbose;
            workerdata[static_cast<size_t>(i)].output_file = args.output_file;
            workerdata[static_cast<size_t>(i)].warp_window = args.warp_window;
            workerdata[static_cast<size_t>(i)].paaU = args.paaU;
            workerdata[static_cast<size_t>(i)].paaL = args.paaL;

            if (pthread_create(&threadid[static_cast<size_t>(i)], nullptr, qa_exact_search_odyssey_worker, (void *)&workerdata[static_cast<size_t>(i)]) != 0)
            {
                printf("[Node %d]: Error creating thread %d for qa_exact_search_odyssey_worker\n", my_rank, i);
                std::exit(EXIT_FAILURE);
            }
        }

        int ws_pqs_stolen = 0;
        int ws_times = 0;
        WorkstealingThreadData ws_th_data;
        if (comm_sz > 1 && static_cast<int>(workstealing_data->ws_type) > 0)
        {
            ws_th_data.query_workers_finished = &query_workers_finished;
            ws_th_data.receiving_workstealing = &receiving_workstealing;
            ws_th_data.priority_queues_filled = &priority_queues_filled;
            ws_th_data.bsf_lock = &bsf_lock;
            ws_th_data.bsf_result = &bsf_result;
            ws_th_data.batchlist = batchlist;
            ws_th_data.pqs_stolen = &ws_pqs_stolen;
            ws_th_data.workstealing_times = &ws_times;
            ws_th_data.final_pq_list = &all_pqs;
            ws_th_data.final_pq_list_size = &all_pqs_size;
            ws_th_data.index = index;
            ws_th_data.my_rank = my_rank;
            ws_th_data.comm_sz = comm_sz;
            ws_th_data.rawfile = rawfile;
            ws_th_data.query_counter = query_counter;
            ws_th_data.merge_offset = merge_offset;
            ws_th_data.replication_data = replication_data;
            ws_th_data.workstealing_data = workstealing_data;
            ws_th_data.bsf_sharing_data = bsf_sharing_data;
            ws_th_data.query_threads = query_threads;
            ws_th_data.pq_th_div_factor = pq_th_div_factor;
            ws_th_data.corr_threshold = args.corr_threshold;
            ws_th_data.verbose = args.verbose;
            ws_th_data.output_file = args.output_file;

            if (pthread_create(&workstealing_thread, nullptr, workstealing_manager, (void *)&ws_th_data) != 0)
            {
                printf("[Node %d]: Error creating workstealing thread\n", my_rank);
                std::exit(EXIT_FAILURE);
            }
        }

        if (comm_data != nullptr && comm_data->mode == DynamicSchedulingMode::STANDALONE_THREAD && comm_sz > 1)
        {
            CoordinatorData c_data;
            c_data.comm_data = comm_data;
            c_data.threads_finished = &query_workers_finished;
            if (pthread_create(&coordinator_thread_id, nullptr, dynamic_query_scheduler, (void *)&c_data) != 0)
            {
                printf("[Node %d]: Error creating scheduler coordinator thread\n", my_rank);
                std::exit(EXIT_FAILURE);
            }
        }

        for (int i = 0; i < query_threads; i++)
        {
            if (pthread_join(threadid[static_cast<size_t>(i)], nullptr) != 0)
            {
                printf("[Node %d]: Error joining qa_exact_search_odyssey_worker %d\n", my_rank, i);
                std::exit(EXIT_FAILURE);
            }
        }

        query_workers_finished = 1;

        if (comm_sz > 1 && static_cast<int>(workstealing_data->ws_type) > 0)
        {
            if (pthread_join(workstealing_thread, nullptr) != 0)
            {
                printf("[Node %d]: Error joining workstealing thread\n", my_rank);
                std::exit(EXIT_FAILURE);
            }
        }

        if (comm_data != nullptr && comm_data->mode == DynamicSchedulingMode::STANDALONE_THREAD && comm_sz > 1)
        {
            if (pthread_join(coordinator_thread_id, nullptr) != 0)
            {
                printf("[Node %d]: Error joining coordinator thread\n", my_rank);
                std::exit(EXIT_FAILURE);
            }
        }

        pthread_barrier_destroy(&sync_barrier);
        pthread_mutex_destroy(&bsf_lock);
        pthread_mutex_destroy(&distances_lock);

        if (all_pqs != nullptr)
        {
            for (int i = 0; i < all_pqs_size; i++)
            {
                pqueue_free(all_pqs[i]);
                all_pqs[i] = nullptr;
            }
            free(all_pqs);
        }
        free(batchlist->batches);
        free(batchlist);

        return bsf_result;
    }

    query_result qa_exact_search_odyssey_knn_workstealing(WsSearchFunctionParams ws_args)
    {
        int query_id = ws_args.query_id;
        ts_type *ts = ws_args.ts;
        ts_type *paa = ws_args.paa;
        isax_index *index = ws_args.index;
        NodeList *nodelist = ws_args.nodelist;
        float minimum_distance = ws_args.minimum_distance;
        double (*estimation_func)(double) = ws_args.estimation_func;
        float bsf = ws_args.bsf;
        file_position_type bsf_pos = ws_args.bsf_pos;
        int *batches_to_create = ws_args.batch_ids;
        std::vector<BsfMessage> *shared_bsf_results = ws_args.shared_bsf_results;
        int top_k = ws_args.k;
        int merge_offset = ws_args.merge_offset;
        int my_rank = ws_args.my_rank;
        int query_threads = ws_args.query_threads;
        int pq_th_div_factor = ws_args.pq_th_div_factor;
        int comm_sz = ws_args.comm_sz;
        WorkstealingData *workstealing_data = ws_args.workstealing_data;
        ReplicationData *replication_data = ws_args.replication_data;
        float *rawfile = ws_args.rawfile;
        BsfSharingData *bsf_sharing_data = ws_args.bsf_sharing_data;
        ws_args.query_counter = query_id;
        int query_counter = ws_args.query_counter;

        query_result bsf_result;
        bsf_result.distance = FLT_MAX;
        bsf_result.node = nullptr;
        bsf_result.pqueue_position = 0;
        bsf_result.pq_bsf = pqueue_bsf_init_from_val(top_k, bsf, bsf_pos);
        bsf_result.total_time = 0.0f;

        int median = estimate_th(bsf, estimation_func);
        int query_th = median / pq_th_div_factor;
        if (query_th <= 0)
            query_th = 1;

        BatchList *batchlist = create_subtree_batches(nodelist, query_threads, query_th);

        std::vector<pthread_t> threadid(static_cast<size_t>(query_threads));
        pthread_barrier_t sync_barrier;
        pthread_barrier_init(&sync_barrier, nullptr, query_threads);
        pthread_mutex_t bsf_lock;
        pthread_mutex_t distances_lock;
        pthread_mutex_init(&bsf_lock, nullptr);
        pthread_mutex_init(&distances_lock, nullptr);

        volatile int batch_counter = 0;
        volatile int pq_counter = 0;
        volatile char query_workers_finished = 0;
        volatile char receiving_workstealing = 1;
        volatile char priority_queues_filled = 0;

        SubtreeBatch *batches = batchlist->batches;
        int total_batches = batchlist->batch_amount;

        SubtreeBatch *ws_batches = nullptr;
        if (workstealing_data->ws_type == WorkstealingType::S_WS)
        {
            ws_batches = static_cast<SubtreeBatch *>(std::malloc(sizeof(SubtreeBatch) * static_cast<size_t>(workstealing_data->items_to_send)));
            if (ws_batches == nullptr)
            {
                free(batchlist->batches);
                free(batchlist);
                pqueue_bsf_destroy(bsf_result.pq_bsf);
                bsf_result.pq_bsf = nullptr;
                return bsf_result;
            }
            total_batches = workstealing_data->items_to_send;
            for (int i = 0; i < workstealing_data->items_to_send; i++)
            {
                int batch_id = batches_to_create[i];
                ws_batches[i] = batchlist->batches[batch_id];
            }
            batches = ws_batches;
        }

        pqueue_t **all_pqs = nullptr;
        int all_pqs_size = 0;
        int pqs_processed = 0;
        int pqs_stolen = 0;

        std::vector<QaWorkerData> workerdata(static_cast<size_t>(query_threads));
        for (int i = 0; i < query_threads; i++)
        {
            workerdata[static_cast<size_t>(i)].workernumber = i;
            workerdata[static_cast<size_t>(i)].ts = ts;
            workerdata[static_cast<size_t>(i)].paa = paa;
            workerdata[static_cast<size_t>(i)].minimum_distance = minimum_distance;
            workerdata[static_cast<size_t>(i)].bsf_result = &bsf_result;
            workerdata[static_cast<size_t>(i)].batches = batches;
            workerdata[static_cast<size_t>(i)].total_batches = total_batches;
            workerdata[static_cast<size_t>(i)].batch_counter = &batch_counter;
            workerdata[static_cast<size_t>(i)].sync_barrier = &sync_barrier;
            workerdata[static_cast<size_t>(i)].bsf_lock = &bsf_lock;
            workerdata[static_cast<size_t>(i)].receiving_workstealing = &receiving_workstealing;
            workerdata[static_cast<size_t>(i)].index = index;
            workerdata[static_cast<size_t>(i)].pq_counter = &pq_counter;
            workerdata[static_cast<size_t>(i)].final_pq_list = &all_pqs;
            workerdata[static_cast<size_t>(i)].final_pq_list_size = &all_pqs_size;
            workerdata[static_cast<size_t>(i)].distances_lock = &distances_lock;
            workerdata[static_cast<size_t>(i)].priority_queues_filled = &priority_queues_filled;
            workerdata[static_cast<size_t>(i)].pqs_stolen = &pqs_stolen;
            workerdata[static_cast<size_t>(i)].processed_pqs = &pqs_processed;
            workerdata[static_cast<size_t>(i)].shared_bsf_results = shared_bsf_results;
            workerdata[static_cast<size_t>(i)].comm_data = nullptr;
            workerdata[static_cast<size_t>(i)].query_counter = query_counter;
            workerdata[static_cast<size_t>(i)].replication_data = replication_data;
            workerdata[static_cast<size_t>(i)].workstealing_data = workstealing_data;
            workerdata[static_cast<size_t>(i)].query_threads = query_threads;
            workerdata[static_cast<size_t>(i)].comm_sz = comm_sz;
            workerdata[static_cast<size_t>(i)].my_rank = my_rank;
            workerdata[static_cast<size_t>(i)].merge_offset = merge_offset;
            workerdata[static_cast<size_t>(i)].corr_threshold = ws_args.corr_threshold;
            workerdata[static_cast<size_t>(i)].rawfile = rawfile;
            workerdata[static_cast<size_t>(i)].bsf_sharing_data = bsf_sharing_data;
            workerdata[static_cast<size_t>(i)].pq_th_div_factor = pq_th_div_factor;
            workerdata[static_cast<size_t>(i)].verbose = ws_args.verbose;
            workerdata[static_cast<size_t>(i)].output_file = ws_args.output_file;
            workerdata[static_cast<size_t>(i)].warp_window = ws_args.warp_window;
            workerdata[static_cast<size_t>(i)].paaU = ws_args.paaU;
            workerdata[static_cast<size_t>(i)].paaL = ws_args.paaL;

            if (pthread_create(&threadid[static_cast<size_t>(i)], nullptr, qa_exact_search_odyssey_worker, (void *)&workerdata[static_cast<size_t>(i)]) != 0)
            {
                printf("[Node %d]: Error creating thread %d for qa_exact_search_odyssey_knn_worker (workstealing)\n", my_rank, i);
                std::exit(EXIT_FAILURE);
            }
        }

        for (int i = 0; i < query_threads; i++)
        {
            if (pthread_join(threadid[static_cast<size_t>(i)], nullptr) != 0)
            {
                printf("[Node %d]: Error joining qa_exact_search_odyssey_knn_worker %d (workstealing)\n", my_rank, i);
                std::exit(EXIT_FAILURE);
            }
        }

        query_workers_finished = 1;

        pthread_barrier_destroy(&sync_barrier);
        pthread_mutex_destroy(&bsf_lock);
        pthread_mutex_destroy(&distances_lock);

        if (all_pqs != nullptr)
        {
            for (int i = 0; i < all_pqs_size; i++)
            {
                pqueue_free(all_pqs[i]);
                all_pqs[i] = nullptr;
            }
            free(all_pqs);
        }

        if (ws_batches != nullptr)
            free(ws_batches);
        free(batchlist->batches);
        free(batchlist);

        (void)pqs_stolen;
        (void)pqs_processed;
        return bsf_result;
    }

    void Odyssey::initializeMPI(int argc, char **argv)
    {
#if ODYSSEY_MPI
        int already_initialized = 0;
        MPI_Initialized(&already_initialized);
        if (!already_initialized)
        {
            int provided;
            MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
            if (provided < MPI_THREAD_MULTIPLE)
            {
                printf("The threading support level is lesser than that demanded.\n");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }
        MPI_Comm_size(MPI_COMM_WORLD, &this->comm_sz);
        MPI_Comm_rank(MPI_COMM_WORLD, &this->my_rank);
#else
        
        this->comm_sz = 1;
        this->my_rank = 0;
#endif
    }

    Odyssey::Odyssey(DistanceType distance_type)
        : SimilaritySearchAlgorithm(distance_type)
    {
        
        int argc = 0;
        char **argv = nullptr;
        initializeMPI(argc, argv);

        if (this->time_series_size % 8 != 0)
        {
            if (this->my_rank == 0)
            {
                std::cerr << "[Node " << this->my_rank
                          << "]: Error: SIMD calculations require query length to be a multiple of 8. Current value is "
                          << this->time_series_size << "\n";
            }
            std::exit(EXIT_FAILURE);
        }
    }

    Odyssey::Odyssey(DistanceType distance_type, int argc, char **argv)
        : SimilaritySearchAlgorithm(distance_type)
    {
        initializeMPI(argc, argv);

        if (this->time_series_size % 8 != 0)
        {
            if (this->my_rank == 0)
            {
                std::cerr << "[Node " << this->my_rank
                          << "]: Error: SIMD calculations require query length to be a multiple of 8. Current value is "
                          << this->time_series_size << "\n";
            }
            std::exit(EXIT_FAILURE);
        }
    }

    Odyssey::Odyssey(const OdysseyConfig &config, DistanceType distance_type, int argc, char **argv)
        : SimilaritySearchAlgorithm(distance_type)
    {
        this->search_workers = config.search_workers;
        this->index_threads = config.index_threads;
        this->warping_window = config.warping_window;
        this->leaf_size = config.leaf_size;
        this->paa_segments = config.paa_segments;
        this->replication_groups = config.replication_groups;
        this->query_threads = config.query_threads;
        this->pq_th_div_factor = config.pq_th_div_factor;
        this->num_threads = config.search_workers;
        
        initializeMPI(argc, argv);

        if (this->time_series_size % 8 != 0)
        {
            if (this->my_rank == 0)
            {
                std::cerr << "[Node " << this->my_rank
                          << "]: Error: SIMD calculations require query length to be a multiple of 8. Current value is "
                          << this->time_series_size << "\n";
            }
            std::exit(EXIT_FAILURE);
        }
    }

    void Odyssey::setNumThreads(int num_threads)
    {
        int max_threads = omp_get_max_threads();

        if (num_threads > max_threads) 
        {
            std::cerr << "[Warning] " << num_threads 
                    << " threads exceeds max available " << max_threads << " Using the max threads available.\n";
            this->num_threads = max_threads;
        } 
        else if (num_threads < 1) 
        {
            std::cerr << "[Warning] Thread count must be >= 1. Using 1.\n";
            this->num_threads = 1;
        } 
        else 
        {
            this->num_threads = num_threads;
        }
    } 

    int Odyssey::getNumThreads() const
    {
        return this->num_threads;
    } 

    void Odyssey::buildIndex(DataSource *data_source)
    {
        
        FileDataSource *file_source = dynamic_cast<FileDataSource *>(data_source);
        if (file_source == nullptr)
        {
            fprintf(stderr, "Error: Odyssey::buildIndex requires FileDataSource\n");
            throw std::runtime_error("Odyssey::buildIndex requires FileDataSource");
        }

        const char *raw_filename = file_source->getFilename();
        if (raw_filename == nullptr)
        {
            fprintf(stderr, "Error: FileDataSource does not have a filename\n");
            throw std::runtime_error("FileDataSource does not have a filename");
        }

        this->dim = data_source->getDim();
        this->time_series_size = static_cast<int>(this->dim);
        this->n_database = data_source->getTotalRecords();

        if (this->dataset_size == 0)
        {
            this->dataset_size = this->n_database;
        }

        odyssey_optimize_params(this);

        odyssey_prepare_structures(this, raw_filename);

        odyssey_log_parameters(this);

        this->rawfile = this->buildIndexSequence();

        if (this->verbose)
        {
            daisy::print_index_stats(this->index, this->my_rank);
        }

        #if ODYSSEY_MPI
                MPI_Barrier(MPI_COMM_WORLD);
        #endif
    }
    
    bool Odyssey::validateSearchParams(const idx_t k, const idx_t n_query) const
    {
        if (k == 0)
        {
            std::cerr << "[Error] k must be greater than 0\n";
            return false;
        }
        if (k > n_database)
        {
            std::cerr << "[Error] k (" << k << ") cannot be greater than database size (" << n_database << ")\n";
            return false;
        }
        if (n_query == 0)
        {
            std::cerr << "[Error] n_query must be greater than 0\n";
            return false;
        }
        if (getIndex() == nullptr)
        {
            std::cerr << "[Error] [Node " << getMyRank() << "] Index must be built before searching.\n";
            return false;
        }
        return true;
    }

    void Odyssey::searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D)
    {
        if (!validateSearchParams(k, n_query))
            return;

        this->top_k = static_cast<int>(k);

        const int q_num = static_cast<int>(n_query);
        const int topk = static_cast<int>(k);
        isax_index *idx = this->index;
        const int my_rank = this->my_rank;
        const int comm_sz = this->comm_sz;

        OdysseyQuery *queries = load_queries_from_buffer(query, q_num, idx, my_rank);

        double (*basis_func)(double) = initialize_basis_function(dataset_type.c_str());

        odyssey_preprocess_and_sort_queries(this, queries, q_num, true);

        query_result *results = (query_result *)malloc(sizeof(query_result) * static_cast<size_t>(q_num));
        if (!results)
        {
            std::cerr << "[Node " << my_rank << "]: Failed to allocate results.\n";
            free_queries(queries, q_num);
            return;
        }

        std::vector<BsfMessage> shared_bsf_results(static_cast<size_t>(q_num));
        for (int i = 0; i < q_num; i++)
        {
            results[i].distance = FLT_MAX;
            results[i].pq_bsf = nullptr;
            results[i].total_time = FLT_MAX;
            shared_bsf_results[i].bsf = FLT_MAX;
            shared_bsf_results[i].position = 0;
            shared_bsf_results[i].q_num = i;
        }

        if (my_rank == 0 && verbose)
        {
            for (int i = 0; i < q_num; i++)
            {
                if (queries[i].initial_pq_bsfs != nullptr)
                    printf("[Node %d]: Query %d, Actual ID: %d, BSF: %f\n", my_rank, i, queries[i].id,
                           queries[i].initial_pq_bsfs->knn[topk - 1]);
            }
        }

        NodeList nodelist = initialize_node_list(idx, my_rank);

        if (this->distance_type == DistanceType::DTW)
            searchIndexDTW(queries, q_num, topk, results, &shared_bsf_results, nodelist, I, D, basis_func);
        else
            searchIndexL2Squared(queries, q_num, topk, results, &shared_bsf_results, nodelist, I, D, basis_func);

        free(results);
        
    }

    void Odyssey::searchIndexL2Squared(OdysseyQuery *queries, int q_num, int topk,
                                      query_result *results, std::vector<BsfMessage> *shared_bsf_results,
                                      NodeList &nodelist, idx_t *I, float *D,
                                      double (*basis_func)(double))
    {
        constexpr bool ENABLE_PRINTS_PER_QUERY = false;  
        const float minimum_distance = FLT_MAX;
        isax_index *index = this->index;
        ReplicationData &replication_data = this->replication_data;
        WorkstealingData *workstealing_data = &this->workstealing_data;
        const DynamicSchedulingMode mode = static_cast<DynamicSchedulingMode>(this->dynamic_scheduling_mode);
        const int num_procs = this->comm_sz;
        const int rank = this->my_rank;

#if ODYSSEY_MPI
        if (num_procs == 1)
        {
            
            for (int q_loaded = 0; q_loaded < q_num; q_loaded++)
            {
                SearchFunctionParams args;
                args.query_id = q_loaded;
                args.ts = queries[q_loaded].query;
                args.paa = queries[q_loaded].paa;
                args.index = index;
                args.nodelist = &nodelist;
                args.minimum_distance = minimum_distance;
                args.comm_data = nullptr;
                args.estimation_func = basis_func;
                args.shared_bsf_results = shared_bsf_results;
                args.k = topk;
                args.precomputed_bsfs = queries[q_loaded].initial_pq_bsfs;
                args.query_threads = query_threads;
                args.my_rank = my_rank;
                args.comm_sz = comm_sz;
                args.verbose = verbose;
                args.rawfile = rawfile;
                args.replication_data = &replication_data;
                args.output_file = output_file;
                args.corr_threshold = corr_threshold;
                args.bsf_sharing_data = &bsf_sharing_data;
                args.workstealing_data = workstealing_data;
                args.pq_th_div_factor = pq_th_div_factor;
                args.merge_offset = merge_offset;
                args.query_counter = q_loaded;
                args.warp_window = 0;
                args.paaU = nullptr;
                args.paaL = nullptr;

                query_result result = qa_exact_search_odyssey_knn(args);
                results[queries[q_loaded].id] = result;
            }
        }
        else
        {
        std::vector<MPI_Request> request(static_cast<size_t>(comm_sz));
        std::vector<MPI_Request> send_request(static_cast<size_t>(comm_sz));
        const int distributed_queries_initial_burst = 1;
        std::vector<int*> process_buffer_initial(static_cast<size_t>(comm_sz));
        for (int i = 0; i < comm_sz; i++)
        {
            process_buffer_initial[static_cast<size_t>(i)] = (int *)malloc(sizeof(int) * static_cast<size_t>(distributed_queries_initial_burst));
            CHECK_ALLOC(process_buffer_initial[static_cast<size_t>(i)], my_rank);
        }
        std::vector<int> process_buffer(static_cast<size_t>(comm_sz), 0);
        int rec_message = 0;
        int buffer_sent = 0;
        int termination_message_id = -1;
        int q_loaded = 0;

        const int DISTRIBUTED_QUERIES_SEND_QUERY = 800;
        const int DISTRIBUTED_QUERIES_REQUEST_QUERY = 801;

        if (my_rank == rep_find_coordinator_node_rank(replication_data, my_rank))
        {
            send_initial_queries_module_coordinator_async_chatzakis(&q_loaded, my_rank, comm_sz,
                                                                  distributed_queries_initial_burst,
                                                                  process_buffer_initial.data(),
                                                                  &rec_message, request.data(), send_request.data(),
                                                                  q_num, &termination_message_id,
                                                                  &replication_data);
        }

        while (1)
        {
            if (my_rank == rep_find_coordinator_node_rank(replication_data, my_rank))
            {
                if (mode == DynamicSchedulingMode::PERIODIC_CHECK || mode == DynamicSchedulingMode::STANDALONE_THREAD)
                {
                    if (q_loaded >= q_num || q_loaded == -1)
                    {
                        if (verbose)
                            printf("[Node %d]: Exiting the dynamic loop\n", my_rank);
                        break;
                    }

                    int query_to_keep_stats = q_loaded;
                    q_loaded++;

                    CommunicationModuleData comm_data;
                    comm_data.module_func = &send_queries_module_coordinator_async_chatzakis;
                    comm_data.q_loaded = &q_loaded;
                    comm_data.rec_message = &rec_message;
                    comm_data.termination_message_id = &termination_message_id;
                    comm_data.q_num = q_num;
                    comm_data.request = request.data();
                    comm_data.send_request = send_request.data();
                    comm_data.process_buffer = process_buffer.data();
                    comm_data.mode = mode;
                    comm_data.my_rank = my_rank;
                    comm_data.comm_sz = comm_sz;
                    comm_data.replication_data = &replication_data;
                    comm_data.verbose = verbose;

                    SearchFunctionParams args;
                    args.query_id = query_to_keep_stats;
                    args.ts = queries[query_to_keep_stats].query;
                    args.paa = queries[query_to_keep_stats].paa;
                    args.index = index;
                    args.nodelist = &nodelist;
                    args.minimum_distance = minimum_distance;
                    args.comm_data = &comm_data;
                    args.estimation_func = basis_func;
                    args.shared_bsf_results = shared_bsf_results;
                    args.k = topk;
                    args.precomputed_bsfs = queries[query_to_keep_stats].initial_pq_bsfs;
                    args.query_threads = query_threads;
                    args.my_rank = my_rank;
                    args.comm_sz = comm_sz;
                    args.verbose = verbose;
                    args.rawfile = rawfile;
                    args.replication_data = &replication_data;
                    args.output_file = output_file;
                    args.corr_threshold = corr_threshold;
                    args.bsf_sharing_data = &bsf_sharing_data;
                    args.workstealing_data = workstealing_data;
                    args.pq_th_div_factor = pq_th_div_factor;
                    args.merge_offset = merge_offset;
                    args.query_counter = query_to_keep_stats;
                    args.warp_window = 0;
                    args.paaU = nullptr;
                    args.paaL = nullptr;

                    query_result result = qa_exact_search_odyssey_knn(args);
                    result.total_time = 0.0;  

                    int query_id = queries[query_to_keep_stats].id;
                    results[query_id] = result;

                    if (ENABLE_PRINTS_PER_QUERY && verbose && results[query_id].pq_bsf != nullptr)
                    {
                        printf("[Node %d]: Processed query %d (id %d) => (1-nn=%f, pos=%llu)\n",
                               my_rank, query_to_keep_stats, query_id,
                               results[query_id].pq_bsf->knn[0],
                               (unsigned long long)results[query_id].pq_bsf->position[0]);
                    }
                }

                if (!send_queries_module_coordinator_async_chatzakis(&q_loaded, q_num, process_buffer.data(),
                                                                     request.data(), &rec_message, send_request.data(),
                                                                     &termination_message_id,
                                                                     &replication_data, my_rank, comm_sz, verbose))
                {
                    break;
                }
            }
            else
            {
                MPI_Recv(&q_loaded, 1, MPI_INT, rep_find_coordinator_node_rank(replication_data, my_rank),
                         DISTRIBUTED_QUERIES_SEND_QUERY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (q_loaded >= q_num || q_loaded == -1)
                {
                    break;
                }

                SearchFunctionParams args;
                args.query_id = q_loaded;
                args.ts = queries[q_loaded].query;
                args.paa = queries[q_loaded].paa;
                args.index = index;
                args.nodelist = &nodelist;
                args.minimum_distance = minimum_distance;
                args.comm_data = nullptr;
                args.estimation_func = basis_func;
                args.shared_bsf_results = shared_bsf_results;
                args.k = topk;
                args.precomputed_bsfs = queries[q_loaded].initial_pq_bsfs;
                args.query_threads = query_threads;
                args.my_rank = my_rank;
                args.comm_sz = comm_sz;
                args.verbose = verbose;
                args.rawfile = rawfile;
                args.replication_data = &replication_data;
                args.output_file = output_file;
                args.corr_threshold = corr_threshold;
                args.bsf_sharing_data = &bsf_sharing_data;
                args.workstealing_data = workstealing_data;
                args.pq_th_div_factor = pq_th_div_factor;
                args.merge_offset = merge_offset;
                args.query_counter = q_loaded;
                args.warp_window = 0;
                args.paaU = nullptr;
                args.paaL = nullptr;

                query_result result = qa_exact_search_odyssey_knn(args);
                result.total_time = 0.0;

                int query_id = queries[q_loaded].id;
                results[query_id] = result;

                if (ENABLE_PRINTS_PER_QUERY && verbose && results[query_id].pq_bsf != nullptr)
                {
                    printf("[Node %d]: Processed query %d (id %d) => (1-nn=%f, pos=%llu)\n",
                           my_rank, q_loaded, query_id,
                           results[query_id].pq_bsf->knn[0],
                           (unsigned long long)results[query_id].pq_bsf->position[0]);
                }

                if (q_loaded == q_num - 1)
                {
                    break;
                }

                MPI_Isend(&buffer_sent, 1, MPI_INT, rep_find_coordinator_node_rank(replication_data, my_rank),
                         DISTRIBUTED_QUERIES_REQUEST_QUERY, MPI_COMM_WORLD,
                         &send_request[static_cast<size_t>(rep_find_coordinator_node_rank(replication_data, my_rank))]);
            }
        }

        for (size_t i = 0; i < static_cast<size_t>(comm_sz); i++)
            free(process_buffer_initial[i]);
        }  
#else
        (void)mode;
        (void)workstealing_data;
        for (int q_loaded = 0; q_loaded < q_num; q_loaded++)
        {
            SearchFunctionParams args;
            args.query_id = q_loaded;
            args.ts = queries[q_loaded].query;
            args.paa = queries[q_loaded].paa;
            args.index = index;
            args.nodelist = &nodelist;
            args.minimum_distance = minimum_distance;
            args.comm_data = nullptr;
            args.estimation_func = basis_func;
            args.shared_bsf_results = shared_bsf_results;
            args.k = topk;
            args.precomputed_bsfs = queries[q_loaded].initial_pq_bsfs;
            args.query_threads = query_threads;
            args.my_rank = my_rank;
            args.comm_sz = comm_sz;
            args.verbose = verbose;
            args.rawfile = rawfile;
            args.replication_data = &replication_data;
            args.output_file = output_file;
            args.corr_threshold = corr_threshold;
            args.bsf_sharing_data = &bsf_sharing_data;
            args.workstealing_data = &workstealing_data;
            args.pq_th_div_factor = pq_th_div_factor;
            args.merge_offset = merge_offset;
            args.query_counter = q_loaded;
            args.warp_window = 0;
            args.paaU = nullptr;
            args.paaL = nullptr;

            query_result result = qa_exact_search_odyssey_knn(args);
            results[queries[q_loaded].id] = result;
        }
#endif

        if (num_procs > 1 && workstealing_data->ws_type != WorkstealingType::DISABLED)
        {
            odyssey_perform_workstealing(this, queries, nodelist,
                                         &qa_exact_search_odyssey_knn_workstealing,
                                         basis_func, results, shared_bsf_results);
        }

#if defined(ODYSSEY_DEBUG_RESULTS) && ODYSSEY_DEBUG_RESULTS
        for (int i = 0; i < q_num; i++)
        {
            if (results[i].pq_bsf != nullptr)
            {
                printf("[Node %d] DEBUG results[%d]: ", my_rank, i);
                for (int j = 0; j < topk; j++)
                    printf("(pos=%llu dist=%.4f) ", (unsigned long long)results[i].pq_bsf->position[j], results[i].pq_bsf->knn[j]);
                printf("\n");
            }
            else
                printf("[Node %d] DEBUG results[%d]: pq_bsf is NULL\n", my_rank, i);
        }
        fflush(stdout);
#endif

        std::memset(I, 0, static_cast<size_t>(q_num) * static_cast<size_t>(topk) * sizeof(idx_t));
        std::memset(D, 0, static_cast<size_t>(q_num) * static_cast<size_t>(topk) * sizeof(float));

        const idx_t ts_offset = static_cast<idx_t>(rep_get_time_series_offset(replication_data, rank));
        const idx_t my_partition_size = rep_get_time_series_of_group(replication_data, rank);

        for (int i = 0; i < q_num; i++)
        {
            if (results[i].pq_bsf != nullptr)
            {
                if (num_procs == 1)
                {
                    
                    std::vector<std::pair<float, idx_t>> pairs;
                    pairs.reserve(static_cast<size_t>(topk));
                    for (int j = 0; j < topk; j++)
                    {
                        const long pos_signed = results[i].pq_bsf->position[j];
                        const float dist = results[i].pq_bsf->knn[j];
                        if (pos_signed >= 0 && dist < FLT_MAX * 0.99f)
                            pairs.emplace_back(dist, static_cast<idx_t>(pos_signed));
                    }
                    std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b) {
                        if (a.first != b.first) return a.first < b.first;
                        return a.second < b.second;
                    });
                    std::unordered_set<idx_t> seen_pos;
                    std::vector<std::pair<float, idx_t>> uniq;
                    uniq.reserve(pairs.size());
                    for (const auto &p : pairs)
                    {
                        if (seen_pos.insert(p.second).second)
                            uniq.push_back(p);
                    }
                    idx_t last_pos = 0;
                    float last_dist = 0.0f;
                    for (int j = 0; j < topk; j++)
                    {
                        if (j < static_cast<int>(uniq.size()))
                        {
                            last_dist = uniq[static_cast<size_t>(j)].first;
                            last_pos = uniq[static_cast<size_t>(j)].second;
                        }
                        I[static_cast<size_t>(i) * static_cast<size_t>(topk) + static_cast<size_t>(j)] = last_pos;
                        D[static_cast<size_t>(i) * static_cast<size_t>(topk) + static_cast<size_t>(j)] = last_dist;
                    }
                }
                else
                {
                    for (int j = 0; j < topk; j++)
                    {
                        const long pos_signed = results[i].pq_bsf->position[j];
                        idx_t pos = static_cast<idx_t>(pos_signed);
                        I[static_cast<size_t>(i) * static_cast<size_t>(topk) + static_cast<size_t>(j)] =
                            (pos < my_partition_size) ? (pos + ts_offset) : pos;
                        D[static_cast<size_t>(i) * static_cast<size_t>(topk) + static_cast<size_t>(j)] = results[i].pq_bsf->knn[j];
                    }
                }
            }
            else if (num_procs > 1)
            {
                
                for (int j = 0; j < topk; j++)
                {
                    I[static_cast<size_t>(i) * static_cast<size_t>(topk) + static_cast<size_t>(j)] = 0;
                    D[static_cast<size_t>(i) * static_cast<size_t>(topk) + static_cast<size_t>(j)] = FLT_MAX;
                }
            }
        }

#if ODYSSEY_MPI
        if (num_procs > 1)
        {
            odyssey_merge_knn_results_mpi(rank, num_procs, q_num, topk, I, D);
        }
#endif

        free(nodelist.nlist);
        free_queries(queries, q_num);
    }

    void Odyssey::searchIndexDTW(OdysseyQuery *queries, int q_num, int topk,
                                 query_result *results, std::vector<BsfMessage> *shared_bsf_results,
                                 NodeList &nodelist, idx_t *I, float *D,
                                 double (*basis_func)(double))
    {
        constexpr bool ENABLE_PRINTS_PER_QUERY = false;
        const float minimum_distance = FLT_MAX;
        const int warp_window = this->warping_window;
        isax_index *index = this->index;
        ReplicationData &replication_data = this->replication_data;
        WorkstealingData *workstealing_data = &this->workstealing_data;
        const DynamicSchedulingMode mode = static_cast<DynamicSchedulingMode>(this->dynamic_scheduling_mode);

#if ODYSSEY_MPI
        std::vector<MPI_Request> request(static_cast<size_t>(comm_sz));
        std::vector<MPI_Request> send_request(static_cast<size_t>(comm_sz));
        const int distributed_queries_initial_burst = 1;
        std::vector<int*> process_buffer_initial(static_cast<size_t>(comm_sz));
        for (int i = 0; i < comm_sz; i++)
        {
            process_buffer_initial[static_cast<size_t>(i)] = (int *)malloc(sizeof(int) * static_cast<size_t>(distributed_queries_initial_burst));
            CHECK_ALLOC(process_buffer_initial[static_cast<size_t>(i)], my_rank);
        }
        std::vector<int> process_buffer(static_cast<size_t>(comm_sz), 0);
        int rec_message = 0;
        int buffer_sent = 0;
        int termination_message_id = -1;
        int q_loaded = 0;

        const int DISTRIBUTED_QUERIES_SEND_QUERY = 800;
        const int DISTRIBUTED_QUERIES_REQUEST_QUERY = 801;

        if (my_rank == rep_find_coordinator_node_rank(replication_data, my_rank))
        {
            send_initial_queries_module_coordinator_async_chatzakis(&q_loaded, my_rank, comm_sz,
                                                                  distributed_queries_initial_burst,
                                                                  process_buffer_initial.data(),
                                                                  &rec_message, request.data(), send_request.data(),
                                                                  q_num, &termination_message_id,
                                                                  &replication_data);
        }

        while (1)
        {
            if (my_rank == rep_find_coordinator_node_rank(replication_data, my_rank))
            {
                if (mode == DynamicSchedulingMode::PERIODIC_CHECK || mode == DynamicSchedulingMode::STANDALONE_THREAD)
                {
                    if (q_loaded >= q_num || q_loaded == -1)
                    {
                        if (verbose)
                            printf("[Node %d]: Exiting the dynamic loop (DTW)\n", my_rank);
                        break;
                    }

                    int query_to_keep_stats = q_loaded;
                    q_loaded++;

                    CommunicationModuleData comm_data;
                    comm_data.module_func = &send_queries_module_coordinator_async_chatzakis;
                    comm_data.q_loaded = &q_loaded;
                    comm_data.rec_message = &rec_message;
                    comm_data.termination_message_id = &termination_message_id;
                    comm_data.q_num = q_num;
                    comm_data.request = request.data();
                    comm_data.send_request = send_request.data();
                    comm_data.process_buffer = process_buffer.data();
                    comm_data.mode = mode;
                    comm_data.my_rank = my_rank;
                    comm_data.comm_sz = comm_sz;
                    comm_data.replication_data = &replication_data;
                    comm_data.verbose = verbose;

                    SearchFunctionParams args;
                    args.query_id = query_to_keep_stats;
                    args.ts = queries[query_to_keep_stats].query;
                    args.paa = queries[query_to_keep_stats].paa;
                    args.index = index;
                    args.nodelist = &nodelist;
                    args.minimum_distance = minimum_distance;
                    args.comm_data = &comm_data;
                    args.estimation_func = basis_func;
                    args.shared_bsf_results = shared_bsf_results;
                    args.k = topk;
                    args.precomputed_bsfs = queries[query_to_keep_stats].initial_pq_bsfs;
                    args.query_threads = query_threads;
                    args.my_rank = my_rank;
                    args.comm_sz = comm_sz;
                    args.verbose = verbose;
                    args.rawfile = rawfile;
                    args.replication_data = &replication_data;
                    args.output_file = output_file;
                    args.corr_threshold = corr_threshold;
                    args.bsf_sharing_data = &bsf_sharing_data;
                    args.workstealing_data = workstealing_data;
                    args.pq_th_div_factor = pq_th_div_factor;
                    args.merge_offset = merge_offset;
                    args.query_counter = query_to_keep_stats;
                    args.warp_window = warp_window;
                    args.paaU = queries[query_to_keep_stats].paaU;
                    args.paaL = queries[query_to_keep_stats].paaL;

                    query_result result = qa_exact_search_odyssey_knn(args);
                    result.total_time = 0.0;

                    int query_id = queries[query_to_keep_stats].id;
                    results[query_id] = result;

                    if (ENABLE_PRINTS_PER_QUERY && verbose && results[query_id].pq_bsf != nullptr)
                    {
                        printf("[Node %d]: Processed query %d (id %d) => (1-nn=%f, pos=%llu)\n",
                               my_rank, query_to_keep_stats, query_id,
                               results[query_id].pq_bsf->knn[0],
                               (unsigned long long)results[query_id].pq_bsf->position[0]);
                    }
                }

                if (!send_queries_module_coordinator_async_chatzakis(&q_loaded, q_num, process_buffer.data(),
                                                                     request.data(), &rec_message, send_request.data(),
                                                                     &termination_message_id,
                                                                     &replication_data, my_rank, comm_sz, verbose))
                {
                    break;
                }
            }
            else
            {
                MPI_Recv(&q_loaded, 1, MPI_INT, rep_find_coordinator_node_rank(replication_data, my_rank),
                         DISTRIBUTED_QUERIES_SEND_QUERY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (q_loaded >= q_num || q_loaded == -1)
                {
                    break;
                }

                SearchFunctionParams args;
                args.query_id = q_loaded;
                args.ts = queries[q_loaded].query;
                args.paa = queries[q_loaded].paa;
                args.index = index;
                args.nodelist = &nodelist;
                args.minimum_distance = minimum_distance;
                args.comm_data = nullptr;
                args.estimation_func = basis_func;
                args.shared_bsf_results = shared_bsf_results;
                args.k = topk;
                args.precomputed_bsfs = queries[q_loaded].initial_pq_bsfs;
                args.query_threads = query_threads;
                args.my_rank = my_rank;
                args.comm_sz = comm_sz;
                args.verbose = verbose;
                args.rawfile = rawfile;
                args.replication_data = &replication_data;
                args.output_file = output_file;
                args.corr_threshold = corr_threshold;
                args.bsf_sharing_data = &bsf_sharing_data;
                args.workstealing_data = workstealing_data;
                args.pq_th_div_factor = pq_th_div_factor;
                args.merge_offset = merge_offset;
                args.query_counter = q_loaded;
                args.warp_window = warp_window;
                args.paaU = queries[q_loaded].paaU;
                args.paaL = queries[q_loaded].paaL;

                query_result result = qa_exact_search_odyssey_knn(args);
                result.total_time = 0.0;

                int query_id = queries[q_loaded].id;
                results[query_id] = result;

                if (ENABLE_PRINTS_PER_QUERY && verbose && results[query_id].pq_bsf != nullptr)
                {
                    printf("[Node %d]: Processed query %d (id %d) => (1-nn=%f, pos=%llu)\n",
                           my_rank, q_loaded, query_id,
                           results[query_id].pq_bsf->knn[0],
                           (unsigned long long)results[query_id].pq_bsf->position[0]);
                }

                if (q_loaded == q_num - 1)
                {
                    break;
                }

                MPI_Isend(&buffer_sent, 1, MPI_INT, rep_find_coordinator_node_rank(replication_data, my_rank),
                         DISTRIBUTED_QUERIES_REQUEST_QUERY, MPI_COMM_WORLD,
                         &send_request[static_cast<size_t>(rep_find_coordinator_node_rank(replication_data, my_rank))]);
            }
        }

        for (size_t i = 0; i < static_cast<size_t>(comm_sz); i++)
            free(process_buffer_initial[i]);
#else
        (void)mode;
        (void)workstealing_data;
        for (int q_loaded = 0; q_loaded < q_num; q_loaded++)
        {
            SearchFunctionParams args;
            args.query_id = q_loaded;
            args.ts = queries[q_loaded].query;
            args.paa = queries[q_loaded].paa;
            args.index = index;
            args.nodelist = &nodelist;
            args.minimum_distance = minimum_distance;
            args.comm_data = nullptr;
            args.estimation_func = basis_func;
            args.shared_bsf_results = shared_bsf_results;
            args.k = topk;
            args.precomputed_bsfs = queries[q_loaded].initial_pq_bsfs;
            args.query_threads = query_threads;
            args.my_rank = my_rank;
            args.comm_sz = comm_sz;
            args.verbose = verbose;
            args.rawfile = rawfile;
            args.replication_data = &replication_data;
            args.output_file = output_file;
            args.corr_threshold = corr_threshold;
            args.bsf_sharing_data = &bsf_sharing_data;
            args.workstealing_data = &workstealing_data;
            args.pq_th_div_factor = pq_th_div_factor;
            args.merge_offset = merge_offset;
            args.query_counter = q_loaded;
            args.warp_window = warp_window;
            args.paaU = queries[q_loaded].paaU;
            args.paaL = queries[q_loaded].paaL;

            query_result result = qa_exact_search_odyssey_knn(args);
            results[queries[q_loaded].id] = result;
        }
#endif

        if (comm_sz > 1 && workstealing_data->ws_type != WorkstealingType::DISABLED)
        {
            odyssey_perform_workstealing(this, queries, nodelist,
                                         &qa_exact_search_odyssey_knn_workstealing,
                                         basis_func, results, shared_bsf_results);
        }

#if defined(ODYSSEY_DEBUG_RESULTS) && ODYSSEY_DEBUG_RESULTS
        for (int i = 0; i < q_num; i++)
        {
            if (results[i].pq_bsf != nullptr)
            {
                printf("[Node %d] DEBUG DTW results[%d]: ", my_rank, i);
                for (int j = 0; j < topk; j++)
                    printf("(pos=%llu dist=%.4f) ", (unsigned long long)results[i].pq_bsf->position[j], results[i].pq_bsf->knn[j]);
                printf("\n");
            }
            else
                printf("[Node %d] DEBUG DTW results[%d]: pq_bsf is NULL\n", my_rank, i);
        }
        fflush(stdout);
#endif

        if (comm_sz > 1)
        {
            std::memset(I, 0, static_cast<size_t>(q_num) * static_cast<size_t>(topk) * sizeof(idx_t));
            std::memset(D, 0, static_cast<size_t>(q_num) * static_cast<size_t>(topk) * sizeof(float));
        }

        const idx_t ts_offset = static_cast<idx_t>(rep_get_time_series_offset(replication_data, my_rank));
        const idx_t my_partition_size = rep_get_time_series_of_group(replication_data, my_rank);

        for (int i = 0; i < q_num; i++)
        {
            if (results[i].pq_bsf != nullptr)
            {
                
                std::vector<std::pair<float, idx_t>> pairs;
                pairs.reserve(static_cast<size_t>(topk));
                for (int j = 0; j < topk; j++)
                {
                    float dist = results[i].pq_bsf->knn[j];
                    idx_t pos = static_cast<idx_t>(results[i].pq_bsf->position[j]);
                    if (dist < FLT_MAX && pos != static_cast<idx_t>(-1))
                    {
                        idx_t global_pos = (pos < my_partition_size) ? (pos + ts_offset) : pos;
                        pairs.emplace_back(dist, global_pos);
                    }
                }

                std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b) {
                    if (a.first != b.first)
                        return a.first < b.first;
                    return a.second < b.second;
                });

                std::unordered_set<idx_t> seen;
                std::vector<std::pair<float, idx_t>> uniq;
                uniq.reserve(pairs.size());
                for (const auto &p : pairs)
                {
                    if (seen.insert(p.second).second)
                        uniq.push_back(p);
                }

                int out_j = 0;
                for (; out_j < topk && out_j < static_cast<int>(uniq.size()); out_j++)
                {
                    I[static_cast<size_t>(i) * static_cast<size_t>(topk) + static_cast<size_t>(out_j)] = uniq[out_j].second;
                    D[static_cast<size_t>(i) * static_cast<size_t>(topk) + static_cast<size_t>(out_j)] = uniq[out_j].first;
                }

                if (out_j > 0 && out_j < topk)
                {
                    idx_t pad_pos = I[static_cast<size_t>(i) * static_cast<size_t>(topk) + static_cast<size_t>(out_j - 1)];
                    float pad_dist = D[static_cast<size_t>(i) * static_cast<size_t>(topk) + static_cast<size_t>(out_j - 1)];
                    for (int j = out_j; j < topk; j++)
                    {
                        I[static_cast<size_t>(i) * static_cast<size_t>(topk) + static_cast<size_t>(j)] = pad_pos;
                        D[static_cast<size_t>(i) * static_cast<size_t>(topk) + static_cast<size_t>(j)] = pad_dist;
                    }
                }
            }
            else if (comm_sz > 1)
            {
                
                for (int j = 0; j < topk; j++)
                {
                    I[static_cast<size_t>(i) * static_cast<size_t>(topk) + static_cast<size_t>(j)] = 0;
                    D[static_cast<size_t>(i) * static_cast<size_t>(topk) + static_cast<size_t>(j)] = FLT_MAX;
                }
            }
        }

#if ODYSSEY_MPI
        if (comm_sz > 1)
        {
            odyssey_merge_knn_results_mpi(my_rank, comm_sz, q_num, topk, I, D);
        }
#endif

        free(nodelist.nlist);
        free_queries(queries, q_num);
    }

    float *Odyssey::buildIndexSequence()
    {

        isax_index *index = this->index;
        int my_rank = this->my_rank;
        ReplicationData *replication_data = &this->replication_data;
        int index_threads = this->index_threads;

        const char *ifilename = nullptr;
        if (this->index_settings && this->index_settings->raw_filename)
        {
            ifilename = this->index_settings->raw_filename;
        }
        else if (index && index->settings && index->settings->raw_filename)
        {
            ifilename = index->settings->raw_filename;
        }

        if (ifilename == nullptr)
        {
            fprintf(stderr, "[Node %d] Error: raw_filename not set in index settings.\n", my_rank);
            std::exit(EXIT_FAILURE);
        }

        idx_t total_samples = this->dataset_size;

        FILE *ifile = std::fopen(ifilename, "rb");
        if (!ifile)
        {
            fprintf(stderr, "[Node %d] File %s not found!\n", my_rank, ifilename);
            std::exit(EXIT_FAILURE);
        }

        std::fseek(ifile, 0L, SEEK_END);
        file_position_type sz = static_cast<file_position_type>(std::ftell(ifile)); 
        std::fseek(ifile, 0L, SEEK_SET);

        idx_t my_time_series = rep_get_time_series_of_group(*replication_data, my_rank);

        file_position_type total_records = sz / static_cast<file_position_type>(index->settings->ts_byte_size);

        if (total_records < static_cast<file_position_type>(total_samples))
        {
            fprintf(stderr,
                    "[Node %d] File %s has only %llu records (expected at least %llu)!\n",
                    my_rank,
                    ifilename,
                    static_cast<unsigned long long>(total_records),
                    static_cast<unsigned long long>(total_samples));
            fprintf(stderr, "[Node %d] Likely cause: multi-node with /tmp or local path. Run on a SINGLE NODE.\n", my_rank);
            fflush(stderr);
            std::fclose(ifile);
            std::exit(EXIT_FAILURE);
        }

        size_t rawfile_bytes =
            static_cast<size_t>(index->settings->ts_byte_size) *
            static_cast<size_t>(my_time_series);

        float *rawfile = static_cast<float *>(std::malloc(rawfile_bytes));
        if (rawfile == nullptr)
        {
            fprintf(stderr, "[Node %d] Error: Memory allocation failed for rawfile.\n", my_rank);
            std::fclose(ifile);
            std::exit(EXIT_FAILURE);
        }

        idx_t ts_offset = rep_get_time_series_offset(*replication_data, my_rank);
        file_position_type position_to_file =
            static_cast<file_position_type>(ts_offset) *
            static_cast<file_position_type>(index->settings->ts_byte_size);

        int returned_val = std::fseek(ifile, static_cast<long>(position_to_file), SEEK_SET);
        if (returned_val != 0)
        {
            fprintf(stderr, "[Node %d] Error on fseek() while positioning to %llu bytes.\n",
                    my_rank,
                    static_cast<unsigned long long>(position_to_file));
            std::fclose(ifile);
            std::exit(EXIT_FAILURE);
        }

        size_t elements_to_read =
            static_cast<size_t>(index->settings->timeseries_size) *
            static_cast<size_t>(my_time_series);

        size_t read_number = std::fread(
            rawfile,
            sizeof(ts_type),
            elements_to_read,
            ifile);

        std::fclose(ifile);

        printf("[Node %d]: Loaded %zu data series starting from %llu.\n",
               my_rank,
               read_number / static_cast<size_t>(index->settings->timeseries_size),
               static_cast<unsigned long long>(position_to_file));

        if ((read_number / static_cast<size_t>(index->settings->timeseries_size)) !=
            static_cast<size_t>(my_time_series))
        {
            fprintf(stderr,
                    "[Node %d] Must read: %llu but node read %zu time series\n",
                    my_rank,
                    static_cast<unsigned long long>(my_time_series),
                    read_number / static_cast<size_t>(index->settings->timeseries_size));
            std::exit(EXIT_FAILURE);
        }

        index->fbl = reinterpret_cast<first_buffer_layer *>(
            initialize_pRecBuf_ekosmas(
                index->settings->initial_fbl_buffer_size,
                static_cast<int>(std::pow(2.0, index->settings->paa_segments)),
                index->settings->max_total_buffer_size +
                    DISK_BUFFER_SIZE * (PROGRESS_CALCULATE_THREAD_NUMBER - 1),
                index,
                index_threads));

        std::vector<pthread_t> threadid(static_cast<size_t>(index_threads));

        daisy::buffer_data_inmemory_ekosmas *input_data =
            static_cast<daisy::buffer_data_inmemory_ekosmas *>(
                std::malloc(sizeof(daisy::buffer_data_inmemory_ekosmas) *
                            static_cast<size_t>(index_threads)));
        if (input_data == nullptr)
        {
            fprintf(stderr,
                    "[Node %d] Error: Memory allocation failed for buffer_data_inmemory_ekosmas.\n",
                    my_rank);
            std::exit(EXIT_FAILURE);
        }

        unsigned long next_block_to_process = 0;
        int node_counter = 0; 

        volatile unsigned long *next_iSAX_group =
            static_cast<volatile unsigned long *>(
                std::calloc(static_cast<size_t>(index->fbl->max_total_size),
                            sizeof(unsigned long)));
        if (next_iSAX_group == nullptr)
        {
            fprintf(stderr,
                    "[Node %d] Error: Memory allocation failed for next_iSAX_group.\n",
                    my_rank);
            std::free(input_data);
            std::exit(EXIT_FAILURE);
        }

        pthread_barrier_t wait_summaries_to_compute;
        pthread_barrier_init(&wait_summaries_to_compute, nullptr,
                             static_cast<unsigned int>(index_threads));

        pthread_mutex_t lock_firstnode = PTHREAD_MUTEX_INITIALIZER;

        for (int i = 0; i < index_threads; i++)
        {
            daisy::buffer_data_inmemory_ekosmas &data = input_data[i];
            data.index = index;
            data.lock_firstnode = &lock_firstnode; 
            data.workernumber = i;
            data.shared_start_number = &next_block_to_process;
            data.ts_num = my_time_series;
            data.wait_summaries_to_compute = &wait_summaries_to_compute;
            data.node_counter = &node_counter; 
            data.parallelism_in_subtree = daisy::NO_PARALLELISM_IN_SUBTREE;
            data.next_iSAX_group = next_iSAX_group; 
            data.rawfile = rawfile;
            data.deterministic_index = this->workstealing_data.deterministic_index;
            data.index_threads = index_threads;
            data.readblock = this->read_block_length;
            data.my_rank = my_rank;
            data.comm_sz = this->comm_sz;
            data.replication_data = replication_data;
            
        }

        for (int i = 0; i < index_threads; i++)
        {
            if (pthread_create(&threadid[static_cast<size_t>(i)],
                               nullptr,
                               daisy::index_creation_sequence_worker,
                               static_cast<void *>(&input_data[i])) != 0)
            {
                fprintf(stderr,
                        "[Node %d] Error: could not create index_creation_sequence_worker thread %d\n",
                        my_rank,
                        i);
                std::free(input_data);
                std::exit(EXIT_FAILURE);
            }
        }

        for (int i = 0; i < index_threads; i++)
        {
            if (pthread_join(threadid[static_cast<size_t>(i)], nullptr) != 0)
            {
                fprintf(stderr,
                        "[Node %d] Error: could not join index_creation_sequence_worker thread %d\n",
                        my_rank,
                        i);
                std::free(input_data);
                std::exit(EXIT_FAILURE);
            }
        }

        std::free(input_data);
        std::free(const_cast<unsigned long *>(next_iSAX_group));

        #if ODYSSEY_MPI
                MPI_Barrier(MPI_COMM_WORLD);
        #endif

        return rawfile;
    }

    Odyssey::~Odyssey()
    {
        delete[] database;
    }

    void odyssey_optimize_params(Odyssey *odyssey)
    {

        if (odyssey->comm_sz == 1)
        {

            if (odyssey->bsf_sharing_data.bsf_sharing_enabled)
            {
                odyssey->bsf_sharing_data.bsf_sharing_enabled = false;
                if (odyssey->my_rank == 0)  
                {
                    printf("[Node %d, OptParams]: Single node execution. Disabling BSF-sharing\n", 
                           odyssey->my_rank);
                }
            }

            if (odyssey->workstealing_data.ws_type != WorkstealingType::DISABLED)
            {
                odyssey->workstealing_data.ws_type = WorkstealingType::DISABLED;
                if (odyssey->my_rank == 0)
                {
                    printf("[Node %d, OptParams]: Single node execution. Disabling Workstealing\n", 
                           odyssey->my_rank);
                }
            }

            if (odyssey->replication_data.total_groups != 1)
            {
                odyssey->replication_data.total_groups = 1;
                if (odyssey->my_rank == 0)
                {
                    printf("[Node %d, OptParams]: Single node execution. Replication groups set to 1\n", 
                           odyssey->my_rank);
                }
            }

            return;  
        }

        if (odyssey->replication_data.total_groups == 1)
        {

            if (odyssey->bsf_sharing_data.bsf_sharing_enabled)
            {
                odyssey->bsf_sharing_data.bsf_sharing_enabled = false;
                if (odyssey->my_rank == 0)
                {
                    printf("[Node %d, OptParams]: Full replication selected. Disabling BSF-sharing\n", 
                           odyssey->my_rank);
                }
            }
        }

        if (odyssey->replication_data.total_groups == odyssey->comm_sz)
        {

            if (odyssey->workstealing_data.ws_type != WorkstealingType::DISABLED)
            {
                odyssey->workstealing_data.ws_type = WorkstealingType::DISABLED;
                if (odyssey->my_rank == 0)
                {
                    printf("[Node %d, OptParams]: No replication selected. Disabling Workstealing\n", 
                           odyssey->my_rank);
                }
            }
        }

    }

    void odyssey_prepare_structures(Odyssey *odyssey, const char *raw_filename)
    {

        odyssey->index_settings = isax_index_settings_init(
            "",                              
            odyssey->time_series_size,       
            odyssey->paa_segments,           
            odyssey->sax_cardinality,        
            odyssey->leaf_size,              
            odyssey->min_leaf_size,          
            odyssey->initial_lbl_size,       
            odyssey->flush_limit,            
            odyssey->initial_fbl_size,      
            1,                               
            0,                               
            0,                               
            1,                               
            1                                
        );

        if (raw_filename != nullptr && odyssey->index_settings != nullptr)
        {
            
            if (odyssey->index_settings->raw_filename != nullptr)
            {
                free(odyssey->index_settings->raw_filename);
            }
            odyssey->index_settings->raw_filename = (char *)malloc(strlen(raw_filename) + 1);
            if (odyssey->index_settings->raw_filename != nullptr)
            {
                strcpy(odyssey->index_settings->raw_filename, raw_filename);
            }
        }

        odyssey->index = isax_index_init_inmemory_ekosmas(odyssey->index_settings);

        bsf_sharing_init(odyssey->bsf_sharing_data, odyssey->my_rank, odyssey->comm_sz);

        if (odyssey->replication_groups == 0)
        {

            odyssey->replication_data.total_groups = odyssey->comm_sz;
        }
        else
        {
            odyssey->replication_data.total_groups = odyssey->replication_groups;
        }

        rep_init(odyssey->replication_data, odyssey->dataset_size, odyssey->my_rank, 
                 odyssey->comm_sz, odyssey->index_threads, odyssey->query_threads);

        if (odyssey->my_rank == 0)
        {
            printf("[Node 0] Replication Groups Configuration:\n");
            printf("  - Total Replication Groups: %d\n", odyssey->replication_data.total_groups);
            printf("  - Total MPI Processes: %d\n", odyssey->comm_sz);
            for (int i = 0; i < odyssey->replication_data.total_groups; i++)
            {
                printf("  - Group[%d]: %llu time series, %d nodes\n",
                       i,
                       static_cast<unsigned long long>(odyssey->replication_data.node_groups[i].total_time_series),
                       odyssey->replication_data.node_groups[i].total_nodes);
            }
        }

        ws_init(odyssey->workstealing_data, odyssey->comm_sz);
    }

    void odyssey_log_parameters(Odyssey *odyssey)
    {
        if (odyssey->my_rank == 0 && odyssey->verbose)  
        {
            const char *scheduling_methods[] = {"Single Node", "Static", "Round Robin", "Dynamic"};
            const char *odyssey_modes[] = {"Subsequence Similarity Search", "Sequence Similarity Search"};
            const char *dynamic_scheduling_modes[] = {"Periodic Check", "Standalone Thread"};
            const char *dis_enab[] = {"Disabled", "Enabled"};
            const char *workstealing_types[] = {"Disabled", "S-WS"};

            printf("================ Odyssey Settings ================\n");
            printf("Total Processes: [%d]\n", odyssey->comm_sz);

            printf("Dataset, Size: [%s, %llu]\n", 
                   "From FileDataSource", 
                   (unsigned long long)odyssey->dataset_size);
            printf("Queries, Size: [%s, %llu]\n", 
                   "Passed to searchIndex()", 
                   0ULL);  
            
            printf("PAA Segments, SAX Cardinality: [%d, %d]\n", 
                   odyssey->index->settings->paa_segments, 
                   odyssey->index->settings->sax_bit_cardinality);
            printf("Time-series Size: [%d]\n", odyssey->index->settings->timeseries_size);
            printf("Leaf Size, Min Leaf Size, Read Block, Flush Limit: [%d, %d, %d, %d]\n", 
                   odyssey->index->settings->max_leaf_size, 
                   odyssey->index->settings->min_leaf_size, 
                   odyssey->read_block_length, 
                   odyssey->index->settings->max_total_full_buffer_size);

            int mode_idx = odyssey->mode;
            int scheduling_idx = odyssey->query_scheduling;

            if (mode_idx < 0 || mode_idx >= 2) mode_idx = 1;  
            if (scheduling_idx < 0 || scheduling_idx >= 4) scheduling_idx = 3;  

            printf("Mode, Scheduling, Method: [%s, %s, KNN]\n",
                   odyssey_modes[mode_idx],
                   scheduling_methods[scheduling_idx]);

            if (odyssey->query_scheduling == 3)  
            {
                int dyn_sched_idx = odyssey->dynamic_scheduling_mode;
                if (dyn_sched_idx < 0 || dyn_sched_idx >= 2) dyn_sched_idx = 1;  
                printf("Dynamic Scheduling: [%s]\n", dynamic_scheduling_modes[dyn_sched_idx]);
            }

            if (odyssey->mode == 0)  
            {
                printf("Merge Offset: [%d]\n", odyssey->merge_offset);
            }

            printf("TH Division Factor: [%d]\n", odyssey->pq_th_div_factor);
            printf("Dataset-Type: [%s]\n", odyssey->dataset_type.c_str());

            int ws_type_idx = static_cast<int>(odyssey->workstealing_data.ws_type);
            if (ws_type_idx < 0 || ws_type_idx >= 2) ws_type_idx = 0;  
            printf("Workstealing: [%s]\n", workstealing_types[ws_type_idx]);
            
            printf("BSF-Sharing: [%s]\n", 
                   dis_enab[odyssey->bsf_sharing_data.bsf_sharing_enabled ? 1 : 0]);
            printf("Density-Aware Distribution: [%s]\n", 
                   dis_enab[odyssey->density_aware_prepro ? 1 : 0]);
            printf("Output File Name: [%s]\n", 
                   odyssey->output_file.empty() ? "Not Provided" : odyssey->output_file.c_str());

            printf("Online znorm: [%s]\n", "Not Supported (handle externally)");
            
            printf("KNN k-size: [%d]\n", odyssey->top_k);
            printf("Verbose: [%s]\n", odyssey->verbose ? "Enabled" : "Disabled");

            rep_log_info(odyssey->replication_data, odyssey->index_threads, odyssey->query_threads);

            printf("==================================================\n");
        }
    }

} 
