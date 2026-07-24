#include "Messi.hpp"

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <unordered_set>
#include <vector>

#include "../isax/iSAXPqueue.hpp"
#include "../isax/iSAXIndex.hpp"
#include "../isax/iSAXSearch.hpp"

#include "../distance_computers/DistanceComputer.hpp"

namespace daisy
{
    void insert_tree_node_m_hybridpqueue(float *paa, isax_node *node, isax_index *index, float bsf, pqueue_t **pq, pthread_mutex_t *lock_queue, int *tnumber, int n_pqueue)
    {

        if (node == NULL || node->isax_values == NULL || node->isax_cardinalities == NULL)
        {
            return;
        }

        float distance = minidist_paa_to_isax(paa, node->isax_values,
                                              node->isax_cardinalities,
                                              index->settings->sax_bit_cardinality,
                                              index->settings->sax_alphabet_cardinality,
                                              index->settings->paa_segments,
                                              MINVAL, MAXVAL,
                                              index->settings->mindist_sqrt);

        if (distance < bsf)
        {
            if (node->is_leaf)
            {
                query_result *mindist_result = (query_result *)malloc(sizeof(query_result));
                mindist_result->node = node;
                mindist_result->distance = distance;
                pthread_mutex_lock(&lock_queue[*tnumber]);
                pqueue_insert(pq[*tnumber], mindist_result);

                pthread_mutex_unlock(&lock_queue[*tnumber]);
                *tnumber = (*tnumber + 1) % n_pqueue;
            }
            else
            {
                if (node->left_child != NULL &&
                    node->left_child->isax_values != NULL &&
                    node->left_child->isax_cardinalities != NULL)
                {
                    insert_tree_node_m_hybridpqueue(paa, node->left_child, index, bsf, pq, lock_queue, tnumber, n_pqueue);
                }
                if (node->right_child != NULL &&
                    node->right_child->isax_values != NULL &&
                    node->right_child->isax_cardinalities != NULL)
                {
                    insert_tree_node_m_hybridpqueue(paa, node->right_child, index, bsf, pq, lock_queue, tnumber, n_pqueue);
                }
            }
        }
    }

    void insert_tree_node_m_hybridpqueue_DTW(float *paaU, float *paaL, isax_node *node, isax_index *index, float bsf, pqueue_t **pq, pthread_mutex_t *lock_queue, int *tnumber, int n_pqueue)
    {

        if (node == NULL || node->isax_values == NULL || node->isax_cardinalities == NULL)
        {
            return;
        }

        float distance = minidist_paa_to_isax_DTW(paaU, paaL, node->isax_values,
                                                  node->isax_cardinalities,
                                                  index->settings->sax_bit_cardinality,
                                                  index->settings->sax_alphabet_cardinality,
                                                  index->settings->paa_segments,
                                                  MINVAL, MAXVAL,
                                                  index->settings->mindist_sqrt);
        if (distance <= bsf)
        {
            if (node->is_leaf)
            {
                query_result *mindist_result = (query_result *)malloc(sizeof(query_result));
                mindist_result->node = node;
                mindist_result->distance = distance;
                pthread_mutex_lock(&lock_queue[*tnumber]);
                pqueue_insert(pq[*tnumber], mindist_result);
                pthread_mutex_unlock(&lock_queue[*tnumber]);
                *tnumber = (*tnumber + 1) % n_pqueue;
            }
            else
            {
                if (node->left_child != NULL &&
                    node->left_child->isax_values != NULL &&
                    node->left_child->isax_cardinalities != NULL)
                {
                    insert_tree_node_m_hybridpqueue_DTW(paaU, paaL, node->left_child, index, bsf, pq, lock_queue, tnumber, n_pqueue);
                }
                if (node->right_child != NULL &&
                    node->right_child->isax_values != NULL &&
                    node->right_child->isax_cardinalities != NULL)
                {
                    insert_tree_node_m_hybridpqueue_DTW(paaU, paaL, node->right_child, index, bsf, pq, lock_queue, tnumber, n_pqueue);
                }
            }
        }
    }

    void calculate_node2_topk_inmemory(isax_index *index, isax_node *node, ts_type *query, ts_type *paa, pqueue_bsf *pq_bsf, pthread_rwlock_t *lock_queue, float *rawfile)
    {

        if (node == NULL || node->buffer == NULL)
        {
            return;
        }

        {
            int i;
            for (i = 0; i < node->buffer->full_buffer_size; i++)
            {
                float dist = ts_euclidean_distance_SIMD(query, node->buffer->full_ts_buffer[i],
                                                        index->settings->timeseries_size, pq_bsf->knn[pq_bsf->k - 1]);

                if (dist <= pq_bsf->knn[pq_bsf->k - 1])
                {
                    file_position_type pos = 0;
                    if (node->buffer->full_position_buffer != nullptr &&
                        node->buffer->full_position_buffer[i] != nullptr)
                    {
                        pos = *node->buffer->full_position_buffer[i] / index->settings->timeseries_size;
                    }
                    pthread_rwlock_wrlock(lock_queue);
                    pqueue_bsf_insert(pq_bsf, dist, static_cast<long int>(pos), node);
                    pthread_rwlock_unlock(lock_queue);
                }
            }

            for (i = 0; i < node->buffer->tmp_full_buffer_size; i++)
            {
                float dist = ts_euclidean_distance_SIMD(query, node->buffer->tmp_full_ts_buffer[i],
                                                        index->settings->timeseries_size, pq_bsf->knn[pq_bsf->k - 1]);

                if (dist <= pq_bsf->knn[pq_bsf->k - 1])
                {
                    file_position_type pos = 0;
                    if (node->buffer->tmp_full_position_buffer != nullptr &&
                        node->buffer->tmp_full_position_buffer[i] != nullptr)
                    {
                        pos = *node->buffer->tmp_full_position_buffer[i] / index->settings->timeseries_size;
                    }
                    pthread_rwlock_wrlock(lock_queue);
                    pqueue_bsf_insert(pq_bsf, dist, static_cast<long int>(pos), node);
                    pthread_rwlock_unlock(lock_queue);
                }
            }
            for (i = 0; i < node->buffer->partial_buffer_size; i++)
            {
                float distmin = minidist_paa_to_isax_rawa_SIMD(paa, node->buffer->partial_sax_buffer[i],
                                                               index->settings->max_sax_cardinalities,
                                                               index->settings->sax_bit_cardinality,
                                                               index->settings->sax_alphabet_cardinality,
                                                               index->settings->paa_segments, MINVAL, MAXVAL,
                                                               index->settings->mindist_sqrt);
                if (distmin <= pq_bsf->knn[pq_bsf->k - 1])
                {
                    float dist = ts_euclidean_distance_SIMD(query, &(rawfile[*node->buffer->partial_position_buffer[i]]),
                                                            index->settings->timeseries_size, pq_bsf->knn[pq_bsf->k - 1]);

                    if (dist <= pq_bsf->knn[pq_bsf->k - 1])
                    {
                        pthread_rwlock_wrlock(lock_queue);

                        pqueue_bsf_insert(pq_bsf, dist, *node->buffer->partial_position_buffer[i] / index->settings->timeseries_size, node);

                        pthread_rwlock_unlock(lock_queue);
                    }
                }
            }
        }
    }

    void calculate_node_DTW2knn_inmemory(isax_index *index, isax_node *node, ts_type *query, float *uo, float *lo, ts_type *paa, ts_type *paaU, ts_type *paaL, float bsf, int warpWind, pqueue_bsf *pq_bsf, pthread_rwlock_t *lock_queue, float *rawfile)
    {

        if (node == NULL || node->buffer == NULL)
        {
            return;
        }

        float distmin;
        int k;
        float *cb = (float *)malloc(sizeof(float) * index->settings->timeseries_size);
        float *cb1 = (float *)malloc(sizeof(float) * index->settings->timeseries_size);
        int length = 2 * warpWind + 1;
        float *tSum = (float *)malloc(sizeof(float) * length);

        float *pCost = (float *)malloc(sizeof(float) * length);

        float *rDist = (float *)malloc(sizeof(float) * length);

        for (int i = 0; i < node->buffer->partial_buffer_size; i++)
        {
            distmin = minidist_paa_to_isax_raw_DTW_SIMD(paaU, paaL, node->buffer->partial_sax_buffer[i],
                                                        index->settings->max_sax_cardinalities,
                                                        index->settings->sax_bit_cardinality,
                                                        index->settings->sax_alphabet_cardinality,
                                                        index->settings->paa_segments, MINVAL, MAXVAL,
                                                        index->settings->mindist_sqrt);

            if (distmin <= bsf)
            {
                distmin = lb_keogh_data_bound(&(rawfile[*node->buffer->partial_position_buffer[i]]), uo, lo, cb1, index->settings->timeseries_size, bsf);

                cb[index->settings->timeseries_size - 1] = cb1[index->settings->timeseries_size - 1];
                for (k = index->settings->timeseries_size - 2; k >= 0; k--)
                    cb[k] = cb[k + 1] + cb1[k];

                float dist = dtwsimdPruned(query, &(rawfile[*node->buffer->partial_position_buffer[i]]), cb, index->settings->timeseries_size, warpWind, bsf, tSum, pCost, rDist);

                if (dist <= pq_bsf->knn[pq_bsf->k - 1])
                {
                    pthread_rwlock_wrlock(lock_queue);
                    pqueue_bsf_insert(pq_bsf, dist, *node->buffer->partial_position_buffer[i] / index->settings->timeseries_size, node);
                    pthread_rwlock_unlock(lock_queue);
                }
            }
        }

        free(tSum);
        free(pCost);
        free(rDist);
        free(cb);
        free(cb1);
    }

    void *MESSI_topk_search_worker_L2Squared(void *rfdata)
    {
        isax_node *current_root_node;
        query_result *n;
        isax_index *index = ((MESSI_workerdata *)rfdata)->index;
        ts_type *paa = ((MESSI_workerdata *)rfdata)->paa;
        ts_type *ts = ((MESSI_workerdata *)rfdata)->ts;
        pqueue_t *pq = ((MESSI_workerdata *)rfdata)->pq;
        query_result *do_not_remove = ((MESSI_workerdata *)rfdata)->bsf_result;
        float minimum_distance = ((MESSI_workerdata *)rfdata)->minimum_distance;
        pqueue_bsf *pq_bsf = ((MESSI_workerdata *)rfdata)->pq_bsf;
        int limit = ((MESSI_workerdata *)rfdata)->limit;
        int checks = 0;
        int n_pqueue = ((MESSI_workerdata *)rfdata)->n_pqueue;
        bool finished = true;
        int current_root_node_number;
        int tight_bound = index->settings->tight_bound;
        int aggressive_check = index->settings->aggressive_check;
        float bsfdisntance = pq_bsf->knn[pq_bsf->k - 1];
        int calculate_node = 0, calculate_node_quque = 0;
        int tnumber = rand() % n_pqueue;
        int startqueuenumber = ((MESSI_workerdata *)rfdata)->startqueuenumber;
        float *rawfile = ((MESSI_workerdata *)rfdata)->rawfile;

        while (1)
        {
            current_root_node_number = __sync_fetch_and_add(((MESSI_workerdata *)rfdata)->node_counter, 1);

            if (current_root_node_number >= ((MESSI_workerdata *)rfdata)->amountnode)
                break;
            current_root_node = ((MESSI_workerdata *)rfdata)->nodelist[current_root_node_number];

            insert_tree_node_m_hybridpqueue(paa, current_root_node, index, bsfdisntance, ((MESSI_workerdata *)rfdata)->allpq, ((MESSI_workerdata *)rfdata)->alllock, &tnumber, n_pqueue);
        }

        pthread_barrier_wait(((MESSI_workerdata *)rfdata)->lock_barrier);
        while (1)
        {
            pthread_mutex_lock(&(((MESSI_workerdata *)rfdata)->alllock[startqueuenumber]));
            n = (query_result *)pqueue_pop(((MESSI_workerdata *)rfdata)->allpq[startqueuenumber]);
            pthread_mutex_unlock(&(((MESSI_workerdata *)rfdata)->alllock[startqueuenumber]));
            if (n == NULL)
                break;

            bsfdisntance = pq_bsf->knn[pq_bsf->k - 1];

            if (n->distance > bsfdisntance || n->distance > minimum_distance)
            {
                break;
            }
            else
            {
                if (n->node->is_leaf)
                {

                    checks++;
                    calculate_node2_topk_inmemory(index, n->node, ts, paa, pq_bsf, ((MESSI_workerdata *)rfdata)->lock_bsf, rawfile);
                }
            }
            free(n);
        }

        if ((((MESSI_workerdata *)rfdata)->allqueuelabel[startqueuenumber]) == 1)
        {
            (((MESSI_workerdata *)rfdata)->allqueuelabel[startqueuenumber]) = 0;
            pthread_mutex_lock(&(((MESSI_workerdata *)rfdata)->alllock[startqueuenumber]));
            while ((n = (query_result *)pqueue_pop(((MESSI_workerdata *)rfdata)->allpq[startqueuenumber])))
            {
                free(n);
            }
            pthread_mutex_unlock(&(((MESSI_workerdata *)rfdata)->alllock[startqueuenumber]));
        }

        while (1)
        {
            int offset = rand() % n_pqueue;
            finished = true;
            for (int i = 0; i < n_pqueue; i++)
            {
                if ((((MESSI_workerdata *)rfdata)->allqueuelabel[i]) == 1)
                {
                    finished = false;
                    while (1)
                    {
                        pthread_mutex_lock(&(((MESSI_workerdata *)rfdata)->alllock[i]));
                        n = (query_result *)pqueue_pop(((MESSI_workerdata *)rfdata)->allpq[i]);
                        pthread_mutex_unlock(&(((MESSI_workerdata *)rfdata)->alllock[i]));
                        if (n == NULL)
                            break;
                        if (n->distance > bsfdisntance || n->distance > minimum_distance)
                        {
                            break;
                        }
                        else
                        {

                            if (n->node->is_leaf)
                            {
                                checks++;

                                calculate_node2_topk_inmemory(index, n->node, ts, paa, pq_bsf, ((MESSI_workerdata *)rfdata)->lock_bsf, rawfile);
                            }
                        }

                        free(n);
                    }

                    ((MESSI_workerdata *)rfdata)->allqueuelabel[i] = 0;
                }
            }
            if (finished)
            {
                break;
            }
        }

        return nullptr;
    }

    void *MESSI_topk_search_worker_DTW(void *rfdata)
    {
        isax_node *current_root_node;
        query_result *n;
        isax_index *index = ((MESSI_workerdata *)rfdata)->index;
        ts_type *paa = ((MESSI_workerdata *)rfdata)->paa;
        ts_type *paaU = ((MESSI_workerdata *)rfdata)->paaU;
        ts_type *paaL = ((MESSI_workerdata *)rfdata)->paaL;
        int warpWind = ((MESSI_workerdata *)rfdata)->warpWind;
        ts_type *ts = ((MESSI_workerdata *)rfdata)->ts;
        ts_type *uo = ((MESSI_workerdata *)rfdata)->uo;
        ts_type *lo = ((MESSI_workerdata *)rfdata)->lo;
        pqueue_t *pq = ((MESSI_workerdata *)rfdata)->pq;
        int n_pqueue = ((MESSI_workerdata *)rfdata)->n_pqueue;
        float *rawfile = ((MESSI_workerdata *)rfdata)->rawfile;
        query_result *do_not_remove = ((MESSI_workerdata *)rfdata)->bsf_result;
        float minimum_distance = ((MESSI_workerdata *)rfdata)->minimum_distance;
        int limit = ((MESSI_workerdata *)rfdata)->limit;
        int checks = 0;
        bool finished = true;
        int current_root_node_number;
        int tight_bound = index->settings->tight_bound;
        int aggressive_check = index->settings->aggressive_check;
        query_result *bsf_result = (((MESSI_workerdata *)rfdata)->bsf_result);
        pqueue_bsf *pq_bsf = ((MESSI_workerdata *)rfdata)->pq_bsf;
        float bsfdisntance = pq_bsf->knn[pq_bsf->k - 1];
        int calculate_node = 0, calculate_node_quque = 0;
        int tnumber = rand() % n_pqueue;
        int startqueuenumber = ((MESSI_workerdata *)rfdata)->startqueuenumber;

        while (1)
        {
            current_root_node_number = __sync_fetch_and_add(((MESSI_workerdata *)rfdata)->node_counter, 1);

            if (current_root_node_number >= ((MESSI_workerdata *)rfdata)->amountnode)
                break;
            current_root_node = ((MESSI_workerdata *)rfdata)->nodelist[current_root_node_number];

            insert_tree_node_m_hybridpqueue_DTW(paaU, paaL, current_root_node, index, bsfdisntance, ((MESSI_workerdata *)rfdata)->allpq, ((MESSI_workerdata *)rfdata)->alllock, &tnumber, n_pqueue);
        }

        pthread_barrier_wait(((MESSI_workerdata *)rfdata)->lock_barrier);

        while (1)
        {
            pthread_mutex_lock(&(((MESSI_workerdata *)rfdata)->alllock[startqueuenumber]));
            n = (query_result *)pqueue_pop(((MESSI_workerdata *)rfdata)->allpq[startqueuenumber]);
            pthread_mutex_unlock(&(((MESSI_workerdata *)rfdata)->alllock[startqueuenumber]));
            if (n == NULL)
                break;

            bsfdisntance = pq_bsf->knn[pq_bsf->k - 1];

            if (n->distance > bsfdisntance || n->distance > minimum_distance)
            {
                break;
            }
            else
            {

                if (n->node->is_leaf)
                {
                    checks++;

                    calculate_node_DTW2knn_inmemory(index, n->node, ts, uo, lo, paa, paaU, paaL, bsfdisntance, warpWind, pq_bsf, ((MESSI_workerdata *)rfdata)->lock_bsf, rawfile);
                }
            }
            free(n);
        }

        if ((((MESSI_workerdata *)rfdata)->allqueuelabel[startqueuenumber]) == 1)
        {
            (((MESSI_workerdata *)rfdata)->allqueuelabel[startqueuenumber]) = 0;
            pthread_mutex_lock(&(((MESSI_workerdata *)rfdata)->alllock[startqueuenumber]));
            while ((n = (query_result *)pqueue_pop(((MESSI_workerdata *)rfdata)->allpq[startqueuenumber])))
            {
                free(n);
            }
            pthread_mutex_unlock(&(((MESSI_workerdata *)rfdata)->alllock[startqueuenumber]));
        }

        while (1)
        {
            int offset = rand() % n_pqueue;
            finished = true;
            for (int i = 0; i < n_pqueue; i++)
            {
                if ((((MESSI_workerdata *)rfdata)->allqueuelabel[i]) == 1)
                {
                    finished = false;
                    while (1)
                    {
                        pthread_mutex_lock(&(((MESSI_workerdata *)rfdata)->alllock[i]));
                        n = (query_result *)pqueue_pop(((MESSI_workerdata *)rfdata)->allpq[i]);
                        pthread_mutex_unlock(&(((MESSI_workerdata *)rfdata)->alllock[i]));
                        if (n == NULL)
                            break;

                        bsfdisntance = pq_bsf->knn[pq_bsf->k - 1];
                        if (n->distance > bsfdisntance || n->distance > minimum_distance)
                        {
                            break;
                        }
                        else
                        {

                            if (n->node->is_leaf)
                            {
                                checks++;
                                calculate_node_DTW2knn_inmemory(index, n->node, ts, uo, lo, paa, paaU, paaL, bsfdisntance, warpWind, pq_bsf, ((MESSI_workerdata *)rfdata)->lock_bsf, rawfile);
                            }
                        }

                        free(n);
                    }

                    ((MESSI_workerdata *)rfdata)->allqueuelabel[i] = 0;
                }
            }
            if (finished)
            {
                break;
            }
        }

        return nullptr;
    }

    void calculate_node_range_inmemory(isax_index *index, isax_node *node, ts_type *query, ts_type *paa,
                                       float r, std::vector<std::pair<float, idx_t>> *results,
                                       pthread_rwlock_t *lock, float *rawfile)
    {
        if (node == NULL || node->buffer == NULL)
            return;

        for (int i = 0; i < node->buffer->full_buffer_size; i++)
        {
            float dist = ts_euclidean_distance_SIMD(query, node->buffer->full_ts_buffer[i],
                                                    index->settings->timeseries_size, r);
            if (dist <= r)
            {
                file_position_type pos = 0;
                if (node->buffer->full_position_buffer != nullptr &&
                    node->buffer->full_position_buffer[i] != nullptr)
                    pos = *node->buffer->full_position_buffer[i] / index->settings->timeseries_size;
                pthread_rwlock_wrlock(lock);
                results->emplace_back(dist, static_cast<idx_t>(pos));
                pthread_rwlock_unlock(lock);
            }
        }

        for (int i = 0; i < node->buffer->tmp_full_buffer_size; i++)
        {
            float dist = ts_euclidean_distance_SIMD(query, node->buffer->tmp_full_ts_buffer[i],
                                                    index->settings->timeseries_size, r);
            if (dist <= r)
            {
                file_position_type pos = 0;
                if (node->buffer->tmp_full_position_buffer != nullptr &&
                    node->buffer->tmp_full_position_buffer[i] != nullptr)
                    pos = *node->buffer->tmp_full_position_buffer[i] / index->settings->timeseries_size;
                pthread_rwlock_wrlock(lock);
                results->emplace_back(dist, static_cast<idx_t>(pos));
                pthread_rwlock_unlock(lock);
            }
        }

        for (int i = 0; i < node->buffer->partial_buffer_size; i++)
        {
            float distmin = minidist_paa_to_isax_rawa_SIMD(paa, node->buffer->partial_sax_buffer[i],
                                                           index->settings->max_sax_cardinalities,
                                                           index->settings->sax_bit_cardinality,
                                                           index->settings->sax_alphabet_cardinality,
                                                           index->settings->paa_segments, MINVAL, MAXVAL,
                                                           index->settings->mindist_sqrt);
            if (distmin <= r)
            {
                float dist = ts_euclidean_distance_SIMD(query, &(rawfile[*node->buffer->partial_position_buffer[i]]),
                                                        index->settings->timeseries_size, r);
                if (dist <= r)
                {
                    pthread_rwlock_wrlock(lock);
                    results->emplace_back(dist, static_cast<idx_t>(*node->buffer->partial_position_buffer[i] / index->settings->timeseries_size));
                    pthread_rwlock_unlock(lock);
                }
            }
        }
    }

    void *MESSI_range_search_worker_L2Squared(void *rfdata)
    {
        MESSI_workerdata *wd = (MESSI_workerdata *)rfdata;
        isax_index *index = wd->index;
        ts_type *paa = wd->paa;
        ts_type *ts = wd->ts;
        float r = wd->r;
        int n_pqueue = wd->n_pqueue;
        float *rawfile = wd->rawfile;
        int tnumber = rand() % n_pqueue;
        int startqueuenumber = wd->startqueuenumber;
        bool finished = true;
        query_result *n;

        while (1)
        {
            int current_root_node_number = __sync_fetch_and_add(wd->node_counter, 1);
            if (current_root_node_number >= wd->amountnode)
                break;
            insert_tree_node_m_hybridpqueue(paa, wd->nodelist[current_root_node_number], index, r,
                                            wd->allpq, wd->alllock, &tnumber, n_pqueue);
        }

        pthread_barrier_wait(wd->lock_barrier);

        while (1)
        {
            pthread_mutex_lock(&wd->alllock[startqueuenumber]);
            n = (query_result *)pqueue_pop(wd->allpq[startqueuenumber]);
            pthread_mutex_unlock(&wd->alllock[startqueuenumber]);
            if (n == NULL)
                break;
            if (n->distance > r)
                break;
            if (n->node->is_leaf)
                calculate_node_range_inmemory(index, n->node, ts, paa, r,
                                             wd->range_results, wd->lock_range_results, rawfile);
            free(n);
        }

        if (wd->allqueuelabel[startqueuenumber] == 1)
        {
            wd->allqueuelabel[startqueuenumber] = 0;
            pthread_mutex_lock(&wd->alllock[startqueuenumber]);
            while ((n = (query_result *)pqueue_pop(wd->allpq[startqueuenumber])))
                free(n);
            pthread_mutex_unlock(&wd->alllock[startqueuenumber]);
        }

        while (1)
        {
            finished = true;
            for (int i = 0; i < n_pqueue; i++)
            {
                if (wd->allqueuelabel[i] == 1)
                {
                    finished = false;
                    while (1)
                    {
                        pthread_mutex_lock(&wd->alllock[i]);
                        n = (query_result *)pqueue_pop(wd->allpq[i]);
                        pthread_mutex_unlock(&wd->alllock[i]);
                        if (n == NULL)
                            break;
                        if (n->distance > r)
                            break;
                        if (n->node->is_leaf)
                            calculate_node_range_inmemory(index, n->node, ts, paa, r,
                                                         wd->range_results, wd->lock_range_results, rawfile);
                        free(n);
                    }
                    wd->allqueuelabel[i] = 0;
                }
            }
            if (finished)
                break;
        }

        return nullptr;
    }

    Messi::Messi(DistanceType distance_type)
        : Messi(distance_type, MessiConfig{})
    {
    }

    Messi::Messi(DistanceType distance_type, const MessiConfig &config)
        : SimilaritySearchAlgorithm(distance_type)
    {
        this->search_workers = config.search_workers;
        this->index_workers = config.index_workers;
        this->warping_window = config.warping_window;
        this->leaf_size = config.leaf_size;
        this->paa_segments = config.paa_segments;
    }

    void *indexCreationWorker(void *transferdata)
    {
        sax_type *sax = (sax_type *)malloc(sizeof(sax_type) * ((buffer_data_inmemory *)transferdata)->index->settings->paa_segments);

        unsigned long start_number;
        unsigned long stop_number = ((buffer_data_inmemory *)transferdata)->stop_number;
        unsigned long roundfinishednumber = stop_number;
        file_position_type *pos = (file_position_type *)malloc(sizeof(file_position_type));
        isax_index *index = ((buffer_data_inmemory *)transferdata)->index;
        ts_type *ts = (ts_type *)malloc(sizeof(ts_type) * index->settings->timeseries_size);
        int paa_segments = ((buffer_data_inmemory *)transferdata)->index->settings->paa_segments;

        int read_block_length = ((buffer_data_inmemory *)transferdata)->read_block_length;

        unsigned long i = 0;
        float *raw_file = ((buffer_data_inmemory *)transferdata)->ts;
        while (1)
        {
            start_number = __sync_fetch_and_add(((buffer_data_inmemory *)transferdata)->shared_start_number, read_block_length);
            if (start_number > stop_number)
            {
                break;
            }
            else if (start_number > stop_number - read_block_length)
            {
                roundfinishednumber = stop_number;
            }
            else
            {
                roundfinishednumber = min(stop_number, (start_number + read_block_length));
            }
            for (i = start_number; i < roundfinishednumber; i++)
            {
                memcpy(ts, &(raw_file[i * index->settings->timeseries_size]), sizeof(ts_type) * index->settings->timeseries_size);
                if (sax_from_ts(ts, sax, index->settings->ts_values_per_paa_segment,
                                index->settings->paa_segments, index->settings->sax_alphabet_cardinality,
                                index->settings->sax_bit_cardinality) == SUCCESS)
                {
                    *pos = (file_position_type)(i * index->settings->timeseries_size);
                    memcpy(&(index->sax_cache[i * index->settings->paa_segments]), sax, sizeof(sax_type) * index->settings->paa_segments);

                    isax_pRecBuf_index_insert_inmemory(index, sax, pos, ((buffer_data_inmemory *)transferdata)->lock_firstnode, ((buffer_data_inmemory *)transferdata)->workernumber, ((buffer_data_inmemory *)transferdata)->total_workernumber);
                }
                else
                {
                    fprintf(stderr, "error: cannot insert record in index, since sax representation\
                    failed to be created");
                }
            }
        }

        free(pos);
        free(sax);
        free(ts);

        pthread_barrier_wait(((buffer_data_inmemory *)transferdata)->lock_barrier1);
        pthread_barrier_wait(((buffer_data_inmemory *)transferdata)->lock_barrier2);
        bool have_record = false;
        int j;
        isax_node_record *r = (isax_node_record *)malloc(sizeof(isax_node_record));

        while (1)
        {

            j = __sync_fetch_and_add(((buffer_data_inmemory *)transferdata)->node_counter, 1);

            if (j >= index->fbl->number_of_buffers)
            {
                break;
            }
            parallel_fbl_soft_buffer *current_fbl_node = &((parallel_first_buffer_layer *)(index->fbl))->soft_buffers[j];
            if (!current_fbl_node->initialized)
            {
                continue;
            }

            int i;
            have_record = false;
            for (int k = 0; k < ((buffer_data_inmemory *)transferdata)->total_workernumber; k++)
            {
                if (current_fbl_node->buffer_size[k] > 0)
                    have_record = true;
                for (i = 0; i < current_fbl_node->buffer_size[k]; i++)
                {
                    r->sax = (sax_type *)&(((current_fbl_node->sax_records[k]))[i * index->settings->paa_segments]);
                    r->position = (file_position_type *)&((file_position_type *)(current_fbl_node->pos_records[k]))[i];
                    r->insertion_mode = (insertion_mode)(NO_TMP | PARTIAL);

                    add_record_to_node(index, current_fbl_node->node, r, 1);
                }
            }
            if (have_record)
            {
                flush_subtree_leaf_buffers_inmemory(index, current_fbl_node->node);
            }
        }
        free(r);

        return NULL;
    }

    void Messi::buildIndex(DataSource *data_source)
    {
        this->dim = data_source->getDim();
        this->n_database = data_source->getTotalRecords();

        if (this->n_database == 0)
        {

            data_source->reset();
            idx_t count = 0;
            float *dummy = new float[this->dim];
            while (data_source->nextRecord(dummy))
            {
                count++;
            }
            delete[] dummy;
            this->n_database = count;
            data_source->reset();
        }

        data_source->reset();
        // If the datasource already wraps a contiguous in-memory buffer, reuse it
        // directly to avoid an unnecessary copy (100M*96 floats ~ 73 GB).
        const float *raw = data_source->rawPointer();
        if (raw != nullptr)
        {
            this->database = const_cast<float *>(raw); // stored as non-owning view
            this->owns_database = false;
        }
        else
        {
            this->database = new float[this->n_database * this->dim];
            float *record = new float[this->dim];
            idx_t idx = 0;
            while (data_source->nextRecord(record))
            {
                std::copy(record, record + this->dim, this->database + idx * this->dim);
                idx++;
            }
            delete[] record;
            this->owns_database = true;
        }

        this->index_settings = isax_index_settings_init("",
                                                        this->dim,
                                                        this->paa_segments,
                                                        this->sax_cardinality,
                                                        this->leaf_size,
                                                        this->min_leaf_size,
                                                        this->initial_lbl_size,
                                                        this->flush_limit,
                                                        this->initial_fbl_size,
                                                        this->total_loaded_leaves,
                                                        this->tight_bound,
                                                        0,
                                                        1,
                                                        1,
                                                        this->bp_mode);

        // Equi-depth breakpoints (no-op in Gaussian mode), installed as active.
        compute_equidepth_breakpoints(this->index_settings, this->database, this->n_database);
        activateBreakpoints();

        this->index = isax_index_init_inmemory(this->index_settings);
        isax_index *index = this->index;

        index->sax_file = NULL;
        long int ts_loaded = 0;
        unsigned long shared_start_number = 0;
        int i, j;
        int node_counter = 0;
        pthread_t threadid[this->index_workers];
        buffer_data_inmemory *input_data = (buffer_data_inmemory *)malloc(sizeof(buffer_data_inmemory) * (this->index_workers));
        index->sax_cache = (sax_type *)malloc(sizeof(sax_type) * index->settings->paa_segments * this->n_database);
        pthread_barrier_t lock_barrier1, lock_barrier2;
        pthread_barrier_init(&lock_barrier1, NULL, this->index_workers + 1);
        pthread_barrier_init(&lock_barrier2, NULL, this->index_workers + 1);

        pthread_mutex_t lock_record = PTHREAD_MUTEX_INITIALIZER, lockfbl = PTHREAD_MUTEX_INITIALIZER, lock_index = PTHREAD_MUTEX_INITIALIZER, lock_firstnode = PTHREAD_MUTEX_INITIALIZER, lock_disk = PTHREAD_MUTEX_INITIALIZER;

        destroy_fbl(index->fbl);
        index->fbl = (first_buffer_layer *)initialize_pRecBuf(index->settings->initial_fbl_buffer_size, pow(2, index->settings->paa_segments), index->settings->max_total_buffer_size + DISK_BUFFER_SIZE * (PROGRESS_CALCULATE_THREAD_NUMBER - 1), index, this->index_workers);

        int *nodeid = (int *)malloc(sizeof(int) * index->fbl->number_of_buffers);
        int *nodesize = (int *)malloc(sizeof(int) * index->fbl->number_of_buffers);

        for (i = 0; i < this->index_workers; i++)
        {
            input_data[i].index = index;
            input_data[i].lock_fbl = &lockfbl;
            input_data[i].lock_record = &lock_record;
            input_data[i].lock_firstnode = &lock_firstnode;
            input_data[i].lock_index = &lock_index;
            input_data[i].ts = this->database;
            input_data[i].lock_disk = &lock_disk;
            input_data[i].workernumber = i;
            input_data[i].total_workernumber = this->index_workers;
            input_data[i].start_number = i * (this->n_database / this->index_workers);
            input_data[i].shared_start_number = &shared_start_number;
            input_data[i].stop_number = this->n_database;
            input_data[i].node_counter = &node_counter;
            input_data[i].lock_barrier1 = &lock_barrier1;
            input_data[i].lock_barrier2 = &lock_barrier2;
            input_data[i].nodeid = nodeid;

            input_data[i].read_block_length = this->read_block_length;
        }

        for (i = 0; i < this->index_workers; i++)
        {
            pthread_create(&(threadid[i]), NULL, indexCreationWorker, (void *)&(input_data[i]));
        }

        pthread_barrier_wait(&lock_barrier1);
        pthread_barrier_wait(&lock_barrier2);

        for (i = 0; i < this->index_workers; i++)
        {
            pthread_join(threadid[i], NULL);
        }
        __sync_fetch_and_add(&(index->total_records), this->n_database);
        index->sax_cache_size = index->total_records;
        fprintf(stderr, ">>> Finished indexing\n");

        pthread_barrier_destroy(&lock_barrier1);
        pthread_barrier_destroy(&lock_barrier2);

        free(input_data);
        free(nodeid);
        free(nodesize);
    }

    void Messi::searchIndexL2Squared(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D)
    {
        ts_type *paa = (ts_type *)malloc(sizeof(ts_type) * index->settings->paa_segments);

        node_list nodelist;
        nodelist.nlist = (isax_node **)malloc(sizeof(isax_node *) * pow(2, index->settings->paa_segments));
        nodelist.node_amount = 0;
        isax_node *current_root_node = index->first_node;

        while (1)
        {
            if (current_root_node != NULL)
            {

                nodelist.nlist[nodelist.node_amount] = current_root_node;
                current_root_node = current_root_node->next;
                nodelist.node_amount++;
            }
            else
                break;
        }

        for (idx_t q_loaded = 0; q_loaded < n_query; q_loaded++)
        {

            const float *ts = query + q_loaded * this->dim;

            paa_from_ts(ts, paa, index->settings->paa_segments, index->settings->ts_values_per_paa_segment);

            pqueue_bsf result = MESSI_search_topk_L2Squared((float *)ts, paa, &nodelist, k);

            std::vector<std::pair<float, long>> pairs;
            pairs.reserve(static_cast<size_t>(k));
            for (idx_t ik = 0; ik < k; ik++)
            {
                if (result.position[ik] >= 0 && result.knn[ik] < FLT_MAX * 0.99f)
                    pairs.emplace_back(result.knn[ik], result.position[ik]);
            }
            std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b)
                      {
                if (a.first != b.first) return a.first < b.first;
                return a.second < b.second; });
            std::unordered_set<long> seen_pos;
            std::vector<std::pair<float, long>> uniq;
            uniq.reserve(pairs.size());
            for (const auto &p : pairs)
            {
                if (seen_pos.insert(p.second).second)
                    uniq.push_back(p);
            }
            long last_pos = 0;
            float last_dist = 0.0f;
            for (idx_t ik = 0; ik < k; ik++)
            {
                if (ik < static_cast<idx_t>(uniq.size()))
                {
                    last_dist = uniq[static_cast<size_t>(ik)].first;
                    last_pos = uniq[static_cast<size_t>(ik)].second;
                }
                I[q_loaded * k + ik] = static_cast<idx_t>(last_pos >= 0 ? last_pos : 0);
                D[q_loaded * k + ik] = last_dist;
            }

            free(result.position);
            free(result.knn);
            free(result.node);
        }

        free(nodelist.nlist);
        free(paa);
        fprintf(stderr, ">>> Finished querying.\n");
        fflush(stdout);
    }

    void Messi::searchIndexDTW(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D)
    {
        isax_index *index = this->index;

        node_list nodelist;
        nodelist.nlist = (isax_node **)malloc(sizeof(isax_node *) * pow(2, index->settings->paa_segments));
        nodelist.node_amount = 0;
        isax_node *current_root_node = index->first_node;
        while (1)
        {
            if (current_root_node != NULL)
            {
                nodelist.nlist[nodelist.node_amount] = current_root_node;
                current_root_node = current_root_node->next;
                nodelist.node_amount++;
            }
            else
            {
                break;
            }
        }

        for (idx_t q_loaded = 0; q_loaded < n_query; q_loaded++)
        {

            float *ts = (float *)&query[q_loaded * this->dim];

            pqueue_bsf result = MESSI_search_topk_DTW((float *)ts, &nodelist, k);

            std::vector<std::pair<float, long>> pairs;
            pairs.reserve(static_cast<size_t>(k));
            for (idx_t ik = 0; ik < k; ik++)
            {
                if (result.position[ik] >= 0 && result.knn[ik] < FLT_MAX * 0.99f)
                    pairs.emplace_back(result.knn[ik], result.position[ik]);
            }
            std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b)
                      {
                if (a.first != b.first) return a.first < b.first;
                return a.second < b.second; });
            std::unordered_set<long> seen_pos;
            std::vector<std::pair<float, long>> uniq;
            uniq.reserve(pairs.size());
            for (const auto &p : pairs)
            {
                if (seen_pos.insert(p.second).second)
                    uniq.push_back(p);
            }
            long last_pos = 0;
            float last_dist = 0.0f;
            for (idx_t ik = 0; ik < k; ik++)
            {
                if (ik < static_cast<idx_t>(uniq.size()))
                {
                    last_dist = uniq[static_cast<size_t>(ik)].first;
                    last_pos = uniq[static_cast<size_t>(ik)].second;
                }
                I[q_loaded * k + ik] = static_cast<idx_t>(last_pos >= 0 ? last_pos : 0);
                D[q_loaded * k + ik] = last_dist;
            }

            free(result.position);
            free(result.knn);
            free(result.node);
        }

        free(nodelist.nlist);
        fprintf(stderr, ">>> Finished querying.\n");
    }

    std::vector<std::pair<float, idx_t>> Messi::MESSI_search_range_L2Squared(ts_type *ts, ts_type *paa, node_list *nodelist, float r)
    {
        std::vector<std::pair<float, idx_t>> results;
        pthread_rwlock_t lock_results = PTHREAD_RWLOCK_INITIALIZER;

        int node_counter = 0;
        pqueue_t **allpq = (pqueue_t **)malloc(sizeof(pqueue_t *) * this->n_pqueue);
        pthread_mutex_t ququelock[this->n_pqueue];
        int queuelabel[this->n_pqueue];

        pthread_t threadid[this->search_workers];
        MESSI_workerdata workerdata[this->search_workers];
        pthread_mutex_t lock_queue = PTHREAD_MUTEX_INITIALIZER;
        pthread_mutex_t lock_current_root_node = PTHREAD_MUTEX_INITIALIZER;
        pthread_barrier_t lock_barrier;
        pthread_barrier_init(&lock_barrier, NULL, this->search_workers);

        for (int i = 0; i < this->n_pqueue; i++)
        {
            allpq[i] = pqueue_init(index->settings->root_nodes_size / this->n_pqueue, cmp_pri, get_pri, set_pri, get_pos, set_pos);
            pthread_mutex_init(&ququelock[i], NULL);
            queuelabel[i] = 1;
        }

        for (int i = 0; i < this->search_workers; i++)
        {
            workerdata[i].paa = paa;
            workerdata[i].ts = ts;
            workerdata[i].lock_queue = &lock_queue;
            workerdata[i].lock_current_root_node = &lock_current_root_node;
            workerdata[i].nodelist = nodelist->nlist;
            workerdata[i].amountnode = nodelist->node_amount;
            workerdata[i].index = index;
            workerdata[i].node_counter = &node_counter;
            workerdata[i].lock_barrier = &lock_barrier;
            workerdata[i].alllock = ququelock;
            workerdata[i].allqueuelabel = queuelabel;
            workerdata[i].allpq = allpq;
            workerdata[i].startqueuenumber = i % this->n_pqueue;
            workerdata[i].n_pqueue = this->n_pqueue;
            workerdata[i].rawfile = this->database;
            workerdata[i].r = r;
            workerdata[i].range_results = &results;
            workerdata[i].lock_range_results = &lock_results;
        }

        for (int i = 0; i < this->search_workers; i++)
            pthread_create(&(threadid[i]), NULL, MESSI_range_search_worker_L2Squared, (void *)&(workerdata[i]));
        for (int i = 0; i < this->search_workers; i++)
            pthread_join(threadid[i], NULL);

        pthread_barrier_destroy(&lock_barrier);
        pthread_rwlock_destroy(&lock_results);

        for (int i = 0; i < this->n_pqueue; i++)
            pqueue_free(allpq[i]);
        free(allpq);

        return results;
    }

    void Messi::searchIndex(const float *query, idx_t n_query, const SearchConfig &config,
                            std::vector<std::vector<idx_t>> &I,
                            std::vector<std::vector<float>> &D)
    {
        activateBreakpoints();
        if (config.type == QueryType::TOP_K)
        {
            SimilaritySearchAlgorithm::searchIndex(query, n_query, config, I, D);
            return;
        }

        ts_type *paa = (ts_type *)malloc(sizeof(ts_type) * index->settings->paa_segments);
        node_list nodelist;
        nodelist.nlist = (isax_node **)malloc(sizeof(isax_node *) * (int)pow(2, index->settings->paa_segments));
        nodelist.node_amount = 0;
        isax_node *current_root_node = index->first_node;
        while (current_root_node != NULL)
        {
            nodelist.nlist[nodelist.node_amount++] = current_root_node;
            current_root_node = current_root_node->next;
        }

        I.resize(n_query);
        D.resize(n_query);

        for (idx_t q_loaded = 0; q_loaded < n_query; q_loaded++)
        {
            const float *ts = query + q_loaded * this->dim;
            paa_from_ts(ts, paa, index->settings->paa_segments, index->settings->ts_values_per_paa_segment);

            std::vector<std::pair<float, idx_t>> hits = MESSI_search_range_L2Squared((float *)ts, paa, &nodelist, config.r);
            std::sort(hits.begin(), hits.end());

            I[q_loaded].resize(hits.size());
            D[q_loaded].resize(hits.size());
            for (size_t j = 0; j < hits.size(); j++)
            {
                D[q_loaded][j] = hits[j].first;
                I[q_loaded][j] = hits[j].second;
            }
        }

        free(nodelist.nlist);
        free(paa);
    }

    void Messi::searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D)
    {
        activateBreakpoints();
        if (this->distance_type == DistanceType::L2_SQUARED)
        {
            searchIndexL2Squared(query, n_query, k, I, D);
        }
        else if (this->distance_type == DistanceType::DTW)
        {
            searchIndexDTW(query, n_query, k, I, D);
        }
        else
        {
            fprintf(stderr, "Error: Unsupported distance type for Messi index.\n");
            exit(1);
        }
    }

    pqueue_bsf Messi::MESSI_search_topk_L2Squared(ts_type *ts, ts_type *paa, node_list *nodelist, idx_t k)
    {
        pqueue_bsf *pq_bsf = pqueue_bsf_init(k);

        approximate_topk_inmemory(ts, paa, index, pq_bsf, this->database);
        this->minimum_distance = pq_bsf->knn[k - 1];

        int tight_bound = index->settings->tight_bound;
        int aggressive_check = index->settings->aggressive_check;
        int node_counter = 0;

        if (this->minimum_distance == FLT_MAX || min_checked_leaves > 1)
        {
            refine_topk_answer_inmemory(ts, paa, index, pq_bsf, this->minimum_distance, this->min_checked_leaves, this->database);
            this->minimum_distance = pq_bsf->knn[k - 1];
        }
        pqueue_t **allpq = (pqueue_t **)malloc(sizeof(pqueue_t *) * this->n_pqueue);

        pthread_mutex_t ququelock[this->n_pqueue];
        int queuelabel[this->n_pqueue];

        isax_node *current_root_node = index->first_node;

        pthread_t threadid[this->search_workers];
        MESSI_workerdata workerdata[this->search_workers];
        pthread_mutex_t lock_queue = PTHREAD_MUTEX_INITIALIZER, lock_current_root_node = PTHREAD_MUTEX_INITIALIZER;
        pthread_rwlock_t lock_bsf = PTHREAD_RWLOCK_INITIALIZER;
        pthread_barrier_t lock_barrier;
        pthread_barrier_init(&lock_barrier, NULL, this->search_workers);

        for (int i = 0; i < this->n_pqueue; i++)
        {
            allpq[i] = pqueue_init(index->settings->root_nodes_size / this->n_pqueue, cmp_pri, get_pri, set_pri, get_pos, set_pos);
            pthread_mutex_init(&ququelock[i], NULL);
            queuelabel[i] = 1;
        }

        for (int i = 0; i < this->search_workers; i++)
        {
            workerdata[i].paa = paa;
            workerdata[i].ts = ts;
            workerdata[i].lock_queue = &lock_queue;
            workerdata[i].lock_current_root_node = &lock_current_root_node;
            workerdata[i].lock_bsf = &lock_bsf;
            workerdata[i].nodelist = nodelist->nlist;
            workerdata[i].amountnode = nodelist->node_amount;
            workerdata[i].index = index;
            workerdata[i].minimum_distance = minimum_distance;
            workerdata[i].node_counter = &node_counter;
            workerdata[i].pq = allpq[i];
            workerdata[i].lock_barrier = &lock_barrier;
            workerdata[i].alllock = ququelock;
            workerdata[i].allqueuelabel = queuelabel;
            workerdata[i].allpq = allpq;
            workerdata[i].startqueuenumber = i % this->n_pqueue;
            workerdata[i].pq_bsf = pq_bsf;

            workerdata[i].n_pqueue = this->n_pqueue;
            workerdata[i].rawfile = this->database;
        }

        query_result *n;

        for (int i = 0; i < this->search_workers; i++)
        {
            pthread_create(&(threadid[i]), NULL, MESSI_topk_search_worker_L2Squared, (void *)&(workerdata[i]));
        }
        for (int i = 0; i < this->search_workers; i++)
        {
            pthread_join(threadid[i], NULL);
        }
        this->minimum_distance = pq_bsf->knn[k - 1];

        pthread_barrier_destroy(&lock_barrier);

        for (int i = 0; i < this->n_pqueue; i++)
        {
            pqueue_free(allpq[i]);
        }
        free(allpq);

        pqueue_bsf result = *pq_bsf;
        free(pq_bsf);
        return result;
    }

    pqueue_bsf Messi::MESSI_search_topk_DTW(ts_type *ts, node_list *nodelist, idx_t k)
    {
        isax_index *index = this->index;
        int warpWind = this->warping_window;
        float *rawfile = this->database;

        ts_type *paa = (ts_type *)malloc(sizeof(ts_type) * index->settings->paa_segments);
        paa_from_ts(ts, paa, index->settings->paa_segments, index->settings->ts_values_per_paa_segment);

        pqueue_bsf *pq_bsf = pqueue_bsf_init(k);
        approximate_DTWtopk_inmemory(ts, paa, index, warpWind, pq_bsf, rawfile);
        this->minimum_distance = pq_bsf->knn[k - 1];

        int tight_bound = index->settings->tight_bound;
        int aggressive_check = index->settings->aggressive_check;
        int node_counter = 0;

        pqueue_t **allpq = (pqueue_t **)malloc(sizeof(pqueue_t *) * this->n_pqueue);

        ts_type *upperLemire = (ts_type *)malloc(sizeof(ts_type) * index->settings->timeseries_size);
        ts_type *lowerLemire = (ts_type *)malloc(sizeof(ts_type) * index->settings->timeseries_size);
        ts_type *paaU = (ts_type *)malloc(sizeof(ts_type) * index->settings->paa_segments);
        ts_type *paaL = (ts_type *)malloc(sizeof(ts_type) * index->settings->paa_segments);

        lower_upper_lemire(ts, index->settings->timeseries_size, warpWind, lowerLemire, upperLemire);
        paa_from_ts(upperLemire, paaU, index->settings->paa_segments, index->settings->ts_values_per_paa_segment);
        paa_from_ts(lowerLemire, paaL, index->settings->paa_segments, index->settings->ts_values_per_paa_segment);

        pthread_mutex_t ququelock[this->n_pqueue];
        int queuelabel[this->n_pqueue];

        isax_node *current_root_node = index->first_node;
        pthread_t threadid[this->search_workers];
        MESSI_workerdata workerdata[this->search_workers];
        pthread_mutex_t lock_queue = PTHREAD_MUTEX_INITIALIZER, lock_current_root_node = PTHREAD_MUTEX_INITIALIZER;
        pthread_rwlock_t lock_bsf = PTHREAD_RWLOCK_INITIALIZER;
        pthread_barrier_t lock_barrier;
        pthread_barrier_init(&lock_barrier, NULL, this->search_workers);

        for (int i = 0; i < this->n_pqueue; i++)
        {
            allpq[i] = pqueue_init(index->settings->root_nodes_size / this->n_pqueue, cmp_pri, get_pri, set_pri, get_pos, set_pos);
            pthread_mutex_init(&ququelock[i], NULL);
            queuelabel[i] = 1;
        }

        for (int i = 0; i < this->search_workers; i++)
        {
            workerdata[i].paa = paa;
            workerdata[i].paaU = paaU;
            workerdata[i].paaL = paaL;
            workerdata[i].ts = ts;
            workerdata[i].uo = upperLemire;
            workerdata[i].lo = lowerLemire;
            workerdata[i].lock_queue = &lock_queue;
            workerdata[i].lock_current_root_node = &lock_current_root_node;
            workerdata[i].lock_bsf = &lock_bsf;
            workerdata[i].nodelist = nodelist->nlist;
            workerdata[i].amountnode = nodelist->node_amount;
            workerdata[i].index = index;
            workerdata[i].minimum_distance = minimum_distance;
            workerdata[i].node_counter = &node_counter;
            workerdata[i].pq = allpq[i];

            workerdata[i].lock_barrier = &lock_barrier;
            workerdata[i].alllock = ququelock;
            workerdata[i].allqueuelabel = queuelabel;
            workerdata[i].allpq = allpq;
            workerdata[i].startqueuenumber = i % this->n_pqueue;
            workerdata[i].warpWind = warpWind;
            workerdata[i].pq_bsf = pq_bsf;

            workerdata[i].n_pqueue = this->n_pqueue;
            workerdata[i].rawfile = this->database;
        }

        for (int i = 0; i < this->search_workers; i++)
        {
            pthread_create(&(threadid[i]), NULL, MESSI_topk_search_worker_DTW, (void *)&(workerdata[i]));
        }
        for (int i = 0; i < this->search_workers; i++)
        {
            pthread_join(threadid[i], NULL);
        }
        this->minimum_distance = pq_bsf->knn[k - 1];

        pthread_barrier_destroy(&lock_barrier);

        for (int i = 0; i < this->n_pqueue; i++)
        {
            pqueue_free(allpq[i]);
        }
        free(allpq);
        free(upperLemire);
        free(lowerLemire);
        free(paaU);
        free(paaL);
        free(paa);

        pqueue_bsf result = *pq_bsf;
        free(pq_bsf);
        return result;
    }

    Messi::~Messi()
    {
        if (owns_database && database != nullptr)
        {
            delete[] database;
        }

        if (index != nullptr)
        {
            if (index->sax_cache != nullptr)
            {
                free(index->sax_cache);
            }
            if (index->answer != nullptr)
            {
                free(index->answer);
            }
            if (index->fbl != nullptr)
            {

                destroy_parallel_fbl((parallel_first_buffer_layer *)index->fbl);
            }
            if (index->sax_file != nullptr)
            {
                fclose(index->sax_file);
            }
            free(index);
        }

        if (index_settings != nullptr)
        {
            if (index_settings->bit_masks != nullptr)
            {
                free(index_settings->bit_masks);
            }
            if (index_settings->max_sax_cardinalities != nullptr)
            {
                free(index_settings->max_sax_cardinalities);
            }
            free(index_settings);
        }
    }
}
