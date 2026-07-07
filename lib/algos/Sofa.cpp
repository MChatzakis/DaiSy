#include "Sofa.hpp"

#ifdef SOFA_FFTW_ENABLED

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <unordered_set>
#include <vector>

#include <fftw3.h>
#include <pthread.h>

#include "../isax/iSAXPqueue.hpp"
#include "../isax/iSAXIndex.hpp"
#include "../isax/iSAXSearch.hpp"
#include "../distance_computers/DistanceComputer.hpp"

namespace daisy
{

// ---- sofa_fft_from_ts / sofa_sfa_from_fft (from sofa/indexing.cpp) ----

void sofa_fft_from_ts(
    int ts_length, float norm_factor, bool is_norm,
    fftwf_complex *ts_out, float *transform, fftwf_plan plan_forward,
    int coeff_number)
{
    fftwf_execute(plan_forward);
    ts_out[0][1] = 0.0f;

    int j = 0;
    int start_offset = is_norm ? 1 : 0;
    for (int k = start_offset; k < coeff_number / 2 + start_offset; ++k, j += 2) {
        transform[j] = ts_out[k][0];
        transform[j + 1] = ts_out[k][1] * -1.0f;
    }
    for (int i = 0; i < coeff_number; ++i)
        transform[i] *= norm_factor;
}

void sofa_sfa_from_fft(
    float *transform, sax_type *sax_out, float **bins,
    int paa_segments, int cardinality)
{
    for (int k = 0; k < paa_segments; ++k) {
        unsigned int c = 0;
        for (; c < (unsigned int)(cardinality - 1); ++c)
            if (transform[k] < bins[k][c]) break;
        sax_out[k] = (sax_type)c;
    }
}

static void *sofa_bins_worker(void *transferdata)
{
    SOFA_bins_worker *d = (SOFA_bins_worker *)transferdata;
    int ts_length = d->index->settings->timeseries_size;

    for (long i = 0; i < d->records; ++i) {
        unsigned long db_idx = d->start_number + (unsigned long)i;
        if (db_idx >= d->stop_number) break;

        memcpy(d->ts_buf, &d->raw_database[db_idx * ts_length], sizeof(float) * ts_length);
        sofa_fft_from_ts(ts_length, d->norm_factor, d->is_norm,
                         d->ts_out, d->fft_transform, d->plan_forward, d->coeff_number);

        long slot = (long)d->workernumber * d->records_offset + i;
        for (int j = 0; j < d->coeff_number; ++j)
            d->dft_mem_array[j][slot] = d->fft_transform[j];
    }
    return nullptr;
}

static int sofa_compare_float(const void *a, const void *b)
{
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa < fb) ? -1 : (fa > fb) ? 1 : 0;
}

static void *sofa_divide_worker(void *transferdata)
{
    SOFA_divide_worker_data *d = (SOFA_divide_worker_data *)transferdata;

    for (unsigned long j = d->start_number; j < d->stop_number; ++j)
        qsort(d->dft_mem_array[j], d->sample_size, sizeof(float), sofa_compare_float);

    for (unsigned long i = d->start_number; i < d->stop_number; ++i) {
        float *col = d->dft_mem_array[i];
        if (d->histogram_type == 1) {
            float depth = (float)d->sample_size / (float)d->num_symbols;
            float bin_index = 0.0f;
            for (int j = 0; j < d->num_symbols - 1; ++j) {
                bin_index += depth;
                d->bins[i][j] = col[(int)bin_index];
            }
        } else {
            float first = col[0];
            float last = col[d->sample_size - 1];
            float width = (last - first) / (float)d->num_symbols;
            for (int j = 0; j < d->num_symbols - 1; ++j)
                d->bins[i][j] = width * (float)(j + 1) + first;
        }
    }
    return nullptr;
}

void *sofa_index_creation_worker(void *transferdata)
{
    SOFA_index_worker  *d = (SOFA_index_worker *)transferdata;
    isax_index         *index = d->index;
    int ts_length = index->settings->timeseries_size;
    int paa_segs = index->settings->paa_segments;

    sax_type           *sax = (sax_type *)malloc(sizeof(sax_type) * paa_segs);
    file_position_type *pos = (file_position_type *)malloc(sizeof(file_position_type));

    unsigned long roundfinish;
    while (1) {
        unsigned long start = __sync_fetch_and_add(d->shared_start_number,
                                                   (unsigned long)d->read_block_length);
        if (start > d->stop_number) break;
        roundfinish = (start > d->stop_number - (unsigned long)d->read_block_length)
            ? d->stop_number
            : std::min(d->stop_number, start + (unsigned long)d->read_block_length);

        for (unsigned long i = start; i < roundfinish; i++) {
            memcpy(d->ts_buf, &d->raw_database[i * ts_length], sizeof(float) * ts_length);
            sofa_fft_from_ts(ts_length, d->norm_factor, d->is_norm,
                             d->ts_out, d->fft_transform, d->plan_forward, paa_segs);
            sofa_sfa_from_fft(d->fft_transform, sax, d->bins,
                              paa_segs, index->settings->sax_alphabet_cardinality);

            *pos = (file_position_type)(i * ts_length);
            memcpy(&index->sax_cache[i * paa_segs], sax, sizeof(sax_type) * paa_segs);
            isax_pRecBuf_index_insert_inmemory(index, sax, pos,
                d->lock_firstnode, d->workernumber, d->total_workernumber);
        }
    }

    free(sax);
    free(pos);

    pthread_barrier_wait(d->lock_barrier1);
    pthread_barrier_wait(d->lock_barrier2);

    isax_node_record *r = (isax_node_record *)malloc(sizeof(isax_node_record));
    while (1) {
        int j = __sync_fetch_and_add(d->node_counter, 1);
        if (j >= index->fbl->number_of_buffers) break;

        parallel_fbl_soft_buffer *cur =
            &((parallel_first_buffer_layer *)(index->fbl))->soft_buffers[j];
        if (!cur->initialized) continue;

        bool have_record = false;
        for (int k = 0; k < d->total_workernumber; k++) {
            if (cur->buffer_size[k] > 0) have_record = true;
            for (int ii = 0; ii < cur->buffer_size[k]; ii++) {
                r->sax = (sax_type *)&(cur->sax_records[k][ii * paa_segs]);
                r->position = (file_position_type *)
                              &((file_position_type *)(cur->pos_records[k]))[ii];
                r->insertion_mode = (insertion_mode)(NO_TMP | PARTIAL);
                add_record_to_node(index, cur->node, r, 1);
            }
        }
        if (have_record)
            flush_subtree_leaf_buffers_inmemory(index, cur->node);
    }
    free(r);
    return nullptr;
}

// ---- sofa lb / minidist (from sofa/Sofa.cpp) ----

static inline float sofa_lb_segment(
    const float *bins_row, float fft_val,
    sax_type sax_val, sax_type sax_card, sax_type max_bit_card,
    int max_card, float factor)
{
    int rl = (int)sax_val << (max_bit_card - sax_card);
    unsigned int shift = (unsigned int)(max_bit_card - sax_card);
    int mask = (shift < 32u) ? ~(0xFFFFFFFF << shift) : 0;
    int ru = mask | rl;

    float bp_lo = (rl == 0)            ? -FLT_MAX : bins_row[rl - 1];
    float bp_hi = (ru >= max_card - 1) ?  FLT_MAX : bins_row[ru];
    float d = 0.0f;
    if      (bp_lo > fft_val) d = bp_lo - fft_val;
    else if (bp_hi < fft_val) d = fft_val - bp_hi;
    return factor * d * d;
}

static float sofa_minidist(
    float **bins, float *fft, sax_type *sax, sax_type *sax_cards,
    sax_type max_bit_card, int max_card, int n_segments, bool is_norm)
{
    float dist = 0.0f;
    int i = 0;
    if (!is_norm) {
        dist += sofa_lb_segment(bins[0], fft[0], sax[0], sax_cards[0],
                                max_bit_card, max_card, 0.5f);
        i = 2;
    }
    for (; i < n_segments; i++)
        dist += sofa_lb_segment(bins[i], fft[i], sax[i], sax_cards[i],
                                max_bit_card, max_card, 1.0f);
    return dist;
}

void insert_tree_node_sofa(
    float **bins, bool is_norm,
    float *fft, isax_node *node, isax_index *index, float bsf,
    pqueue_t **pq, pthread_mutex_t *lock_queue, int *tnumber, int n_pqueue)
{
    if (node == NULL || node->isax_values == NULL || node->isax_cardinalities == NULL)
        return;

    float distance = sofa_minidist(bins, fft,
        node->isax_values, node->isax_cardinalities,
        index->settings->sax_bit_cardinality,
        index->settings->sax_alphabet_cardinality,
        index->settings->paa_segments, is_norm);

    if (distance < bsf) {
        if (node->is_leaf) {
            query_result *r = (query_result *)malloc(sizeof(query_result));
            r->node = node;
            r->distance = distance;
            pthread_mutex_lock(&lock_queue[*tnumber]);
            pqueue_insert(pq[*tnumber], r);
            pthread_mutex_unlock(&lock_queue[*tnumber]);
            *tnumber = (*tnumber + 1) % n_pqueue;
        } else {
            if (node->left_child  && node->left_child->isax_values  && node->left_child->isax_cardinalities)
                insert_tree_node_sofa(bins, is_norm, fft, node->left_child,  index, bsf, pq, lock_queue, tnumber, n_pqueue);
            if (node->right_child && node->right_child->isax_values && node->right_child->isax_cardinalities)
                insert_tree_node_sofa(bins, is_norm, fft, node->right_child, index, bsf, pq, lock_queue, tnumber, n_pqueue);
        }
    }
}

void calculate_node_topk_sofa(
    isax_index *index, isax_node *node, ts_type *query, float *fft,
    float **bins, bool is_norm, pqueue_bsf *pq_bsf,
    pthread_rwlock_t *lock_bsf, float *rawfile)
{
    if (node == NULL || node->buffer == NULL) return;

    for (int i = 0; i < node->buffer->full_buffer_size; i++) {
        float dist = ts_euclidean_distance_SIMD(query, node->buffer->full_ts_buffer[i],
            index->settings->timeseries_size, pq_bsf->knn[pq_bsf->k - 1]);
        if (dist <= pq_bsf->knn[pq_bsf->k - 1]) {
            file_position_type pos = 0;
            if (node->buffer->full_position_buffer && node->buffer->full_position_buffer[i])
                pos = *node->buffer->full_position_buffer[i] / index->settings->timeseries_size;
            pthread_rwlock_wrlock(lock_bsf);
            pqueue_bsf_insert(pq_bsf, dist, static_cast<long>(pos), node);
            pthread_rwlock_unlock(lock_bsf);
        }
    }
    for (int i = 0; i < node->buffer->tmp_full_buffer_size; i++) {
        float dist = ts_euclidean_distance_SIMD(query, node->buffer->tmp_full_ts_buffer[i],
            index->settings->timeseries_size, pq_bsf->knn[pq_bsf->k - 1]);
        if (dist <= pq_bsf->knn[pq_bsf->k - 1]) {
            file_position_type pos = 0;
            if (node->buffer->tmp_full_position_buffer && node->buffer->tmp_full_position_buffer[i])
                pos = *node->buffer->tmp_full_position_buffer[i] / index->settings->timeseries_size;
            pthread_rwlock_wrlock(lock_bsf);
            pqueue_bsf_insert(pq_bsf, dist, static_cast<long>(pos), node);
            pthread_rwlock_unlock(lock_bsf);
        }
    }
    for (int i = 0; i < node->buffer->partial_buffer_size; i++) {
        if (node->buffer->partial_position_buffer[i] == nullptr) continue;
        float distmin = sofa_minidist(bins, fft,
            node->buffer->partial_sax_buffer[i],
            index->settings->max_sax_cardinalities,
            index->settings->sax_bit_cardinality,
            index->settings->sax_alphabet_cardinality,
            index->settings->paa_segments, is_norm);
        if (distmin <= pq_bsf->knn[pq_bsf->k - 1]) {
            float dist = ts_euclidean_distance_SIMD(query,
                &rawfile[*node->buffer->partial_position_buffer[i]],
                index->settings->timeseries_size, pq_bsf->knn[pq_bsf->k - 1]);
            if (dist <= pq_bsf->knn[pq_bsf->k - 1]) {
                pthread_rwlock_wrlock(lock_bsf);
                pqueue_bsf_insert(pq_bsf, dist,
                    *node->buffer->partial_position_buffer[i] /
                        index->settings->timeseries_size, node);
                pthread_rwlock_unlock(lock_bsf);
            }
        }
    }
}

void *SOFA_topk_search_worker(void *rfdata)
{
    SOFA_workerdata *d = (SOFA_workerdata *)rfdata;
    isax_index *index = d->index;
    float  *fft = d->fft;
    ts_type *ts = d->ts;
    pqueue_bsf *pq_bsf = d->pq_bsf;
    float minimum_distance = d->minimum_distance;
    int n_pqueue = d->n_pqueue;
    bool is_norm = d->is_norm;
    float **bins = d->bins;
    float  *rawfile = d->rawfile;

    float bsfdist = pq_bsf->knn[pq_bsf->k - 1];
    int tnumber = rand() % n_pqueue;
    int startq = d->startqueuenumber;

    while (1) {
        int cur = __sync_fetch_and_add(d->node_counter, 1);
        if (cur >= d->amountnode) break;
        insert_tree_node_sofa(bins, is_norm, fft,
            d->nodelist[cur], index, bsfdist,
            d->allpq, d->alllock, &tnumber, n_pqueue);
    }

    pthread_barrier_wait(d->lock_barrier);

    query_result *n;
    while (1) {
        pthread_mutex_lock(&d->alllock[startq]);
        n = (query_result *)pqueue_pop(d->allpq[startq]);
        pthread_mutex_unlock(&d->alllock[startq]);
        if (n == NULL) break;

        bsfdist = pq_bsf->knn[pq_bsf->k - 1];
        if (n->distance > bsfdist || n->distance > minimum_distance) { free(n); break; }
        if (n->node->is_leaf)
            calculate_node_topk_sofa(index, n->node, ts, fft, bins, is_norm,
                                     pq_bsf, d->lock_bsf, rawfile);
        free(n);
    }

    if (d->allqueuelabel[startq] == 1) {
        d->allqueuelabel[startq] = 0;
        pthread_mutex_lock(&d->alllock[startq]);
        while ((n = (query_result *)pqueue_pop(d->allpq[startq]))) free(n);
        pthread_mutex_unlock(&d->alllock[startq]);
    }

    while (1) {
        bool finished = true;
        for (int i = 0; i < n_pqueue; i++) {
            if (d->allqueuelabel[i] != 1) continue;
            finished = false;
            while (1) {
                pthread_mutex_lock(&d->alllock[i]);
                n = (query_result *)pqueue_pop(d->allpq[i]);
                pthread_mutex_unlock(&d->alllock[i]);
                if (n == NULL) break;
                bsfdist = pq_bsf->knn[pq_bsf->k - 1];
                if (n->distance > bsfdist || n->distance > minimum_distance) { free(n); break; }
                if (n->node->is_leaf)
                    calculate_node_topk_sofa(index, n->node, ts, fft, bins, is_norm,
                                             pq_bsf, d->lock_bsf, rawfile);
                free(n);
            }
            d->allqueuelabel[i] = 0;
        }
        if (finished) break;
    }
    return nullptr;
}

void calculate_node_range_sofa(
    isax_index *index, isax_node *node, ts_type *query, float *fft,
    float **bins, bool is_norm, float r,
    std::vector<std::pair<float, idx_t>> *results,
    pthread_rwlock_t *lock_results, float *rawfile)
{
    if (node == NULL || node->buffer == NULL) return;

    for (int i = 0; i < node->buffer->full_buffer_size; i++) {
        float dist = ts_euclidean_distance_SIMD(query, node->buffer->full_ts_buffer[i],
            index->settings->timeseries_size, r);
        if (dist <= r) {
            file_position_type pos = 0;
            if (node->buffer->full_position_buffer && node->buffer->full_position_buffer[i])
                pos = *node->buffer->full_position_buffer[i] / index->settings->timeseries_size;
            pthread_rwlock_wrlock(lock_results);
            results->emplace_back(dist, (idx_t)pos);
            pthread_rwlock_unlock(lock_results);
        }
    }
    for (int i = 0; i < node->buffer->tmp_full_buffer_size; i++) {
        float dist = ts_euclidean_distance_SIMD(query, node->buffer->tmp_full_ts_buffer[i],
            index->settings->timeseries_size, r);
        if (dist <= r) {
            file_position_type pos = 0;
            if (node->buffer->tmp_full_position_buffer && node->buffer->tmp_full_position_buffer[i])
                pos = *node->buffer->tmp_full_position_buffer[i] / index->settings->timeseries_size;
            pthread_rwlock_wrlock(lock_results);
            results->emplace_back(dist, (idx_t)pos);
            pthread_rwlock_unlock(lock_results);
        }
    }
    for (int i = 0; i < node->buffer->partial_buffer_size; i++) {
        if (node->buffer->partial_position_buffer[i] == nullptr) continue;
        float distmin = sofa_minidist(bins, fft,
            node->buffer->partial_sax_buffer[i],
            index->settings->max_sax_cardinalities,
            index->settings->sax_bit_cardinality,
            index->settings->sax_alphabet_cardinality,
            index->settings->paa_segments, is_norm);
        if (distmin <= r) {
            float dist = ts_euclidean_distance_SIMD(query,
                &rawfile[*node->buffer->partial_position_buffer[i]],
                index->settings->timeseries_size, r);
            if (dist <= r) {
                pthread_rwlock_wrlock(lock_results);
                results->emplace_back(dist,
                    (idx_t)(*node->buffer->partial_position_buffer[i] /
                        index->settings->timeseries_size));
                pthread_rwlock_unlock(lock_results);
            }
        }
    }
}

void *SOFA_range_search_worker(void *rfdata)
{
    SOFA_workerdata *d = (SOFA_workerdata *)rfdata;
    isax_index *index = d->index;
    float *fft = d->fft;
    ts_type *ts = d->ts;
    float r = d->r;
    int n_pqueue = d->n_pqueue;
    bool is_norm = d->is_norm;
    float **bins = d->bins;
    float *rawfile = d->rawfile;

    int tnumber = rand() % n_pqueue;
    int startq = d->startqueuenumber;

    while (1) {
        int cur = __sync_fetch_and_add(d->node_counter, 1);
        if (cur >= d->amountnode) break;
        insert_tree_node_sofa(bins, is_norm, fft,
            d->nodelist[cur], index, r,
            d->allpq, d->alllock, &tnumber, n_pqueue);
    }

    pthread_barrier_wait(d->lock_barrier);

    query_result *n;
    while (1) {
        pthread_mutex_lock(&d->alllock[startq]);
        n = (query_result *)pqueue_pop(d->allpq[startq]);
        pthread_mutex_unlock(&d->alllock[startq]);
        if (n == NULL) break;
        if (n->distance > r) { free(n); break; }
        if (n->node->is_leaf)
            calculate_node_range_sofa(index, n->node, ts, fft, bins, is_norm,
                                      r, d->range_results, d->lock_range_results, rawfile);
        free(n);
    }

    if (d->allqueuelabel[startq] == 1) {
        d->allqueuelabel[startq] = 0;
        pthread_mutex_lock(&d->alllock[startq]);
        while ((n = (query_result *)pqueue_pop(d->allpq[startq]))) free(n);
        pthread_mutex_unlock(&d->alllock[startq]);
    }

    while (1) {
        bool finished = true;
        for (int i = 0; i < n_pqueue; i++) {
            if (d->allqueuelabel[i] != 1) continue;
            finished = false;
            while (1) {
                pthread_mutex_lock(&d->alllock[i]);
                n = (query_result *)pqueue_pop(d->allpq[i]);
                pthread_mutex_unlock(&d->alllock[i]);
                if (n == NULL) break;
                if (n->distance > r) { free(n); break; }
                if (n->node->is_leaf)
                    calculate_node_range_sofa(index, n->node, ts, fft, bins, is_norm,
                                              r, d->range_results, d->lock_range_results, rawfile);
                free(n);
            }
            d->allqueuelabel[i] = 0;
        }
        if (finished) break;
    }
    return nullptr;
}

// ---- Sofa class implementation ----

static void sofa_approximate_topk(
    ts_type *ts, float *fft, isax_index *index, pqueue_bsf *pq_bsf,
    float *rawfile, float **bins, int paa_segments, int cardinality, bool is_norm)
{
    sax_type *sax = (sax_type *)malloc(sizeof(sax_type) * paa_segments);
    sofa_sfa_from_fft(fft, sax, bins, paa_segments, cardinality);

    root_mask_type root_mask = 0;
    CREATE_MASK(root_mask, index, sax);

    if ((&((parallel_first_buffer_layer *)(index->fbl))->soft_buffers[(int)root_mask])->initialized) {
        isax_node *node =
            (&((parallel_first_buffer_layer *)(index->fbl))->soft_buffers[(int)root_mask])->node;
        while (node != NULL && !node->is_leaf) {
            if (node->split_data == NULL) break;
            int location = index->settings->sax_bit_cardinality - 1 -
                           node->split_data->split_mask[node->split_data->splitpoint];
            root_mask_type mask = index->settings->bit_masks[location];
            node = (sax[node->split_data->splitpoint] & mask)
                 ? node->right_child : node->left_child;
        }
        if (node != NULL)
            calculate_node_topk_inmemory(index, node, ts, pq_bsf, rawfile);
    }
    free(sax);
}

static void sofa_refine_topk(
    ts_type *ts, float *fft, isax_index *index, pqueue_bsf *pq_bsf,
    float minimum_distance, int limit, float *rawfile,
    float **bins, bool is_norm)
{
    pqueue_t *pq = pqueue_init(index->settings->root_nodes_size,
                               cmp_pri, get_pri, set_pri, get_pos, set_pos);

    for (isax_node *cur = index->first_node; cur != NULL; cur = cur->next) {
        if (cur->isax_values == NULL || cur->isax_cardinalities == NULL) continue;
        query_result *r = (query_result *)malloc(sizeof(query_result));
        r->distance = sofa_minidist(bins, fft,
            cur->isax_values, cur->isax_cardinalities,
            index->settings->sax_bit_cardinality,
            index->settings->sax_alphabet_cardinality,
            index->settings->paa_segments, is_norm);
        r->node = cur;
        pqueue_insert(pq, r);
    }

    query_result *n;
    while ((n = (query_result *)pqueue_pop(pq))) {
        if (n->distance >= pq_bsf->knn[pq_bsf->k - 1] || n->distance > minimum_distance) {
            pqueue_insert(pq, n);
            break;
        }
        if (n->node->is_leaf) {
            calculate_node_topk_inmemory(index, n->node, ts, pq_bsf, rawfile);
            if (pq_bsf->knn[pq_bsf->k - 1] < FLT_MAX) {
                pqueue_insert(pq, n);
                break;
            }
        } else {
            auto push_child = [&](isax_node *child) {
                if (child == NULL || child->isax_values == NULL ||
                    child->isax_cardinalities == NULL) return;
                query_result *cr = (query_result *)malloc(sizeof(query_result));
                cr->distance = sofa_minidist(bins, fft,
                    child->isax_values, child->isax_cardinalities,
                    index->settings->sax_bit_cardinality,
                    index->settings->sax_alphabet_cardinality,
                    index->settings->paa_segments, is_norm);
                cr->node = child;
                pqueue_insert(pq, cr);
            };
            push_child(n->node->left_child);
            push_child(n->node->right_child);
        }
        free(n);
    }
    while ((n = (query_result *)pqueue_pop(pq))) free(n);
    pqueue_free(pq);
}

Sofa::Sofa(DistanceType distance_type)
    : Sofa(distance_type, SofaConfig{}) {}

Sofa::Sofa(DistanceType distance_type, const SofaConfig &config)
    : SimilaritySearchAlgorithm(distance_type)
{
    search_workers = config.search_workers;
    index_workers = config.index_workers;
    leaf_size = config.leaf_size;
    min_leaf_size = config.min_leaf_size;
    initial_lbl_size = config.leaf_size;
    paa_segments = config.word_length;
    sax_cardinality= config.alphabet_size;
    sample_size = config.sample_size;
    histogram_type = config.histogram_type;
    coeff_number = config.coeff_number;
    is_norm = config.is_norm;
}

pqueue_bsf Sofa::sofaSearchTopkL2Squared(float *ts, float *fft, node_list *nodelist, idx_t k)
{
    pqueue_bsf *pq_bsf = pqueue_bsf_init(k);

    sofa_approximate_topk(ts, fft, index, pq_bsf, this->database,
                          bins, paa_segments, sax_cardinality, is_norm);
    this->minimum_distance = pq_bsf->knn[k - 1];

    if (this->minimum_distance == FLT_MAX || min_checked_leaves > 1) {
        sofa_refine_topk(ts, fft, index, pq_bsf, this->minimum_distance,
                        this->min_checked_leaves, this->database, bins, is_norm);
        this->minimum_distance = pq_bsf->knn[k - 1];
    }

    int node_counter = 0;
    pqueue_t                    **allpq = (pqueue_t **)malloc(sizeof(pqueue_t *) * n_pqueue);
    std::vector<pthread_mutex_t>  ququelock(n_pqueue);
    std::vector<int>              queuelabel(n_pqueue);
    std::vector<pthread_t>        tids(search_workers);
    std::vector<SOFA_workerdata>  workerdata(search_workers);
    pthread_mutex_t  lock_queue = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t  lock_current_root = PTHREAD_MUTEX_INITIALIZER;
    pthread_rwlock_t lock_bsf = PTHREAD_RWLOCK_INITIALIZER;
    pthread_barrier_t lock_barrier;
    pthread_barrier_init(&lock_barrier, nullptr, search_workers);

    for (int i = 0; i < n_pqueue; i++) {
        allpq[i] = pqueue_init(index->settings->root_nodes_size / n_pqueue,
                               cmp_pri, get_pri, set_pri, get_pos, set_pos);
        pthread_mutex_init(&ququelock[i], nullptr);
        queuelabel[i] = 1;
    }

    for (int i = 0; i < search_workers; i++) {
        workerdata[i].fft = fft;
        workerdata[i].ts = ts;
        workerdata[i].lock_queue = &lock_queue;
        workerdata[i].lock_current_root_node= &lock_current_root;
        workerdata[i].lock_bsf = &lock_bsf;
        workerdata[i].nodelist = nodelist->nlist;
        workerdata[i].amountnode = nodelist->node_amount;
        workerdata[i].index = index;
        workerdata[i].minimum_distance = minimum_distance;
        workerdata[i].node_counter = &node_counter;
        workerdata[i].pq = allpq[i];
        workerdata[i].lock_barrier = &lock_barrier;
        workerdata[i].alllock = ququelock.data();
        workerdata[i].allqueuelabel = queuelabel.data();
        workerdata[i].allpq = allpq;
        workerdata[i].startqueuenumber = i % n_pqueue;
        workerdata[i].pq_bsf = pq_bsf;
        workerdata[i].n_pqueue = n_pqueue;
        workerdata[i].rawfile = this->database;
        workerdata[i].bins = this->bins;
        workerdata[i].is_norm = this->is_norm;
    }

    for (int i = 0; i < search_workers; i++)
        pthread_create(&tids[i], nullptr, SOFA_topk_search_worker, &workerdata[i]);
    for (int i = 0; i < search_workers; i++)
        pthread_join(tids[i], nullptr);

    this->minimum_distance = pq_bsf->knn[k - 1];
    pthread_barrier_destroy(&lock_barrier);
    for (int i = 0; i < n_pqueue; i++) {
        query_result *r;
        while ((r = (query_result *)pqueue_pop(allpq[i]))) free(r);
        pqueue_free(allpq[i]);
    }
    free(allpq);

    pqueue_bsf result = *pq_bsf;
    free(pq_bsf);
    return result;
}

void Sofa::searchIndexL2Squared(const float *query, idx_t n_query, idx_t k, idx_t *I, float *D)
{
    node_list nodelist;
    nodelist.nlist = (isax_node **)malloc(
        sizeof(isax_node *) * (size_t)pow(2, index->settings->paa_segments));
    nodelist.node_amount = 0;
    for (isax_node *cur = index->first_node; cur != NULL; cur = cur->next)
        nodelist.nlist[nodelist.node_amount++] = cur;

    float norm_factor = std::sqrt(2.0f / (float)this->dim);
    int ts_length = index->settings->timeseries_size;

    float         *ts_buf = (float *)fftwf_malloc(sizeof(float) * ts_length);
    fftwf_complex *ts_out = (fftwf_complex *)fftwf_malloc(
                                sizeof(fftwf_complex) * (ts_length / 2 + 1));
    fftwf_plan plan = fftwf_plan_dft_r2c_1d(ts_length, ts_buf, ts_out, FFTW_ESTIMATE);
    float *fft_buf = (float *)fftwf_malloc(sizeof(float) * ts_length);

    for (idx_t q = 0; q < n_query; q++) {
        const float *ts = query + q * this->dim;

        memcpy(ts_buf, ts, sizeof(float) * ts_length);
        sofa_fft_from_ts(ts_length, norm_factor, is_norm,
                         ts_out, fft_buf, plan, paa_segments);

        pqueue_bsf result = sofaSearchTopkL2Squared((float *)ts, fft_buf, &nodelist, k);

        std::vector<std::pair<float, long>> pairs;
        pairs.reserve(k);
        for (idx_t ik = 0; ik < k; ik++)
            if (result.position[ik] >= 0 && result.knn[ik] < FLT_MAX * 0.99f)
                pairs.emplace_back(result.knn[ik], result.position[ik]);

        std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b) {
            return a.first != b.first ? a.first < b.first : a.second < b.second;
        });

        std::unordered_set<long> seen;
        std::vector<std::pair<float, long>> uniq;
        uniq.reserve(pairs.size());
        for (const auto &p : pairs)
            if (seen.insert(p.second).second) uniq.push_back(p);

        long last_pos = 0;
        float last_dist = 0.0f;
        for (idx_t ik = 0; ik < k; ik++) {
            if (ik < (idx_t)uniq.size()) {
                last_dist = uniq[ik].first;
                last_pos = uniq[ik].second;
            }
            I[q * k + ik] = (idx_t)(last_pos >= 0 ? last_pos : 0);
            D[q * k + ik] = last_dist;
        }

        free(result.position);
        free(result.knn);
        free(result.node);
    }

    fftwf_destroy_plan(plan);
    fftwf_free(ts_buf);
    fftwf_free(ts_out);
    fftwf_free(fft_buf);
    free(nodelist.nlist);
    fprintf(stderr, ">>> SOFA: Finished querying.\n");
}

std::vector<std::pair<float, idx_t>> Sofa::sofaSearchRangeL2Squared(
    float *ts, float *fft, node_list *nodelist, float r)
{
    std::vector<std::pair<float, idx_t>> results;
    pthread_rwlock_t lock_results = PTHREAD_RWLOCK_INITIALIZER;

    int node_counter = 0;
    pqueue_t                    **allpq = (pqueue_t **)malloc(sizeof(pqueue_t *) * n_pqueue);
    std::vector<pthread_mutex_t>  ququelock(n_pqueue);
    std::vector<int>              queuelabel(n_pqueue);
    std::vector<pthread_t>        tids(search_workers);
    std::vector<SOFA_workerdata>  workerdata(search_workers);
    pthread_mutex_t  lock_queue = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t  lock_current_root = PTHREAD_MUTEX_INITIALIZER;
    pthread_barrier_t lock_barrier;
    pthread_barrier_init(&lock_barrier, nullptr, search_workers);

    for (int i = 0; i < n_pqueue; i++) {
        allpq[i] = pqueue_init(index->settings->root_nodes_size / n_pqueue,
                               cmp_pri, get_pri, set_pri, get_pos, set_pos);
        pthread_mutex_init(&ququelock[i], nullptr);
        queuelabel[i] = 1;
    }

    for (int i = 0; i < search_workers; i++) {
        workerdata[i].fft = fft;
        workerdata[i].ts = ts;
        workerdata[i].lock_queue = &lock_queue;
        workerdata[i].lock_current_root_node = &lock_current_root;
        workerdata[i].lock_bsf = nullptr;
        workerdata[i].nodelist = nodelist->nlist;
        workerdata[i].amountnode = nodelist->node_amount;
        workerdata[i].index = index;
        workerdata[i].minimum_distance = r;
        workerdata[i].node_counter = &node_counter;
        workerdata[i].pq = allpq[i];
        workerdata[i].lock_barrier = &lock_barrier;
        workerdata[i].alllock = ququelock.data();
        workerdata[i].allqueuelabel = queuelabel.data();
        workerdata[i].allpq = allpq;
        workerdata[i].startqueuenumber = i % n_pqueue;
        workerdata[i].pq_bsf = nullptr;
        workerdata[i].n_pqueue = n_pqueue;
        workerdata[i].rawfile = this->database;
        workerdata[i].bins = this->bins;
        workerdata[i].is_norm = this->is_norm;
        workerdata[i].r = r;
        workerdata[i].range_results = &results;
        workerdata[i].lock_range_results = &lock_results;
    }

    for (int i = 0; i < search_workers; i++)
        pthread_create(&tids[i], nullptr, SOFA_range_search_worker, &workerdata[i]);
    for (int i = 0; i < search_workers; i++)
        pthread_join(tids[i], nullptr);

    pthread_barrier_destroy(&lock_barrier);
    pthread_rwlock_destroy(&lock_results);
    for (int i = 0; i < n_pqueue; i++) {
        query_result *r_item;
        while ((r_item = (query_result *)pqueue_pop(allpq[i]))) free(r_item);
        pqueue_free(allpq[i]);
    }
    free(allpq);

    return results;
}

void Sofa::searchIndexRangeL2Squared(const float *query, idx_t n_query, float r,
                                      std::vector<std::vector<idx_t>> &I,
                                      std::vector<std::vector<float>> &D)
{
    node_list nodelist;
    nodelist.nlist = (isax_node **)malloc(
        sizeof(isax_node *) * (size_t)pow(2, index->settings->paa_segments));
    nodelist.node_amount = 0;
    for (isax_node *cur = index->first_node; cur != NULL; cur = cur->next)
        nodelist.nlist[nodelist.node_amount++] = cur;

    float norm_factor = std::sqrt(2.0f / (float)this->dim);
    int ts_length = index->settings->timeseries_size;

    float         *ts_buf = (float *)fftwf_malloc(sizeof(float) * ts_length);
    fftwf_complex *ts_out = (fftwf_complex *)fftwf_malloc(
                                sizeof(fftwf_complex) * (ts_length / 2 + 1));
    fftwf_plan plan = fftwf_plan_dft_r2c_1d(ts_length, ts_buf, ts_out, FFTW_ESTIMATE);
    float *fft_buf = (float *)fftwf_malloc(sizeof(float) * ts_length);

    I.resize(n_query);
    D.resize(n_query);

    for (idx_t q = 0; q < n_query; q++) {
        const float *ts = query + q * this->dim;

        memcpy(ts_buf, ts, sizeof(float) * ts_length);
        sofa_fft_from_ts(ts_length, norm_factor, is_norm,
                         ts_out, fft_buf, plan, paa_segments);

        auto hits = sofaSearchRangeL2Squared((float *)ts, fft_buf, &nodelist, r);

        std::sort(hits.begin(), hits.end());
        I[q].resize(hits.size());
        D[q].resize(hits.size());
        for (size_t j = 0; j < hits.size(); j++) {
            D[q][j] = hits[j].first;
            I[q][j] = hits[j].second;
        }
    }

    fftwf_destroy_plan(plan);
    fftwf_free(ts_buf);
    fftwf_free(ts_out);
    fftwf_free(fft_buf);
    free(nodelist.nlist);
}

void Sofa::searchIndexDTW(const float *query, idx_t n_query, idx_t k, idx_t *I, float *D)
{
    throw std::runtime_error("SOFA does not support DTW distance.");
}

void Sofa::searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D)
{
    if (this->distance_type == DistanceType::L2_SQUARED)
        searchIndexL2Squared(query, n_query, k, I, D);
    else if (this->distance_type == DistanceType::DTW)
        searchIndexDTW(query, n_query, k, I, D);
    else
        throw std::runtime_error("SOFA: unsupported distance type.");
}

void Sofa::searchIndex(const float *query, idx_t n_query, const SearchConfig &config,
                       std::vector<std::vector<idx_t>> &I,
                       std::vector<std::vector<float>> &D)
{
    if (config.type == QueryType::TOP_K) {
        SimilaritySearchAlgorithm::searchIndex(query, n_query, config, I, D);
        return;
    }
    if (this->distance_type != DistanceType::L2_SQUARED)
        throw std::runtime_error("SOFA range search only supports L2_SQUARED.");
    searchIndexRangeL2Squared(query, n_query, config.r, I, D);
}

Sofa::~Sofa()
{
    if (owns_database && database != nullptr)
        delete[] database;

    sfaFreeBins();

    if (index != nullptr) {
        if (index->sax_cache) free(index->sax_cache);
        if (index->answer)    free(index->answer);
        if (index->fbl)       destroy_parallel_fbl((parallel_first_buffer_layer *)index->fbl);
        if (index->sax_file)  fclose(index->sax_file);
        free(index);
    }

    if (index_settings != nullptr) {
        if (index_settings->bit_masks)              free(index_settings->bit_masks);
        if (index_settings->max_sax_cardinalities)  free(index_settings->max_sax_cardinalities);
        free(index_settings);
    }
}

void Sofa::sfaBinsInit()
{
    bins = new float *[paa_segments];
    for (int i = 0; i < paa_segments; ++i) {
        bins[i] = new float[sax_cardinality - 1];
        for (int j = 0; j < sax_cardinality - 1; ++j)
            bins[i][j] = FLT_MAX;
    }
}

void Sofa::sfaFreeBins()
{
    if (bins) {
        for (int i = 0; i < paa_segments; ++i) delete[] bins[i];
        delete[] bins;
        bins = nullptr;
    }
    delete[] coefficients;
    coefficients = nullptr;
}

void Sofa::sfaSetBins()
{
    unsigned int actual_sample =
        (unsigned int)std::min((long)sample_size, (long)n_database);
    float norm_factor = std::sqrt(2.0f / (float)dim);

    float **dft_mem = new float *[paa_segments];
    for (int k = 0; k < paa_segments; ++k)
        dft_mem[k] = new float[actual_sample]();

    int nw = index_workers;
    long per_worker = actual_sample / nw;

    SOFA_bins_worker *bw = new SOFA_bins_worker[nw];
    pthread_t        *tids = new pthread_t[nw];

    for (int i = 0; i < nw; i++) {
        bw[i].index = index;
        bw[i].dft_mem_array = dft_mem;
        bw[i].raw_database = database;
        bw[i].start_number = (unsigned long)i * per_worker;
        bw[i].stop_number = (unsigned long)(i + 1) * per_worker;
        bw[i].records = per_worker;
        bw[i].records_offset = per_worker;
        bw[i].workernumber = i;
        bw[i].norm_factor = norm_factor;
        bw[i].is_norm = is_norm;
        bw[i].coeff_number = paa_segments;

        bw[i].ts_buf = (float *)fftwf_malloc(sizeof(float) * dim);
        bw[i].ts_out = (fftwf_complex *)fftwf_malloc(
                                   sizeof(fftwf_complex) * (dim / 2 + 1));
        bw[i].plan_forward = fftwf_plan_dft_r2c_1d(
                                   (int)dim, bw[i].ts_buf, bw[i].ts_out, FFTW_ESTIMATE);
        bw[i].fft_transform = (float *)fftwf_malloc(sizeof(float) * dim);
    }
    bw[nw - 1].records = actual_sample - (long)(nw - 1) * per_worker;
    bw[nw - 1].stop_number = actual_sample;

    for (int i = 0; i < nw; i++)
        pthread_create(&tids[i], nullptr, sofa_bins_worker, &bw[i]);
    for (int i = 0; i < nw; i++)
        pthread_join(tids[i], nullptr);

    for (int i = 0; i < nw; i++) {
        fftwf_destroy_plan(bw[i].plan_forward);
        fftwf_free(bw[i].ts_buf);
        fftwf_free(bw[i].ts_out);
        fftwf_free(bw[i].fft_transform);
    }

    SOFA_divide_worker_data *dw = new SOFA_divide_worker_data[nw];
    long segs_pw = paa_segments / nw;
    for (int i = 0; i < nw; i++) {
        dw[i].dft_mem_array = dft_mem;
        dw[i].bins = bins;
        dw[i].start_number = (unsigned long)i * segs_pw;
        dw[i].stop_number = (i < nw - 1)
                             ? (unsigned long)(i + 1) * segs_pw
                             : (unsigned long)paa_segments;
        dw[i].sample_size = actual_sample;
        dw[i].num_symbols = sax_cardinality;
        dw[i].histogram_type= histogram_type;
        pthread_create(&tids[i], nullptr, sofa_divide_worker, &dw[i]);
    }
    for (int i = 0; i < nw; i++)
        pthread_join(tids[i], nullptr);

    for (int k = 0; k < paa_segments; ++k) delete[] dft_mem[k];
    delete[] dft_mem;
    delete[] bw;
    delete[] dw;
    delete[] tids;

    fprintf(stderr, ">>> SOFA: Finished binning (sample=%u, %s)\n",
            actual_sample, histogram_type == 1 ? "equi-depth" : "equi-width");
}

void Sofa::buildIndex(DataSource *data_source)
{
    this->dim = data_source->getDim();
    this->n_database = data_source->getTotalRecords();

    if (this->n_database == 0) {
        data_source->reset();
        idx_t count = 0;
        float *dummy = new float[this->dim];
        while (data_source->nextRecord(dummy)) count++;
        delete[] dummy;
        this->n_database = count;
        data_source->reset();
    }

    data_source->reset();
    const float *raw = data_source->rawPointer();
    if (raw != nullptr) {
        this->database = const_cast<float *>(raw);
        this->owns_database = false;
    } else {
        this->database = new float[this->n_database * this->dim];
        float *record = new float[this->dim];
        idx_t idx = 0;
        while (data_source->nextRecord(record))
            std::copy(record, record + this->dim, this->database + idx++ * this->dim);
        delete[] record;
        this->owns_database = true;
    }

    int sax_bit_card = static_cast<int>(std::round(std::log2(static_cast<double>(this->sax_cardinality))));
    this->index_settings = isax_index_settings_init("",
        this->dim, this->paa_segments, sax_bit_card,
        this->leaf_size, this->min_leaf_size,
        this->initial_lbl_size, this->flush_limit,
        this->initial_fbl_size, this->total_loaded_leaves,
        this->tight_bound, 0, 1, 1);

    this->index = isax_index_init_inmemory(this->index_settings);
    isax_index *index = this->index;
    index->sax_file = NULL;

    sfaBinsInit();
    sfaSetBins();

    float norm_factor = std::sqrt(2.0f / (float)this->dim);
    int read_block_length = this->read_block_length;
    unsigned long shared_start = 0;
    int node_counter = 0;

    index->sax_cache = (sax_type *)malloc(
        sizeof(sax_type) * index->settings->paa_segments * this->n_database);

    pthread_barrier_t lock_barrier1, lock_barrier2;
    pthread_barrier_init(&lock_barrier1, nullptr, this->index_workers + 1);
    pthread_barrier_init(&lock_barrier2, nullptr, this->index_workers + 1);
    pthread_mutex_t lock_firstnode = PTHREAD_MUTEX_INITIALIZER;

    destroy_fbl(index->fbl);
    index->fbl = (first_buffer_layer *)initialize_pRecBuf(
        index->settings->initial_fbl_buffer_size,
        (int)pow(2, index->settings->paa_segments),
        index->settings->max_total_buffer_size +
            DISK_BUFFER_SIZE * (PROGRESS_CALCULATE_THREAD_NUMBER - 1),
        index, this->index_workers);

    int *nodeid = (int *)malloc(sizeof(int) * index->fbl->number_of_buffers);
    int *nodesize = (int *)malloc(sizeof(int) * index->fbl->number_of_buffers);

    pthread_t        *tids = new pthread_t[this->index_workers];
    SOFA_index_worker *iw = new SOFA_index_worker[this->index_workers];

    for (int i = 0; i < this->index_workers; i++) {
        iw[i].index = index;
        iw[i].raw_database = this->database;
        iw[i].shared_start_number = &shared_start;
        iw[i].stop_number = this->n_database;
        iw[i].workernumber = i;
        iw[i].total_workernumber = this->index_workers;
        iw[i].lock_barrier1 = &lock_barrier1;
        iw[i].lock_barrier2 = &lock_barrier2;
        iw[i].lock_firstnode = &lock_firstnode;
        iw[i].node_counter = &node_counter;
        iw[i].nodeid = nodeid;
        iw[i].read_block_length = read_block_length;
        iw[i].bins = this->bins;
        iw[i].norm_factor = norm_factor;
        iw[i].is_norm = this->is_norm;

        iw[i].ts_buf = (float *)fftwf_malloc(sizeof(float) * this->dim);
        iw[i].ts_out = (fftwf_complex *)fftwf_malloc(
                                   sizeof(fftwf_complex) * (this->dim / 2 + 1));
        iw[i].plan_forward = fftwf_plan_dft_r2c_1d(
                                   (int)this->dim, iw[i].ts_buf, iw[i].ts_out, FFTW_ESTIMATE);
        iw[i].fft_transform = (float *)fftwf_malloc(sizeof(float) * this->dim);

        pthread_create(&tids[i], nullptr, sofa_index_creation_worker, &iw[i]);
    }

    pthread_barrier_wait(&lock_barrier1);
    pthread_barrier_wait(&lock_barrier2);

    for (int i = 0; i < this->index_workers; i++)
        pthread_join(tids[i], nullptr);

    for (int i = 0; i < this->index_workers; i++) {
        fftwf_destroy_plan(iw[i].plan_forward);
        fftwf_free(iw[i].ts_buf);
        fftwf_free(iw[i].ts_out);
        fftwf_free(iw[i].fft_transform);
    }

    __sync_fetch_and_add(&index->total_records, this->n_database);
    index->sax_cache_size = index->total_records;
    fprintf(stderr, ">>> SOFA: Finished indexing\n");

    pthread_barrier_destroy(&lock_barrier1);
    pthread_barrier_destroy(&lock_barrier2);

    delete[] tids;
    delete[] iw;
    free(nodeid);
    free(nodesize);
}

} // namespace daisy

#endif // SOFA_FFTW_ENABLED
