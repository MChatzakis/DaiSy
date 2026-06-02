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
    static inline float sofa_lb_segment(
        const float *bins_row, float fft_val,
        sax_type sax_val, sax_type sax_card, sax_type max_bit_card,
        int max_card, float factor)
    {
        int rl = (int)sax_val << (max_bit_card - sax_card);
        unsigned int shift = (unsigned int)(max_bit_card - sax_card);
        int mask = (shift < 32u) ? ~(0xFFFFFFFF << shift) : 0;
        int ru   = mask | rl;

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
                                    max_bit_card, max_card, 1.0f);
            i = 2;
        }
        for (; i < n_segments; i++)
            dist += sofa_lb_segment(bins[i], fft[i], sax[i], sax_cards[i],
                                    max_bit_card, max_card, 2.0f);
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
                r->node     = node;
                r->distance = distance;
                pthread_mutex_lock(&lock_queue[*tnumber]);
                pqueue_insert(pq[*tnumber], r);
                pthread_mutex_unlock(&lock_queue[*tnumber]);
                *tnumber = (*tnumber + 1) % n_pqueue;
            } else {
                if (node->left_child  &&
                    node->left_child->isax_values  &&
                    node->left_child->isax_cardinalities)
                    insert_tree_node_sofa(bins, is_norm, fft, node->left_child,
                                          index, bsf, pq, lock_queue, tnumber, n_pqueue);
                if (node->right_child &&
                    node->right_child->isax_values &&
                    node->right_child->isax_cardinalities)
                    insert_tree_node_sofa(bins, is_norm, fft, node->right_child,
                                          index, bsf, pq, lock_queue, tnumber, n_pqueue);
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
        SOFA_workerdata *d  = (SOFA_workerdata *)rfdata;
        isax_index *index   = d->index;
        float  *fft = d->fft;
        ts_type    *ts  = d->ts;
        pqueue_bsf *pq_bsf  = d->pq_bsf;
        float minimum_distance = d->minimum_distance;
        int   n_pqueue = d->n_pqueue;
        bool  is_norm  = d->is_norm;
        float **bins = d->bins;
        float  *rawfile     = d->rawfile;

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
        leaf_size  = config.leaf_size;
        min_leaf_size = config.min_leaf_size;
        initial_lbl_size = config.leaf_size;
        paa_segments = config.word_length;
        sax_cardinality = config.alphabet_size;
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
        pqueue_t                    **allpq     = (pqueue_t **)malloc(sizeof(pqueue_t *) * n_pqueue);
        std::vector<pthread_mutex_t>  ququelock(n_pqueue);
        std::vector<int>  queuelabel(n_pqueue);
        std::vector<pthread_t>  tids(search_workers);
        std::vector<SOFA_workerdata>  workerdata(search_workers);
        pthread_mutex_t  lock_queue  = PTHREAD_MUTEX_INITIALIZER;
        pthread_mutex_t  lock_current_root = PTHREAD_MUTEX_INITIALIZER;
        pthread_rwlock_t lock_bsf  = PTHREAD_RWLOCK_INITIALIZER;
        pthread_barrier_t lock_barrier;
        pthread_barrier_init(&lock_barrier, nullptr, search_workers);

        for (int i = 0; i < n_pqueue; i++) {
            allpq[i] = pqueue_init(index->settings->root_nodes_size / n_pqueue,
                                   cmp_pri, get_pri, set_pri, get_pos, set_pos);
            pthread_mutex_init(&ququelock[i], nullptr);
            queuelabel[i] = 1;
        }

        for (int i = 0; i < search_workers; i++) {
            workerdata[i].fft  = fft;
            workerdata[i].ts  = ts;
            workerdata[i].lock_queue  = &lock_queue;
            workerdata[i].lock_current_root_node = &lock_current_root;
            workerdata[i].lock_bsf  = &lock_bsf;
            workerdata[i].nodelist  = nodelist->nlist;
            workerdata[i].amountnode = nodelist->node_amount;
            workerdata[i].index = index;
            workerdata[i].minimum_distance = minimum_distance;
            workerdata[i].node_counter  = &node_counter;
            workerdata[i].pq  = allpq[i];
            workerdata[i].lock_barrier = &lock_barrier;
            workerdata[i].alllock  = ququelock.data();
            workerdata[i].allqueuelabel = queuelabel.data();
            workerdata[i].allpq = allpq;
            workerdata[i].startqueuenumber = i % n_pqueue;
            workerdata[i].pq_bsf  = pq_bsf;
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
        int   ts_length = index->settings->timeseries_size;

        float         *ts_buf = (float *)fftwf_malloc(sizeof(float) * ts_length);
        fftwf_complex *ts_out = (fftwf_complex *)fftwf_malloc(
                                    sizeof(fftwf_complex) * (ts_length / 2 + 1));
        fftwf_plan plan  = fftwf_plan_dft_r2c_1d(ts_length, ts_buf, ts_out, FFTW_ESTIMATE);
        float *fft_buf  = (float *)fftwf_malloc(sizeof(float) * ts_length);

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

    void Sofa::searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D)
    {
        if (this->distance_type == DistanceType::L2_SQUARED) {
            searchIndexL2Squared(query, n_query, k, I, D);
        } else {
            fprintf(stderr, "Error: SOFA only supports L2_SQUARED distance.\n");
            exit(1);
        }
    }

    Sofa::~Sofa()
    {
        if (owns_database && database != nullptr)
            delete[] database;

        sfaFreeBins();

        if (index != nullptr) {
            if (index->sax_cache) free(index->sax_cache);
            if (index->answer)  free(index->answer);
            if (index->fbl)   destroy_parallel_fbl((parallel_first_buffer_layer *)index->fbl);
            if (index->sax_file)  fclose(index->sax_file);
            free(index);
        }

        if (index_settings != nullptr) {
            if (index_settings->bit_masks)  free(index_settings->bit_masks);
            if (index_settings->max_sax_cardinalities) free(index_settings->max_sax_cardinalities);
            free(index_settings);
        }
    }

}


#endif
